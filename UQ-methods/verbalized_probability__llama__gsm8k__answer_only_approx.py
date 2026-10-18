#!/usr/bin/env python3
# Reference repository: https://github.com/IINemo/lm-polygraph

from __future__ import annotations

if __name__ == "__main__" and not globals().get("_UQ_EXECUTING", False):
    
    import ast as _uq_ast
    from pathlib import Path as _uq_Path
    _uq_tree = _uq_ast.parse(_uq_Path(__file__).read_text())
    _uq_helper = next(n for n in _uq_tree.body if isinstance(n, _uq_ast.FunctionDef) and n.name == "_uq_standalone")
    exec(compile(_uq_ast.Module(body=[_uq_helper], type_ignores=[]), __file__, "exec"), globals())
    _uq_standalone()
    raise SystemExit(0)


import argparse
import json
import math
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from math_verify import parse, verify


BINS: List[Tuple[str, float]] = [
    ("A", 0.0),
    ("B", 0.1),
    ("C", 0.2),
    ("D", 0.3),
    ("E", 0.4),
    ("F", 0.5),
    ("G", 0.6),
    ("H", 0.7),
    ("I", 0.8),
    ("J", 0.9),
    ("K", 1.0),
]





def safe_print(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def load_records(path: str) -> List[Dict[str, Any]]:
    
    text = open(path, "r", encoding="utf-8").read().strip()
    if not text:
        return []

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass

    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception as e:
                safe_print(f"[warn] skip bad json line {line_no}: {e}", file=sys.stderr)
    return out


def load_done_ids(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "id" in obj and "error" not in obj:
                    done.add(str(obj["id"]))
            except Exception:
                continue
    return done


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()





def get_id(record: Dict[str, Any], idx: int) -> str:
    for k in ("id", "question_id", "qid"):
        if record.get(k) is not None:
            return str(record[k])
    return f"line_{idx}"


def get_question(record: Dict[str, Any]) -> str:
    for k in ("question", "problem", "prompt", "input"):
        if record.get(k) is not None:
            return str(record[k]).strip()
    return ""


def normalize_ground_truth(gt_field: Any) -> Dict[str, Any]:
    if isinstance(gt_field, str):
        value = gt_field.strip()
        return {"value": value, "aliases": [value]}
    if isinstance(gt_field, dict):
        value = str(gt_field.get("value", "")).strip()
        aliases = [str(x).strip() for x in gt_field.get("aliases", []) if str(x).strip()]
        if value and value not in aliases:
            aliases = [value] + aliases
        return {"value": value, "aliases": aliases}
    value = str(gt_field).strip()
    return {"value": value, "aliases": [value]}


def get_gt(record: Dict[str, Any]) -> Dict[str, Any]:
    for k in ("gt", "ground_truth"):
        if record.get(k) is not None:
            return normalize_ground_truth(record[k])

    for k in ("correct answer", "correct_answer", "answer", "gold", "target", "label"):
        if record.get(k) is not None:
            return normalize_ground_truth(record[k])

    return {"value": "", "aliases": [""]}


def extract_text_field(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        for key in ("text", "raw_text", "answer", "response", "final_answer", "llm answer", "llm_answer"):
            if x.get(key) is not None:
                return str(x[key]).strip()
    return str(x).strip()


def get_greedy_answer(record: Dict[str, Any]) -> str:
    if record.get("greedy_answer") is not None:
        return str(record["greedy_answer"]).strip()
    if record.get("greedy") is not None:
        text = extract_text_field(record["greedy"])
        if text:
            return text
    
    responses = record.get("responses")
    if isinstance(responses, list) and responses:
        return extract_text_field(responses[0])
    return ""


def get_target_reasoning(record: Dict[str, Any]) -> str:
    
    greedy = record.get("greedy")
    if isinstance(greedy, dict):
        for k in ("reason", "rationale", "reasoning_augmented", "raw_text", "text"):
            if greedy.get(k) is not None:
                return str(greedy[k]).strip()

    responses = record.get("responses")
    if isinstance(responses, list) and responses:
        r = responses[0]
        if isinstance(r, dict):
            for k in ("llm reason", "llm_reason", "reason", "rationale", "text"):
                if r.get(k) is not None:
                    return str(r[k]).strip()
    return ""





def normalize_number_string(text: str) -> str:
    text = str(text).strip().replace(",", "").replace("$", "")
    try:
        val = float(text)
        if val.is_integer():
            return str(int(val))
        return str(val)
    except Exception:
        return text.strip()


def extract_numeric_answer(text: str) -> str:
    text = str(text).strip()

    m = re.search(r"####\s*\$?(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return normalize_number_string(m.group(1))

    m = re.search(r"(?i)answer\s*:\s*\$?(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return normalize_number_string(m.group(1))

    
    m = re.search(r"\\boxed\{\s*\$?(-?[\d,]+(?:\.\d+)?)\s*\}", text)
    if m:
        return normalize_number_string(m.group(1))

    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if nums:
        return normalize_number_string(nums[-1])

    return text


def mathv_correct(ref_text: str, target_text: str) -> bool:
    try:
        return bool(verify(parse(str(ref_text)), parse(str(target_text))))
    except Exception:
        return False


def math_or_numeric_equal(ref_text: str, target_text: str) -> bool:
    if mathv_correct(ref_text, target_text):
        return True
    ref_num = extract_numeric_answer(ref_text)
    target_num = extract_numeric_answer(target_text)
    if mathv_correct(ref_num, target_num):
        return True
    return normalize_number_string(ref_num) == normalize_number_string(target_num)


def is_correct_gsm8k(prediction: str, gt: Dict[str, Any]) -> int:
    refs: List[str] = []
    if gt.get("value"):
        refs.append(str(gt["value"]))
    refs.extend([str(x) for x in gt.get("aliases", []) if str(x).strip()])

    candidates = [prediction, extract_numeric_answer(prediction)]
    for ref in refs:
        for cand in candidates:
            if ref and cand and math_or_numeric_equal(ref, cand):
                return 1
    return 0





def resolve_dtype(dtype: str):
    if dtype == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    if dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype in ("fp16", "float16", "half"):
        return torch.float16
    if dtype in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


class QwenScoredBinJudge:
    def __init__(self, model_path: str, device: str = "cuda", dtype: str = "auto") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=resolve_dtype(dtype),
            device_map="auto" if device == "cuda" and torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if not (device == "cuda" and torch.cuda.is_available()):
            self.model.to("cpu")
        self.model.eval()
        self.option_ids = self._get_option_ids()

    def _single_token_id(self, s: str) -> Optional[int]:
        
        
        for cand in (s, " " + s, "\n" + s):
            ids = self.tokenizer.encode(cand, add_special_tokens=False)
            if len(ids) == 1:
                return int(ids[0])
        return None

    def _get_option_ids(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        missing: List[str] = []
        for letter, _ in BINS:
            tid = self._single_token_id(letter)
            if tid is None:
                missing.append(letter)
            else:
                out[letter] = tid
        if missing:
            raise ValueError(f"Cannot map option letters to single tokens: {missing}")
        return out

    def _messages(self, question: str, proposed_answer: str, reasoning: str = "") -> List[Dict[str, str]]:
        opts = "\n".join(f"{letter}. {int(value * 100)}%" for letter, value in BINS)
        reasoning_block = ""
        if reasoning and reasoning.strip() and reasoning.strip() != proposed_answer.strip():
            reasoning_block = f"Reasoning trace:\n{reasoning.strip()}\n\n"

        user = (
            f"Problem:\n{question.strip()}\n\n"
            f"{reasoning_block}"
            f"Proposed final answer:\n{proposed_answer.strip()}\n\n"
            "Estimate the probability that the proposed final answer is mathematically correct.\n"
            "Choose exactly one option. Do not explain.\n"
            f"{opts}\n\n"
            "Answer with only the option letter."
        )
        return [
            {"role": "system", "content": "You are a calibrated judge for mathematical reasoning correctness."},
            {"role": "user", "content": user},
        ]

    def _prompt(self, messages: List[Dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **({"enable_thinking": False} if os.environ.get("UQ_FAMILY") == "qwen" else {}),
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\n\nASSISTANT:"

    @torch.no_grad()
    def score(self, question: str, proposed_answer: str, reasoning: str = "") -> Dict[str, Any]:
        prompt = self._prompt(self._messages(question, proposed_answer, reasoning))
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        logits = self.model(**inputs).logits[0, -1, :].float()

        letters: List[str] = []
        values: List[float] = []
        raw_logits: List[float] = []
        for letter, value in BINS:
            letters.append(letter)
            values.append(value)
            raw_logits.append(float(logits[self.option_ids[letter]].item()))

        probs = F.softmax(torch.tensor(raw_logits, dtype=torch.float32, device=logits.device), dim=-1).detach().cpu().tolist()
        confidence = float(sum(p * v for p, v in zip(probs, values)))
        confidence = min(1.0, max(0.0, confidence))
        best_i = max(range(len(probs)), key=lambda i: probs[i])

        return {
            "verbalized_confidence_raw": confidence,
            "verbalized_confidence_0_1": confidence,
            "verbalized_probability_uncertainty": 1.0 - confidence,
            "uncertainty": 1.0 - confidence,
            "parse_mode": "scored_bins_expected_value",
            "best_option": letters[best_i],
            "best_option_value": values[best_i],
            "best_option_prob": float(probs[best_i]),
            "option_probs": {
                letters[i]: {"value": values[i], "prob": float(probs[i]), "logit": raw_logits[i]}
                for i in range(len(letters))
            },
            "prompt_len": int(inputs["input_ids"].shape[1]),
        }





def build_output_record(record: Dict[str, Any], idx: int, judge: QwenScoredBinJudge, args) -> Dict[str, Any]:
    rid = get_id(record, idx)
    question = get_question(record)
    gt = get_gt(record)
    greedy_answer = get_greedy_answer(record)
    reasoning = get_target_reasoning(record)

    correct = is_correct_gsm8k(greedy_answer, gt)
    score = judge.score(question=question, proposed_answer=greedy_answer, reasoning=reasoning)

    out: Dict[str, Any] = {
        "id": rid,
        "gt": gt,
        "greedy_answer": greedy_answer,
        "greedy_answer_final": extract_numeric_answer(greedy_answer),
        "correct": int(correct),
        "verbalized_confidence_raw": score["verbalized_confidence_raw"],
        "verbalized_confidence_0_1": score["verbalized_confidence_0_1"],
        "verbalized_probability_uncertainty": score["verbalized_probability_uncertainty"],
        "uncertainty": score["uncertainty"],
        "parse_mode": score["parse_mode"],
        "best_option": score["best_option"],
        "best_option_value": score["best_option_value"],
        "best_option_prob": score["best_option_prob"],
        "option_probs": score["option_probs"],
        "prompt_len": score["prompt_len"],
        "meta": {
            "method": "Verbalized Probability - scored confidence bins",
            "dataset": "gsm8k",
            "model_path": args.model_path,
            "model_family": "qwen",
            "correctness_checker": "math_verify + numeric_fallback",
            "bins": {letter: value for letter, value in BINS},
        },
    }

    if args.keep_question:
        out["question"] = question
    if args.keep_reasoning:
        out["target_reasoning"] = reasoning
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute Verbalized Probability uncertainty for Qwen on GSM8K.")
    p.add_argument("--input-file", default="autodl-tmp/uq_Qwen3-1.7B-gsm8k.jsonl")
    p.add_argument("--output-file", default="autodl-tmp/method_result/verbalized_probability_qwen3-1.7B_gsm8k.jsonl")
    p.add_argument("--model-path", default="autodl-tmp/models/Qwen/Qwen3-1.7B")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--dtype", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"], default="auto")
    p.add_argument("--hf-endpoint", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--keep-question", action="store_true")
    p.add_argument("--keep-reasoning", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    if os.path.exists(args.output_file) and args.overwrite and not args.resume:
        os.remove(args.output_file)

    done = load_done_ids(args.output_file) if args.resume else set()

    safe_print(f"INPUT_FILE = {args.input_file}")
    safe_print(f"OUTPUT_FILE = {args.output_file}")
    safe_print(f"Judge model = {args.model_path}")
    safe_print("Dataset = GSM8K")
    safe_print("Elicitation mode = scored confidence bins A-K")
    safe_print("Loading Qwen judge model...")

    judge = QwenScoredBinJudge(args.model_path, args.device, args.dtype)
    records = load_records(args.input_file)
    if args.limit is not None:
        records = records[: args.limit]

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    mode = "a" if args.resume and os.path.exists(args.output_file) else "w"

    processed = 0
    skipped = 0
    with open(args.output_file, mode, encoding="utf-8") as fout:
        for idx, record in enumerate(records, start=1):
            rid = get_id(record, idx)
            if rid in done:
                skipped += 1
                continue

            try:
                out = build_output_record(record, idx, judge, args)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                out = {
                    "id": rid,
                    "question": get_question(record),
                    "error": str(e),
                    "meta": {"method": "Verbalized Probability - scored confidence bins", "dataset": "gsm8k"},
                }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            processed += 1

            if args.verbose:
                if "error" in out:
                    safe_print(f"error: {rid} | {out['error']}")
                else:
                    safe_print(
                        f"done: {rid} | correct={out['correct']} | "
                        f"conf={out['verbalized_confidence_0_1']:.4f} | "
                        f"unc={out['verbalized_probability_uncertainty']:.4f} | "
                        f"best={out['best_option']}({out['best_option_value']:.1f})"
                    )

    safe_print(f"Finished. processed={processed}, skipped={skipped}")
    safe_print(f"Saved to: {args.output_file}")


if __name__ == "__main__":
    main()

def _uq_standalone():
    
    import argparse as _ap
    import ast as _ast
    import json as _json
    import os as _os
    from pathlib import Path as _Path
    import sys as _sys
    import tempfile as _tempfile
    import types as _types

    _parser = _ap.ArgumentParser(description="verbalized_probability / llama / gsm8k / answer_only (standalone)")
    _parser.add_argument("--input-file", required=True)
    _parser.add_argument("--output-file", required=True)
    _parser.add_argument("--model-path")
    _parser.add_argument("--correct-file")
    _parser.add_argument("--problem-file")
    _parser.add_argument("--nli-model")
    _parser.add_argument("--device", choices=["cpu", "cuda"])
    _parser.add_argument("--overwrite", action="store_true")
    _parser.add_argument("--dry-run", action="store_true")
    _argsv = _sys.argv[1:]
    _cut = _argsv.index("--") if "--" in _argsv else len(_argsv)
    _args = _parser.parse_args(_argsv[:_cut])
    _extra = _argsv[_cut + 1:]
    _source = _Path(__file__).resolve()
    _input = _Path(_args.input_file).resolve()
    _output = _Path(_args.output_file).resolve()
    if not _input.is_file():
        _parser.error("input file does not exist")
    if _input == _output:
        _parser.error("output must differ from input")
    if _output.exists() and not _args.overwrite:
        _parser.error("output exists; use --overwrite")
    for _name in ["correct_file", "problem_file", "model_path", "nli_model"]:
        _value = getattr(_args, _name)
        if _value and _Path(_value).exists():
            setattr(_args, _name, str(_Path(_value).resolve()))
    if _args.correct_file and _output == _Path(_args.correct_file).resolve():
        _parser.error("output must differ from correctness input")
    if _args.problem_file and _output == _Path(_args.problem_file).resolve():
        _parser.error("output must differ from problem input")
    _tree = _ast.parse(_source.read_text(), filename=str(_source))
    _tree.body = [n for n in _tree.body
                  if not (isinstance(n, _ast.FunctionDef) and n.name == "_uq_standalone")
                  and not (isinstance(n, _ast.If) and "_UQ_EXECUTING" in _ast.unparse(n.test))]
    _options = {}
    for _node in _ast.walk(_tree):
        if isinstance(_node, _ast.Call) and isinstance(_node.func, _ast.Attribute) and _node.func.attr == "add_argument":
            for _opt in _node.args:
                if isinstance(_opt, _ast.Constant) and isinstance(_opt.value, str) and _opt.value.startswith("--"):
                    _options[_opt.value] = _node
    _argv = [str(_source)]
    def _add(_names, _value):
        if _value is not None:
            _opt = next((x for x in _names if x in _options), None)
            if _opt:
                _argv.extend([_opt, str(_value)])
    _add(["--input-file", "--input_file"], str(_input))
    if "verbalized_probability" not in {"max_seq_prob", "mean_tok_ent", "max_tok_prob", "ppl"}:
        _add(["--model-path", "--model_name_or_path", "--judge-model-path"], _args.model_path)
    _add(["--api-model"], _args.model_path)
    _add(["--model-family", "--judge-family"], "llama")
    _add(["--correct-file", "--correct_file"], _args.correct_file)
    _add(["--problem-file"], _args.problem_file)
    _add(["--nli-model", "--nli-model-path"], _args.nli_model)
    _add(["--device"], _args.device)
    _protected = {"--input-file", "--input_file", "--output-file", "--output_file", "--model-path",
                  "--model_name_or_path", "--judge-model-path", "--model-family", "--judge-family", "--api-model"}
    if any(x.split("=")[0] in _protected for x in _extra):
        _parser.error("use the main CLI options for input/output/model/family")
    _argv.extend(_extra)
    
    _provided = {x.split("=")[0] for x in _argv if x.startswith("--")}
    _provided.update({"--output-file", "--output_file"})
    for _opt, _node in _options.items():
        _required = any(k.arg == "required" and isinstance(k.value, _ast.Constant) and k.value.value for k in _node.keywords)
        _aliases = [x.value for x in _node.args if isinstance(x, _ast.Constant)]
        if _required and not _provided.intersection(_aliases):
            _parser.error("missing backend argument: " + _opt)
    if "verbalized_probability" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "verbalized_probability", "family": "llama",
                           "dataset": "gsm8k", "setting": "answer_only", "model_loading": False}, indent=2))
        return
    _output.parent.mkdir(parents=True, exist_ok=True)
    with _tempfile.TemporaryDirectory(prefix="uq-", dir=_output.parent) as _tmp:
        _raw = _Path(_tmp) / "result.jsonl"
        _add(["--output-file", "--output_file"], str(_raw))
        _overrides = {"INPUT_FILE": str(_input), "OUTPUT_FILE": str(_raw),
                      "MODEL_NAME": _args.model_path, "JUDGE_MODEL_NAME": _args.model_path,
                      "API_MODEL": _args.model_path, "MODEL_llama": "llama"}
        if _args.nli_model:
            _overrides.update(NLI_MODEL_NAME=_args.nli_model, NLI_MODEL=_args.nli_model)
        if _args.device:
            _overrides["DEVICE"] = _args.device
        for _node in _tree.body:
            if isinstance(_node, (_ast.Assign, _ast.AnnAssign)):
                _targets = _node.targets if isinstance(_node, _ast.Assign) else [_node.target]
                for _target in _targets:
                    if isinstance(_target, _ast.Name) and _overrides.get(_target.id) is not None:
                        _node.value = _ast.Constant(_overrides[_target.id])
        _ast.fix_missing_locations(_tree)
        _os.environ.update(UQ_llama="llama", UQ_MODEL_PATH=_args.model_path or "",
                           UQ_INPUT_FILE=str(_input), UQ_OUTPUT_FILE=str(_raw))
        _module = _types.ModuleType("__main__")
        _module.__file__ = str(_source)
        _sys.modules["__main__"] = _module
        _sys.argv = _argv
        exec(compile(_tree, str(_source), "exec"), _module.__dict__)
        _count = 0
        with _raw.open() as _stream:
            for _line in _stream:
                if _line.strip():
                    _row = _json.loads(_line)
                    if _row.get("error"):
                        raise RuntimeError(str(_row["error"]))
                    _count += 1
        if not _count:
            raise RuntimeError("backend produced no records")
        _os.replace(_raw, _output)
        print(f"Wrote {_count} records to {_output}")

