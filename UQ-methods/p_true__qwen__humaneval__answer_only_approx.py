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
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


_EPS = 1e-12
print_lock = threading.Lock()
write_lock = threading.Lock()


def safe_print(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    with print_lock:
        print(*args, **kwargs)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception as e:
                print(f"[warn] skip bad json line {line_no}: {e}", file=sys.stderr)
    return records


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f.flush()


def load_done_ids(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "id" in obj:
                    done.add(str(obj["id"]))
            except Exception:
                continue
    return done


def build_correct_map(records: List[Dict[str, Any]], question_id_key: str, correct_key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for rec in records:
        qid = rec.get(question_id_key)
        if qid is None and question_id_key != "id":
            qid = rec.get("id")
        if qid is None:
            qid = rec.get("question_id") or rec.get("qid")
        if qid is None:
            continue
        try:
            correct = int(rec.get(correct_key, 0))
        except Exception:
            correct = 0
        out[str(qid)] = 1 if correct else 0
    return out


def resolve_record_id(record: Dict[str, Any]) -> Optional[str]:
    for key in ["id", "question_id", "qid"]:
        if key in record and record[key] is not None:
            return str(record[key])
    return None


def extract_text_field(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        for key in ["text", "raw_text", "answer", "response", "prediction"]:
            if key in x and x[key] is not None:
                return str(x[key]).strip()
    return str(x).strip()


def get_greedy_answer(record: Dict[str, Any]) -> str:
    if record.get("greedy_answer") is not None:
        return str(record["greedy_answer"]).strip()
    greedy = record.get("greedy")
    if greedy is not None:
        text = extract_text_field(greedy)
        if text:
            return text
    if record.get("prediction") is not None:
        return str(record["prediction"]).strip()
    return ""


def get_entry_point(record: Dict[str, Any]) -> str:
    gt = record.get("gt") or record.get("ground_truth") or {}
    if isinstance(gt, dict) and gt.get("entry_point") is not None:
        return str(gt["entry_point"])
    if record.get("entry_point") is not None:
        return str(record["entry_point"])
    return ""


def extract_code_answer(text: str) -> str:
    
    text = "" if text is None else str(text)
    text = text.strip()

    m = re.search(r"<CODE>\s*(.*?)\s*</CODE>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip("\n")

    m = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip("\n")

    
    text = re.sub(r"^<CODE>\s*", "", text, flags=re.IGNORECASE).strip("\n")
    text = re.sub(r"\s*</CODE>$", "", text, flags=re.IGNORECASE).strip("\n")
    return text.strip("\n")


def extract_function_signature(question: str, entry_point: str = "") -> str:
    
    question = "" if question is None else str(question)
    if entry_point:
        pattern = rf"^\s*def\s+{re.escape(entry_point)}\s*\([^\n]*\)\s*(?:->\s*[^:]+)?\s*:"
        m = re.search(pattern, question, flags=re.MULTILINE)
        if m:
            return m.group(0).strip()

    m = re.search(r"^\s*def\s+\w+\s*\([^\n]*\)\s*(?:->\s*[^:]+)?\s*:", question, flags=re.MULTILINE)
    if m:
        return m.group(0).strip()
    return ""


def indent_code_block(code: str, spaces: int = 4) -> str:
    prefix = " " * spaces
    lines = str(code).splitlines()
    if not lines:
        return prefix + "pass"
    
    out = []
    for line in lines:
        if not line.strip():
            out.append("")
        elif line.startswith(prefix):
            out.append(line)
        else:
            out.append(prefix + line)
    return "\n".join(out)


def build_code_for_display(question: str, prediction: str, entry_point: str = "") -> str:
    
    code = extract_code_answer(prediction)
    if re.search(r"^\s*def\s+\w+\s*\(", code, flags=re.MULTILINE):
        return code.strip()

    signature = extract_function_signature(question, entry_point)
    if signature:
        return signature + "\n" + indent_code_block(code)
    return code.strip()


def truncate_middle(text: str, max_chars: int) -> str:
    text = "" if text is None else str(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n... [truncated] ...\n" + text[-half:]


class PTrueCalculator:
    def __init__(
        self,
        model_path: str,
        model_family: str = "auto",
        device: str = "cuda",
        dtype: str = "auto",
        trust_remote_code: bool = True,
        use_device_map: bool = True,
        use_p_true_only: bool = False,
    ) -> None:
        self.model_path = model_path
        self.model_family = model_family.lower()
        self.use_p_true_only = use_p_true_only

        torch_dtype = self._resolve_dtype(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
        }
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        if use_device_map and device == "cuda" and torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self.model.eval()

        if not (use_device_map and device == "cuda" and torch.cuda.is_available()):
            if device == "cuda" and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
            self.model.to(self.device)
        else:
            self.device = next(self.model.parameters()).device

    @staticmethod
    def _resolve_dtype(dtype: str):
        dtype = str(dtype).lower()
        if dtype == "auto":
            return None
        if dtype in {"float16", "fp16", "half"}:
            return torch.float16
        if dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if dtype in {"float32", "fp32"}:
            return torch.float32
        raise ValueError(f"Unsupported dtype: {dtype}")

    def _build_messages(self, question: str, proposed_code: str) -> List[Dict[str, str]]:
        system = (
            "You are a strict Python code correctness judge for HumanEval programming tasks. "
            "Given a task and a proposed solution, decide whether the proposed solution is correct. "
            "Consider edge cases and hidden tests."
        )
        user = (
            "Task:\n"
            f"{question.strip()}\n\n"
            "Proposed solution:\n"
            "```python\n"
            f"{proposed_code.strip()}\n"
            "```\n\n"
            "Question: Will this proposed solution pass all tests for the task?\n"
            "Answer with exactly one word: True or False."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        try:
            if self.model_family == "qwen":
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            chunks = []
            for m in messages:
                chunks.append(f"{m['role'].upper()}: {m['content']}")
            chunks.append("ASSISTANT:")
            return "\n\n".join(chunks)

    def _encode_prompt(self, prompt: str) -> List[int]:
        return self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

    def _candidate_logprob(self, prompt_ids: List[int], candidate_text: str) -> float:
        cand_ids = self.tokenizer(candidate_text, add_special_tokens=False)["input_ids"]
        if not cand_ids:
            return -math.inf

        full_ids = prompt_ids + cand_ids
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.device)

        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[:, :-1, :].float()
            logprobs = torch.log_softmax(logits, dim=-1)

        start = len(prompt_ids)
        total = 0.0
        for k, tok_id in enumerate(cand_ids):
            lp_pos = start + k - 1
            if lp_pos < 0 or lp_pos >= logprobs.shape[1]:
                return -math.inf
            total += float(logprobs[0, lp_pos, tok_id].item())
        return total

    @staticmethod
    def _logsumexp(values: Sequence[float]) -> float:
        vals = [v for v in values if math.isfinite(v)]
        if not vals:
            return -math.inf
        m = max(vals)
        return m + math.log(sum(math.exp(v - m) for v in vals))

    def compute(self, question: str, proposed_code: str) -> Dict[str, Any]:
        messages = self._build_messages(question, proposed_code)
        prompt = self._format_prompt(messages)
        prompt_ids = self._encode_prompt(prompt)

        
        true_forms = ["True", " True", "true", " true"]
        false_forms = ["False", " False", "false", " false"]

        true_lps = [self._candidate_logprob(prompt_ids, s) for s in true_forms]
        false_lps = [self._candidate_logprob(prompt_ids, s) for s in false_forms]

        lp_true = self._logsumexp(true_lps)
        lp_false = self._logsumexp(false_lps)
        denom = self._logsumexp([lp_true, lp_false])

        if not math.isfinite(denom):
            p_true = None
            p_false = None
            uncertainty = None
        else:
            p_true = float(math.exp(lp_true - denom))
            p_false = float(math.exp(lp_false - denom))
            uncertainty = float(1.0 - p_true)

        return {
            "p_true_confidence": p_true,
            "p_true_uncertainty": uncertainty,
            "uncertainty": uncertainty,
            "p_true": p_true,
            "p_false": p_false,
            "true_logprob": lp_true if math.isfinite(lp_true) else None,
            "false_logprob": lp_false if math.isfinite(lp_false) else None,
            "candidate_logprobs": {
                "true_forms": [None if not math.isfinite(x) else float(x) for x in true_lps],
                "false_forms": [None if not math.isfinite(x) else float(x) for x in false_lps],
            },
            "prompt_num_tokens": len(prompt_ids),
        }


def process_one_record(
    record: Dict[str, Any],
    ptrue_calc: PTrueCalculator,
    correct_map: Dict[str, int],
    args: argparse.Namespace,
    line_idx: int,
) -> Dict[str, Any]:
    qid = resolve_record_id(record) or f"line_{line_idx}"
    question = str(record.get("question", ""))
    entry_point = get_entry_point(record)
    greedy_answer = get_greedy_answer(record)
    proposed_code = build_code_for_display(question, greedy_answer, entry_point)

    correct = correct_map.get(qid, 0)

    if args.max_question_chars and args.max_question_chars > 0:
        question_for_prompt = truncate_middle(question, args.max_question_chars)
    else:
        question_for_prompt = question

    if args.max_code_chars and args.max_code_chars > 0:
        code_for_prompt = truncate_middle(proposed_code, args.max_code_chars)
    else:
        code_for_prompt = proposed_code

    metric = ptrue_calc.compute(question_for_prompt, code_for_prompt)

    out: Dict[str, Any] = {
        "id": qid,
        "correct": int(correct),
        "p_true_confidence": metric["p_true_confidence"],
        "p_true_uncertainty": metric["p_true_uncertainty"],
        "uncertainty": metric["uncertainty"],
        "p_true": metric["p_true"],
        "p_false": metric["p_false"],
        "true_logprob": metric["true_logprob"],
        "false_logprob": metric["false_logprob"],
        "candidate_logprobs": metric["candidate_logprobs"],
        "prompt_num_tokens": metric["prompt_num_tokens"],
        "meta": {
            "method": "P(True)",
            "dataset": "humaneval",
            "model_path": args.model_path,
            "model_family": args.model_family,
            "correctness_source": "external_correct_file_only",
            "candidate_labels": ["True", "False"],
            "score_definition": "P(True) normalized over True/False candidate likelihoods",
        },
    }

    if args.keep_question:
        out["question"] = question
    if args.keep_prediction:
        out["prediction"] = greedy_answer
        out["proposed_code_for_judge"] = proposed_code

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute P(True) uncertainty for HumanEval JSONL results.")
    parser.add_argument("--input-file", required=True, help="HumanEval stage1 model output JSONL")
    parser.add_argument("--correct-file", required=True, help="Correctness JSONL, e.g. question_id/correct")
    parser.add_argument("--output-file", required=True, help="Output JSONL path")
    parser.add_argument("--model-path", required=True, help="Local model path used for P(True) judging")
    parser.add_argument("--model-family", default="auto", choices=["auto", "qwen", "llama"], help="Model chat-template family")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--dtype", default="auto", help="auto/fp16/bf16/fp32")
    parser.add_argument("--no-device-map", action="store_true", help="Disable device_map='auto'")
    parser.add_argument("--no-trust-remote-code", action="store_true", help="Disable trust_remote_code")
    parser.add_argument("--question-id-key", default="question_id", help="Question id key in correct file")
    parser.add_argument("--correct-key", default="correct", help="Correct label key in correct file")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N not-done records")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output instead of resuming")
    parser.add_argument("--keep-question", action="store_true", help="Keep question in output")
    parser.add_argument("--keep-prediction", action="store_true", help="Keep greedy prediction in output")
    parser.add_argument("--max-question-chars", type=int, default=0, help="Truncate question in judge prompt if >0")
    parser.add_argument("--max-code-chars", type=int, default=0, help="Truncate proposed code in judge prompt if >0")
    parser.add_argument("--hf-endpoint", default=None, help="Optional HF_ENDPOINT mirror, e.g. https://hf-mirror.com")
    parser.add_argument("--verbose", action="store_true", help="Print per-record progress")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    safe_print(f"INPUT_FILE = {args.input_file}")
    safe_print(f"CORRECT_FILE = {args.correct_file}")
    safe_print(f"OUTPUT_FILE = {args.output_file}")
    safe_print("Correctness source: external correct file only")
    safe_print(f"P(True) judge model: {args.model_path}")
    safe_print(f"Model family: {args.model_family}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    if args.overwrite and os.path.exists(args.output_file):
        os.remove(args.output_file)

    correct_records = load_jsonl(args.correct_file)
    correct_map = build_correct_map(correct_records, args.question_id_key, args.correct_key)

    done = load_done_ids(args.output_file)
    if done:
        safe_print(f"existing records: {len(done)}")

    safe_print("Loading local P(True) judge model...")
    ptrue_calc = PTrueCalculator(
        model_path=args.model_path,
        model_family=args.model_family,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=not args.no_trust_remote_code,
        use_device_map=not args.no_device_map,
    )

    processed = 0
    missing_correct = 0

    with open(args.input_file, "r", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin, start=1):
            if args.limit is not None and processed >= args.limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except Exception as e:
                safe_print(f"skip invalid json line: {line_idx} | {e}")
                continue

            qid = resolve_record_id(record) or f"line_{line_idx}"
            if qid in done:
                continue

            if qid not in correct_map:
                missing_correct += 1

            try:
                result = process_one_record(record, ptrue_calc, correct_map, args, line_idx)
                append_jsonl(args.output_file, result)
                processed += 1

                if args.verbose:
                    conf = result["p_true_confidence"]
                    unc = result["p_true_uncertainty"]
                    conf_s = "None" if conf is None else f"{conf:.6f}"
                    unc_s = "None" if unc is None else f"{unc:.6f}"
                    safe_print(
                        f"done: {result['id']} | line={line_idx} | "
                        f"correct={result['correct']} | p_true={conf_s} | unc={unc_s}"
                    )
            except KeyboardInterrupt:
                safe_print("KeyboardInterrupt detected. Exiting gracefully.")
                raise
            except Exception as e:
                err = {
                    "id": qid,
                    "correct": int(correct_map.get(qid, 0)),
                    "error": str(e),
                    "meta": {
                        "method": "P(True)",
                        "dataset": "humaneval",
                        "correctness_source": "external_correct_file_only",
                    },
                }
                if args.keep_question:
                    err["question"] = record.get("question", "")
                append_jsonl(args.output_file, err)
                safe_print(f"error: {qid} | line={line_idx} | {e}")

    safe_print(f"Finished. processed={processed}")
    safe_print(f"Saved to: {args.output_file}")
    if missing_correct > 0:
        safe_print(f"[warn] {missing_correct} records missing in correct file; set correct=0")


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

    _parser = _ap.ArgumentParser(description="p_true / qwen / humaneval / answer_only (standalone)")
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
    if "p_true" not in {"max_seq_prob", "mean_tok_ent", "max_tok_prob", "ppl"}:
        _add(["--model-path", "--model_name_or_path", "--judge-model-path"], _args.model_path)
    _add(["--api-model"], _args.model_path)
    _add(["--model-family", "--judge-family"], "qwen")
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
    if "p_true" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "p_true", "family": "qwen",
                           "dataset": "humaneval", "setting": "answer_only", "model_loading": False}, indent=2))
        return
    _output.parent.mkdir(parents=True, exist_ok=True)
    with _tempfile.TemporaryDirectory(prefix="uq-", dir=_output.parent) as _tmp:
        _raw = _Path(_tmp) / "result.jsonl"
        _add(["--output-file", "--output_file"], str(_raw))
        _overrides = {"INPUT_FILE": str(_input), "OUTPUT_FILE": str(_raw),
                      "MODEL_NAME": _args.model_path, "JUDGE_MODEL_NAME": _args.model_path,
                      "API_MODEL": _args.model_path, "MODEL_qwen": "qwen"}
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
        _os.environ.update(UQ_qwen="qwen", UQ_MODEL_PATH=_args.model_path or "",
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

