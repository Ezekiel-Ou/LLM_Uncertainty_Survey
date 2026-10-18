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
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


SPECIAL_TOKEN_PATTERNS = (
    re.compile(r"^<\|.*\|>$"),
    re.compile(r"^</?think>$"),
    re.compile(r"^<tool_.*>$"),
    re.compile(r"^<\|endoftext\|>$"),
)


def is_special_token_text(token_text: str) -> bool:
    if not token_text:
        return False
    return any(p.match(token_text) for p in SPECIAL_TOKEN_PATTERNS)


@dataclass
class StepInfo:
    token_text: str
    chosen_prob: float
    chosen_logprob: float
    entropy: Optional[float]


def sanitize_prob(p: Any) -> float:
    try:
        p = float(p)
    except Exception:
        return 0.0
    if math.isnan(p) or p < 0.0:
        return 0.0
    return min(max(p, 0.0), 1.0)


_EPS = 1e-12


def estimate_entropy_from_top_logprobs(top_logprobs: Sequence[Dict[str, Any]]) -> Optional[float]:
    if not top_logprobs:
        return None

    probs: List[float] = []
    for item in top_logprobs:
        p = sanitize_prob(item.get("prob", 0.0))
        if p > 0.0:
            probs.append(p)

    if not probs:
        return None

    s = sum(probs)
    if s > 1.0 + 1e-6:
        probs = [p / s for p in probs]
        s = 1.0

    tail = max(0.0, 1.0 - s)
    entropy = 0.0
    for p in probs:
        entropy -= p * math.log(max(p, _EPS))
    if tail > 0.0:
        entropy -= tail * math.log(max(tail, _EPS))
    return entropy



def extract_answer_steps(greedy: Dict[str, Any]) -> List[StepInfo]:
    raw_steps = greedy.get("steps") or greedy.get("raw_steps") or []
    answer_steps: List[StepInfo] = []
    for step in raw_steps:
        token_text = str(step.get("token_text", ""))
        if is_special_token_text(token_text):
            continue
        chosen_prob = sanitize_prob(step.get("chosen_prob", 0.0))
        chosen_logprob = step.get("chosen_logprob", None)
        try:
            chosen_logprob = float(chosen_logprob)
        except Exception:
            chosen_logprob = math.log(max(chosen_prob, _EPS))
        
        entropy = step.get("entropy")
        if entropy is None:
            entropy = estimate_entropy_from_top_logprobs(step.get("top_logprobs") or [])
        answer_steps.append(
            StepInfo(
                token_text=token_text,
                chosen_prob=chosen_prob,
                chosen_logprob=chosen_logprob,
                entropy=entropy,
            )
        )
    return answer_steps



def build_correct_map(records: List[Dict[str, Any]], question_id_key: str, correct_key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for rec in records:
        qid = rec.get(question_id_key)
        if qid is None and question_id_key != "id":
            qid = rec.get("id")
        if qid is None:
            continue
        try:
            correct = int(rec.get(correct_key, 0))
        except Exception:
            correct = 0
        out[str(qid)] = 1 if correct else 0
    return out



def compute_metrics_from_steps(greedy: Dict[str, Any]) -> Dict[str, Any]:
    steps = extract_answer_steps(greedy)

    if not steps:
        return {
            "maximum_sequence_probability": None,
            "maximum_token_probability": None,
            "perplexity": None,
            "mean_token_entropy": None,
            "token_entropy": [],
            "num_answer_tokens": 0,
            "avg_logprob_answer_only": None,
            "seq_logprob_answer_only": None,
        }

    token_probs = [s.chosen_prob for s in steps]
    token_logprobs = [math.log(max(p, _EPS)) for p in token_probs]
    token_entropies = [s.entropy for s in steps if s.entropy is not None]

    seq_logprob = sum(token_logprobs)
    avg_logprob = seq_logprob / len(token_logprobs)
    seq_prob = math.exp(seq_logprob)
    max_token_prob = max(token_probs)
    ppl = math.exp(-avg_logprob)

    return {
        "maximum_sequence_probability": seq_prob,
        "maximum_token_probability": max_token_prob,
        "perplexity": ppl,
        "mean_token_entropy": (sum(token_entropies) / len(token_entropies)) if token_entropies else None,
        "token_entropy": token_entropies,
        "num_answer_tokens": len(steps),
        "avg_logprob_answer_only": avg_logprob,
        "seq_logprob_answer_only": seq_logprob,
    }


class PMICalculator:
    def __init__(self, model_path: str, device: str = "cuda", dtype: str = "auto") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        torch_dtype = self._resolve_dtype(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self.model.eval()

        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

    def _resolve_dtype(self, dtype: str):
        import torch

        if dtype == "auto":
            return None
        if dtype in {"float16", "fp16", "half"}:
            return torch.float16
        if dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if dtype in {"float32", "fp32"}:
            return torch.float32
        raise ValueError(f"Unsupported dtype: {dtype}")

    def _token_logprobs_for_suffix(self, prefix_ids: List[int], suffix_ids: List[int]) -> List[float]:
        import torch

        if not suffix_ids:
            return []

        full_ids = prefix_ids + suffix_ids
        input_ids = torch.tensor([full_ids], device=self.device, dtype=torch.long)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[:, :-1, :]
            logprobs = torch.log_softmax(logits, dim=-1)

        result: List[float] = []
        start = len(prefix_ids)
        for pos, tok_id in enumerate(suffix_ids):
            lp_pos = start + pos - 1
            if lp_pos < 0:
                continue
            lp = float(logprobs[0, lp_pos, tok_id].item())
            result.append(lp)
        return result

    def compute_pmi(self, prompt_text: str, answer_text: str) -> Tuple[List[float], Optional[float]]:
        prefix_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        suffix_ids = self.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        answer_only_prefix_ids: List[int] = []

        cond_lps = self._token_logprobs_for_suffix(prefix_ids, suffix_ids)
        base_lps = self._token_logprobs_for_suffix(answer_only_prefix_ids, suffix_ids)

        if not cond_lps or not base_lps or len(cond_lps) != len(base_lps):
            return [], None

        pmi_per_token = [c - b for c, b in zip(cond_lps, base_lps)]
        mean_pmi = sum(pmi_per_token) / len(pmi_per_token) if pmi_per_token else None
        return pmi_per_token, mean_pmi



def resolve_record_id(record: Dict[str, Any]) -> Optional[str]:
    rid = record.get("id")
    if rid is None:
        rid = record.get("question_id")
    if rid is None:
        return None
    return str(rid)



def main() -> None:
    parser = argparse.ArgumentParser(description="Compute UQ metrics for HumanEval result files.")
    parser.add_argument("--input-file", required=True, help="Model output JSONL file")
    parser.add_argument("--correct-file", required=True, help="Correctness JSONL file, e.g. question_id/correct")
    parser.add_argument("--output-file", required=True, help="Output JSONL file")
    parser.add_argument("--model-path", default=None, help="Optional model path for PMI / MeanPMI")
    parser.add_argument("--device", default="cuda", help="Device for PMI model loading")
    parser.add_argument("--dtype", default="auto", help="Model dtype for PMI model loading")
    parser.add_argument("--question-id-key", default="question_id", help="Key name in correct file for question id")
    parser.add_argument("--correct-key", default="correct", help="Key name in correct file for correctness label")
    parser.add_argument("--keep-question", action="store_true", help="Keep question field in output")
    parser.add_argument("--keep-prediction", action="store_true", help="Keep greedy prediction text in output")
    parser.add_argument("--verbose", action="store_true", help="Print per-line progress")
    args = parser.parse_args()

    data = load_jsonl(args.input_file)
    correct_records = load_jsonl(args.correct_file)
    correct_map = build_correct_map(correct_records, args.question_id_key, args.correct_key)

    pmi_calc: Optional[PMICalculator] = None
    if args.model_path:
        pmi_calc = PMICalculator(args.model_path, device=args.device, dtype=args.dtype)

    output_records: List[Dict[str, Any]] = []
    missing_correct = 0

    for idx, rec in enumerate(data):
        rid = resolve_record_id(rec)
        if rid is None:
            rid = f"line_{idx}"

        greedy = rec.get("greedy", {}) or {}
        metrics = compute_metrics_from_steps(greedy)

        if rid in correct_map:
            correct = correct_map[rid]
        else:
            correct = 0
            missing_correct += 1

        out: Dict[str, Any] = {
            "id": rid,
            "correct": correct,
            **metrics,
            "pointwise_mutual_information": None,
            "mean_pointwise_mutual_information": None,
        }

        if args.keep_question:
            out["question"] = rec.get("question")
        if args.keep_prediction:
            out["prediction"] = greedy.get("text")

        if pmi_calc is not None:
            prompt_text = str(greedy.get("prompt", ""))
            answer_text = str(greedy.get("text", ""))
            pmi_per_token, mean_pmi = pmi_calc.compute_pmi(prompt_text, answer_text)
            out["pointwise_mutual_information"] = pmi_per_token
            out["mean_pointwise_mutual_information"] = mean_pmi

        output_records.append(out)

        if args.verbose:
            print(
                f"done: {rid} | line={idx+1} | correct={correct} | "
                f"msp={out['maximum_sequence_probability']} | ppl={out['perplexity']}",
                flush=True,
            )

    write_jsonl(args.output_file, output_records)

    print(f"[info] wrote {len(output_records)} records to {args.output_file}", file=sys.stderr)
    if missing_correct > 0:
        print(
            f"[warn] {missing_correct} records were missing in the correctness file and were set to correct=0",
            file=sys.stderr,
        )


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

    _parser = _ap.ArgumentParser(description="mean_tok_ent / qwen / humaneval / answer_only (standalone)")
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
    if "mean_tok_ent" not in {"max_seq_prob", "mean_tok_ent", "max_tok_prob", "ppl"}:
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
    if "mean_tok_ent" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "mean_tok_ent", "family": "qwen",
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

