#!/usr/bin/env python3
# Original paper/author repository: https://github.com/potsawee/selfcheckgpt

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
import ast
import json
import math
import os
import re
import sys
import textwrap
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer






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
                print(f"[warn] skip bad json line {line_no} in {path}: {e}", file=sys.stderr)
    return records


def append_jsonl(path: str, obj: Dict[str, Any], lock: Optional[threading.Lock] = None) -> None:
    def _write() -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f.flush()

    if lock is None:
        _write()
    else:
        with lock:
            _write()


def load_done_ids(path: str) -> Set[str]:
    done: Set[str] = set()
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






def build_correct_map(records: Sequence[Dict[str, Any]], question_id_key: str, correct_key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for rec in records:
        qid = rec.get(question_id_key)
        if qid is None and question_id_key != "id":
            qid = rec.get("id")
        if qid is None and question_id_key != "question_id":
            qid = rec.get("question_id")
        if qid is None:
            continue

        try:
            correct = int(rec.get(correct_key, 0))
        except Exception:
            correct = 0
        out[str(qid)] = 1 if correct else 0
    return out






def resolve_record_id(obj: Dict[str, Any], fallback: str = "") -> str:
    for key in ("id", "question_id", "qid", "task_id"):
        if obj.get(key) is not None:
            return str(obj[key])
    return fallback


def extract_text_field(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        for key in ("text", "raw_text", "answer", "response", "completion", "code", "prediction"):
            if x.get(key) is not None:
                return str(x[key]).strip()
    return str(x).strip()


def get_greedy_answer(obj: Dict[str, Any]) -> str:
    for key in ("greedy_answer", "prediction", "answer"):
        if obj.get(key) is not None and not isinstance(obj.get(key), dict):
            text = str(obj[key]).strip()
            if text:
                return text

    greedy = obj.get("greedy")
    text = extract_text_field(greedy)
    if text:
        return text

    return ""


def get_sampled_answers(obj: Dict[str, Any]) -> List[str]:
    outs: List[str] = []

    for key in ("sampled_answers", "samples", "sample_outputs", "sampled", "responses"):
        val = obj.get(key)
        if isinstance(val, list):
            for item in val:
                text = extract_text_field(item)
                if text:
                    outs.append(text)
            if outs:
                return outs

    return outs


def get_source_model(obj: Dict[str, Any]) -> str:
    meta = obj.get("meta")
    if isinstance(meta, dict):
        for key in ("model", "source_model", "llm_model"):
            if meta.get(key) is not None:
                return str(meta[key])
    return "unknown"






_CODE_FENCE_RE = re.compile(r"^```(?:python|py)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def strip_code_wrappers(text: str) -> str:
    
    if text is None:
        return ""
    s = str(text).strip()

    
    m = re.search(r"<CODE>\s*(.*?)\s*</CODE>", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        s = m.group(1).strip()

    s = _CODE_FENCE_RE.sub("", s).strip()

    
    s = re.sub(r"</CODE>\s*$", "", s, flags=re.IGNORECASE).strip()
    return s


def normalize_code_text(text: str) -> str:
    
    s = strip_code_wrappers(text)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = textwrap.dedent(s).strip()

    
    lines: List[str] = []
    for line in s.split("\n"):
        line = line.rstrip()
        if not line.strip():
            continue
        if line.strip().startswith("#"):
            continue
        lines.append(line.strip())

    return "\n".join(lines).strip()


def _try_ast_dump_as_module(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
        return ast.dump(tree, include_attributes=False)
    except Exception:
        return None


def _try_ast_dump_as_function_body(code: str) -> Optional[str]:
    try:
        body = textwrap.indent(textwrap.dedent(code).strip() or "pass", "    ")
        wrapped = "def __candidate__():\n" + body + "\n"
        tree = ast.parse(wrapped)
        if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
            return None
        return ast.dump(tree.body[0].body, include_attributes=False)
    except Exception:
        return None


def ast_signature(text: str) -> Optional[str]:
    code = strip_code_wrappers(text)
    code = textwrap.dedent(code).strip()
    if not code:
        return None

    
    return _try_ast_dump_as_module(code) or _try_ast_dump_as_function_body(code)


def code_exact_match_score(a: str, b: str) -> float:
    a_norm = normalize_code_text(a)
    b_norm = normalize_code_text(b)
    if a_norm and b_norm and a_norm == b_norm:
        return 1.0

    a_ast = ast_signature(a)
    b_ast = ast_signature(b)
    if a_ast is not None and b_ast is not None and a_ast == b_ast:
        return 1.0

    return 0.0


def format_code_for_nli(code: str) -> str:
    code_norm = strip_code_wrappers(code)
    code_norm = textwrap.dedent(code_norm).strip()
    if not code_norm:
        return "Empty Python code."
    return "Python code:\n" + code_norm






class SelfCheckNLI:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_length: int = 512,
        batch_size: int = 32,
        use_safetensors: Optional[bool] = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"[info] loading NLI model: {model_name} on {self.device}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        kwargs: Dict[str, Any] = {"torch_dtype": torch.float32}
        if use_safetensors is not None:
            kwargs["use_safetensors"] = use_safetensors

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
        self.model.to(self.device)
        self.model.eval()

        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.entail_idx: Optional[int] = None
        self.contra_idx: Optional[int] = None
        for idx, label in id2label.items():
            if "entail" in label:
                self.entail_idx = idx
            if "contradiction" in label:
                self.contra_idx = idx

        if self.entail_idx is None or self.contra_idx is None:
            raise ValueError(f"Cannot find entailment/contradiction labels in {id2label}")

    def probs_batch(self, pairs: List[Tuple[str, str]]) -> Tuple[List[float], List[float]]:
        if not pairs:
            return [], []

        entail_probs: List[float] = []
        contra_probs: List[float] = []

        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            premises = [x[0] for x in batch]
            hypotheses = [x[1] for x in batch]

            inputs = self.tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                logits = self.model(**inputs).logits.float()
                probs = F.softmax(logits, dim=-1)

            entail_probs.extend(probs[:, self.entail_idx].detach().cpu().tolist())
            contra_probs.extend(probs[:, self.contra_idx].detach().cpu().tolist())

        return [float(x) for x in entail_probs], [float(x) for x in contra_probs]






def clamp01(x: float) -> float:
    if math.isnan(x):
        return 0.5
    return max(0.0, min(1.0, float(x)))


def compute_selfcheck_score(
    greedy_answer: str,
    sampled_answers: List[str],
    nli: SelfCheckNLI,
    max_stage1_samples: int,
    use_exact_match_mix: bool,
    alpha: float,
) -> Tuple[float, List[Dict[str, Any]]]:
    greedy_clean = strip_code_wrappers(greedy_answer)
    if not greedy_clean.strip():
        return 0.5, []

    used_samples = [strip_code_wrappers(x) for x in sampled_answers if strip_code_wrappers(x).strip()]
    used_samples = used_samples[:max_stage1_samples]
    if not used_samples:
        return 0.5, []

    greedy_for_nli = format_code_for_nli(greedy_clean)
    samples_for_nli = [format_code_for_nli(s) for s in used_samples]

    
    forward_pairs = [(s, greedy_for_nli) for s in samples_for_nli]
    backward_pairs = [(greedy_for_nli, s) for s in samples_for_nli]

    p_ent_f, p_con_f = nli.probs_batch(forward_pairs)
    p_ent_b, p_con_b = nli.probs_batch(backward_pairs)

    details: List[Dict[str, Any]] = []
    vals: List[float] = []

    for s, ef, cf, eb, cb in zip(used_samples, p_ent_f, p_con_f, p_ent_b, p_con_b):
        support_forward = 0.5 * (ef + (1.0 - cf))
        support_backward = 0.5 * (eb + (1.0 - cb))
        nli_support = 0.5 * (support_forward + support_backward)

        exact_match = code_exact_match_score(s, greedy_clean)

        if use_exact_match_mix:
            support = alpha * nli_support + (1.0 - alpha) * exact_match
        else:
            support = nli_support

        support = clamp01(support)
        vals.append(support)
        details.append(
            {
                "sample_answer": s,
                "entail_forward": float(ef),
                "contradiction_forward": float(cf),
                "entail_backward": float(eb),
                "contradiction_backward": float(cb),
                "nli_support": clamp01(nli_support),
                "code_exact_or_ast_match": float(exact_match),
                "single_support": support,
            }
        )

    confidence = clamp01(sum(vals) / len(vals)) if vals else 0.5
    return confidence, details


def compute_item(
    obj: Dict[str, Any],
    correct_map: Dict[str, int],
    nli: SelfCheckNLI,
    args: argparse.Namespace,
    line_idx: int,
) -> Tuple[Dict[str, Any], bool]:
    qid = resolve_record_id(obj, fallback=f"line_{line_idx}")
    greedy_answer = get_greedy_answer(obj)
    sampled_answers = get_sampled_answers(obj)

    missing_correct = qid not in correct_map
    correct = correct_map.get(qid, 0)

    confidence, details = compute_selfcheck_score(
        greedy_answer=greedy_answer,
        sampled_answers=sampled_answers,
        nli=nli,
        max_stage1_samples=args.max_stage1_samples,
        use_exact_match_mix=not args.no_exact_match_mix,
        alpha=args.alpha,
    )
    uncertainty = float(1.0 - confidence)

    used_samples = [strip_code_wrappers(x) for x in sampled_answers if strip_code_wrappers(x).strip()]
    used_samples = used_samples[:args.max_stage1_samples]

    out: Dict[str, Any] = {
        "id": qid,
        "correct": int(correct),
        "selfcheck_confidence": confidence,
        "selfcheck_uncertainty": uncertainty,
        "uncertainty": uncertainty,
        "num_samples_total": len(sampled_answers),
        "num_samples_used": len(used_samples),
        "meta": {
            "method": "SelfCheckGPT-style",
            "dataset": "humaneval",
            "source_model": get_source_model(obj),
            "correctness_source": "external_correct_file",
            "nli_model": args.nli_model,
            "nli_max_length": args.nli_max_length,
            "nli_batch_size": args.nli_batch_size,
            "max_stage1_samples": args.max_stage1_samples,
            "use_exact_match_mix": not args.no_exact_match_mix,
            "alpha": args.alpha if not args.no_exact_match_mix else None,
            "score_definition": "avg bidirectional NLI support from sampled code to greedy code; uncertainty = 1 - confidence",
        },
    }

    if args.keep_question:
        out["question"] = obj.get("question", "")
    if args.keep_prediction:
        out["greedy_answer"] = strip_code_wrappers(greedy_answer)
    if args.keep_samples:
        out["sampled_answers"] = used_samples
    if not args.no_details:
        out["selfcheck_details"] = details

    return out, missing_correct






def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SelfCheckGPT-style uncertainty for HumanEval JSONL outputs."
    )
    parser.add_argument("--input-file", required=True, help="HumanEval stage1 generation JSONL")
    parser.add_argument("--correct-file", required=True, help="Correctness JSONL; correct is loaded from here only")
    parser.add_argument("--output-file", required=True, help="Output JSONL")

    parser.add_argument("--nli-model", default="microsoft/deberta-large-mnli", help="NLI model name/path")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"], help="Device for NLI model")
    parser.add_argument("--nli-max-length", type=int, default=512, help="Max token length for NLI input")
    parser.add_argument("--nli-batch-size", type=int, default=32, help="Batch size for NLI scoring")
    parser.add_argument(
        "--use-safetensors",
        action="store_true",
        help="Force use_safetensors=True when loading NLI model. By default transformers decides.",
    )
    parser.add_argument(
        "--no-safetensors",
        action="store_true",
        help="Force use_safetensors=False when loading NLI model, useful for some mirrored checkpoints.",
    )

    parser.add_argument("--max-stage1-samples", type=int, default=10, help="Max samples used per question")
    parser.add_argument("--alpha", type=float, default=0.8, help="Mix weight: alpha*NLI + (1-alpha)*code exact/AST match")
    parser.add_argument("--no-exact-match-mix", action="store_true", help="Use NLI support only")

    parser.add_argument("--question-id-key", default="question_id", help="Question id key in correct file")
    parser.add_argument("--correct-key", default="correct", help="Correct label key in correct file")
    parser.add_argument("--hf-endpoint", default=None, help="Optional HF mirror, e.g. https://hf-mirror.com")

    parser.add_argument("--keep-question", action="store_true", help="Keep question in output")
    parser.add_argument("--keep-prediction", action="store_true", help="Keep greedy answer in output")
    parser.add_argument("--keep-samples", action="store_true", help="Keep sampled answers in output")
    parser.add_argument("--no-details", action="store_true", help="Do not write per-sample NLI details")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output instead of resuming")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    if args.use_safetensors and args.no_safetensors:
        raise ValueError("Cannot set both --use-safetensors and --no-safetensors")
    use_safetensors: Optional[bool]
    if args.use_safetensors:
        use_safetensors = True
    elif args.no_safetensors:
        use_safetensors = False
    else:
        use_safetensors = None

    if args.overwrite and os.path.exists(args.output_file):
        os.remove(args.output_file)

    data = load_jsonl(args.input_file)
    correct_records = load_jsonl(args.correct_file)
    correct_map = build_correct_map(correct_records, args.question_id_key, args.correct_key)

    nli = SelfCheckNLI(
        model_name=args.nli_model,
        device=args.device,
        max_length=args.nli_max_length,
        batch_size=args.nli_batch_size,
        use_safetensors=use_safetensors,
    )

    done = load_done_ids(args.output_file)
    write_lock = threading.Lock()

    missing_correct = 0
    processed = 0
    skipped = 0

    for line_idx, obj in enumerate(data, start=1):
        qid = resolve_record_id(obj, fallback=f"line_{line_idx}")
        if qid in done:
            skipped += 1
            continue

        try:
            out, miss = compute_item(obj, correct_map, nli, args, line_idx)
            missing_correct += int(miss)
            append_jsonl(args.output_file, out, lock=write_lock)
            processed += 1

            if args.verbose:
                print(
                    f"done: {qid} | line={line_idx} | correct={out['correct']} | "
                    f"samples={out['num_samples_used']} | "
                    f"conf={out['selfcheck_confidence']:.4f} | "
                    f"unc={out['selfcheck_uncertainty']:.4f}",
                    flush=True,
                )
        except KeyboardInterrupt:
            print("[info] KeyboardInterrupt detected. Exiting gracefully.", file=sys.stderr)
            raise
        except Exception as e:
            err = {
                "id": qid,
                "correct": correct_map.get(qid, 0),
                "error": str(e),
                "meta": {
                    "method": "SelfCheckGPT-style",
                    "dataset": "humaneval",
                    "correctness_source": "external_correct_file",
                },
            }
            if args.keep_question:
                err["question"] = obj.get("question", "")
            append_jsonl(args.output_file, err, lock=write_lock)
            processed += 1
            print(f"[error] {qid} | line={line_idx} | {e}", file=sys.stderr, flush=True)

    print(
        f"[info] wrote/updated {processed} records to {args.output_file} | skipped={skipped}",
        file=sys.stderr,
    )
    if missing_correct > 0:
        print(
            f"[warn] {missing_correct} records were missing in correctness file and were set to correct=0",
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

    _parser = _ap.ArgumentParser(description="selfcheckgpt / qwen / humaneval / answer_only (standalone)")
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
    if "selfcheckgpt" not in {"max_seq_prob", "mean_tok_ent", "max_tok_prob", "ppl"}:
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
    if "selfcheckgpt" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "selfcheckgpt", "family": "qwen",
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

