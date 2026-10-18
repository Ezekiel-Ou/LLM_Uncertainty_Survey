#!/usr/bin/env python3
# Original paper/author repository: https://github.com/Zhiyuan-GG/Conformal-Uncertainty-Criterion

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
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from math_verify import parse, verify





def safe_print(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def load_records(path: str) -> List[Dict[str, Any]]:
    
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return []

    
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass

    
    records: List[Dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
            else:
                safe_print(f"[warn] skip non-object json line {line_no}", file=sys.stderr)
        except Exception as e:
            safe_print(f"[warn] skip invalid json line {line_no}: {e}", file=sys.stderr)
    return records


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")





_PARSE_CACHE: Dict[str, Any] = {}


def safe_parse_math(text: Any) -> Any:
    key = "" if text is None else str(text)
    if key in _PARSE_CACHE:
        return _PARSE_CACHE[key]
    try:
        parsed = parse(key)
    except Exception:
        parsed = None
    _PARSE_CACHE[key] = parsed
    return parsed


def mathv_equal(a: Any, b: Any) -> bool:
    
    try:
        pa = safe_parse_math(a)
        pb = safe_parse_math(b)
        if pa is None or pb is None:
            return False
        return bool(verify(pa, pb))
    except Exception:
        return False


def mathv_correct(ref_text: Any, target_text: Any) -> bool:
    return mathv_equal(ref_text, target_text)





def get_record_id(record: Dict[str, Any], fallback_idx: int) -> str:
    for key in ["id", "question_id", "qid"]:
        if key in record and record[key] is not None:
            return str(record[key])
    return f"line_{fallback_idx}"


def get_question(record: Dict[str, Any]) -> str:
    for key in ["question", "problem", "prompt"]:
        if key in record and record[key] is not None:
            return str(record[key])
    return ""


def get_correct_answer(record: Dict[str, Any]) -> str:
    for key in ["correct answer", "correct_answer", "answer", "gt", "ground_truth"]:
        if key in record and record[key] is not None:
            val = record[key]
            if isinstance(val, dict):
                if val.get("value") is not None:
                    return str(val["value"])
                aliases = val.get("aliases")
                if isinstance(aliases, list) and aliases:
                    return str(aliases[0])
            return str(val)
    return ""


def get_responses(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    responses = record.get("responses", [])
    if isinstance(responses, list):
        return [r for r in responses if isinstance(r, dict)]
    return []


def get_response_answer(resp: Dict[str, Any]) -> str:
    for key in ["llm answer", "llm_answer", "answer", "text", "response"]:
        if key in resp and resp[key] is not None:
            return str(resp[key]).strip()
    return ""


def get_response_reason(resp: Dict[str, Any]) -> str:
    for key in ["llm reason", "llm_reason", "reason", "rationale", "full_response"]:
        if key in resp and resp[key] is not None:
            return str(resp[key])
    return ""


def get_answer_info_text(resp: Dict[str, Any]) -> str:
    ans_info = resp.get("answer info") or resp.get("answer_info") or {}
    if isinstance(ans_info, dict):
        return str(ans_info.get("text", "") or "")
    return ""


def get_answer_token_logprobs(resp: Dict[str, Any]) -> List[float]:
    ans_info = resp.get("answer info") or resp.get("answer_info") or {}
    if not isinstance(ans_info, dict):
        return []

    prob_info = ans_info.get("prob info") or ans_info.get("prob_info") or {}
    if not isinstance(prob_info, dict):
        return []

    vals = (
        prob_info.get("token logprob")
        or prob_info.get("token_logprob")
        or prob_info.get("chosen_logprob")
        or prob_info.get("logprob")
        or []
    )
    if not isinstance(vals, list):
        return []

    out: List[float] = []
    for x in vals:
        try:
            fx = float(x)
            if math.isfinite(fx):
                out.append(fx)
        except Exception:
            continue
    return out


def get_all_token_logprobs(resp: Dict[str, Any]) -> List[float]:
    
    vals: List[float] = []

    steps = resp.get("step-wise info") or resp.get("step_wise_info") or []
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            prob_info = step.get("prob info") or step.get("prob_info") or {}
            if not isinstance(prob_info, dict):
                continue
            xs = prob_info.get("token logprob") or prob_info.get("token_logprob") or []
            if isinstance(xs, list):
                for x in xs:
                    try:
                        fx = float(x)
                        if math.isfinite(fx):
                            vals.append(fx)
                    except Exception:
                        pass

    vals.extend(get_answer_token_logprobs(resp))
    return vals





def softmax_from_logprobs(logprobs: List[float]) -> List[float]:
    if not logprobs:
        return []
    x = torch.tensor(logprobs, dtype=torch.float32)
    return [float(p) for p in F.softmax(x, dim=0).tolist()]


def compute_response_logprob(
    resp: Dict[str, Any],
    prob_source: str,
    length_normalization: str,
) -> Optional[float]:
    if prob_source == "answer":
        lps = get_answer_token_logprobs(resp)
    elif prob_source == "all":
        lps = get_all_token_logprobs(resp)
    else:
        raise ValueError(f"Unsupported prob_source: {prob_source}")

    if not lps:
        return None

    if length_normalization == "none":
        return float(sum(lps))
    if length_normalization == "mean":
        return float(sum(lps) / max(1, len(lps)))

    raise ValueError(f"Unsupported length_normalization: {length_normalization}")


def compute_sample_probs(
    responses: List[Dict[str, Any]],
    prob_source: str,
    length_normalization: str,
    probability_mode: str,
) -> Tuple[List[float], List[Optional[float]], str]:
    logps: List[Optional[float]] = [
        compute_response_logprob(r, prob_source=prob_source, length_normalization=length_normalization)
        for r in responses
    ]

    valid = [x is not None and math.isfinite(float(x)) for x in logps]
    if all(valid) and logps:
        probs = softmax_from_logprobs([float(x) for x in logps if x is not None])
        return probs, logps, f"{prob_source}_logprob_softmax_{length_normalization}"

    if probability_mode == "strict":
        missing = sum(1 for v in valid if not v)
        raise ValueError(
            f"Missing valid logprobs for {missing}/{len(logps)} responses under strict mode."
        )

    
    n = len(responses)
    if n == 0:
        return [], logps, "empty"
    return [1.0 / n] * n, logps, "uniform_fallback_missing_logprob"


def cluster_answers_by_math(
    answers: List[str],
) -> Tuple[List[int], List[Dict[str, Any]]]:
    
    reps: List[str] = []
    cluster_ids: List[int] = []
    pairwise_debug: List[Dict[str, Any]] = []

    for idx, ans in enumerate(answers):
        assigned = False
        for cid, rep in enumerate(reps):
            matched = mathv_equal(ans, rep)
            pairwise_debug.append(
                {
                    "sample_idx": idx,
                    "sample_answer": ans,
                    "rep_cluster": cid,
                    "rep_answer": rep,
                    "matched": bool(matched),
                    "matched_by": "math_verify",
                }
            )
            if matched:
                cluster_ids.append(cid)
                assigned = True
                break

        if not assigned:
            cid = len(reps)
            reps.append(ans)
            cluster_ids.append(cid)
            pairwise_debug.append(
                {
                    "sample_idx": idx,
                    "sample_answer": ans,
                    "rep_cluster": cid,
                    "rep_answer": ans,
                    "matched": True,
                    "matched_by": "new_cluster",
                }
            )

    return cluster_ids, pairwise_debug


def compute_cluster_summary(
    cluster_ids: List[int],
    sample_probs: List[float],
    answers: List[str],
    correctness: List[int],
    response_logprobs: List[Optional[float]],
) -> List[Dict[str, Any]]:
    cluster_mass: Dict[int, float] = {}
    cluster_members: Dict[int, List[int]] = {}

    for i, cid in enumerate(cluster_ids):
        cluster_mass[cid] = cluster_mass.get(cid, 0.0) + float(sample_probs[i])
        cluster_members.setdefault(cid, []).append(i)

    summary: List[Dict[str, Any]] = []
    for cid in sorted(cluster_members.keys()):
        members = cluster_members[cid]
        correct_count = sum(int(correctness[i]) for i in members)
        summary.append(
            {
                "cluster_id": int(cid),
                "prob": float(cluster_mass.get(cid, 0.0)),
                "size": int(len(members)),
                "correct_count": int(correct_count),
                "correct_rate": float(correct_count / len(members)) if members else 0.0,
                "member_indices": [int(i) for i in members],
                "member_answers": [answers[i] for i in members],
                "member_probs": [float(sample_probs[i]) for i in members],
                "member_logprobs": [
                    None if response_logprobs[i] is None else float(response_logprobs[i])
                    for i in members
                ],
                "representative": answers[members[0]] if members else "",
            }
        )
    return summary


def compute_conu_from_clusters(cluster_summary: List[Dict[str, Any]]) -> Tuple[float, float, int]:
    if not cluster_summary:
        return 0.0, 1.0, -1

    best = max(cluster_summary, key=lambda x: float(x["prob"]))
    conf = max(0.0, min(1.0, float(best["prob"])))
    unc = 1.0 - conf
    if abs(conf - 1.0) < 1e-10:
        conf = 1.0
    if abs(unc) < 1e-10:
        unc = 0.0
    return conf, unc, int(best["cluster_id"])


def choose_target_index(
    target_mode: str,
    cluster_summary: List[Dict[str, Any]],
    responses: List[Dict[str, Any]],
) -> int:
    if not responses:
        return -1
    if target_mode == "first":
        return 0
    if target_mode == "majority":
        if not cluster_summary:
            return 0
        best = max(cluster_summary, key=lambda x: float(x["prob"]))
        members = best.get("member_indices") or [0]
        return int(members[0])
    raise ValueError(f"Unsupported target_mode: {target_mode}")





def process_one_record(
    record: Dict[str, Any],
    record_idx: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    qid = get_record_id(record, record_idx)
    question = get_question(record)
    correct_answer = get_correct_answer(record)
    responses = get_responses(record)

    if args.max_responses is not None:
        responses = responses[: args.max_responses]

    answers = [get_response_answer(r) for r in responses]
    nonempty_pairs = [(r, a) for r, a in zip(responses, answers) if a.strip()]
    responses = [x[0] for x in nonempty_pairs]
    answers = [x[1] for x in nonempty_pairs]

    if not responses:
        return {
            "id": qid,
            "question": question if args.keep_question else None,
            "correct_answer": correct_answer,
            "correct": 0,
            "conu_confidence": None,
            "conu_uncertainty": None,
            "uncertainty": None,
            "error": "no_valid_responses",
            "meta": {
                "method": "ConU",
                "dataset": "reasoning_gsm8k",
            },
        }

    sample_probs, response_logprobs, prob_mode_used = compute_sample_probs(
        responses,
        prob_source=args.prob_source,
        length_normalization=args.length_normalization,
        probability_mode=args.probability_mode,
    )

    response_correctness = [
        1 if mathv_correct(correct_answer, ans) else 0
        for ans in answers
    ]

    cluster_ids, pairwise_debug = cluster_answers_by_math(answers)
    cluster_summary = compute_cluster_summary(
        cluster_ids=cluster_ids,
        sample_probs=sample_probs,
        answers=answers,
        correctness=response_correctness,
        response_logprobs=response_logprobs,
    )

    conu_conf, conu_unc, majority_cluster_id = compute_conu_from_clusters(cluster_summary)

    target_idx = choose_target_index(args.target_mode, cluster_summary, responses)
    target_answer = answers[target_idx] if 0 <= target_idx < len(answers) else ""
    target_correct = 1 if mathv_correct(correct_answer, target_answer) else 0

    majority_cluster = None
    for c in cluster_summary:
        if int(c["cluster_id"]) == majority_cluster_id:
            majority_cluster = c
            break

    out: Dict[str, Any] = {
        "id": qid,
        "correct_answer": correct_answer,
        "target_response_index": int(target_idx),
        "target_answer": target_answer,
        "correct": int(target_correct),

        "conu_confidence": float(conu_conf),
        "conu_uncertainty": float(conu_unc),
        "uncertainty": float(conu_unc),

        "majority_cluster_id": int(majority_cluster_id),
        "majority_cluster_representative": majority_cluster.get("representative") if majority_cluster else None,
        "majority_cluster_correct_rate": majority_cluster.get("correct_rate") if majority_cluster else None,

        "num_responses": int(len(responses)),
        "num_clusters": int(len(cluster_summary)),
        "num_correct_responses": int(sum(response_correctness)),
        "answer_correct_rate": float(sum(response_correctness) / len(response_correctness))
        if response_correctness else 0.0,

        "cluster_summary": cluster_summary,
        "meta": {
            "method": "ConU",
            "dataset": "reasoning_gsm8k",
            "answer_equivalence_judge": "math_verify",
            "correctness_source": "record['correct answer']",
            "prob_source": args.prob_source,
            "length_normalization": args.length_normalization,
            "probability_mode_requested": args.probability_mode,
            "probability_mode_used": prob_mode_used,
            "target_mode": args.target_mode,
            "max_responses": args.max_responses,
        },
    }

    if args.keep_question:
        out["question"] = question
    if args.keep_responses:
        out["responses"] = [
            {
                "index": i,
                "llm_answer": answers[i],
                "correct": int(response_correctness[i]),
                "sample_prob": float(sample_probs[i]),
                "response_logprob": None if response_logprobs[i] is None else float(response_logprobs[i]),
                "cluster_id": int(cluster_ids[i]),
                "llm_reason": get_response_reason(responses[i]) if args.keep_reason else None,
                "answer_info_text": get_answer_info_text(responses[i]),
            }
            for i in range(len(responses))
        ]
    if args.keep_pair_details:
        out["debug"] = {
            "sample_probs": sample_probs,
            "response_logprobs": response_logprobs,
            "cluster_ids": cluster_ids,
            "pairwise_details": pairwise_debug,
        }

    return out





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ConU uncertainty for reasoning-sampling GSM8K-style files."
    )
    parser.add_argument("--input-file", required=True, help="Input JSON/JSONL file")
    parser.add_argument("--output-file", required=True, help="Output JSONL file")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N records")

    parser.add_argument(
        "--prob-source",
        choices=["answer", "all"],
        default="answer",
        help=(
            "Which token logprobs to use for response probability. "
            "'answer' uses response['answer info']['prob info']['token logprob']; "
            "'all' uses step-wise info plus answer info."
        ),
    )
    parser.add_argument(
        "--length-normalization",
        choices=["none", "mean"],
        default="none",
        help=(
            "How to aggregate token logprobs before softmax. "
            "'none' = sum token logprobs, matching sequence probability; "
            "'mean' = average token logprob, less length-biased."
        ),
    )
    parser.add_argument(
        "--probability-mode",
        choices=["auto", "strict"],
        default="auto",
        help=(
            "auto: fallback to uniform if logprobs are missing; "
            "strict: raise an error if any response lacks logprobs."
        ),
    )
    parser.add_argument(
        "--target-mode",
        choices=["first", "majority"],
        default="first",
        help="Which response is used for the output correct field.",
    )
    parser.add_argument("--max-responses", type=int, default=None, help="Use at most this many responses per record")

    parser.add_argument("--keep-question", action="store_true", help="Keep question field if present")
    parser.add_argument("--keep-responses", action="store_true", help="Keep per-response info in output")
    parser.add_argument("--keep-reason", action="store_true", help="When --keep-responses, also keep llm reason")
    parser.add_argument("--keep-pair-details", action="store_true", help="Keep pairwise clustering debug details")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    safe_print(f"INPUT_FILE = {args.input_file}")
    safe_print(f"OUTPUT_FILE = {args.output_file}")
    safe_print("Dataset format: reasoning GSM8K sampling")
    safe_print("Correctness source: record['correct answer'] with math_verify")
    safe_print(f"Probability source: {args.prob_source}")
    safe_print(f"Length normalization: {args.length_normalization}")
    safe_print(f"Probability mode: {args.probability_mode}")

    records = load_records(args.input_file)
    if args.limit is not None:
        records = records[: args.limit]

    outputs: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records):
        try:
            out = process_one_record(rec, idx, args)
        except Exception as e:
            qid = get_record_id(rec, idx) if isinstance(rec, dict) else f"line_{idx}"
            out = {
                "id": qid,
                "error": str(e),
                "conu_confidence": None,
                "conu_uncertainty": None,
                "uncertainty": None,
                "meta": {
                    "method": "ConU",
                    "dataset": "reasoning_gsm8k",
                },
            }

        outputs.append(out)

        if args.verbose:
            unc = out.get("conu_uncertainty")
            unc_s = "None" if unc is None else f"{float(unc):.6f}"
            safe_print(
                f"done: {out.get('id')} | line={idx+1} | "
                f"correct={out.get('correct')} | "
                f"clusters={out.get('num_clusters')} | "
                f"conf={out.get('conu_confidence')} | unc={unc_s}"
            )

    write_jsonl(args.output_file, outputs)
    safe_print(f"[info] wrote {len(outputs)} records to {args.output_file}", file=sys.stderr)


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

    _parser = _ap.ArgumentParser(description="conu / llama / gsm8k / reasoning_augmented (standalone)")
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
    if "conu" not in {"max_seq_prob", "mean_tok_ent", "max_tok_prob", "ppl"}:
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
    if "conu" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "conu", "family": "llama",
                           "dataset": "gsm8k", "setting": "reasoning_augmented", "model_loading": False}, indent=2))
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

