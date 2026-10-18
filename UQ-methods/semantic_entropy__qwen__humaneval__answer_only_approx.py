#!/usr/bin/env python3
# Original paper/author repository: https://github.com/jlko/semantic_uncertainty
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
import ast
import json
import math
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional, Tuple





def safe_print(*args, **kwargs) -> None:
    print(*args, **kwargs, flush=True)


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


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()


def load_done_ids(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "id" in obj and obj["id"] is not None:
                    done.add(str(obj["id"]))
            except Exception:
                continue
    return done


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        val = float(x)
        if math.isnan(val) or math.isinf(val):
            return default
        return val
    except Exception:
        return default


def softmax_from_logprobs(logprobs: List[float]) -> List[float]:
    if not logprobs:
        return []
    m = max(logprobs)
    exps = [math.exp(x - m) for x in logprobs]
    s = sum(exps)
    if s <= 0:
        return [1.0 / len(logprobs)] * len(logprobs)
    return [float(x / s) for x in exps]


def build_correct_map(records: List[Dict[str, Any]], question_id_key: str, correct_key: str) -> Dict[str, int]:
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





def get_question_id(obj: Dict[str, Any]) -> str:
    for key in ["id", "question_id", "qid", "task_id"]:
        if key in obj and obj[key] is not None:
            return str(obj[key])
    return "unknown_id"


def get_question(obj: Dict[str, Any]) -> str:
    return str(obj.get("question", obj.get("prompt", ""))).strip()


def get_gt(obj: Dict[str, Any]) -> Any:
    if "gt" in obj:
        return obj["gt"]
    if "ground_truth" in obj:
        return obj["ground_truth"]
    return None


def get_entry_point(obj: Dict[str, Any]) -> Optional[str]:
    gt = obj.get("gt") or obj.get("ground_truth") or {}
    if isinstance(gt, dict):
        ep = gt.get("entry_point")
        if ep is not None:
            return str(ep)
    ep = obj.get("entry_point")
    if ep is not None:
        return str(ep)
    return None


def extract_text_field(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        for key in ["text", "raw_text", "answer", "response", "completion", "code"]:
            if key in x and x[key] is not None:
                return str(x[key]).strip()
    return str(x).strip()


def get_greedy_answer(obj: Dict[str, Any]) -> str:
    if "greedy_answer" in obj and obj["greedy_answer"] is not None:
        return str(obj["greedy_answer"]).strip()

    if "greedy" in obj:
        text = extract_text_field(obj["greedy"])
        if text:
            return text

    for key in ["prediction", "answer", "response", "completion", "text"]:
        if key in obj and obj[key] is not None:
            return str(obj[key]).strip()

    return ""


def get_sample_records(obj: Dict[str, Any], max_samples: Optional[int]) -> List[Dict[str, Any]]:
    
    outs: List[Dict[str, Any]] = []

    if "sampled_answers" in obj and isinstance(obj["sampled_answers"], list):
        for i, s in enumerate(obj["sampled_answers"]):
            t = extract_text_field(s)
            if t:
                outs.append({"sample_index": i, "text": t, "seq_logprob": None})
    elif "samples" in obj and isinstance(obj["samples"], list):
        for i, s in enumerate(obj["samples"]):
            t = extract_text_field(s)
            if not t:
                continue
            lp = None
            if isinstance(s, dict):
                for key in ["seq_logprob", "sequence_logprob", "sum_logprob", "logprob"]:
                    lp = safe_float(s.get(key), None)
                    if lp is not None:
                        break
            outs.append({"sample_index": i, "text": t, "seq_logprob": lp})
    elif "sampled_responses" in obj and isinstance(obj["sampled_responses"], list):
        for i, s in enumerate(obj["sampled_responses"]):
            t = extract_text_field(s)
            if t:
                outs.append({"sample_index": i, "text": t, "seq_logprob": None})

    if max_samples is not None:
        outs = outs[:max_samples]
    return outs


def get_verbalized_confidence(obj: Dict[str, Any]) -> Optional[float]:
    verbalized = obj.get("verbalized")
    if isinstance(verbalized, dict):
        return safe_float(verbalized.get("confidence"), None)
    return None





_CODE_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] = {}


def strip_code_wrappers(text: str) -> str:
    
    text = "" if text is None else str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    
    m = re.search(r"<CODE>\s*(.*?)\s*</CODE>", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        text = m.group(1).strip()
    else:
        text = re.sub(r"</?CODE>", "", text, flags=re.IGNORECASE).strip()

    
    m = re.search(r"```(?:python|py)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        text = m.group(1).strip()
    else:
        text = re.sub(r"^```(?:python|py)?\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*```$", "", text).strip()

    text = re.sub(r"^\s*(?:Output|Answer|Code)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.split("</CODE>")[0].strip()
    return text


def extract_signature_from_question(question: str, entry_point: Optional[str]) -> Optional[str]:
    
    question = "" if question is None else str(question)

    
    try:
        tree = ast.parse(question)
        target_func: Optional[ast.FunctionDef] = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                if entry_point is None or node.name == entry_point:
                    target_func = node
                    break
        if target_func is not None:
            target_func.name = "__candidate__"
            target_func.body = [ast.Pass()]
            target_func.decorator_list = []
            mod = ast.Module(body=[target_func], type_ignores=[])
            ast.fix_missing_locations(mod)
            return ast.unparse(mod).split("\n", 1)[0]
    except Exception:
        pass

    
    if entry_point:
        pattern = rf"^\s*def\s+{re.escape(entry_point)}\s*\([^\n]*\)\s*(?:->\s*[^:\n]+)?\s*:"
    else:
        pattern = r"^\s*def\s+\w+\s*\([^\n]*\)\s*(?:->\s*[^:\n]+)?\s*:"
    m = re.search(pattern, question, flags=re.MULTILINE)
    if not m:
        return None
    sig = m.group(0).strip()
    sig = re.sub(r"def\s+\w+", "def __candidate__", sig, count=1)
    return sig


def indent_body(code: str) -> str:
    code = code.replace("\t", "    ")
    stripped = code.strip("\n")
    if not stripped.strip():
        return "    pass"
    dedented = textwrap.dedent(stripped)
    return textwrap.indent(dedented, "    ")


def remove_docstring_nodes(node: ast.AST) -> ast.AST:
    for child in ast.walk(node):
        if isinstance(child, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(child, "body", None)
            if body and isinstance(body[0], ast.Expr):
                val = body[0].value
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    child.body = body[1:]
    return node


def canonicalize_ast(tree: ast.AST, entry_point: Optional[str]) -> ast.AST:
    
    selected: Optional[ast.FunctionDef] = None
    if isinstance(tree, ast.Module):
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                if entry_point is None or node.name == entry_point or selected is None:
                    selected = node
                    if entry_point is not None and node.name == entry_point:
                        break
        if selected is not None:
            selected.name = "__candidate__"
            selected.decorator_list = []
            tree = ast.Module(body=[selected], type_ignores=[])

    remove_docstring_nodes(tree)
    ast.fix_missing_locations(tree)
    return tree


def normalize_text_for_fallback(code: str) -> str:
    code = strip_code_wrappers(code)
    code = textwrap.dedent(code).strip()
    lines: List[str] = []
    for line in code.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(re.sub(r"\s+", " ", line))
    return "\n".join(lines)


def code_signature(code: str, question: str, entry_point: Optional[str]) -> Dict[str, Any]:
    cache_key = (code or "", question or "", entry_point or "")
    if cache_key in _CODE_CACHE:
        return _CODE_CACHE[cache_key]

    cleaned = strip_code_wrappers(code)
    fallback = normalize_text_for_fallback(cleaned)
    result: Dict[str, Any] = {
        "raw": code or "",
        "cleaned": cleaned,
        "fallback_text": fallback,
        "ast_dump": None,
        "parse_ok": False,
        "wrapped": False,
        "canonical_source": None,
    }

    candidates: List[Tuple[str, bool]] = []
    if cleaned.strip():
        candidates.append((textwrap.dedent(cleaned).strip(), False))
        if not re.search(r"^\s*def\s+", cleaned, flags=re.MULTILINE):
            signature = extract_signature_from_question(question, entry_point)
            if signature is not None:
                candidates.append((signature + "\n" + indent_body(cleaned), True))

    for candidate, wrapped in candidates:
        try:
            tree = ast.parse(candidate)
            tree = canonicalize_ast(tree, entry_point)
            dump = ast.dump(tree, annotate_fields=True, include_attributes=False)
            result.update(
                {
                    "ast_dump": dump,
                    "parse_ok": True,
                    "wrapped": wrapped,
                    "canonical_source": ast.unparse(tree),
                }
            )
            break
        except Exception:
            continue

    _CODE_CACHE[cache_key] = result
    return result


def codes_equivalent(
    code_a: str,
    code_b: str,
    question: str,
    entry_point: Optional[str],
    mode: str = "ast_or_text",
) -> bool:
    sig_a = code_signature(code_a, question, entry_point)
    sig_b = code_signature(code_b, question, entry_point)

    if mode in {"ast", "ast_or_text"}:
        if sig_a.get("parse_ok") and sig_b.get("parse_ok"):
            return sig_a.get("ast_dump") == sig_b.get("ast_dump")
        if mode == "ast":
            return False

    return bool(sig_a.get("fallback_text")) and sig_a.get("fallback_text") == sig_b.get("fallback_text")





def cluster_samples_by_code(
    texts: List[str],
    question: str,
    entry_point: Optional[str],
    equivalence_mode: str,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    
    cluster_reps: List[str] = []
    cluster_ids: List[int] = []
    pairwise_debug: List[Dict[str, Any]] = []

    for idx, text in enumerate(texts):
        assigned = False
        for cid, rep in enumerate(cluster_reps):
            matched = codes_equivalent(
                text,
                rep,
                question=question,
                entry_point=entry_point,
                mode=equivalence_mode,
            )
            pairwise_debug.append(
                {
                    "sample_idx": idx,
                    "sample_text": text,
                    "rep_cluster": cid,
                    "rep_text": rep,
                    "matched": bool(matched),
                    "matched_by": f"humaneval_code_{equivalence_mode}",
                }
            )
            if matched:
                cluster_ids.append(cid)
                assigned = True
                break

        if not assigned:
            new_cid = len(cluster_reps)
            cluster_reps.append(text)
            cluster_ids.append(new_cid)
            pairwise_debug.append(
                {
                    "sample_idx": idx,
                    "sample_text": text,
                    "rep_cluster": new_cid,
                    "rep_text": text,
                    "matched": True,
                    "matched_by": "new_cluster",
                }
            )

    return cluster_ids, pairwise_debug


def compute_sample_probs(sample_records: List[Dict[str, Any]], prob_mode: str) -> Tuple[List[float], str]:
    n = len(sample_records)
    if n == 0:
        return [], "none"

    logprobs = [safe_float(s.get("seq_logprob"), None) for s in sample_records]
    has_all_logprobs = all(x is not None for x in logprobs)

    if prob_mode == "uniform":
        return [1.0 / n] * n, "uniform_forced"

    if prob_mode == "seq_logprob":
        if not has_all_logprobs:
            raise ValueError("--prob-mode seq_logprob requires every used sample to have seq_logprob")
        return softmax_from_logprobs([float(x) for x in logprobs]), "softmax_seq_logprob"

    
    if has_all_logprobs:
        return softmax_from_logprobs([float(x) for x in logprobs]), "softmax_seq_logprob"
    return [1.0 / n] * n, "uniform_fallback_no_complete_seq_logprob"


def compute_cluster_summary(
    cluster_ids: List[int],
    sample_probs: List[float],
    texts: List[str],
) -> List[Dict[str, Any]]:
    cluster_mass: Dict[int, float] = {}
    cluster_members: Dict[int, List[int]] = {}

    for i, cid in enumerate(cluster_ids):
        cluster_mass[cid] = cluster_mass.get(cid, 0.0) + sample_probs[i]
        cluster_members.setdefault(cid, []).append(i)

    summary: List[Dict[str, Any]] = []
    for cid in sorted(cluster_mass.keys()):
        members = cluster_members[cid]
        summary.append(
            {
                "cluster_id": int(cid),
                "prob": float(cluster_mass[cid]),
                "size": int(len(members)),
                "member_indices": [int(x) for x in members],
                "member_texts": [texts[i] for i in members],
                "representative": texts[members[0]] if members else "",
            }
        )
    return summary


def compute_semantic_entropy(cluster_summary: List[Dict[str, Any]]) -> Tuple[float, float]:
    probs = [max(0.0, float(c["prob"])) for c in cluster_summary]
    probs = [p for p in probs if p > 0.0]

    if not probs:
        return 0.0, 0.0

    
    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]

    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    if abs(entropy) < 1e-10:
        entropy = 0.0

    k = len(probs)
    normalized = entropy / math.log(k + 1e-12) if k > 1 else 0.0
    if abs(normalized) < 1e-10:
        normalized = 0.0

    normalized = max(0.0, min(1.0, normalized))
    return float(entropy), float(normalized)


def compute_semantic_density(cluster_summary: List[Dict[str, Any]]) -> Tuple[float, float]:
    probs = [max(0.0, float(c["prob"])) for c in cluster_summary]
    if not probs:
        return 1.0, 0.0

    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]

    density_confidence = sum(p * p for p in probs)
    density_confidence = max(0.0, min(1.0, density_confidence))
    density_uncertainty = 1.0 - density_confidence

    if abs(density_uncertainty) < 1e-10:
        density_uncertainty = 0.0
    if abs(density_confidence - 1.0) < 1e-10:
        density_confidence = 1.0

    return float(density_confidence), float(density_uncertainty)





def process_one_record(
    record: Dict[str, Any],
    correct_map: Dict[str, int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    qid = get_question_id(record)
    question = get_question(record)
    gt = get_gt(record)
    entry_point = get_entry_point(record)

    greedy_answer = get_greedy_answer(record)
    sample_records = get_sample_records(record, args.max_samples)

    if args.include_greedy_in_clusters:
        sample_records = [{"sample_index": "greedy", "text": greedy_answer, "seq_logprob": None}] + sample_records

    if not sample_records:
        sample_records = [{"sample_index": "greedy_fallback", "text": greedy_answer, "seq_logprob": None}]

    sampled_answers = [s["text"] for s in sample_records]
    sample_probs, prob_source = compute_sample_probs(sample_records, args.prob_mode)

    cluster_ids, pairwise_debug = cluster_samples_by_code(
        sampled_answers,
        question=question,
        entry_point=entry_point,
        equivalence_mode=args.equivalence_mode,
    )

    cluster_summary = compute_cluster_summary(
        cluster_ids=cluster_ids,
        sample_probs=sample_probs,
        texts=sampled_answers,
    )

    semantic_entropy, semantic_entropy_normalized = compute_semantic_entropy(cluster_summary)
    semantic_density_confidence, semantic_density_uncertainty = compute_semantic_density(cluster_summary)

    if args.primary_uncertainty == "entropy":
        uncertainty_alias = semantic_entropy_normalized
    else:
        uncertainty_alias = semantic_density_uncertainty

    if qid in correct_map:
        correct = correct_map[qid]
    else:
        correct = 0

    result: Dict[str, Any] = {
        "id": qid,
        "correct": int(correct),
        "semantic_entropy": semantic_entropy,
        "semantic_entropy_normalized": semantic_entropy_normalized,
        "semantic_entropy_uncertainty": semantic_entropy_normalized,
        "semantic_density_confidence": semantic_density_confidence,
        "semantic_density_uncertainty": semantic_density_uncertainty,
        "uncertainty": uncertainty_alias,
        "primary_uncertainty": args.primary_uncertainty,
        "num_samples": len(sampled_answers),
        "num_clusters": len(cluster_summary),
        "cluster_summary": cluster_summary,
        "debug": {
            "sample_probs": sample_probs,
            "sample_prob_source": prob_source,
            "sampled_answers": sampled_answers,
            "cluster_ids": [int(x) for x in cluster_ids],
            "pairwise_details": pairwise_debug,
        },
        "meta": {
            "method": "Semantic Density + Semantic Entropy",
            "dataset": "humaneval",
            "answer_equivalence_judge": f"humaneval_code_{args.equivalence_mode}",
            "correctness_source": "external_correct_file",
            "max_samples_used": args.max_samples,
            "prob_mode": args.prob_mode,
            "include_greedy_in_clusters": args.include_greedy_in_clusters,
            "entropy_log_base": "natural",
        },
    }

    if args.keep_question:
        result["question"] = question
    if args.keep_prediction:
        result["greedy_answer"] = greedy_answer
        result["greedy_text"] = greedy_answer
    if args.keep_gt:
        result["gt"] = gt
        result["entry_point"] = entry_point
    if args.keep_samples:
        result["sampled_answers"] = sampled_answers

    verbalized_conf = get_verbalized_confidence(record)
    if verbalized_conf is not None:
        result["verbalized_confidence_raw"] = verbalized_conf
        result["verbalized_confidence_0_1"] = verbalized_conf / 100.0 if verbalized_conf > 1.0 else verbalized_conf

    return result





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Semantic Density and Semantic Entropy uncertainty for HumanEval stage1 JSONL files."
    )
    parser.add_argument("--input-file", "--input_file", dest="input_file", required=True, help="HumanEval stage1 JSONL file")
    parser.add_argument("--correct-file", "--correct_file", dest="correct_file", required=True, help="External correctness JSONL file")
    parser.add_argument("--output-file", "--output_file", dest="output_file", required=True, help="Output JSONL file")
    parser.add_argument("--question-id-key", default="question_id", help="Question id key in correctness file")
    parser.add_argument("--correct-key", default="correct", help="Correctness key in correctness file")
    parser.add_argument("--max-samples", type=int, default=10, help="Maximum number of stage1 samples used for clustering")
    parser.add_argument(
        "--prob-mode",
        choices=["auto", "uniform", "seq_logprob"],
        default="auto",
        help="How to assign sample probabilities",
    )
    parser.add_argument(
        "--equivalence-mode",
        choices=["ast", "text", "ast_or_text"],
        default="ast_or_text",
        help="Code equivalence mode for clustering",
    )
    parser.add_argument(
        "--primary-uncertainty",
        choices=["entropy", "density"],
        default="entropy",
        help="Which metric is copied to the generic `uncertainty` field",
    )
    parser.add_argument("--include-greedy-in-clusters", action="store_true", help="Include greedy answer as an additional cluster sample")
    parser.add_argument("--keep-question", action="store_true", help="Keep question field in output")
    parser.add_argument("--keep-prediction", action="store_true", help="Keep greedy answer in output")
    parser.add_argument("--keep-samples", action="store_true", help="Keep used sampled answers in output")
    parser.add_argument("--keep-gt", action="store_true", help="Keep gt and entry_point in output")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N records")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file instead of resuming")
    parser.add_argument("--verbose", action="store_true", help="Print per-record progress")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    if args.overwrite and os.path.exists(args.output_file):
        os.remove(args.output_file)

    done = load_done_ids(args.output_file)
    if args.verbose:
        safe_print(f"existing records: {len(done)}")
        safe_print(f"INPUT_FILE = {args.input_file}")
        safe_print(f"CORRECT_FILE = {args.correct_file}")
        safe_print(f"OUTPUT_FILE = {args.output_file}")
        safe_print("Correctness source: external correct file only")
        safe_print(f"Equivalence mode: {args.equivalence_mode}")
        safe_print(f"Probability mode: {args.prob_mode}")
        safe_print(f"Primary uncertainty alias: {args.primary_uncertainty}")

    correct_records = load_jsonl(args.correct_file)
    correct_map = build_correct_map(correct_records, args.question_id_key, args.correct_key)

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
                safe_print(f"skip line {line_idx}: json parse error: {e}")
                continue

            qid = get_question_id(record)
            if qid in done:
                continue

            try:
                result = process_one_record(record, correct_map, args)
                if result["id"] not in correct_map:
                    missing_correct += 1
                append_jsonl(args.output_file, result)
                processed += 1

                if args.verbose:
                    safe_print(
                        f"done: {result['id']} | line={line_idx} | "
                        f"correct={result['correct']} | "
                        f"clusters={result['num_clusters']} | "
                        f"sd_conf={result['semantic_density_confidence']:.4f} | "
                        f"sd_unc={result['semantic_density_uncertainty']:.4f} | "
                        f"se={result['semantic_entropy']:.4f} | "
                        f"se_norm={result['semantic_entropy_normalized']:.4f}"
                    )
            except KeyboardInterrupt:
                safe_print("KeyboardInterrupt detected. Exiting gracefully.")
                raise
            except Exception as e:
                err = {
                    "id": qid,
                    "question": record.get("question", ""),
                    "correct": int(correct_map.get(qid, 0)),
                    "error": str(e),
                }
                append_jsonl(args.output_file, err)
                processed += 1
                safe_print(f"error: {qid} | line={line_idx} | {e}")

    print(f"[info] wrote/processed {processed} new records to {args.output_file}", file=sys.stderr)
    if missing_correct > 0:
        print(
            f"[warn] {missing_correct} processed records were missing in the correctness file and were set to correct=0",
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

    _parser = _ap.ArgumentParser(description="semantic_entropy / qwen / humaneval / answer_only (standalone)")
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
    if "semantic_entropy" not in {"max_seq_prob", "mean_tok_ent", "max_tok_prob", "ppl"}:
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
    if "semantic_entropy" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "semantic_entropy", "family": "qwen",
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

