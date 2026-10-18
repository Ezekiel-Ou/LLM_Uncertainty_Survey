#!/usr/bin/env python3
# Original paper/author repository: https://github.com/zlin7/UQ-NLG
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
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np





write_lock = threading.Lock()
print_lock = threading.Lock()





def safe_print(*args, **kwargs) -> None:
    with print_lock:
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
    with write_lock:
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
    return ""


def get_question(obj: Dict[str, Any]) -> str:
    return str(obj.get("question", obj.get("prompt", ""))).strip()


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


def get_sampled_answers(obj: Dict[str, Any]) -> List[str]:
    outs: List[str] = []

    if "sampled_answers" in obj and isinstance(obj["sampled_answers"], list):
        for s in obj["sampled_answers"]:
            t = extract_text_field(s)
            if t:
                outs.append(t)
        return outs

    if "samples" in obj and isinstance(obj["samples"], list):
        for s in obj["samples"]:
            t = extract_text_field(s)
            if t:
                outs.append(t)
        return outs

    if "sampled_responses" in obj and isinstance(obj["sampled_responses"], list):
        for s in obj["sampled_responses"]:
            t = extract_text_field(s)
            if t:
                outs.append(t)
        return outs

    return outs


def get_verbalized_confidence(obj: Dict[str, Any]) -> Optional[float]:
    verbalized = obj.get("verbalized")
    if isinstance(verbalized, dict):
        conf = verbalized.get("confidence", None)
        if conf is None:
            return None
        try:
            return float(conf)
        except Exception:
            return None
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
            original_name = target_func.name
            target_func.name = "__candidate__"
            
            target_func.body = [ast.Pass()]
            mod = ast.Module(body=[target_func], type_ignores=[])
            ast.fix_missing_locations(mod)
            source = ast.unparse(mod)
            first_line = source.split("\n", 1)[0]
            target_func.name = original_name
            return first_line
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


def canonicalize_function_ast(tree: ast.AST, entry_point: Optional[str]) -> ast.AST:
    
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
    lines = []
    for line in code.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
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
    }

    candidates: List[Tuple[str, bool]] = []
    if cleaned.strip():
        candidates.append((textwrap.dedent(cleaned).strip(), False))

        
        if not re.search(r"^\s*def\s+", cleaned, flags=re.MULTILINE):
            signature = extract_signature_from_question(question, entry_point)
            if signature is not None:
                wrapped = signature + "\n" + indent_body(cleaned)
                candidates.append((wrapped, True))

    for candidate, wrapped in candidates:
        try:
            tree = ast.parse(candidate)
            tree = canonicalize_function_ast(tree, entry_point)
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





def build_equivalence_graph_matrix(
    answers: List[str],
    question: str,
    entry_point: Optional[str],
    equivalence_mode: str,
    use_self_loop: bool,
) -> Tuple[np.ndarray, List[Tuple[int, int, bool]]]:
    
    n = len(answers)
    A = np.zeros((n, n), dtype=float)
    pair_details: List[Tuple[int, int, bool]] = []

    for i in range(n):
        if use_self_loop:
            A[i, i] = 1.0

    for i in range(n):
        for j in range(i + 1, n):
            same = codes_equivalent(
                answers[i],
                answers[j],
                question=question,
                entry_point=entry_point,
                mode=equivalence_mode,
            )
            pair_details.append((i, j, same))
            if same:
                A[i, j] = 1.0
                A[j, i] = 1.0

    return A, pair_details





def connected_components_count(A: np.ndarray) -> int:
    n = A.shape[0]
    if n == 0:
        return 0

    visited = [False] * n
    comp = 0

    for start in range(n):
        if visited[start]:
            continue
        comp += 1
        stack = [start]
        visited[start] = True
        while stack:
            u = stack.pop()
            neighbors = np.where(A[u] > 0)[0].tolist()
            for v in neighbors:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
    return comp


def degree_matrix(A: np.ndarray) -> np.ndarray:
    deg = A.sum(axis=1)
    return np.diag(deg)


def degree_scores(A: np.ndarray) -> Dict[str, Any]:
    deg = A.sum(axis=1)
    if len(deg) == 0:
        return {
            "degree_matrix": [],
            "degree_vector": [],
            "degree_sum": 0.0,
            "degree_mean": 0.0,
            "degree_max": 0.0,
            "degree_min": 0.0,
        }
    D = np.diag(deg)
    return {
        "degree_matrix": D.tolist(),
        "degree_vector": deg.tolist(),
        "degree_sum": float(deg.sum()),
        "degree_mean": float(deg.mean()),
        "degree_max": float(deg.max()),
        "degree_min": float(deg.min()),
    }


def laplacian_matrix(A: np.ndarray) -> np.ndarray:
    return degree_matrix(A) - A


def laplacian_eigenvalue_sum(A: np.ndarray) -> Dict[str, Any]:
    if A.shape[0] == 0:
        return {
            "laplacian_matrix": [],
            "laplacian_eigenvalues": [],
            "laplacian_eigenvalue_sum": 0.0,
        }

    L = laplacian_matrix(A)
    eigvals = np.linalg.eigvalsh(L)
    eigvals = np.real(eigvals)
    
    eigvals = np.where(np.abs(eigvals) < 1e-12, 0.0, eigvals)
    return {
        "laplacian_matrix": L.tolist(),
        "laplacian_eigenvalues": eigvals.tolist(),
        "laplacian_eigenvalue_sum": float(eigvals.sum()),
    }


def shortest_path_matrix(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    dist = np.full((n, n), np.inf, dtype=float)
    for i in range(n):
        dist[i, i] = 0.0

    edges = np.where(A > 0)
    for i, j in zip(edges[0], edges[1]):
        if i != j:
            dist[i, j] = 1.0

    for k in range(n):
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])

    return dist


def _json_safe_float(x: float) -> Optional[float]:
    if x is None:
        return None
    if not np.isfinite(x):
        return None
    return float(x)


def eccentricity_scores(A: np.ndarray) -> Dict[str, Any]:
    n = A.shape[0]
    if n == 0:
        return {
            "shortest_path_matrix": [],
            "eccentricity_vector": [],
            "eccentricity_mean": None,
            "eccentricity_max": None,
            "eccentricity_min": None,
            "graph_disconnected": False,
        }

    dist = shortest_path_matrix(A)
    ecc: List[Optional[float]] = []
    disconnected = False

    for i in range(n):
        row = dist[i]
        finite = row[np.isfinite(row)]
        if len(finite) < n:
            disconnected = True
        if len(finite) == 0:
            ecc_i = None
        else:
            ecc_i = float(finite.max())
        ecc.append(ecc_i)

    finite_ecc = [x for x in ecc if x is not None and np.isfinite(x)]
    if len(finite_ecc) == 0:
        mean_ecc = None
        max_ecc = None
        min_ecc = None
    else:
        mean_ecc = float(np.mean(finite_ecc))
        max_ecc = float(np.max(finite_ecc))
        min_ecc = float(np.min(finite_ecc))

    
    dist_safe: List[List[Optional[float]]] = []
    for row in dist.tolist():
        dist_safe.append([_json_safe_float(float(x)) for x in row])

    return {
        "shortest_path_matrix": dist_safe,
        "eccentricity_vector": ecc,
        "eccentricity_mean": mean_ecc,
        "eccentricity_max": max_ecc,
        "eccentricity_min": min_ecc,
        "graph_disconnected": disconnected,
    }


def normalized_graph_scores(num_semantic_sets: int, deg_info: Dict[str, Any], lap_info: Dict[str, Any], ecc_info: Dict[str, Any], n: int) -> Dict[str, Any]:
    
    if n <= 1:
        return {
            "semantic_set_uncertainty_0_1": 0.0,
            "degree_mean_normalized": 0.0,
            "laplacian_sum_normalized": 0.0,
            "eccentricity_mean_normalized": None,
        }

    max_degree = float(n - 1)
    max_degree_sum = float(n * (n - 1))
    ecc_mean = ecc_info.get("eccentricity_mean")

    return {
        
        "semantic_set_uncertainty_0_1": float((num_semantic_sets - 1) / (n - 1)),
        
        "degree_mean_normalized": float(deg_info.get("degree_mean", 0.0) / max_degree),
        "laplacian_sum_normalized": float(lap_info.get("laplacian_eigenvalue_sum", 0.0) / max_degree_sum),
        
        "eccentricity_mean_normalized": None if ecc_mean is None else float(ecc_mean / (n - 1)),
    }





def compute_graph_methods_for_item(
    obj: Dict[str, Any],
    correct_map: Dict[str, int],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], bool]:
    qid = get_question_id(obj)
    question = get_question(obj)
    entry_point = get_entry_point(obj)
    greedy_answer = get_greedy_answer(obj)
    sampled_answers = get_sampled_answers(obj)[: args.max_samples]
    verbalized_conf = get_verbalized_confidence(obj)

    missing_correct = False
    if qid in correct_map:
        correct = correct_map[qid]
    else:
        correct = 0
        missing_correct = True

    if args.include_greedy_in_graph:
        node_answers = [greedy_answer] + sampled_answers
        node_types = ["greedy"] + ["sample"] * len(sampled_answers)
    else:
        node_answers = sampled_answers
        node_types = ["sample"] * len(sampled_answers)

    A, pair_details = build_equivalence_graph_matrix(
        node_answers,
        question=question,
        entry_point=entry_point,
        equivalence_mode=args.equivalence_mode,
        use_self_loop=args.use_self_loop,
    )

    num_semantic_sets = connected_components_count(A)
    lap_info = laplacian_eigenvalue_sum(A)
    deg_info = degree_scores(A)
    ecc_info = eccentricity_scores(A)
    n = int(A.shape[0])
    normalized = normalized_graph_scores(num_semantic_sets, deg_info, lap_info, ecc_info, n)

    metrics = {
        "number_of_semantic_sets": int(num_semantic_sets),
        "sum_of_eigenvalues_of_graph_laplacian": lap_info["laplacian_eigenvalue_sum"],
        "laplacian_eigenvalues": lap_info["laplacian_eigenvalues"],
        "laplacian_matrix": lap_info["laplacian_matrix"],
        "the_degree_matrix": {
            "degree_matrix": deg_info["degree_matrix"],
            "degree_vector": deg_info["degree_vector"],
            "degree_sum": deg_info["degree_sum"],
            "degree_mean": deg_info["degree_mean"],
            "degree_max": deg_info["degree_max"],
            "degree_min": deg_info["degree_min"],
        },
        "eccentricity": {
            "shortest_path_matrix": ecc_info["shortest_path_matrix"],
            "eccentricity_vector": ecc_info["eccentricity_vector"],
            "eccentricity_mean": ecc_info["eccentricity_mean"],
            "eccentricity_max": ecc_info["eccentricity_max"],
            "eccentricity_min": ecc_info["eccentricity_min"],
            "graph_disconnected": ecc_info["graph_disconnected"],
        },
        "normalized": normalized,
    }

    result: Dict[str, Any] = {
        "id": qid,
        "correct": correct,
        
        "number_of_semantic_sets": int(num_semantic_sets),
        "sum_of_eigenvalues_of_graph_laplacian": lap_info["laplacian_eigenvalue_sum"],
        "degree_mean": deg_info["degree_mean"],
        "degree_sum": deg_info["degree_sum"],
        "eccentricity_mean": ecc_info["eccentricity_mean"],
        "semantic_set_uncertainty_0_1": normalized["semantic_set_uncertainty_0_1"],
        "metrics": metrics,
        "semantic_graph": {
            "num_nodes": n,
            "adjacency_matrix": A.tolist(),
            "pair_equivalence": [
                {
                    "i": int(i),
                    "j": int(j),
                    "same_semantics": bool(same),
                }
                for i, j, same in pair_details
            ],
        },
        "meta": {
            "method": "MAP-style graph uncertainty",
            "dataset": "humaneval",
            "answer_equivalence_judge": f"code_{args.equivalence_mode}",
            "max_samples_used": args.max_samples,
            "num_samples_available": len(get_sampled_answers(obj)),
            "num_samples_used": len(sampled_answers),
            "include_greedy_in_graph": bool(args.include_greedy_in_graph),
            "use_self_loop": bool(args.use_self_loop),
            "correctness_source": args.correct_file,
            "correctness_recomputed": False,
        },
    }

    if args.keep_question:
        result["question"] = question
    if args.keep_prediction:
        result["greedy_answer"] = greedy_answer
    if args.keep_samples:
        result["sampled_answers"] = sampled_answers
    if args.keep_nodes:
        result["graph_nodes"] = [
            {
                "node_index": i,
                "node_type": node_types[i],
                "answer": node_answers[i],
                "parse_ok": bool(code_signature(node_answers[i], question, entry_point).get("parse_ok")),
                "wrapped": bool(code_signature(node_answers[i], question, entry_point).get("wrapped")),
            }
            for i in range(len(node_answers))
        ]
    if verbalized_conf is not None:
        result["verbalized_confidence_raw"] = verbalized_conf
        result["verbalized_confidence_0_1"] = verbalized_conf / 100.0 if verbalized_conf > 1.0 else verbalized_conf

    return result, missing_correct





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute HumanEval MAP-style graph uncertainty metrics."
    )
    parser.add_argument("--input-file", required=True, help="HumanEval stage1 model output JSONL")
    parser.add_argument("--correct-file", required=True, help="External correctness JSONL file")
    parser.add_argument("--output-file", required=True, help="Output JSONL path")

    parser.add_argument("--question-id-key", default="question_id", help="ID key in correct file")
    parser.add_argument("--correct-key", default="correct", help="Correctness key in correct file")

    parser.add_argument("--max-samples", type=int, default=10, help="Maximum samples used to build graph")
    parser.add_argument(
        "--include-greedy-in-graph",
        action="store_true",
        help="Use nodes = [greedy] + samples instead of samples only",
    )
    parser.add_argument("--use-self-loop", action="store_true", help="Set A[i,i]=1")
    parser.add_argument(
        "--equivalence-mode",
        choices=["ast", "text", "ast_or_text"],
        default="ast_or_text",
        help="Code equivalence judge for HumanEval completions",
    )

    parser.add_argument("--keep-question", action="store_true", help="Keep question in output")
    parser.add_argument("--keep-prediction", action="store_true", help="Keep greedy answer in output")
    parser.add_argument("--keep-samples", action="store_true", help="Keep sampled answers in output")
    parser.add_argument("--keep-nodes", action="store_true", help="Keep graph node details in output")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file instead of resuming")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.overwrite and os.path.exists(args.output_file):
        os.remove(args.output_file)

    correct_records = load_jsonl(args.correct_file)
    correct_map = build_correct_map(correct_records, args.question_id_key, args.correct_key)

    done = load_done_ids(args.output_file)
    if args.verbose:
        safe_print(f"existing records: {len(done)}")
        safe_print(f"correct records loaded: {len(correct_map)}")

    total = 0
    written = 0
    missing_correct = 0

    with open(args.input_file, "r", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                obj = json.loads(line)
            except Exception as e:
                safe_print(f"skip invalid json line: {line_idx} | {e}")
                continue

            qid = get_question_id(obj)
            if not qid:
                qid = f"line_{line_idx}"
                obj["id"] = qid

            if qid in done:
                continue

            try:
                result, miss = compute_graph_methods_for_item(obj, correct_map, args)
                if miss:
                    missing_correct += 1
                append_jsonl(args.output_file, result)
                written += 1

                if args.verbose:
                    safe_print(
                        f"done: {qid} | line={line_idx} | correct={result['correct']} | "
                        f"semantic_sets={result['number_of_semantic_sets']} | "
                        f"lap_sum={result['sum_of_eigenvalues_of_graph_laplacian']:.4f} | "
                        f"degree_mean={result['degree_mean']:.4f} | "
                        f"ecc_mean={result['eccentricity_mean']}"
                    )
            except KeyboardInterrupt:
                safe_print("KeyboardInterrupt detected. Exiting gracefully.")
                raise
            except Exception as e:
                err = {
                    "id": qid,
                    "correct": correct_map.get(qid, 0),
                    "question": obj.get("question", ""),
                    "error": str(e),
                }
                append_jsonl(args.output_file, err)
                written += 1
                safe_print(f"error: {qid} | line={line_idx} | {e}")

    print(f"[info] processed={total}, wrote={written}, output={args.output_file}", file=sys.stderr)
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

    _parser = _ap.ArgumentParser(description="deg / llama / humaneval / answer_only (standalone)")
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
    if "deg" not in {"max_seq_prob", "mean_tok_ent", "max_tok_prob", "ppl"}:
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
    if "deg" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "deg", "family": "llama",
                           "dataset": "humaneval", "setting": "answer_only", "model_loading": False}, indent=2))
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

