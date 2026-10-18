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
import json
import math
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from math_verify import parse, verify





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
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[warn] skip bad json line {line_no}: {e}", file=sys.stderr)
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


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





def get_record_id(record: Dict[str, Any], fallback: int) -> str:
    for key in ["id", "question_id", "qid", "task_id"]:
        if key in record and record[key] is not None:
            return str(record[key])
    return f"line_{fallback}"


def get_question(record: Dict[str, Any]) -> str:
    for key in ["question", "problem", "prompt", "input"]:
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def get_correct_answer(record: Dict[str, Any]) -> str:
    for key in ["correct answer", "correct_answer", "answer", "ground_truth", "gt"]:
        if key in record and record[key] is not None:
            val = record[key]
            if isinstance(val, dict):
                if val.get("value") is not None:
                    return str(val["value"]).strip()
                aliases = val.get("aliases")
                if isinstance(aliases, list) and aliases:
                    return str(aliases[0]).strip()
            return str(val).strip()
    return ""


def get_responses(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    responses = record.get("responses")
    if isinstance(responses, list):
        return [r for r in responses if isinstance(r, dict)]

    
    samples = record.get("samples")
    if isinstance(samples, list):
        out = []
        for s in samples:
            if isinstance(s, dict):
                out.append({
                    "llm reason": str(s.get("reason", s.get("text", s.get("raw_text", ""))) or ""),
                    "llm answer": str(s.get("answer", s.get("llm answer", s.get("text", s.get("raw_text", "")))) or ""),
                })
            else:
                out.append({"llm reason": "", "llm answer": str(s)})
        return out

    
    greedy = record.get("greedy")
    if isinstance(greedy, dict):
        text = str(greedy.get("text", greedy.get("raw_text", "")) or "")
        return [{"llm reason": text, "llm answer": text}]

    return []


def extract_response_answer(resp: Dict[str, Any]) -> str:
    for key in ["llm answer", "llm_answer", "answer", "final_answer", "prediction"]:
        val = resp.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()

    
    for key in ["llm reason", "llm_reason", "reason", "text", "raw_text"]:
        val = resp.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def extract_response_reason(resp: Dict[str, Any]) -> str:
    for key in ["llm reason", "llm_reason", "reason", "text", "raw_text"]:
        val = resp.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""





_parse_cache: Dict[str, Any] = {}
_equiv_cache: Dict[Tuple[str, str], bool] = {}


def normalize_for_cache(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def safe_parse_math(text: str) -> Any:
    text = str(text or "")
    key = normalize_for_cache(text)
    if key in _parse_cache:
        return _parse_cache[key]
    try:
        parsed = parse(text)
    except Exception:
        parsed = None
    _parse_cache[key] = parsed
    return parsed


def mathv_correct(ref_text: str, target_text: str) -> bool:
    try:
        ref = safe_parse_math(ref_text)
        target = safe_parse_math(target_text)
        if ref is None or target is None:
            return False
        return bool(verify(ref, target))
    except Exception:
        return False


def mathv_equivalent(a: str, b: str) -> bool:
    ka = normalize_for_cache(a)
    kb = normalize_for_cache(b)
    key = (ka, kb) if ka <= kb else (kb, ka)
    if key in _equiv_cache:
        return _equiv_cache[key]
    val = mathv_correct(a, b)
    _equiv_cache[key] = val
    return val





def build_equivalence_graph(answers: List[str], self_loop: bool = False) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    n = len(answers)
    A = np.zeros((n, n), dtype=float)
    pair_details: List[Dict[str, Any]] = []

    if self_loop:
        for i in range(n):
            A[i, i] = 1.0

    for i in range(n):
        for j in range(i + 1, n):
            same = mathv_equivalent(answers[i], answers[j])
            if same:
                A[i, j] = 1.0
                A[j, i] = 1.0
            pair_details.append({
                "i": i,
                "j": j,
                "answer_i": answers[i],
                "answer_j": answers[j],
                "same_semantics": bool(same),
                "judge": "math_verify",
            })
    return A, pair_details


def connected_components_count(A: np.ndarray) -> Tuple[int, List[int], List[List[int]]]:
    n = int(A.shape[0])
    if n == 0:
        return 0, [], []

    visited = [False] * n
    comp_ids = [-1] * n
    comps: List[List[int]] = []

    for start in range(n):
        if visited[start]:
            continue
        cid = len(comps)
        comp_nodes: List[int] = []
        stack = [start]
        visited[start] = True
        comp_ids[start] = cid
        while stack:
            u = stack.pop()
            comp_nodes.append(u)
            neighbors = np.where(A[u] > 0)[0].tolist()
            for v in neighbors:
                if not visited[v]:
                    visited[v] = True
                    comp_ids[v] = cid
                    stack.append(v)
        comps.append(sorted(comp_nodes))

    return len(comps), comp_ids, comps


def degree_matrix(A: np.ndarray) -> np.ndarray:
    deg = A.sum(axis=1)
    return np.diag(deg)


def degree_scores(A: np.ndarray) -> Dict[str, Any]:
    deg = A.sum(axis=1)
    if len(deg) == 0:
        return {
            "degree_vector": [],
            "degree_matrix": [],
            "degree_sum": 0.0,
            "degree_mean": 0.0,
            "degree_max": 0.0,
            "degree_min": 0.0,
        }
    D = np.diag(deg)
    return {
        "degree_vector": [float(x) for x in deg.tolist()],
        "degree_matrix": D.tolist(),
        "degree_sum": float(deg.sum()),
        "degree_mean": float(deg.mean()),
        "degree_max": float(deg.max()),
        "degree_min": float(deg.min()),
    }


def laplacian_info(A: np.ndarray) -> Dict[str, Any]:
    n = int(A.shape[0])
    if n == 0:
        return {
            "laplacian_matrix": [],
            "laplacian_eigenvalues": [],
            "laplacian_eigenvalue_sum": 0.0,
        }
    D = degree_matrix(A)
    L = D - A
    try:
        eigvals = np.linalg.eigvalsh(L)
    except Exception:
        eigvals = np.linalg.eigvals(L).real
    eigvals = np.real(eigvals)
    return {
        "laplacian_matrix": L.tolist(),
        "laplacian_eigenvalues": [float(x) for x in eigvals.tolist()],
        "laplacian_eigenvalue_sum": float(eigvals.sum()),
    }


def shortest_path_matrix(A: np.ndarray) -> np.ndarray:
    n = int(A.shape[0])
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    dist = np.full((n, n), np.inf, dtype=float)
    for i in range(n):
        dist[i, i] = 0.0

    rows, cols = np.where(A > 0)
    for i, j in zip(rows, cols):
        if i != j:
            dist[i, j] = 1.0

    
    for k in range(n):
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
    return dist


def clean_inf_for_json(x: Any) -> Any:
    if isinstance(x, float):
        if math.isinf(x) or math.isnan(x):
            return None
        return x
    if isinstance(x, list):
        return [clean_inf_for_json(v) for v in x]
    return x


def eccentricity_scores(A: np.ndarray) -> Dict[str, Any]:
    n = int(A.shape[0])
    if n == 0:
        return {
            "eccentricity_vector": [],
            "eccentricity_mean": None,
            "eccentricity_max": None,
            "eccentricity_min": None,
            "graph_disconnected": False,
            "shortest_path_matrix": [],
        }

    dist = shortest_path_matrix(A)
    ecc: List[float] = []
    disconnected = False

    for i in range(n):
        row = dist[i]
        finite = row[np.isfinite(row)]
        if len(finite) < n:
            disconnected = True
        if len(finite) == 0:
            ecc_i = math.inf
        else:
            ecc_i = float(finite.max())
        ecc.append(ecc_i)

    finite_ecc = [x for x in ecc if np.isfinite(x)]
    if not finite_ecc:
        mean_ecc = max_ecc = min_ecc = None
    else:
        mean_ecc = float(np.mean(finite_ecc))
        max_ecc = float(np.max(finite_ecc))
        min_ecc = float(np.min(finite_ecc))

    return {
        "eccentricity_vector": clean_inf_for_json(ecc),
        "eccentricity_mean": mean_ecc,
        "eccentricity_max": max_ecc,
        "eccentricity_min": min_ecc,
        "graph_disconnected": bool(disconnected),
        "shortest_path_matrix": clean_inf_for_json(dist.tolist()),
    }


def component_summary(comps: List[List[int]], answers: List[str], correctness: List[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    n = len(answers)
    for cid, members in enumerate(comps):
        size = len(members)
        out.append({
            "semantic_set_id": cid,
            "size": size,
            "fraction": float(size / n) if n else 0.0,
            "member_indices": members,
            "member_answers": [answers[i] for i in members],
            "representative_answer": answers[members[0]] if members else "",
            "num_correct_members": int(sum(correctness[i] for i in members)),
        })
    return out





def compute_map_for_record(
    record: Dict[str, Any],
    record_index: int,
    target_index: int,
    max_responses: Optional[int],
    self_loop: bool,
    keep_reasons: bool,
    keep_pair_details: bool,
    keep_matrices: bool,
) -> Dict[str, Any]:
    rid = get_record_id(record, record_index)
    question = get_question(record)
    correct_answer = get_correct_answer(record)
    responses = get_responses(record)

    if max_responses is not None and max_responses > 0:
        responses = responses[:max_responses]

    answers = [extract_response_answer(r) for r in responses]
    reasons = [extract_response_reason(r) for r in responses]
    valid_nodes = [(i, a) for i, a in enumerate(answers) if str(a).strip()]

    original_indices = [i for i, _ in valid_nodes]
    node_answers = [a for _, a in valid_nodes]
    node_reasons = [reasons[i] if i < len(reasons) else "" for i in original_indices]

    response_correctness = [int(mathv_correct(correct_answer, a)) for a in node_answers]
    num_correct_responses = int(sum(response_correctness))
    num_nodes = len(node_answers)

    
    if 0 <= target_index < len(answers):
        target_answer = answers[target_index]
        target_correct = int(mathv_correct(correct_answer, target_answer)) if target_answer else 0
    else:
        target_answer = node_answers[0] if node_answers else ""
        target_correct = int(mathv_correct(correct_answer, target_answer)) if target_answer else 0
        target_index = original_indices[0] if original_indices else -1

    A, pair_details = build_equivalence_graph(node_answers, self_loop=self_loop)
    num_sets, comp_ids, comps = connected_components_count(A)
    deg_info = degree_scores(A)
    lap_info = laplacian_info(A)
    ecc_info = eccentricity_scores(A)
    sem_sets = component_summary(comps, node_answers, response_correctness)

    
    
    if num_nodes <= 1:
        semantic_set_uncertainty_0_1 = 0.0
    else:
        semantic_set_uncertainty_0_1 = float((num_sets - 1) / (num_nodes - 1))

    answer_correct_rate = float(num_correct_responses / num_nodes) if num_nodes else None

    graph_nodes: List[Dict[str, Any]] = []
    for new_idx, old_idx in enumerate(original_indices):
        item = {
            "node_index": new_idx,
            "response_index": old_idx,
            "answer": node_answers[new_idx],
            "correct": response_correctness[new_idx],
            "semantic_set_id": comp_ids[new_idx] if new_idx < len(comp_ids) else None,
        }
        if keep_reasons:
            item["reason"] = node_reasons[new_idx]
        graph_nodes.append(item)

    semantic_graph = {
        "num_nodes": int(num_nodes),
        "num_edges": int(np.sum(A) / 2) if not self_loop else int((np.sum(A) - np.trace(A)) / 2),
        "self_loop": bool(self_loop),
        "component_ids": comp_ids,
    }
    if keep_matrices:
        semantic_graph["adjacency_matrix"] = A.tolist()
    if keep_pair_details:
        semantic_graph["pair_equivalence"] = pair_details

    metrics = {
        "number_of_semantic_sets": int(num_sets),
        "sum_of_eigenvalues_of_graph_laplacian": lap_info["laplacian_eigenvalue_sum"],
        "laplacian_eigenvalues": lap_info["laplacian_eigenvalues"],
        "the_degree_matrix": {
            "degree_vector": deg_info["degree_vector"],
            "degree_sum": deg_info["degree_sum"],
            "degree_mean": deg_info["degree_mean"],
            "degree_max": deg_info["degree_max"],
            "degree_min": deg_info["degree_min"],
        },
        "eccentricity": {
            "eccentricity_vector": ecc_info["eccentricity_vector"],
            "eccentricity_mean": ecc_info["eccentricity_mean"],
            "eccentricity_max": ecc_info["eccentricity_max"],
            "eccentricity_min": ecc_info["eccentricity_min"],
            "graph_disconnected": ecc_info["graph_disconnected"],
        },
    }
    if keep_matrices:
        metrics["the_degree_matrix"]["degree_matrix"] = deg_info["degree_matrix"]
        metrics["laplacian_matrix"] = lap_info["laplacian_matrix"]
        metrics["eccentricity"]["shortest_path_matrix"] = ecc_info["shortest_path_matrix"]

    result: Dict[str, Any] = {
        "id": rid,
        "correct_answer": correct_answer,
        "target_response_index": int(target_index),
        "target_answer": target_answer,
        "correct": int(target_correct),
        "num_responses": len(responses),
        "num_graph_nodes": int(num_nodes),
        "num_correct_responses": num_correct_responses,
        "answer_correct_rate": answer_correct_rate,

        
        "number_of_semantic_sets": int(num_sets),
        "sum_of_eigenvalues_of_graph_laplacian": lap_info["laplacian_eigenvalue_sum"],
        "degree_mean": deg_info["degree_mean"],
        "degree_sum": deg_info["degree_sum"],
        "degree_max": deg_info["degree_max"],
        "degree_min": deg_info["degree_min"],
        "eccentricity_mean": ecc_info["eccentricity_mean"],
        "eccentricity_max": ecc_info["eccentricity_max"],
        "eccentricity_min": ecc_info["eccentricity_min"],
        "semantic_set_uncertainty_0_1": semantic_set_uncertainty_0_1,
        "uncertainty": semantic_set_uncertainty_0_1,

        "semantic_sets": sem_sets,
        "graph_nodes": graph_nodes,
        "semantic_graph": semantic_graph,
        "metrics": metrics,
        "meta": {
            "method": "MAP-graph-methods",
            "dataset_type": "reasoning_sampling",
            "answer_equivalence_judge": "math_verify",
            "correctness_source": "record['correct answer']",
            "node_source": "responses[*]['llm answer']",
            "max_responses": max_responses,
            "self_loop": self_loop,
        },
    }

    if question:
        result["question"] = question

    return result





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute MAP / graph-based uncertainty metrics for reasoning-sampling results."
    )
    parser.add_argument("--input-file", required=True, help="Input JSONL / JSON file")
    parser.add_argument("--output-file", required=True, help="Output JSONL file")
    parser.add_argument("--target-index", type=int, default=0, help="Response index used for reporting correct field")
    parser.add_argument("--max-responses", type=int, default=None, help="Use at most this many responses per record")
    parser.add_argument("--self-loop", action="store_true", help="Include self loops in adjacency matrix")
    parser.add_argument("--keep-reasons", action="store_true", help="Keep full llm reason text in graph_nodes")
    parser.add_argument("--keep-pair-details", action="store_true", help="Keep pairwise equivalence details")
    parser.add_argument("--keep-matrices", action="store_true", help="Keep adjacency/degree/laplacian/shortest-path matrices")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N records")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file instead of resuming")
    parser.add_argument("--verbose", action="store_true", help="Print per-record progress")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"INPUT_FILE = {args.input_file}", flush=True)
    print(f"OUTPUT_FILE = {args.output_file}", flush=True)
    print("Correctness source: record['correct answer'] + math_verify", flush=True)
    print("Equivalence judge: math_verify over responses[*]['llm answer']", flush=True)
    print("MAP metrics: Number of Semantic Sets, Laplacian eigenvalue sum, Degree Matrix, Eccentricity", flush=True)

    records = load_records(args.input_file)
    if args.limit is not None:
        records = records[: args.limit]

    if args.overwrite and os.path.exists(args.output_file):
        os.remove(args.output_file)

    done = load_done_ids(args.output_file) if not args.overwrite else set()
    if done:
        print(f"Resume mode: found {len(done)} existing ids in output", flush=True)

    processed = 0
    skipped = 0

    for idx, record in enumerate(records, start=1):
        rid = get_record_id(record, idx)
        if str(rid) in done:
            skipped += 1
            continue
        try:
            result = compute_map_for_record(
                record=record,
                record_index=idx,
                target_index=args.target_index,
                max_responses=args.max_responses,
                self_loop=args.self_loop,
                keep_reasons=args.keep_reasons,
                keep_pair_details=args.keep_pair_details,
                keep_matrices=args.keep_matrices,
            )
            append_jsonl(args.output_file, result)
            processed += 1

            if args.verbose:
                print(
                    f"done: {result['id']} | line={idx} | correct={result['correct']} | "
                    f"nodes={result['num_graph_nodes']} | sets={result['number_of_semantic_sets']} | "
                    f"lap_sum={result['sum_of_eigenvalues_of_graph_laplacian']:.4f} | "
                    f"degree_mean={result['degree_mean']:.4f} | ecc_mean={result['eccentricity_mean']}",
                    flush=True,
                )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            err = {
                "id": rid,
                "error": str(e),
                "meta": {
                    "method": "MAP-graph-methods",
                    "dataset_type": "reasoning_sampling",
                },
            }
            append_jsonl(args.output_file, err)
            print(f"error: {rid} | line={idx} | {e}", flush=True)

    print(f"Finished. processed={processed}, skipped={skipped}, total_input_records={len(records)}", flush=True)
    print(f"Saved to: {args.output_file}", flush=True)


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

    _parser = _ap.ArgumentParser(description="ecc / llama / gsm8k / reasoning_augmented (standalone)")
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
    if "ecc" not in {"max_seq_prob", "mean_tok_ent", "max_tok_prob", "ppl"}:
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
    if "ecc" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "ecc", "family": "llama",
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

