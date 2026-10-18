#!/usr/bin/env python3
# Original paper/author repository: https://github.com/zlin7/UQ-NLG
# Reference repository: https://github.com/IINemo/lm-polygraph

if __name__ == "__main__" and not globals().get("_UQ_EXECUTING", False):
    
    import ast as _uq_ast
    from pathlib import Path as _uq_Path
    _uq_tree = _uq_ast.parse(_uq_Path(__file__).read_text())
    _uq_helper = next(n for n in _uq_tree.body if isinstance(n, _uq_ast.FunctionDef) and n.name == "_uq_standalone")
    exec(compile(_uq_ast.Module(body=[_uq_helper], type_ignores=[]), __file__, "exec"), globals())
    _uq_standalone()
    raise SystemExit(0)











import json
import math
import os
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import eigvalsh
from math_verify import parse, verify





INPUT_FILE = __import__("os").environ.get("UQ_INPUT_FILE", "")
OUTPUT_FILE = __import__("os").environ.get("UQ_OUTPUT_FILE", "")


MAX_SAMPLES = 10




INCLUDE_GREEDY_IN_GRAPH = False



USE_SELF_LOOP = False


ENABLE_PARSE_CACHE = True





write_lock = threading.Lock()
print_lock = threading.Lock()





def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs, flush=True)


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
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
                    done.add(obj["id"])
            except Exception:
                continue
    return done





_parse_cache: Dict[str, Any] = {}


def safe_parse(text: str):
    text = "" if text is None else str(text)
    if ENABLE_PARSE_CACHE and text in _parse_cache:
        return _parse_cache[text]

    try:
        parsed = parse(text)
    except Exception:
        parsed = None

    if ENABLE_PARSE_CACHE:
        _parse_cache[text] = parsed
    return parsed


def mathv_equivalent(text_a: str, text_b: str) -> bool:
    
    try:
        a = safe_parse(text_a)
        b = safe_parse(text_b)
        if a is None or b is None:
            return False
        return bool(verify(a, b))
    except Exception:
        return False


def mathv_correct(ref_text: str, target_text: str) -> bool:
    
    try:
        ref = safe_parse(ref_text)
        target = safe_parse(target_text)
        if ref is None or target is None:
            return False
        return bool(verify(ref, target))
    except Exception:
        return False





def get_question_id(obj: Dict[str, Any]) -> str:
    for key in ["id", "question_id", "qid"]:
        if key in obj and obj[key] is not None:
            return str(obj[key])
    return ""


def get_question(obj: Dict[str, Any]) -> str:
    return str(obj.get("question", "")).strip()


def get_gt_text(obj: Dict[str, Any]) -> str:
    gt = obj.get("gt", obj.get("ground_truth", obj.get("answer", "")))
    if isinstance(gt, dict):
        if gt.get("value") is not None:
            return str(gt["value"]).strip()
        aliases = gt.get("aliases", [])
        if isinstance(aliases, list) and len(aliases) > 0:
            return str(aliases[0]).strip()
        return ""
    return str(gt).strip()


def extract_text_field(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        for key in ["text", "raw_text", "answer", "response"]:
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
    return ""


def get_sampled_answers(obj: Dict[str, Any]) -> List[str]:
    outs = []

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





def build_equivalence_graph_matrix(answers: List[str]) -> Tuple[np.ndarray, List[Tuple[int, int, bool]]]:
    
    n = len(answers)
    A = np.zeros((n, n), dtype=float)
    pair_details = []

    for i in range(n):
        if USE_SELF_LOOP:
            A[i, i] = 1.0

    for i in range(n):
        for j in range(i + 1, n):
            same = mathv_equivalent(answers[i], answers[j])
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
            "degree_vector": [],
            "degree_sum": 0.0,
            "degree_mean": 0.0,
            "degree_max": 0.0,
            "degree_min": 0.0,
        }
    return {
        "degree_vector": deg.tolist(),
        "degree_sum": float(deg.sum()),
        "degree_mean": float(deg.mean()),
        "degree_max": float(deg.max()),
        "degree_min": float(deg.min()),
    }


def laplacian_matrix(A: np.ndarray) -> np.ndarray:
    D = degree_matrix(A)
    return D - A


def laplacian_eigenvalue_sum(A: np.ndarray) -> Dict[str, Any]:
    
    if A.shape[0] == 0:
        return {
            "laplacian_eigenvalues": [],
            "laplacian_eigenvalue_sum": 0.0,
        }

    L = laplacian_matrix(A)
    eigvals = eigvalsh(L)
    eigvals = np.real(eigvals)
    return {
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


def eccentricity_scores(A: np.ndarray) -> Dict[str, Any]:
    
    n = A.shape[0]
    if n == 0:
        return {
            "eccentricity_vector": [],
            "eccentricity_mean": None,
            "eccentricity_max": None,
            "eccentricity_min": None,
            "graph_disconnected": False,
        }

    dist = shortest_path_matrix(A)
    ecc = []
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
    if len(finite_ecc) == 0:
        mean_ecc = None
        max_ecc = None
        min_ecc = None
    else:
        mean_ecc = float(np.mean(finite_ecc))
        max_ecc = float(np.max(finite_ecc))
        min_ecc = float(np.min(finite_ecc))

    return {
        "eccentricity_vector": ecc,
        "eccentricity_mean": mean_ecc,
        "eccentricity_max": max_ecc,
        "eccentricity_min": min_ecc,
        "graph_disconnected": disconnected,
    }





def compute_graph_methods_for_item(obj: Dict[str, Any]) -> Dict[str, Any]:
    qid = get_question_id(obj)
    question = get_question(obj)
    gt_text = get_gt_text(obj)
    greedy_answer = get_greedy_answer(obj)
    sampled_answers = get_sampled_answers(obj)[:MAX_SAMPLES]
    verbalized_conf = get_verbalized_confidence(obj)

    greedy_correct = int(mathv_correct(gt_text, greedy_answer))

    if INCLUDE_GREEDY_IN_GRAPH:
        node_answers = [greedy_answer] + sampled_answers
        node_types = ["greedy"] + ["sample"] * len(sampled_answers)
    else:
        node_answers = sampled_answers
        node_types = ["sample"] * len(sampled_answers)

    A, pair_details = build_equivalence_graph_matrix(node_answers)

    
    num_semantic_sets = connected_components_count(A)

    
    lap_info = laplacian_eigenvalue_sum(A)

    
    deg_info = degree_scores(A)

    
    ecc_info = eccentricity_scores(A)

    result = {
        "id": qid,
        "question": question,
        "gt": gt_text,
        "greedy_answer": greedy_answer,
        "sampled_answers": sampled_answers,
        "greedy_correct": greedy_correct,
        "graph_nodes": [
            {"node_index": i, "node_type": node_types[i], "answer": node_answers[i]}
            for i in range(len(node_answers))
        ],
        "semantic_graph": {
            "num_nodes": int(A.shape[0]),
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
        "metrics": {
            "number_of_semantic_sets": int(num_semantic_sets),
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
        },
        "meta": {
            "dataset": "gsm8k",
            "answer_equivalence_judge": "math_verify",
            "max_samples_used": MAX_SAMPLES,
            "include_greedy_in_graph": INCLUDE_GREEDY_IN_GRAPH,
            "use_self_loop": USE_SELF_LOOP,
        },
    }

    if verbalized_conf is not None:
        result["verbalized_confidence_raw"] = verbalized_conf
        result["verbalized_confidence_0_1"] = verbalized_conf / 100.0 if verbalized_conf > 1.0 else verbalized_conf

    return result





def main():
    out_dir = os.path.dirname(OUTPUT_FILE)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    done = load_done_ids(OUTPUT_FILE)
    safe_print(f"existing records: {len(done)}")

    with open(INPUT_FILE, "r", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as e:
                safe_print(f"skip invalid json line: {line_idx} | {e}")
                continue

            qid = get_question_id(obj)
            if qid in done:
                continue

            try:
                result = compute_graph_methods_for_item(obj)
                append_jsonl(OUTPUT_FILE, result)
                safe_print(
                    f"done: {qid} | line={line_idx} | "
                    f"correct={result['greedy_correct']} | "
                    f"semantic_sets={result['metrics']['number_of_semantic_sets']} | "
                    f"lap_sum={result['metrics']['sum_of_eigenvalues_of_graph_laplacian']:.4f} | "
                    f"degree_mean={result['metrics']['the_degree_matrix']['degree_mean']:.4f} | "
                    f"ecc_mean={result['metrics']['eccentricity']['eccentricity_mean']}"
                    )
            except Exception as e:
                err = {
                    "id": qid,
                    "question": obj.get("question", ""),
                    "error": str(e),
                }
                append_jsonl(OUTPUT_FILE, err)
                safe_print(f"error: {qid} | line={line_idx} | {e}")


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

    _parser = _ap.ArgumentParser(description="ecc / qwen / gsm8k / answer_only (standalone)")
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
    if "ecc" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "ecc", "family": "qwen",
                           "dataset": "gsm8k", "setting": "answer_only", "model_loading": False}, indent=2))
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

