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
import textwrap
from collections import deque
from typing import Any, Dict, List, Optional, Tuple


def log(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
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
    out = []
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception as e:
            print(f"[warn] skip invalid json line {line_no}: {e}", file=sys.stderr)
    return out


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
                pass
    return done


def record_id(obj: Dict[str, Any], line_idx: int = 0) -> str:
    for k in ["id", "question_id", "task_id", "qid"]:
        if obj.get(k) is not None:
            return str(obj[k])
    return f"line_{line_idx}"


def load_correct_map(path: str) -> Dict[str, int]:
    mp = {}
    for obj in load_records(path):
        qid = None
        for k in ["question_id", "id", "task_id", "qid"]:
            if obj.get(k) is not None:
                qid = str(obj[k])
                break
        if qid is None:
            continue
        val = obj.get("correct", obj.get("is_correct", obj.get("passed", None)))
        try:
            mp[qid] = int(val)
        except Exception:
            mp[qid] = 1 if bool(val) else 0
    return mp


def load_problem_map(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    mp = {}
    for obj in load_records(path):
        qid = None
        for k in ["question_id", "id", "task_id", "qid"]:
            if obj.get(k) is not None:
                qid = str(obj[k])
                break
        if qid is not None:
            mp[qid] = obj
    return mp


def get_question(record: Dict[str, Any], qid: str, problem_map: Dict[str, Dict[str, Any]]) -> str:
    for k in ["question", "prompt", "problem", "input", "query"]:
        if record.get(k) is not None:
            return str(record[k])
    prob = problem_map.get(qid, {})
    for k in ["question", "prompt", "problem", "input", "query"]:
        if prob.get(k) is not None:
            return str(prob[k])
    return ""


def get_responses(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    xs = record.get("responses", [])
    return [x for x in xs if isinstance(x, dict)] if isinstance(xs, list) else []


def get_reason(resp: Dict[str, Any]) -> str:
    for k in ["llm reason", "llm_reason", "reason", "rationale", "solution"]:
        if resp.get(k) is not None:
            return str(resp[k]).strip()
    return ""


def get_answer(resp: Dict[str, Any]) -> str:
    for k in ["llm answer", "llm_answer", "answer", "text", "raw_text", "prediction"]:
        if resp.get(k) is not None:
            return str(resp[k]).strip()
    return ""


def extract_boxed(text: str) -> str:
    s = str(text)
    idx = s.find(r"\boxed")
    if idx < 0:
        return s
    start = s.find("{", idx)
    if start < 0:
        return s
    depth = 0
    for pos in range(start, len(s)):
        if s[pos] == "{":
            depth += 1
        elif s[pos] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1:pos]
    return s[start + 1:]


def strip_code(text: str) -> str:
    s = extract_boxed(str(text)).strip()
    m = re.search(r"<CODE>\s*(.*?)\s*</CODE>", s, re.S | re.I)
    if m:
        s = m.group(1).strip()
    m = re.search(r"```(?:python|py)?\s*(.*?)```", s, re.S | re.I)
    if m:
        s = m.group(1).strip()
    s = re.sub(r"^\s*(?:answer|output|code)\s*:\s*", "", s, flags=re.I)
    return s.strip("\n")


def function_header(prompt: str) -> Optional[str]:
    pattern = re.compile(r"^def\s+\w+\s*\(.*?\)\s*(?:->\s*[^:]+)?\s*:", re.M)
    ms = list(pattern.finditer(prompt or ""))
    return ms[-1].group(0) if ms else None


def wrap_code(code: str, prompt: str) -> str:
    code = strip_code(code)
    if re.search(r"^\s*def\s+\w+\s*\(", code, re.M):
        return textwrap.dedent(code).strip()
    header = function_header(prompt)
    if not header:
        return textwrap.dedent(code).strip()
    body = textwrap.dedent(code).strip("\n") or "pass"
    return header.rstrip() + "\n" + textwrap.indent(body, "    ")


def trunc(s: str, max_chars: int) -> str:
    s = str(s).strip()
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...[truncated]"


def node_text(resp: Dict[str, Any], question: str, mode: str, max_chars: int) -> str:
    reason = get_reason(resp)
    code = wrap_code(get_answer(resp), question)
    if mode == "reasoning_augmented":
        text = reason
    elif mode == "answer":
        text = f"Final Python solution:\n```python\n{code}\n```"
    else:
        parts = []
        if reason:
            parts.append("Reasoning:\n" + reason)
        if code:
            parts.append("Final Python solution:\n```python\n" + code + "\n```")
        text = "\n\n".join(parts)
    return trunc(text, max_chars)


class NLIScorer:
    def __init__(self, model_name: str, device: str, dtype: str, max_length: int, batch_size: int):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        torch_dtype = self._dtype(dtype)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch_dtype, trust_remote_code=True
        )
        self.model.to(torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"))
        self.model.eval()
        self.con_id = self._label_id(["contradiction", "contradictory"], 0)
        self.neu_id = self._label_id(["neutral"], 1)
        self.ent_id = self._label_id(["entailment", "entails"], 2)

    def _dtype(self, dtype: str):
        import torch
        if dtype in {"auto", "fp32", "float32"}:
            return torch.float32
        if dtype in {"fp16", "float16", "half"}:
            return torch.float16
        if dtype in {"bf16", "bfloat16"}:
            return torch.bfloat16
        raise ValueError(f"Unsupported dtype: {dtype}")

    def _label_id(self, names: List[str], default: int) -> int:
        label2id = getattr(self.model.config, "label2id", {}) or {}
        for k, v in label2id.items():
            lk = str(k).lower()
            if any(n in lk for n in names):
                return int(v)
        return default

    def score_pairs(self, premises: List[str], hypotheses: List[str]) -> List[Dict[str, float]]:
        import torch
        assert len(premises) == len(hypotheses)
        out = []
        for st in range(0, len(premises), self.batch_size):
            p = premises[st:st + self.batch_size]
            h = hypotheses[st:st + self.batch_size]
            inputs = self.tokenizer(
                p, h, truncation=True, max_length=self.max_length,
                padding=True, return_tensors="pt"
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.inference_mode():
                probs = torch.softmax(self.model(**inputs).logits.float(), dim=-1).cpu().tolist()
            for row in probs:
                con = row[self.con_id] if self.con_id < len(row) else row[0]
                ent = row[self.ent_id] if self.ent_id < len(row) else row[-1]
                neu = row[self.neu_id] if self.neu_id < len(row) else max(0.0, 1.0 - con - ent)
                out.append({"contradiction": float(con), "neutral": float(neu), "entailment": float(ent)})
        return out


def make_pair(question: str, a: str, b: str, include_question: bool) -> Tuple[str, str]:
    if include_question and question.strip():
        p = "HumanEval task:\n" + question.strip() + "\n\nResponse A:\n" + a
        h = "For the same HumanEval task, Response B states:\n" + b
    else:
        p = "Response A:\n" + a
        h = "Response B states:\n" + b
    return p, h


def build_nli_graph(nodes: List[str], question: str, nli: NLIScorer, include_q: bool,
                    ent_thr: float, con_thr: float) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
    n = len(nodes)
    adj = [[0] * n for _ in range(n)]
    premises, hypotheses = [], []
    for i in range(n):
        for j in range(i + 1, n):
            p1, h1 = make_pair(question, nodes[i], nodes[j], include_q)
            p2, h2 = make_pair(question, nodes[j], nodes[i], include_q)
            premises.extend([p1, p2])
            hypotheses.extend([h1, h2])
    scores = nli.score_pairs(premises, hypotheses) if premises else []
    details = []
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            sij, sji = scores[k], scores[k + 1]
            k += 2
            min_ent = min(sij["entailment"], sji["entailment"])
            max_con = max(sij["contradiction"], sji["contradiction"])
            same = (min_ent >= ent_thr and max_con <= con_thr)
            adj[i][j] = adj[j][i] = 1 if same else 0
            details.append({
                "i": i, "j": j, "same_semantic_set": bool(same),
                "entailment_i_to_j": sij["entailment"],
                "contradiction_i_to_j": sij["contradiction"],
                "neutral_i_to_j": sij["neutral"],
                "entailment_j_to_i": sji["entailment"],
                "contradiction_j_to_i": sji["contradiction"],
                "neutral_j_to_i": sji["neutral"],
                "min_bidirectional_entailment": float(min_ent),
                "max_bidirectional_contradiction": float(max_con),
            })
    return adj, details


def components(adj: List[List[int]]) -> List[List[int]]:
    n = len(adj)
    seen = [False] * n
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        comp = []
        q = deque([s])
        seen[s] = True
        while q:
            u = q.popleft()
            comp.append(u)
            for v, e in enumerate(adj[u]):
                if e and not seen[v]:
                    seen[v] = True
                    q.append(v)
        comps.append(comp)
    return comps


def shortest(adj: List[List[int]], src: int) -> List[Optional[int]]:
    n = len(adj)
    dist: List[Optional[int]] = [None] * n
    dist[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for v, e in enumerate(adj[u]):
            if e and dist[v] is None:
                dist[v] = int(dist[u]) + 1
                q.append(v)
    return dist


def graph_metrics(adj: List[List[int]]) -> Dict[str, Any]:
    n = len(adj)
    deg = [int(sum(r)) for r in adj]
    deg_mat = [[0] * n for _ in range(n)]
    lap = [[0] * n for _ in range(n)]
    for i in range(n):
        deg_mat[i][i] = deg[i]
        for j in range(n):
            lap[i][j] = deg[i] if i == j else -adj[i][j]
    comps = components(adj)
    ecc = []
    disconnected_directed = 0
    for i in range(n):
        d = shortest(adj, i)
        finite = [x for x in d if x is not None]
        disconnected_directed += sum(1 for j, x in enumerate(d) if j != i and x is None)
        ecc.append(max(finite) if len(finite) > 1 else 0)
    deg_mean = float(sum(deg) / n) if n else None
    sets = len(comps)
    if n <= 1:
        sem_unc = 0.0
        deg_unc = 0.0
    else:
        sem_unc = float((sets - 1) / (n - 1))
        deg_unc = float(1.0 - ((deg_mean or 0.0) / (n - 1)))
    return {
        "number_of_semantic_sets": sets,
        "sum_of_eigenvalues_of_graph_laplacian": float(sum(deg)),
        "degree": deg,
        "degree_mean": deg_mean,
        "degree_sum": float(sum(deg)) if n else None,
        "degree_max": max(deg) if deg else None,
        "degree_min": min(deg) if deg else None,
        "degree_uncertainty_0_1": deg_unc,
        "eccentricity": ecc,
        "eccentricity_mean": float(sum(ecc) / len(ecc)) if ecc else None,
        "eccentricity_max": max(ecc) if ecc else None,
        "eccentricity_min": min(ecc) if ecc else None,
        "eccentricity_mean_finite": float(sum(ecc) / len(ecc)) if ecc else None,
        "num_disconnected_pairs": int(disconnected_directed // 2),
        "semantic_set_uncertainty_0_1": sem_unc,
        "components": comps,
        "degree_matrix": deg_mat,
        "laplacian_matrix": lap,
    }


def process(record: Dict[str, Any], idx: int, args, correct_map, problem_map, nli) -> Dict[str, Any]:
    qid = record_id(record, idx)
    correct = correct_map.get(qid, None)
    correct = int(correct) if correct is not None else None
    question = get_question(record, qid, problem_map)
    responses = get_responses(record)
    if not responses:
        return {"id": qid, "correct": correct, "error": "no_responses"}
    nodes = [node_text(r, question, args.node_text_mode, args.max_node_chars) for r in responses]
    adj, pair_details = build_nli_graph(
        nodes, question, nli, args.include_question_in_nli,
        args.entailment_threshold, args.contradiction_threshold
    )
    m = graph_metrics(adj)
    target_answer = get_answer(responses[0])
    out = {
        "id": qid,
        "correct": correct,
        "target_response_index": 0,
        "target_answer": strip_code(target_answer),
        "num_responses": len(responses),
        "num_graph_nodes": len(nodes),
        "number_of_semantic_sets": m["number_of_semantic_sets"],
        "sum_of_eigenvalues_of_graph_laplacian": m["sum_of_eigenvalues_of_graph_laplacian"],
        "degree": m["degree"],
        "degree_mean": m["degree_mean"],
        "degree_sum": m["degree_sum"],
        "degree_max": m["degree_max"],
        "degree_min": m["degree_min"],
        "degree_uncertainty_0_1": m["degree_uncertainty_0_1"],
        "eccentricity": m["eccentricity"],
        "eccentricity_mean": m["eccentricity_mean"],
        "eccentricity_max": m["eccentricity_max"],
        "eccentricity_min": m["eccentricity_min"],
        "eccentricity_mean_finite": m["eccentricity_mean_finite"],
        "num_disconnected_pairs": m["num_disconnected_pairs"],
        "semantic_set_uncertainty_0_1": m["semantic_set_uncertainty_0_1"],
        "uncertainty": m["semantic_set_uncertainty_0_1"],
        "metrics": {
            "Number of Semantic Sets": m["number_of_semantic_sets"],
            "Sum of Eigenvalues of the Graph Laplacian": m["sum_of_eigenvalues_of_graph_laplacian"],
            "The Degree Matrix": {
                "degree": m["degree"],
                "degree_mean": m["degree_mean"],
                "degree_sum": m["degree_sum"],
                "degree_max": m["degree_max"],
                "degree_min": m["degree_min"],
                "degree_uncertainty_0_1": m["degree_uncertainty_0_1"],
            },
            "Eccentricity": {
                "eccentricity": m["eccentricity"],
                "eccentricity_mean": m["eccentricity_mean"],
                "eccentricity_max": m["eccentricity_max"],
                "eccentricity_min": m["eccentricity_min"],
                "eccentricity_mean_finite": m["eccentricity_mean_finite"],
                "num_disconnected_pairs": m["num_disconnected_pairs"],
            },
        },
        "meta": {
            "method_group": "MAP_graph_methods_NLI",
            "dataset": "humaneval_reasoning_sampling",
            "correctness_source": "external_correct_file_only_by_question_id",
            "question_source": "input_question_or_problem_file_by_question_id",
            "ignored_field": "correct answer",
            "uses_nli": True,
            "nli_model": args.nli_model,
            "node_text_mode": args.node_text_mode,
            "include_question_in_nli": bool(args.include_question_in_nli),
            "entailment_threshold": args.entailment_threshold,
            "contradiction_threshold": args.contradiction_threshold,
            "graph_edge_definition": "bidirectional NLI entailment with contradiction filter",
            "default_uncertainty": "semantic_set_uncertainty_0_1",
        },
    }
    if args.keep_question:
        out["question"] = question
    if args.keep_prediction:
        out["prediction"] = wrap_code(target_answer, question)
    if args.keep_components:
        out["components"] = m["components"]
    if args.keep_matrices:
        out["adjacency_matrix"] = adj
        out["degree_matrix"] = m["degree_matrix"]
        out["laplacian_matrix"] = m["laplacian_matrix"]
    if args.keep_pair_details:
        out["pair_details"] = pair_details
    if args.keep_nodes:
        out["nodes"] = nodes
    return out


def parse_args():
    p = argparse.ArgumentParser(description="NLI-based MAP/graph UQ for HumanEval reasoning-sampling data")
    p.add_argument("--input-file", required=True)
    p.add_argument("--correct-file", required=True)
    p.add_argument("--problem-file", default=None)
    p.add_argument("--output-file", required=True)
    p.add_argument("--nli-model", default="microsoft/deberta-large-mnli")
    p.add_argument("--hf-endpoint", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="fp32", help="Default fp32 avoids DeBERTa fp16 dtype mismatch")
    p.add_argument("--nli-max-length", type=int, default=512)
    p.add_argument("--nli-batch-size", type=int, default=8)
    p.add_argument("--node-text-mode", choices=["all", "reasoning_augmented", "answer"], default="all")
    p.add_argument("--max-node-chars", type=int, default=4000)
    p.add_argument("--include-question-in-nli", action="store_true")
    p.add_argument("--entailment-threshold", type=float, default=0.50)
    p.add_argument("--contradiction-threshold", type=float, default=0.50)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--keep-question", action="store_true")
    p.add_argument("--keep-prediction", action="store_true")
    p.add_argument("--keep-components", action="store_true")
    p.add_argument("--keep-matrices", action="store_true")
    p.add_argument("--keep-pair-details", action="store_true")
    p.add_argument("--keep-nodes", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    log(f"INPUT_FILE = {args.input_file}")
    log(f"CORRECT_FILE = {args.correct_file}")
    log(f"PROBLEM_FILE = {args.problem_file}")
    log(f"OUTPUT_FILE = {args.output_file}")
    log("Correctness source: external correct file only, matched by input id -> correct_file.question_id")
    log("Question source: input question/prompt first; otherwise --problem-file matched by input id -> problem_file.question_id")
    log("Ignored field: record['correct answer']")
    log(f"NLI model = {args.nli_model}")
    log(f"Node text mode = {args.node_text_mode}")
    log(f"include_question_in_nli = {args.include_question_in_nli}")
    log(f"entailment_threshold = {args.entailment_threshold}")
    log(f"contradiction_threshold = {args.contradiction_threshold}")
    log(f"dtype = {args.dtype}")

    records = load_records(args.input_file)
    correct_map = load_correct_map(args.correct_file)
    problem_map = load_problem_map(args.problem_file)
    if not correct_map:
        raise ValueError("No correctness labels loaded from correct file.")
    log(f"Loaded records: {len(records)}")
    log(f"Loaded correctness labels: {len(correct_map)}")
    if problem_map:
        log(f"Loaded problem prompts: {len(problem_map)}")
    log("Loading NLI model...")
    nli = NLIScorer(args.nli_model, args.device, args.dtype, args.nli_max_length, args.nli_batch_size)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    done = set() if args.overwrite else load_done_ids(args.output_file)
    mode = "w" if args.overwrite else "a"
    processed = 0
    with open(args.output_file, mode, encoding="utf-8") as fout:
        for line_idx, rec in enumerate(records, 1):
            if args.limit is not None and processed >= args.limit:
                break
            qid = record_id(rec, line_idx)
            if qid in done:
                continue
            try:
                res = process(rec, line_idx, args, correct_map, problem_map, nli)
                fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                fout.flush()
                processed += 1
                if args.verbose:
                    log(f"done: {qid} | line={line_idx} | correct={res.get('correct')} | sets={res.get('number_of_semantic_sets')} | lap_sum={res.get('sum_of_eigenvalues_of_graph_laplacian')} | degree_mean={res.get('degree_mean')} | ecc_mean={res.get('eccentricity_mean')} | unc={res.get('uncertainty')}")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                fout.write(json.dumps({"id": qid, "error": str(e)}, ensure_ascii=False) + "\n")
                fout.flush()
                if args.verbose:
                    log(f"error: {qid} | line={line_idx} | {e}")
    log(f"Finished. processed={processed}")
    log(f"Saved to: {args.output_file}")


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

    _parser = _ap.ArgumentParser(description="num_set / qwen / humaneval / reasoning_augmented (standalone)")
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
    if "num_set" not in {"max_seq_prob", "mean_tok_ent", "max_tok_prob", "ppl"}:
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
    if "num_set" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "num_set", "family": "qwen",
                           "dataset": "humaneval", "setting": "reasoning_augmented", "model_loading": False}, indent=2))
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

