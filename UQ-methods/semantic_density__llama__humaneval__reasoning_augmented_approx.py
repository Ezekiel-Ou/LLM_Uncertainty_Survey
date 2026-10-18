#!/usr/bin/env python3
# Original paper/author repository: https://github.com/cognizant-ai-labs/semantic-density-paper
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
from collections import deque
from typing import Any, Dict, List, Optional, Tuple


print_lock = threading.Lock()


def safe_print(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    with print_lock:
        print(*args, **kwargs)


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def clamp01(x: float) -> float:
    if math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))





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

    
    records: List[Dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
        except Exception as e:
            safe_print(f"[warn] skip invalid json line {line_no}: {e}", file=sys.stderr)
    return records


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


def load_correct_map(path: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for obj in load_records(path):
        qid = None
        for key in ["question_id", "id", "task_id", "qid"]:
            if obj.get(key) is not None:
                qid = str(obj[key])
                break
        if qid is None:
            continue

        val = obj.get("correct", obj.get("is_correct", obj.get("passed", None)))
        try:
            out[qid] = int(val)
        except Exception:
            out[qid] = 1 if bool(val) else 0
    return out


def load_problem_map(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"problem file does not exist: {path}")

    out: Dict[str, Dict[str, Any]] = {}
    for obj in load_records(path):
        qid = None
        for key in ["question_id", "task_id", "id", "qid"]:
            if obj.get(key) is not None:
                qid = str(obj[key])
                break
        if qid is not None:
            out[qid] = obj
    return out





def get_record_id(record: Dict[str, Any], line_idx: int) -> str:
    for key in ["id", "question_id", "task_id", "qid"]:
        if record.get(key) is not None:
            return str(record[key])
    return f"line_{line_idx}"


def get_responses(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    responses = record.get("responses", [])
    if isinstance(responses, list):
        return [r for r in responses if isinstance(r, dict)]
    return []


def get_llm_reason(response: Dict[str, Any]) -> str:
    for key in ["llm reason", "llm_reason", "reason", "rationale", "solution"]:
        if response.get(key) is not None:
            return str(response[key]).strip()
    return ""


def get_llm_answer(response: Dict[str, Any]) -> str:
    for key in ["llm answer", "llm_answer", "answer", "text", "raw_text", "prediction"]:
        if response.get(key) is not None:
            return str(response[key]).strip()
    return ""


def get_question_from_record_or_problem(
    record: Dict[str, Any],
    qid: str,
    problem_map: Dict[str, Dict[str, Any]],
) -> str:
    for key in ["question", "prompt", "problem", "input", "query"]:
        if record.get(key) is not None:
            return str(record[key])

    prob = problem_map.get(qid, {})
    for key in ["question", "prompt", "problem", "input", "query"]:
        if prob.get(key) is not None:
            return str(prob[key])

    return ""





def extract_boxed_content(text: str) -> str:
    s = str(text)
    marker = r"\boxed"
    idx = s.find(marker)
    if idx < 0:
        return s

    brace_start = s.find("{", idx + len(marker))
    if brace_start < 0:
        return s

    depth = 0
    for pos in range(brace_start, len(s)):
        ch = s[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[brace_start + 1:pos]

    return s[brace_start + 1:]


def strip_code_wrappers(text: str) -> str:
    s = extract_boxed_content(str(text)).strip()

    
    m = re.search(r"<CODE>\s*(.*?)\s*</CODE>", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1).strip()

    
    m = re.search(r"```(?:python|py)?\s*(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1).strip()

    
    s = re.sub(r"^\s*(?:answer|output|code)\s*:\s*", "", s, flags=re.IGNORECASE)

    return s.strip("\n")


def find_function_header(prompt: str) -> Optional[str]:
    if not prompt:
        return None

    pattern = re.compile(r"^def\s+\w+\s*\(.*?\)\s*(?:->\s*[^:]+)?\s*:", re.MULTILINE)
    matches = list(pattern.finditer(prompt))
    if not matches:
        return None
    return matches[-1].group(0)


def looks_like_full_function(code: str) -> bool:
    return bool(re.search(r"^\s*def\s+\w+\s*\(", code, flags=re.MULTILINE))


def wrap_body_with_header(code: str, prompt: str) -> str:
    code = strip_code_wrappers(code)

    if looks_like_full_function(code):
        return textwrap.dedent(code).strip()

    header = find_function_header(prompt)
    if not header:
        return textwrap.dedent(code).strip()

    body = textwrap.dedent(code).strip("\n")
    if not body.strip():
        body = "pass"

    return header.rstrip() + "\n" + textwrap.indent(body, "    ")


def normalize_text_code(code: str) -> str:
    s = strip_code_wrappers(code)
    s = textwrap.dedent(s).strip()
    lines = [ln.rstrip() for ln in s.splitlines()]
    s = "\n".join([ln for ln in lines if ln.strip()])
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def ast_key(code: str, prompt: str) -> Optional[str]:
    full = wrap_body_with_header(code, prompt)
    try:
        tree = ast.parse(full)
        return ast.dump(tree, annotate_fields=True, include_attributes=False)
    except Exception:
        return None


def code_equivalent(a: str, b: str, prompt: str, mode: str = "ast_or_text") -> bool:
    if mode in {"ast", "ast_or_text"}:
        ka = ast_key(a, prompt)
        kb = ast_key(b, prompt)

        if ka is not None and kb is not None:
            return ka == kb

        if mode == "ast":
            return False

    return normalize_text_code(a) == normalize_text_code(b)


def normalize_code_for_output(code: str, prompt: str) -> str:
    return wrap_body_with_header(code, prompt)





def get_prob_info(unit_obj: Any) -> Dict[str, Any]:
    if not isinstance(unit_obj, dict):
        return {}
    for key in ["prob info", "prob_info"]:
        pi = unit_obj.get(key)
        if isinstance(pi, dict):
            return pi
    return {}


def list_floats(x: Any) -> List[float]:
    vals: List[float] = []
    if isinstance(x, list):
        for item in x:
            v = safe_float(item, None)
            if v is not None:
                vals.append(float(v))
    else:
        v = safe_float(x, None)
        if v is not None:
            vals.append(float(v))
    return vals


def extract_token_logprobs(response: Dict[str, Any], prob_source: str) -> List[float]:
    token_logprobs: List[float] = []

    def add_unit(unit_obj: Dict[str, Any]) -> None:
        pi = get_prob_info(unit_obj)
        lp_raw = None
        for k in ["token logprob", "token_logprob", "chosen_logprob", "logprob"]:
            if k in pi:
                lp_raw = pi[k]
                break
        token_logprobs.extend(list_floats(lp_raw))

    if prob_source in {"all", "reasoning_augmented"}:
        steps = response.get("step-wise info", response.get("step_wise_info", []))
        if isinstance(steps, list):
            for step in steps:
                if isinstance(step, dict):
                    add_unit(step)

    if prob_source in {"all", "answer"}:
        answer_info = response.get("answer info", response.get("answer_info", None))
        if isinstance(answer_info, dict):
            add_unit(answer_info)

    return token_logprobs


def response_score_from_logprobs(
    response: Dict[str, Any],
    prob_source: str,
    length_normalization: str,
) -> Tuple[Optional[float], Dict[str, Any]]:
    lps = extract_token_logprobs(response, prob_source=prob_source)

    if not lps:
        return None, {
            "num_logprob_tokens": 0,
            "seq_logprob": None,
            "avg_logprob": None,
            "score": None,
        }

    seq_logprob = float(sum(lps))
    avg_logprob = float(seq_logprob / len(lps))

    if length_normalization == "sum":
        score = seq_logprob
    elif length_normalization == "mean":
        score = avg_logprob
    else:
        raise ValueError(f"Unsupported length_normalization: {length_normalization}")

    return float(score), {
        "num_logprob_tokens": len(lps),
        "seq_logprob": seq_logprob,
        "avg_logprob": avg_logprob,
        "score": float(score),
    }


def softmax(scores: List[float]) -> List[float]:
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    denom = sum(exps)
    if denom <= 0:
        return [1.0 / len(scores)] * len(scores)
    return [float(x / denom) for x in exps]





def truncate_text(text: str, max_chars: int) -> str:
    s = str(text).strip()
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...[truncated]"


def build_node_text(response: Dict[str, Any], question: str, node_text_mode: str, max_chars: int) -> str:
    reason = get_llm_reason(response)
    answer = wrap_body_with_header(get_llm_answer(response), question)

    if node_text_mode == "reasoning_augmented":
        text = reason
    elif node_text_mode == "answer":
        text = "Final Python solution:\n```python\n" + answer.strip() + "\n```"
    else:
        parts = []
        if reason.strip():
            parts.append("Reasoning:\n" + reason.strip())
        if answer.strip():
            parts.append("Final Python solution:\n```python\n" + answer.strip() + "\n```")
        text = "\n\n".join(parts)

    return truncate_text(text, max_chars)


def build_nli_pair(question: str, node_a: str, node_b: str, include_question: bool) -> Tuple[str, str]:
    if include_question and question.strip():
        q = "HumanEval task:\n" + question.strip() + "\n\n"
        premise = q + "Response A:\n" + node_a
        hypothesis = "For the same HumanEval task, Response B states:\n" + node_b
    else:
        premise = "Response A:\n" + node_a
        hypothesis = "Response B states:\n" + node_b
    return premise, hypothesis


class NLIScorer:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: str = "fp32",
        max_length: int = 512,
        batch_size: int = 8,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        torch_dtype = self.resolve_dtype(dtype)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        if device == "cuda" and torch.cuda.is_available():
            self.model.to(torch.device("cuda"))
        else:
            self.model.to(torch.device("cpu"))

        self.model.eval()

        self.contradiction_id = self.find_label_id(["contradiction", "contradictory"], default=0)
        self.neutral_id = self.find_label_id(["neutral"], default=1)
        self.entailment_id = self.find_label_id(["entailment", "entails"], default=2)

    def resolve_dtype(self, dtype: str):
        import torch

        if dtype == "auto":
            
            return torch.float32
        if dtype in {"fp32", "float32"}:
            return torch.float32
        if dtype in {"fp16", "float16", "half"}:
            return torch.float16
        if dtype in {"bf16", "bfloat16"}:
            return torch.bfloat16
        raise ValueError(f"Unsupported dtype: {dtype}")

    def find_label_id(self, names: List[str], default: int) -> int:
        label2id = getattr(self.model.config, "label2id", {}) or {}
        lower = {str(k).lower(): int(v) for k, v in label2id.items()}
        for k, v in lower.items():
            if any(name in k for name in names):
                return v
        return default

    def score_pairs(self, premises: List[str], hypotheses: List[str]) -> List[Dict[str, float]]:
        import torch

        assert len(premises) == len(hypotheses)
        results: List[Dict[str, float]] = []

        for start in range(0, len(premises), self.batch_size):
            batch_p = premises[start:start + self.batch_size]
            batch_h = hypotheses[start:start + self.batch_size]

            inputs = self.tokenizer(
                batch_p,
                batch_h,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.inference_mode():
                logits = self.model(**inputs).logits.float()
                probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()

            for row in probs:
                contradiction = row[self.contradiction_id] if self.contradiction_id < len(row) else row[0]
                entailment = row[self.entailment_id] if self.entailment_id < len(row) else row[-1]
                neutral = row[self.neutral_id] if self.neutral_id < len(row) else (
                    1.0 - contradiction - entailment
                )
                results.append(
                    {
                        "contradiction": float(contradiction),
                        "neutral": float(neutral),
                        "entailment": float(entailment),
                    }
                )

        return results





def build_code_similarity_and_adjacency(
    responses: List[Dict[str, Any]],
    question: str,
    equivalence_mode: str,
) -> Tuple[List[List[float]], List[List[int]], List[Dict[str, Any]]]:
    n = len(responses)
    sim = [[0.0 for _ in range(n)] for _ in range(n)]
    adj = [[0 for _ in range(n)] for _ in range(n)]
    details: List[Dict[str, Any]] = []

    for i in range(n):
        sim[i][i] = 1.0
        adj[i][i] = 0

    answers = [get_llm_answer(r) for r in responses]

    for i in range(n):
        for j in range(i + 1, n):
            same = code_equivalent(answers[i], answers[j], question, equivalence_mode)
            val = 1.0 if same else 0.0
            sim[i][j] = sim[j][i] = val
            adj[i][j] = adj[j][i] = 1 if same else 0
            details.append(
                {
                    "i": i,
                    "j": j,
                    "same_semantic_set": bool(same),
                    "similarity": val,
                    "answer_i": strip_code_wrappers(answers[i]),
                    "answer_j": strip_code_wrappers(answers[j]),
                }
            )

    return sim, adj, details


def build_nli_similarity_and_adjacency(
    responses: List[Dict[str, Any]],
    question: str,
    args: argparse.Namespace,
    nli: NLIScorer,
) -> Tuple[List[List[float]], List[List[int]], List[Dict[str, Any]], List[str]]:
    n = len(responses)
    sim = [[0.0 for _ in range(n)] for _ in range(n)]
    adj = [[0 for _ in range(n)] for _ in range(n)]
    details: List[Dict[str, Any]] = []

    nodes = [
        build_node_text(
            response=r,
            question=question,
            node_text_mode=args.node_text_mode,
            max_chars=args.max_node_chars,
        )
        for r in responses
    ]

    for i in range(n):
        sim[i][i] = 1.0
        adj[i][i] = 0

    premises: List[str] = []
    hypotheses: List[str] = []

    for i in range(n):
        for j in range(i + 1, n):
            p1, h1 = build_nli_pair(question, nodes[i], nodes[j], args.include_question_in_nli)
            p2, h2 = build_nli_pair(question, nodes[j], nodes[i], args.include_question_in_nli)
            premises.extend([p1, p2])
            hypotheses.extend([h1, h2])

    scores = nli.score_pairs(premises, hypotheses) if premises else []

    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            s_ij = scores[idx]
            s_ji = scores[idx + 1]
            idx += 2

            mean_entailment = 0.5 * (s_ij["entailment"] + s_ji["entailment"])
            mean_contradiction = 0.5 * (s_ij["contradiction"] + s_ji["contradiction"])
            min_entailment = min(s_ij["entailment"], s_ji["entailment"])
            max_contradiction = max(s_ij["contradiction"], s_ji["contradiction"])

            
            semantic_similarity = clamp01(
                args.entailment_weight * mean_entailment
                + (1.0 - args.entailment_weight) * (1.0 - mean_contradiction)
            )

            same = (
                min_entailment >= args.entailment_threshold
                and max_contradiction <= args.contradiction_threshold
            )

            sim[i][j] = sim[j][i] = semantic_similarity
            adj[i][j] = adj[j][i] = 1 if same else 0

            details.append(
                {
                    "i": i,
                    "j": j,
                    "same_semantic_set": bool(same),
                    "similarity": semantic_similarity,
                    "entailment_i_to_j": s_ij["entailment"],
                    "contradiction_i_to_j": s_ij["contradiction"],
                    "neutral_i_to_j": s_ij["neutral"],
                    "entailment_j_to_i": s_ji["entailment"],
                    "contradiction_j_to_i": s_ji["contradiction"],
                    "neutral_j_to_i": s_ji["neutral"],
                    "mean_bidirectional_entailment": float(mean_entailment),
                    "mean_bidirectional_contradiction": float(mean_contradiction),
                    "min_bidirectional_entailment": float(min_entailment),
                    "max_bidirectional_contradiction": float(max_contradiction),
                }
            )

    return sim, adj, details, nodes


def connected_components(adjacency: List[List[int]]) -> List[List[int]]:
    n = len(adjacency)
    seen = [False] * n
    comps: List[List[int]] = []

    for start in range(n):
        if seen[start]:
            continue

        q = deque([start])
        seen[start] = True
        comp: List[int] = []

        while q:
            u = q.popleft()
            comp.append(u)
            for v, edge in enumerate(adjacency[u]):
                if edge and not seen[v]:
                    seen[v] = True
                    q.append(v)

        comps.append(comp)

    return comps





def compute_response_probabilities(
    responses: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[float], List[Dict[str, Any]]]:
    scores: List[float] = []
    debug: List[Dict[str, Any]] = []
    missing: List[int] = []

    for idx, resp in enumerate(responses):
        score, dbg = response_score_from_logprobs(
            response=resp,
            prob_source=args.prob_source,
            length_normalization=args.length_normalization,
        )
        dbg["response_index"] = idx

        if score is None:
            missing.append(idx)
            if args.missing_prob == "error":
                raise ValueError(
                    f"Missing logprob for response {idx} under prob_source={args.prob_source}. "
                    f"Use --missing-prob uniform only if you explicitly want a fallback."
                )
            score = 0.0

        scores.append(float(score))
        debug.append(dbg)

    if missing and args.missing_prob == "uniform":
        probs = [1.0 / len(responses)] * len(responses) if responses else []
    else:
        probs = softmax(scores)

    return probs, debug


def compute_semantic_entropy(cluster_probs: List[float]) -> Tuple[float, float]:
    probs = [p for p in cluster_probs if p > 0]
    entropy = float(-sum(p * math.log(p) for p in probs))
    k = len(cluster_probs)
    if k <= 1:
        normalized = 0.0
    else:
        normalized = float(entropy / math.log(k))
    normalized = clamp01(normalized)
    return entropy, normalized


def compute_semantic_density(
    response_probs: List[float],
    similarity: List[List[float]],
) -> Tuple[float, float, List[float]]:
    n = len(response_probs)
    if n == 0:
        return 0.0, 1.0, []

    per_response_density: List[float] = []
    for i in range(n):
        density_i = 0.0
        for j in range(n):
            density_i += response_probs[j] * similarity[i][j]
        per_response_density.append(clamp01(density_i))

    density_confidence = 0.0
    for i in range(n):
        density_confidence += response_probs[i] * per_response_density[i]
    density_confidence = clamp01(density_confidence)

    density_uncertainty = float(1.0 - density_confidence)
    return density_confidence, density_uncertainty, per_response_density





def process_item(
    record: Dict[str, Any],
    line_idx: int,
    args: argparse.Namespace,
    correct_map: Dict[str, int],
    problem_map: Dict[str, Dict[str, Any]],
    nli: Optional[NLIScorer],
) -> Dict[str, Any]:
    qid = get_record_id(record, line_idx)
    question = get_question_from_record_or_problem(record, qid, problem_map)
    responses = get_responses(record)

    correct = correct_map.get(qid, None)
    if correct is not None:
        correct = int(correct)

    if not responses:
        return {
            "id": qid,
            "correct": correct,
            "error": "no_responses",
        }

    response_probs, prob_debug = compute_response_probabilities(responses, args)

    nodes: List[str] = []
    if args.cluster_mode == "code":
        similarity, adjacency, pair_details = build_code_similarity_and_adjacency(
            responses=responses,
            question=question,
            equivalence_mode=args.equivalence_mode,
        )
    elif args.cluster_mode == "nli":
        if nli is None:
            raise ValueError("NLI scorer is required when --cluster-mode nli.")
        similarity, adjacency, pair_details, nodes = build_nli_similarity_and_adjacency(
            responses=responses,
            question=question,
            args=args,
            nli=nli,
        )
    else:
        raise ValueError(f"Unsupported cluster_mode: {args.cluster_mode}")

    clusters = connected_components(adjacency)
    cluster_probs: List[float] = []
    for comp in clusters:
        cluster_probs.append(float(sum(response_probs[i] for i in comp)))

    semantic_entropy, semantic_entropy_norm = compute_semantic_entropy(cluster_probs)
    density_conf, density_unc, per_response_density = compute_semantic_density(response_probs, similarity)

    if args.primary_uncertainty == "entropy":
        uncertainty = semantic_entropy_norm
    else:
        uncertainty = density_unc

    target_answer = get_llm_answer(responses[0])

    result: Dict[str, Any] = {
        "id": qid,
        "correct": correct,

        "target_response_index": 0,
        "target_answer": strip_code_wrappers(target_answer),

        "semantic_entropy": semantic_entropy,
        "semantic_entropy_normalized": semantic_entropy_norm,
        "semantic_entropy_uncertainty": semantic_entropy_norm,

        "semantic_density_confidence": density_conf,
        "semantic_density_uncertainty": density_unc,

        "uncertainty": uncertainty,

        "num_responses": len(responses),
        "num_clusters": len(clusters),
        "cluster_probs": cluster_probs,
        "cluster_sizes": [len(c) for c in clusters],

        "meta": {
            "method_group": "Semantic_Entropy_Semantic_Density",
            "dataset": "humaneval_reasoning_sampling",
            "correctness_source": "external_correct_file_only_by_question_id",
            "question_source": "input_question_or_problem_file_by_question_id",
            "ignored_field": "correct answer",
            "cluster_mode": args.cluster_mode,
            "nli_model": args.nli_model if args.cluster_mode == "nli" else None,
            "node_text_mode": args.node_text_mode if args.cluster_mode == "nli" else None,
            "include_question_in_nli": bool(args.include_question_in_nli) if args.cluster_mode == "nli" else None,
            "equivalence_mode": args.equivalence_mode if args.cluster_mode == "code" else None,
            "prob_source": args.prob_source,
            "length_normalization": args.length_normalization,
            "missing_prob": args.missing_prob,
            "primary_uncertainty": args.primary_uncertainty,
        },
    }

    if args.keep_question:
        result["question"] = question
    if args.keep_prediction:
        result["prediction"] = normalize_code_for_output(target_answer, question)
    if args.keep_components:
        result["components"] = clusters
    if args.keep_matrices:
        result["adjacency_matrix"] = adjacency
        result["similarity_matrix"] = similarity
    if args.keep_pair_details:
        result["pair_details"] = pair_details
    if args.keep_nodes and nodes:
        result["nodes"] = nodes
    if args.keep_responses:
        result["responses_info"] = [
            {
                "response_index": i,
                "response_prob": response_probs[i],
                "density": per_response_density[i],
                "answer": strip_code_wrappers(get_llm_answer(resp)),
                "prob_debug": prob_debug[i],
            }
            for i, resp in enumerate(responses)
        ]
    if args.keep_cluster_details:
        result["clusters"] = [
            {
                "cluster_id": cid,
                "response_indices": comp,
                "cluster_prob": cluster_probs[cid],
                "cluster_size": len(comp),
            }
            for cid, comp in enumerate(clusters)
        ]

    return result





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Semantic Entropy and Semantic Density for HumanEval reasoning-sampling data."
    )

    parser.add_argument("--input-file", required=True, help="Input HumanEval reasoning JSONL/JSON file")
    parser.add_argument("--correct-file", required=True, help="External correct JSONL/JSON file")
    parser.add_argument("--output-file", required=True, help="Output JSONL file")
    parser.add_argument(
        "--problem-file",
        default=None,
        help="Optional HumanEval question file with question_id/question/ground_truth.",
    )

    parser.add_argument(
        "--cluster-mode",
        choices=["nli", "code"],
        default="nli",
        help="Semantic clustering mode. Default nli uses full reasoning+answer with NLI.",
    )
    parser.add_argument(
        "--equivalence-mode",
        choices=["ast", "text", "ast_or_text"],
        default="ast_or_text",
        help="Only used when --cluster-mode code.",
    )

    
    parser.add_argument(
        "--nli-model",
        default="microsoft/deberta-large-mnli",
        help="NLI model path/name. Examples: microsoft/deberta-large-mnli, potsawee/deberta-v3-large-mnli",
    )
    parser.add_argument("--hf-endpoint", default=None, help="Optional HF mirror, e.g. https://hf-mirror.com")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="fp32",
        help="NLI model dtype. Default fp32 avoids DeBERTa fp16 dtype mismatch.",
    )
    parser.add_argument("--nli-max-length", type=int, default=512)
    parser.add_argument("--nli-batch-size", type=int, default=8)
    parser.add_argument(
        "--node-text-mode",
        choices=["all", "reasoning_augmented", "answer"],
        default="all",
        help="Node text for NLI. Default all = reasoning + final code answer.",
    )
    parser.add_argument(
        "--max-node-chars",
        type=int,
        default=4000,
        help="Max chars per node before NLI tokenization.",
    )
    parser.add_argument(
        "--include-question-in-nli",
        action="store_true",
        help="Prepend HumanEval question/prompt to NLI pairs. Recommended.",
    )
    parser.add_argument("--entailment-threshold", type=float, default=0.50)
    parser.add_argument("--contradiction-threshold", type=float, default=0.50)
    parser.add_argument(
        "--entailment-weight",
        type=float,
        default=0.5,
        help="Similarity = w*mean_entailment + (1-w)*(1-mean_contradiction).",
    )

    
    parser.add_argument(
        "--prob-source",
        choices=["answer", "reasoning_augmented", "all"],
        default="all",
        help="Stored logprobs used for response probabilities. Default all.",
    )
    parser.add_argument(
        "--length-normalization",
        choices=["sum", "mean"],
        default="mean",
        help="How to turn token logprobs into response scores. Default mean for reasoning length variance.",
    )
    parser.add_argument(
        "--missing-prob",
        choices=["error", "uniform"],
        default="error",
        help="Missing logprob behavior. Default error avoids silent fallback.",
    )
    parser.add_argument(
        "--primary-uncertainty",
        choices=["entropy", "density"],
        default="entropy",
        help="Which method fills the generic uncertainty field.",
    )

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--keep-question", action="store_true")
    parser.add_argument("--keep-prediction", action="store_true")
    parser.add_argument("--keep-components", action="store_true")
    parser.add_argument("--keep-matrices", action="store_true")
    parser.add_argument("--keep-pair-details", action="store_true")
    parser.add_argument("--keep-nodes", action="store_true")
    parser.add_argument("--keep-responses", action="store_true")
    parser.add_argument("--keep-cluster-details", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    safe_print(f"INPUT_FILE = {args.input_file}")
    safe_print(f"CORRECT_FILE = {args.correct_file}")
    safe_print(f"PROBLEM_FILE = {args.problem_file}")
    safe_print(f"OUTPUT_FILE = {args.output_file}")
    safe_print("Correctness source: external correct file only, matched by input id -> correct_file.question_id")
    safe_print("Question source: input question/prompt first; otherwise --problem-file matched by input id -> problem_file.question_id")
    safe_print("Ignored field: record['correct answer']")
    safe_print(f"Cluster mode = {args.cluster_mode}")
    safe_print(f"Probability source = {args.prob_source}")
    safe_print(f"Length normalization = {args.length_normalization}")
    safe_print(f"Primary uncertainty = {args.primary_uncertainty}")

    records = load_records(args.input_file)
    correct_map = load_correct_map(args.correct_file)
    problem_map = load_problem_map(args.problem_file)

    if not correct_map:
        raise ValueError("No correctness labels loaded from correct file.")

    safe_print(f"Loaded records: {len(records)}")
    safe_print(f"Loaded correctness labels: {len(correct_map)}")
    if problem_map:
        safe_print(f"Loaded problem prompts: {len(problem_map)}")

    nli = None
    if args.cluster_mode == "nli":
        safe_print(f"Loading NLI model: {args.nli_model}")
        nli = NLIScorer(
            model_name_or_path=args.nli_model,
            device=args.device,
            dtype=args.dtype,
            max_length=args.nli_max_length,
            batch_size=args.nli_batch_size,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    done_ids = set()
    if not args.overwrite:
        done_ids = load_done_ids(args.output_file)

    mode = "w" if args.overwrite else "a"
    processed = 0

    with open(args.output_file, mode, encoding="utf-8") as fout:
        for line_idx, record in enumerate(records, start=1):
            if args.limit is not None and processed >= args.limit:
                break

            qid = get_record_id(record, line_idx)
            if qid in done_ids:
                continue

            try:
                result = process_item(
                    record=record,
                    line_idx=line_idx,
                    args=args,
                    correct_map=correct_map,
                    problem_map=problem_map,
                    nli=nli,
                )
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                processed += 1

                if args.verbose:
                    safe_print(
                        f"done: {qid} | line={line_idx} | "
                        f"correct={result.get('correct')} | "
                        f"clusters={result.get('num_clusters')} | "
                        f"entropy={result.get('semantic_entropy')} | "
                        f"entropy_norm={result.get('semantic_entropy_normalized')} | "
                        f"density_unc={result.get('semantic_density_uncertainty')} | "
                        f"unc={result.get('uncertainty')}"
                    )

            except KeyboardInterrupt:
                safe_print("KeyboardInterrupt detected. Exiting.")
                raise
            except Exception as e:
                err = {"id": qid, "error": str(e)}
                fout.write(json.dumps(err, ensure_ascii=False) + "\n")
                fout.flush()
                if args.verbose:
                    safe_print(f"error: {qid} | line={line_idx} | {e}")

    safe_print(f"Finished. processed={processed}")
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

    _parser = _ap.ArgumentParser(description="semantic_density / llama / humaneval / reasoning_augmented (standalone)")
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
    if "semantic_density" not in {"max_seq_prob", "mean_tok_ent", "max_tok_prob", "ppl"}:
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
    if "semantic_density" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "semantic_density", "family": "llama",
                           "dataset": "humaneval", "setting": "reasoning_augmented", "model_loading": False}, indent=2))
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

