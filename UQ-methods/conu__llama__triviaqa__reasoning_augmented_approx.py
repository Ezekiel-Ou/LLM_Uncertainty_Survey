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
import re
import string
import sys
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





def get_record_id(record: Dict[str, Any], line_idx: int) -> str:
    for key in ["id", "question_id", "qid", "task_id"]:
        if record.get(key) is not None:
            return str(record[key])
    return f"line_{line_idx}"


def get_question(record: Dict[str, Any]) -> str:
    for key in ["question", "prompt", "input", "query", "problem"]:
        if record.get(key) is not None:
            return str(record[key])
    return ""


def get_responses(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    responses = record.get("responses", [])
    if isinstance(responses, list):
        return [x for x in responses if isinstance(x, dict)]
    return []


def get_llm_answer(response: Dict[str, Any]) -> str:
    for key in ["llm answer", "llm_answer", "answer", "text", "raw_text", "prediction"]:
        if response.get(key) is not None:
            return str(response[key]).strip()
    return ""


def get_llm_reason(response: Dict[str, Any]) -> str:
    for key in ["llm reason", "llm_reason", "reason", "rationale", "solution"]:
        if response.get(key) is not None:
            return str(response[key]).strip()
    return ""





_ARTICLES = {"a", "an", "the"}


def normalize_trivia_answer(text: str) -> str:
    s = str(text).lower()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("_", " ")

    table = str.maketrans({ch: " " for ch in string.punctuation})
    s = s.translate(table)

    tokens = [tok for tok in s.split() if tok not in _ARTICLES]
    return " ".join(tokens)


def get_gold_answers(record: Dict[str, Any]) -> List[str]:
    ca = record.get("correct answer", record.get("correct_answer", None))
    answers: List[str] = []

    if isinstance(ca, dict):
        if ca.get("value") is not None:
            answers.append(str(ca["value"]))
        aliases = ca.get("aliases", [])
        if isinstance(aliases, list):
            answers.extend(str(a) for a in aliases if a is not None)
        elif aliases is not None:
            answers.append(str(aliases))
    elif ca is not None:
        answers.append(str(ca))

    if not answers:
        for key in ["answer", "gold_answer", "reference"]:
            if record.get(key) is not None:
                answers.append(str(record[key]))
        aliases = record.get("aliases", [])
        if isinstance(aliases, list):
            answers.extend(str(a) for a in aliases if a is not None)

    seen = set()
    out: List[str] = []
    for a in answers:
        key = normalize_trivia_answer(a)
        if key and key not in seen:
            seen.add(key)
            out.append(a)
    return out


def extract_all_boxed(text: str) -> List[str]:
    
    s = str(text)
    out: List[str] = []
    marker = r"\boxed"
    start = 0

    while True:
        idx = s.find(marker, start)
        if idx < 0:
            break

        brace_start = s.find("{", idx + len(marker))
        if brace_start < 0:
            start = idx + len(marker)
            continue

        depth = 0
        found = False

        for pos in range(brace_start, len(s)):
            ch = s[pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out.append(s[brace_start + 1:pos].strip())
                    start = pos + 1
                    found = True
                    break

        if not found:
            out.append(s[brace_start + 1:].strip())
            break

    return out


def clean_prediction_text(text: str) -> str:
    s = str(text).strip()

    m = re.search(r"```(?:text|answer)?\s*(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1).strip()

    s = re.sub(r"\s*####\s*$", "", s).strip()
    s = re.sub(r"^\s*(?:final\s*answer|answer)\s*:\s*", "", s, flags=re.IGNORECASE).strip()

    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines:
        s = lines[0]

    return s.strip(" \t\n\r\"'`")


def extract_prediction_candidates(llm_answer: str) -> List[str]:
    
    raw = str(llm_answer)
    candidates: List[str] = []

    boxed = extract_all_boxed(raw)
    if boxed:
        candidates.extend(reversed(boxed))

    m = re.search(r"final\s*answer\s*:\s*(.*)", raw, flags=re.IGNORECASE | re.DOTALL)
    if m:
        candidates.append(m.group(1).strip())

    candidates.append(raw)

    cleaned: List[str] = []
    seen = set()
    for c in candidates:
        cc = clean_prediction_text(c)
        key = normalize_trivia_answer(cc)
        if key and key not in seen:
            cleaned.append(cc)
            seen.add(key)

    return cleaned


def primary_prediction(response: Dict[str, Any]) -> str:
    candidates = extract_prediction_candidates(get_llm_answer(response))
    return candidates[0] if candidates else clean_prediction_text(get_llm_answer(response))


def answer_match(prediction: str, gold: str, match_mode: str) -> bool:
    p = normalize_trivia_answer(prediction)
    g = normalize_trivia_answer(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    if match_mode == "exact_or_contains":
        return len(g) >= 3 and (g in p or p in g)
    return False


def trivia_correct(prediction_candidates: List[str], gold_answers: List[str], match_mode: str) -> Tuple[int, Optional[str], Optional[str]]:
    for pred in prediction_candidates:
        for gold in gold_answers:
            if answer_match(pred, gold, match_mode):
                return 1, pred, gold
    return 0, prediction_candidates[0] if prediction_candidates else None, None


def correctness_for_response(record: Dict[str, Any], response: Dict[str, Any], match_mode: str) -> Dict[str, Any]:
    gold_answers = get_gold_answers(record)
    pred_candidates = extract_prediction_candidates(get_llm_answer(response))

    if not gold_answers:
        return {
            "correct": None,
            "matched_prediction": pred_candidates[0] if pred_candidates else None,
            "matched_gold": None,
            "prediction_candidates": pred_candidates,
            "gold_answers": [],
            "error": "missing_gold_answer",
        }

    correct, matched_pred, matched_gold = trivia_correct(pred_candidates, gold_answers, match_mode)

    return {
        "correct": int(correct),
        "matched_prediction": matched_pred,
        "matched_gold": matched_gold,
        "prediction_candidates": pred_candidates,
        "gold_answers": gold_answers,
    }





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
    lps = extract_token_logprobs(response, prob_source)

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


def compute_response_probs(
    responses: List[Dict[str, Any]],
    prob_source: str,
    length_normalization: str,
    missing_prob: str,
) -> Tuple[List[float], List[Dict[str, Any]]]:
    scores: List[float] = []
    debug: List[Dict[str, Any]] = []
    missing_indices: List[int] = []

    for idx, response in enumerate(responses):
        score, dbg = response_score_from_logprobs(response, prob_source, length_normalization)
        dbg["response_index"] = idx

        if score is None:
            missing_indices.append(idx)
            if missing_prob == "error":
                raise ValueError(
                    f"Missing logprob for response {idx} under prob_source={prob_source}. "
                    f"Use --missing-prob uniform only if you explicitly want a fallback."
                )
            score = 0.0

        scores.append(float(score))
        debug.append(dbg)

    if missing_indices and missing_prob == "uniform":
        probs = [1.0 / len(responses)] * len(responses) if responses else []
    else:
        probs = softmax(scores)

    return probs, debug





def choose_target_response(responses: List[Dict[str, Any]], target_mode: str) -> Tuple[int, Dict[str, Any]]:
    if not responses:
        raise ValueError("No responses found.")

    if target_mode == "first":
        return 0, responses[0]

    if target_mode == "majority_answer":
        counts: Dict[str, int] = {}
        first_idx: Dict[str, int] = {}

        for idx, resp in enumerate(responses):
            key = normalize_trivia_answer(primary_prediction(resp))
            if key not in counts:
                counts[key] = 0
                first_idx[key] = idx
            counts[key] += 1

        best_key = max(counts.keys(), key=lambda k: (counts[k], -first_idx[k]))
        return first_idx[best_key], responses[first_idx[best_key]]

    raise ValueError(f"Unsupported target_mode: {target_mode}")





class NLIScorer:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: str = "fp32",
        max_length: int = 256,
        batch_size: int = 16,
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
                results.append({
                    "contradiction": float(contradiction),
                    "neutral": float(neutral),
                    "entailment": float(entailment),
                })

        return results


def make_answer_statement(answer: str, question: str = "", include_question: bool = False) -> str:
    ans = clean_prediction_text(answer)
    if include_question and question.strip():
        return f"Question: {question.strip()}\nAnswer: {ans}"
    return f"The answer is: {ans}."


def build_nli_adjacency(
    predictions: List[str],
    question: str,
    nli: NLIScorer,
    include_question_in_nli: bool,
    entailment_threshold: float,
    contradiction_threshold: float,
) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
    n = len(predictions)
    adjacency = [[0 for _ in range(n)] for _ in range(n)]
    norm_preds = [normalize_trivia_answer(p) for p in predictions]

    pair_details: List[Dict[str, Any]] = []
    premises: List[str] = []
    hypotheses: List[str] = []
    pairs: List[Tuple[int, int]] = []

    for i in range(n):
        adjacency[i][i] = 0

    for i in range(n):
        for j in range(i + 1, n):
            
            if norm_preds[i] and norm_preds[i] == norm_preds[j]:
                adjacency[i][j] = 1
                adjacency[j][i] = 1
                pair_details.append({
                    "i": i,
                    "j": j,
                    "same_semantic_set": True,
                    "shortcut": "normalized_exact_match",
                    "answer_i": predictions[i],
                    "answer_j": predictions[j],
                    "normalized_answer_i": norm_preds[i],
                    "normalized_answer_j": norm_preds[j],
                    "entailment_i_to_j": None,
                    "contradiction_i_to_j": None,
                    "neutral_i_to_j": None,
                    "entailment_j_to_i": None,
                    "contradiction_j_to_i": None,
                    "neutral_j_to_i": None,
                })
                continue

            a_i = make_answer_statement(predictions[i], question, include_question_in_nli)
            a_j = make_answer_statement(predictions[j], question, include_question_in_nli)

            
            premises.extend([a_i, a_j])
            hypotheses.extend([a_j, a_i])
            pairs.append((i, j))

    scores = nli.score_pairs(premises, hypotheses) if premises else []

    score_idx = 0
    for i, j in pairs:
        s_ij = scores[score_idx]
        s_ji = scores[score_idx + 1]
        score_idx += 2

        min_entailment = min(s_ij["entailment"], s_ji["entailment"])
        max_contradiction = max(s_ij["contradiction"], s_ji["contradiction"])
        mean_entailment = 0.5 * (s_ij["entailment"] + s_ji["entailment"])
        mean_contradiction = 0.5 * (s_ij["contradiction"] + s_ji["contradiction"])

        same = (
            min_entailment >= entailment_threshold
            and max_contradiction <= contradiction_threshold
        )

        adjacency[i][j] = 1 if same else 0
        adjacency[j][i] = 1 if same else 0

        pair_details.append({
            "i": i,
            "j": j,
            "same_semantic_set": bool(same),
            "shortcut": None,
            "answer_i": predictions[i],
            "answer_j": predictions[j],
            "normalized_answer_i": norm_preds[i],
            "normalized_answer_j": norm_preds[j],
            "entailment_i_to_j": s_ij["entailment"],
            "contradiction_i_to_j": s_ij["contradiction"],
            "neutral_i_to_j": s_ij["neutral"],
            "entailment_j_to_i": s_ji["entailment"],
            "contradiction_j_to_i": s_ji["contradiction"],
            "neutral_j_to_i": s_ji["neutral"],
            "min_bidirectional_entailment": float(min_entailment),
            "max_bidirectional_contradiction": float(max_contradiction),
            "mean_bidirectional_entailment": float(mean_entailment),
            "mean_bidirectional_contradiction": float(mean_contradiction),
        })

    return adjacency, pair_details


def build_normalized_adjacency(predictions: List[str]) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
    n = len(predictions)
    adjacency = [[0 for _ in range(n)] for _ in range(n)]
    norm_preds = [normalize_trivia_answer(p) for p in predictions]
    pair_details: List[Dict[str, Any]] = []

    for i in range(n):
        for j in range(i + 1, n):
            same = bool(norm_preds[i] and norm_preds[i] == norm_preds[j])
            adjacency[i][j] = 1 if same else 0
            adjacency[j][i] = 1 if same else 0
            pair_details.append({
                "i": i,
                "j": j,
                "same_semantic_set": bool(same),
                "answer_i": predictions[i],
                "answer_j": predictions[j],
                "normalized_answer_i": norm_preds[i],
                "normalized_answer_j": norm_preds[j],
            })

    return adjacency, pair_details


def connected_components(adjacency: List[List[int]]) -> List[List[int]]:
    n = len(adjacency)
    seen = [False] * n
    comps: List[List[int]] = []

    for start in range(n):
        if seen[start]:
            continue

        comp: List[int] = []
        q = deque([start])
        seen[start] = True

        while q:
            u = q.popleft()
            comp.append(u)
            for v, edge in enumerate(adjacency[u]):
                if edge and not seen[v]:
                    seen[v] = True
                    q.append(v)

        comps.append(comp)

    return comps





def compute_conu(
    responses: List[Dict[str, Any]],
    question: str,
    args: argparse.Namespace,
    nli: Optional[NLIScorer],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    predictions = [primary_prediction(r) for r in responses]
    normalized_predictions = [normalize_trivia_answer(p) for p in predictions]

    response_probs, response_prob_debug = compute_response_probs(
        responses=responses,
        prob_source=args.prob_source,
        length_normalization=args.length_normalization,
        missing_prob=args.missing_prob,
    )

    if args.cluster_mode == "nli_answer":
        if nli is None:
            raise ValueError("NLI scorer is required when --cluster-mode nli_answer.")
        adjacency, pair_details = build_nli_adjacency(
            predictions=predictions,
            question=question,
            nli=nli,
            include_question_in_nli=args.include_question_in_nli,
            entailment_threshold=args.entailment_threshold,
            contradiction_threshold=args.contradiction_threshold,
        )
    elif args.cluster_mode == "normalized":
        adjacency, pair_details = build_normalized_adjacency(predictions)
    else:
        raise ValueError(f"Unsupported cluster_mode: {args.cluster_mode}")

    clusters = connected_components(adjacency)
    cluster_probs: List[float] = []
    for comp in clusters:
        cluster_probs.append(float(sum(response_probs[i] for i in comp)))

    if cluster_probs:
        majority_cluster_id = max(range(len(cluster_probs)), key=lambda c: cluster_probs[c])
        conu_confidence = float(cluster_probs[majority_cluster_id])
        conu_uncertainty = float(1.0 - conu_confidence)
    else:
        majority_cluster_id = None
        conu_confidence = None
        conu_uncertainty = None

    conu = {
        "conu_confidence": conu_confidence,
        "conu_uncertainty": conu_uncertainty,
        "uncertainty": conu_uncertainty,
        "majority_cluster_id": majority_cluster_id,
        "num_clusters": len(clusters),
        "cluster_probs": cluster_probs,
        "cluster_sizes": [len(c) for c in clusters],
        "clusters": clusters,
    }

    details = {
        "predictions": predictions,
        "normalized_predictions": normalized_predictions,
        "response_probs": response_probs,
        "response_prob_debug": response_prob_debug,
        "adjacency": adjacency,
        "pair_details": pair_details,
    }

    return conu, details





def process_item(
    record: Dict[str, Any],
    line_idx: int,
    args: argparse.Namespace,
    nli: Optional[NLIScorer],
) -> Dict[str, Any]:
    rid = get_record_id(record, line_idx)
    question = get_question(record)
    responses = get_responses(record)

    if not responses:
        return {
            "id": rid,
            "correct": None,
            "error": "no_responses",
        }

    target_idx, target_resp = choose_target_response(responses, args.target_mode)
    correctness = correctness_for_response(record, target_resp, args.match_mode)

    conu, details = compute_conu(
        responses=responses,
        question=question,
        args=args,
        nli=nli,
    )

    target_prediction = details["predictions"][target_idx] if target_idx < len(details["predictions"]) else primary_prediction(target_resp)

    result: Dict[str, Any] = {
        "id": rid,
        "correct": correctness.get("correct"),

        "target_response_index": target_idx,
        "target_answer": target_prediction,
        "matched_prediction": correctness.get("matched_prediction"),
        "matched_gold": correctness.get("matched_gold"),

        "conu_confidence": conu["conu_confidence"],
        "conu_uncertainty": conu["conu_uncertainty"],
        "uncertainty": conu["uncertainty"],

        "majority_cluster_id": conu["majority_cluster_id"],
        "num_responses": len(responses),
        "num_clusters": conu["num_clusters"],
        "cluster_probs": conu["cluster_probs"],
        "cluster_sizes": conu["cluster_sizes"],

        "meta": {
            "method": "ConU",
            "dataset": "trivialqa_reasoning_sampling",
            "correctness_source": "input_correct_answer_value_aliases",
            "correctness_rule": "trivialqa_normalized_alias_match",
            "match_mode": args.match_mode,
            "target_mode": args.target_mode,
            "cluster_mode": args.cluster_mode,
            "cluster_definition": (
                "bidirectional_nli_over_extracted_boxed_final_answers"
                if args.cluster_mode == "nli_answer"
                else "normalized_extracted_final_answer_exact_match"
            ),
            "uses_nli": args.cluster_mode == "nli_answer",
            "nli_model": args.nli_model if args.cluster_mode == "nli_answer" else None,
            "entailment_threshold": args.entailment_threshold if args.cluster_mode == "nli_answer" else None,
            "contradiction_threshold": args.contradiction_threshold if args.cluster_mode == "nli_answer" else None,
            "include_question_in_nli": bool(args.include_question_in_nli) if args.cluster_mode == "nli_answer" else None,
            "prob_source": args.prob_source,
            "length_normalization": args.length_normalization,
            "missing_prob": args.missing_prob,
        },
    }

    if args.keep_question:
        result["question"] = question
    if args.keep_prediction:
        result["prediction"] = target_prediction
    if args.keep_reasoning:
        result["target_reasoning"] = get_llm_reason(target_resp)
    if args.keep_gold:
        result["gold_answers"] = correctness.get("gold_answers", [])
    if args.keep_candidates:
        result["prediction_candidates"] = correctness.get("prediction_candidates", [])
    if args.keep_cluster_details:
        result["clusters"] = [
            {
                "cluster_id": cid,
                "response_indices": comp,
                "cluster_prob": conu["cluster_probs"][cid],
                "cluster_size": len(comp),
                "answers": [details["predictions"][i] for i in comp],
                "normalized_answers": [details["normalized_predictions"][i] for i in comp],
            }
            for cid, comp in enumerate(conu["clusters"])
        ]
    if args.keep_responses:
        result["responses_info"] = [
            {
                "response_index": i,
                "response_prob": details["response_probs"][i],
                "answer": details["predictions"][i],
                "normalized_answer": details["normalized_predictions"][i],
                "prob_debug": details["response_prob_debug"][i],
            }
            for i in range(len(responses))
        ]
    if args.keep_pair_details:
        result["pair_details"] = details["pair_details"]
    if args.keep_adjacency:
        result["adjacency_matrix"] = details["adjacency"]

    return result





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ConU uncertainty for TrivialQA reasoning-sampling data."
    )

    parser.add_argument("--input-file", required=True, help="Input TrivialQA reasoning JSONL/JSON file")
    parser.add_argument("--output-file", required=True, help="Output JSONL file")

    parser.add_argument(
        "--target-mode",
        choices=["first", "majority_answer"],
        default="first",
        help="Target response used for correctness/prediction fields.",
    )
    parser.add_argument(
        "--match-mode",
        choices=["exact", "exact_or_contains"],
        default="exact",
        help="Correctness match mode against gold aliases.",
    )

    parser.add_argument(
        "--cluster-mode",
        choices=["nli_answer", "normalized"],
        default="nli_answer",
        help="Semantic clustering mode. Default uses NLI over extracted final answers.",
    )

    
    parser.add_argument("--nli-model", default="microsoft/deberta-large-mnli")
    parser.add_argument("--hf-endpoint", default=None, help="Optional HF mirror, e.g. https://hf-mirror.com")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="fp32",
        help="NLI dtype. Default fp32 avoids DeBERTa fp16 dtype mismatch.",
    )
    parser.add_argument("--nli-max-length", type=int, default=256)
    parser.add_argument("--nli-batch-size", type=int, default=16)
    parser.add_argument("--entailment-threshold", type=float, default=0.50)
    parser.add_argument("--contradiction-threshold", type=float, default=0.50)
    parser.add_argument(
        "--include-question-in-nli",
        action="store_true",
        help="Include question context in answer-only NLI statements.",
    )

    
    parser.add_argument(
        "--prob-source",
        choices=["answer", "reasoning_augmented", "all"],
        default="all",
        help="Stored logprobs used for response probabilities. Default all = reasoning + answer.",
    )
    parser.add_argument(
        "--length-normalization",
        choices=["sum", "mean"],
        default="mean",
        help="How to turn token logprobs into response scores. Default mean avoids length bias.",
    )
    parser.add_argument(
        "--missing-prob",
        choices=["error", "uniform"],
        default="error",
        help="Missing logprob behavior. Default error avoids silent fallback.",
    )

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--keep-question", action="store_true")
    parser.add_argument("--keep-prediction", action="store_true")
    parser.add_argument("--keep-reasoning", action="store_true")
    parser.add_argument("--keep-gold", action="store_true")
    parser.add_argument("--keep-candidates", action="store_true")
    parser.add_argument("--keep-cluster-details", action="store_true")
    parser.add_argument("--keep-responses", action="store_true")
    parser.add_argument("--keep-pair-details", action="store_true")
    parser.add_argument("--keep-adjacency", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    safe_print(f"INPUT_FILE = {args.input_file}")
    safe_print(f"OUTPUT_FILE = {args.output_file}")
    safe_print("Correctness source: record['correct answer']['value'] + aliases")
    safe_print(f"TrivialQA match mode = {args.match_mode}")
    safe_print(f"Cluster mode = {args.cluster_mode}")
    safe_print(f"Probability source = {args.prob_source}")
    safe_print(f"Length normalization = {args.length_normalization}")
    safe_print(f"Missing prob mode = {args.missing_prob}")

    records = load_records(args.input_file)
    safe_print(f"Loaded records: {len(records)}")

    nli = None
    if args.cluster_mode == "nli_answer":
        safe_print("Loading NLI model...")
        safe_print(f"NLI model = {args.nli_model}")
        safe_print(f"dtype = {args.dtype}")
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

            rid = get_record_id(record, line_idx)
            if rid in done_ids:
                continue

            try:
                result = process_item(record=record, line_idx=line_idx, args=args, nli=nli)
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                processed += 1

                if args.verbose:
                    safe_print(
                        f"done: {rid} | line={line_idx} | "
                        f"correct={result.get('correct')} | "
                        f"clusters={result.get('num_clusters')} | "
                        f"conf={result.get('conu_confidence')} | "
                        f"unc={result.get('conu_uncertainty')} | "
                        f"pred={result.get('matched_prediction')} | "
                        f"gold={result.get('matched_gold')}"
                    )

            except KeyboardInterrupt:
                safe_print("KeyboardInterrupt detected. Exiting.")
                raise
            except Exception as e:
                err = {"id": rid, "error": str(e)}
                fout.write(json.dumps(err, ensure_ascii=False) + "\n")
                fout.flush()
                if args.verbose:
                    safe_print(f"error: {rid} | line={line_idx} | {e}")

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

    _parser = _ap.ArgumentParser(description="conu / llama / triviaqa / reasoning_augmented (standalone)")
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
                           "dataset": "triviaqa", "setting": "reasoning_augmented", "model_loading": False}, indent=2))
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

