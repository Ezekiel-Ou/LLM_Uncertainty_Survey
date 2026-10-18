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
import json
import math
import os
import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from math_verify import parse, verify





print_lock = threading.Lock()
write_lock = threading.Lock()


def safe_print(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    with print_lock:
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

    
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
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


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
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
                    done.add(str(obj["id"]))
            except Exception:
                continue
    return done





def mathv_correct(ref_text: str, target_text: str) -> bool:
    try:
        ref = parse(str(ref_text))
        target = parse(str(target_text))
        return bool(verify(ref, target))
    except Exception:
        return False


def normalize_response_answer(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()





def get_record_id(record: Dict[str, Any], fallback_idx: int) -> str:
    for key in ["id", "question_id", "qid"]:
        if key in record and record[key] is not None:
            return str(record[key])
    return f"line_{fallback_idx}"


def get_correct_answer(record: Dict[str, Any]) -> str:
    for key in ["correct answer", "correct_answer", "answer", "gt", "ground_truth"]:
        if key in record and record[key] is not None:
            value = record[key]
            if isinstance(value, dict):
                if value.get("value") is not None:
                    return str(value["value"])
                aliases = value.get("aliases")
                if isinstance(aliases, list) and aliases:
                    return str(aliases[0])
            return str(value)
    return ""


def get_responses(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    responses = record.get("responses", [])
    if isinstance(responses, list):
        return [x for x in responses if isinstance(x, dict)]
    return []


def get_response_answer(resp: Dict[str, Any]) -> str:
    for key in ["llm answer", "llm_answer", "answer", "text", "raw_text"]:
        if key in resp and resp[key] is not None:
            return normalize_response_answer(resp[key])
    return ""


def get_response_reason(resp: Dict[str, Any]) -> str:
    for key in ["llm reason", "llm_reason", "reason", "rationale", "explanation"]:
        if key in resp and resp[key] is not None:
            return str(resp[key]).strip()
    return ""


def build_full_response_text(resp: Dict[str, Any], include_answer: bool = True) -> str:
    reason = get_response_reason(resp)
    answer = get_response_answer(resp)
    if include_answer and answer:
        if reason:
            return f"{reason}\nFinal Answer: {answer}".strip()
        return f"Final Answer: {answer}".strip()
    return reason.strip() if reason else answer.strip()


def response_token_logprobs(resp: Dict[str, Any], prob_source: str = "all") -> List[float]:
    
    logprobs: List[float] = []

    if prob_source in {"all", "reasoning_augmented"}:
        steps = resp.get("step-wise info", []) or resp.get("step_wise_info", []) or []
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict):
                    continue
                prob_info = step.get("prob info") or step.get("prob_info") or {}
                if isinstance(prob_info, dict):
                    vals = prob_info.get("token logprob") or prob_info.get("token_logprob") or []
                    if isinstance(vals, list):
                        for v in vals:
                            try:
                                logprobs.append(float(v))
                            except Exception:
                                pass

    if prob_source in {"all", "answer"}:
        ans_info = resp.get("answer info") or resp.get("answer_info") or {}
        if isinstance(ans_info, dict):
            prob_info = ans_info.get("prob info") or ans_info.get("prob_info") or {}
            if isinstance(prob_info, dict):
                vals = prob_info.get("token logprob") or prob_info.get("token_logprob") or []
                if isinstance(vals, list):
                    for v in vals:
                        try:
                            logprobs.append(float(v))
                        except Exception:
                            pass

    return logprobs


def response_logprob_score(resp: Dict[str, Any], prob_source: str, length_normalization: str) -> Optional[float]:
    lps = response_token_logprobs(resp, prob_source=prob_source)
    if not lps:
        return None
    if length_normalization == "sum":
        return float(sum(lps))
    if length_normalization == "mean":
        return float(sum(lps) / max(1, len(lps)))
    raise ValueError(f"Unsupported length_normalization: {length_normalization}")


def softmax_from_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    x = torch.tensor(scores, dtype=torch.float32)
    probs = F.softmax(x, dim=0).detach().cpu().tolist()
    return [float(p) for p in probs]


def compute_sample_probs(
    responses: List[Dict[str, Any]],
    prob_source: str,
    length_normalization: str,
    missing_prob: str,
) -> Tuple[Optional[List[float]], List[Optional[float]], str]:
    scores = [response_logprob_score(r, prob_source, length_normalization) for r in responses]
    if all(s is not None for s in scores):
        probs = softmax_from_scores([float(s) for s in scores])
        return probs, scores, f"softmax_{length_normalization}_logprob_{prob_source}"

    if missing_prob == "uniform":
        n = len(responses)
        if n == 0:
            return [], scores, "uniform_no_responses"
        return [1.0 / n] * n, scores, "uniform_due_to_missing_logprob"

    
    return None, scores, "missing_logprob"





@dataclass
class NLIResult:
    entail: float
    neutral: float
    contradiction: float


class NLIJudge:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 8,
        dtype: str = "float32",
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        torch_dtype = torch.float32
        if dtype in {"float16", "fp16", "half"}:
            torch_dtype = torch.float16
        elif dtype in {"bfloat16", "bf16"}:
            torch_dtype = torch.bfloat16
        elif dtype in {"float32", "fp32"}:
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.entail_idx = None
        self.neutral_idx = None
        self.contra_idx = None
        for idx, label in id2label.items():
            if "entail" in label:
                self.entail_idx = idx
            elif "neutral" in label:
                self.neutral_idx = idx
            elif "contradiction" in label or "contra" in label:
                self.contra_idx = idx

        if self.entail_idx is None or self.contra_idx is None:
            raise ValueError(f"Cannot infer entailment/contradiction labels from {id2label}")
        if self.neutral_idx is None:
            
            used = {self.entail_idx, self.contra_idx}
            candidates = [i for i in range(self.model.config.num_labels) if i not in used]
            self.neutral_idx = candidates[0] if candidates else self.contra_idx

        self._cache: Dict[Tuple[str, str], NLIResult] = {}

    @torch.inference_mode()
    def classify_pairs(self, pairs: List[Tuple[str, str]]) -> List[NLIResult]:
        results: List[Optional[NLIResult]] = [None] * len(pairs)
        to_run: List[Tuple[int, str, str]] = []

        for i, (premise, hypothesis) in enumerate(pairs):
            key = (premise, hypothesis)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                to_run.append((i, premise, hypothesis))

        for start in range(0, len(to_run), self.batch_size):
            batch = to_run[start:start + self.batch_size]
            idxs = [x[0] for x in batch]
            premises = [x[1] for x in batch]
            hypotheses = [x[2] for x in batch]
            inputs = self.tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logits = self.model(**inputs).logits.float()
            probs = F.softmax(logits, dim=-1).detach().cpu()
            for row_idx, orig_idx in enumerate(idxs):
                res = NLIResult(
                    entail=float(probs[row_idx, self.entail_idx].item()),
                    neutral=float(probs[row_idx, self.neutral_idx].item()),
                    contradiction=float(probs[row_idx, self.contra_idx].item()),
                )
                key = (pairs[orig_idx][0], pairs[orig_idx][1])
                self._cache[key] = res
                results[orig_idx] = res

        return [r if r is not None else NLIResult(0.0, 1.0, 0.0) for r in results]

    def bidirectional_similarity(self, a: str, b: str) -> Dict[str, float]:
        r_ab, r_ba = self.classify_pairs([(a, b), (b, a)])
        support_ab = 0.5 * (r_ab.entail + (1.0 - r_ab.contradiction))
        support_ba = 0.5 * (r_ba.entail + (1.0 - r_ba.contradiction))
        similarity = 0.5 * (support_ab + support_ba)
        return {
            "entail_ab": r_ab.entail,
            "neutral_ab": r_ab.neutral,
            "contradiction_ab": r_ab.contradiction,
            "entail_ba": r_ba.entail,
            "neutral_ba": r_ba.neutral,
            "contradiction_ba": r_ba.contradiction,
            "support_ab": support_ab,
            "support_ba": support_ba,
            "bidirectional_similarity": similarity,
            "max_contradiction": max(r_ab.contradiction, r_ba.contradiction),
        }





def nli_semantic_equal(
    nli: NLIJudge,
    a: str,
    b: str,
    similarity_threshold: float,
    max_contradiction_threshold: float,
) -> Tuple[bool, Dict[str, float]]:
    if not a.strip() or not b.strip():
        return False, {
            "bidirectional_similarity": 0.0,
            "max_contradiction": 1.0,
            "entail_ab": 0.0,
            "entail_ba": 0.0,
            "contradiction_ab": 1.0,
            "contradiction_ba": 1.0,
        }
    info = nli.bidirectional_similarity(a, b)
    same = (
        info["bidirectional_similarity"] >= similarity_threshold
        and info["max_contradiction"] <= max_contradiction_threshold
    )
    return bool(same), info


def cluster_responses_by_nli(
    texts: List[str],
    nli: NLIJudge,
    similarity_threshold: float,
    max_contradiction_threshold: float,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    cluster_reps: List[str] = []
    cluster_ids: List[int] = []
    pairwise_debug: List[Dict[str, Any]] = []

    for idx, text in enumerate(texts):
        assigned = False
        for cid, rep in enumerate(cluster_reps):
            matched, nli_info = nli_semantic_equal(
                nli,
                text,
                rep,
                similarity_threshold=similarity_threshold,
                max_contradiction_threshold=max_contradiction_threshold,
            )
            pairwise_debug.append({
                "sample_idx": idx,
                "rep_cluster": cid,
                "matched": matched,
                "matched_by": "nli_full_reasoning_bidirectional",
                **nli_info,
            })
            if matched:
                cluster_ids.append(cid)
                assigned = True
                break

        if not assigned:
            new_cid = len(cluster_reps)
            cluster_reps.append(text)
            cluster_ids.append(new_cid)
            pairwise_debug.append({
                "sample_idx": idx,
                "rep_cluster": new_cid,
                "matched": True,
                "matched_by": "new_cluster",
                "bidirectional_similarity": 1.0,
                "max_contradiction": 0.0,
            })

    return cluster_ids, pairwise_debug





def compute_cluster_summary(
    cluster_ids: List[int],
    sample_probs: List[float],
    texts: List[str],
    answers: List[str],
    keep_texts: bool = False,
) -> List[Dict[str, Any]]:
    cluster_mass: Dict[int, float] = {}
    cluster_members: Dict[int, List[int]] = {}

    for i, cid in enumerate(cluster_ids):
        p = float(sample_probs[i]) if i < len(sample_probs) else 0.0
        cluster_mass[cid] = cluster_mass.get(cid, 0.0) + p
        cluster_members.setdefault(cid, []).append(i)

    summary: List[Dict[str, Any]] = []
    for cid in sorted(cluster_mass.keys()):
        members = cluster_members[cid]
        item: Dict[str, Any] = {
            "cluster_id": int(cid),
            "prob": float(cluster_mass[cid]),
            "size": int(len(members)),
            "member_indices": [int(x) for x in members],
            "representative_answer": answers[members[0]] if members else "",
        }
        if keep_texts:
            item["member_answers"] = [answers[i] for i in members]
            item["representative_text"] = texts[members[0]] if members else ""
            item["member_texts"] = [texts[i] for i in members]
        summary.append(item)
    return summary


def compute_semantic_entropy(cluster_summary: List[Dict[str, Any]]) -> Tuple[float, float]:
    probs = [max(0.0, float(c["prob"])) for c in cluster_summary]
    total = sum(probs)
    if total <= 0:
        return 0.0, 0.0
    probs = [p / total for p in probs if p > 0]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    if abs(entropy) < 1e-10:
        entropy = 0.0
    k = len(probs)
    normalized = entropy / math.log(k + 1e-12) if k > 1 else 0.0
    if abs(normalized) < 1e-10:
        normalized = 0.0
    return float(entropy), float(normalized)


def compute_semantic_density(cluster_summary: List[Dict[str, Any]]) -> Tuple[float, float]:
    probs = [max(0.0, float(c["prob"])) for c in cluster_summary]
    total = sum(probs)
    if total <= 0:
        return 1.0, 0.0
    probs = [p / total for p in probs]
    density_confidence = sum(p * p for p in probs)
    density_uncertainty = 1.0 - density_confidence
    if abs(density_uncertainty) < 1e-10:
        density_uncertainty = 0.0
    if abs(density_confidence - 1.0) < 1e-10:
        density_confidence = 1.0
    return float(density_confidence), float(density_uncertainty)





def select_target_index(record: Dict[str, Any], responses: List[Dict[str, Any]], target_mode: str) -> int:
    if not responses:
        return -1
    if target_mode == "first":
        return 0
    if target_mode == "majority_answer":
        
        reps: List[str] = []
        counts: List[int] = []
        first_indices: List[int] = []
        for i, r in enumerate(responses):
            ans = get_response_answer(r)
            assigned = False
            for cid, rep in enumerate(reps):
                if mathv_correct(rep, ans):
                    counts[cid] += 1
                    assigned = True
                    break
            if not assigned:
                reps.append(ans)
                counts.append(1)
                first_indices.append(i)
        best_cid = max(range(len(counts)), key=lambda j: counts[j]) if counts else 0
        return first_indices[best_cid]
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def process_record(record: Dict[str, Any], idx: int, nli: NLIJudge, args: argparse.Namespace) -> Dict[str, Any]:
    rid = get_record_id(record, idx)
    correct_answer = get_correct_answer(record)
    responses_all = get_responses(record)
    responses = responses_all[:args.max_responses] if args.max_responses is not None else responses_all

    if not responses:
        return {
            "id": rid,
            "correct_answer": correct_answer,
            "correct": 0,
            "error": "no_responses",
            "semantic_entropy": None,
            "semantic_entropy_normalized": None,
            "semantic_density_confidence": None,
            "semantic_density_uncertainty": None,
            "uncertainty": None,
        }

    target_idx = select_target_index(record, responses, args.target_mode)
    target_answer = get_response_answer(responses[target_idx]) if target_idx >= 0 else ""
    correct = int(mathv_correct(correct_answer, target_answer)) if correct_answer else 0

    answers = [get_response_answer(r) for r in responses]
    texts = [build_full_response_text(r, include_answer=True) for r in responses]

    probs, raw_scores, prob_mode = compute_sample_probs(
        responses,
        prob_source=args.prob_source,
        length_normalization=args.length_normalization,
        missing_prob=args.missing_prob,
    )

    if probs is None:
        return {
            "id": rid,
            "correct_answer": correct_answer,
            "target_response_index": int(target_idx),
            "target_answer": target_answer,
            "correct": correct,
            "num_responses": len(responses),
            "semantic_entropy": None,
            "semantic_entropy_normalized": None,
            "semantic_entropy_uncertainty": None,
            "semantic_density_confidence": None,
            "semantic_density_uncertainty": None,
            "uncertainty": None,
            "error": "missing_logprob_for_probability_weighted_semantic_entropy_density",
            "debug": {
                "prob_mode": prob_mode,
                "raw_logprob_scores": raw_scores,
            },
            "meta": {
                "method": "Semantic Entropy / Semantic Density over full reasoning chains using NLI",
                "nli_model": args.nli_model,
                "prob_source": args.prob_source,
                "length_normalization": args.length_normalization,
                "missing_prob": args.missing_prob,
            },
        }

    cluster_ids, pairwise_debug = cluster_responses_by_nli(
        texts,
        nli=nli,
        similarity_threshold=args.similarity_threshold,
        max_contradiction_threshold=args.max_contradiction_threshold,
    )

    cluster_summary = compute_cluster_summary(
        cluster_ids=cluster_ids,
        sample_probs=probs,
        texts=texts,
        answers=answers,
        keep_texts=args.keep_texts,
    )

    semantic_entropy, semantic_entropy_normalized = compute_semantic_entropy(cluster_summary)
    density_conf, density_unc = compute_semantic_density(cluster_summary)

    
    answer_correct_flags = [int(mathv_correct(correct_answer, a)) if correct_answer else 0 for a in answers]
    num_correct_responses = int(sum(answer_correct_flags))
    answer_correct_rate = float(num_correct_responses / len(answers)) if answers else 0.0
    target_consistency_flags = [int(mathv_correct(target_answer, a)) if target_answer else 0 for a in answers]
    answer_consistency_with_target = float(sum(target_consistency_flags) / len(target_consistency_flags)) if target_consistency_flags else 0.0

    if args.primary_uncertainty == "entropy":
        uncertainty = semantic_entropy_normalized
    elif args.primary_uncertainty == "density":
        uncertainty = density_unc
    else:
        raise ValueError(f"Unsupported primary_uncertainty: {args.primary_uncertainty}")

    result: Dict[str, Any] = {
        "id": rid,
        "correct_answer": correct_answer,
        "target_response_index": int(target_idx),
        "target_answer": target_answer,
        "correct": correct,

        "semantic_entropy": semantic_entropy,
        "semantic_entropy_normalized": semantic_entropy_normalized,
        "semantic_entropy_uncertainty": semantic_entropy_normalized,

        "semantic_density_confidence": density_conf,
        "semantic_density_uncertainty": density_unc,

        "uncertainty": float(uncertainty),
        "primary_uncertainty": args.primary_uncertainty,

        "num_responses": len(responses),
        "num_clusters": len(cluster_summary),
        "num_correct_responses": num_correct_responses,
        "answer_correct_rate": answer_correct_rate,
        "answer_consistency_with_target": answer_consistency_with_target,

        "cluster_summary": cluster_summary,
        "sample_probabilities": probs,
        "response_logprob_scores": raw_scores,
        "cluster_ids": cluster_ids,

        "meta": {
            "method": "Semantic Entropy / Semantic Density over full reasoning chains using NLI",
            "dataset_format": "reasoning_sampling_gsm8k_style",
            "semantic_unit": "full_response_reasoning_chain_plus_final_answer",
            "nli_model": args.nli_model,
            "similarity_definition": "bidirectional_nli_support",
            "similarity_threshold": args.similarity_threshold,
            "max_contradiction_threshold": args.max_contradiction_threshold,
            "prob_source": args.prob_source,
            "length_normalization": args.length_normalization,
            "prob_mode": prob_mode,
            "correctness_checker": "math_verify(correct answer, target llm answer)",
            "target_mode": args.target_mode,
        },
    }

    if args.keep_responses:
        result["responses"] = [
            {
                "response_index": i,
                "answer": answers[i],
                "full_text": texts[i],
                "sample_probability": probs[i],
                "logprob_score": raw_scores[i],
                "cluster_id": cluster_ids[i],
                "correct": answer_correct_flags[i],
                "consistent_with_target_answer": target_consistency_flags[i],
            }
            for i in range(len(responses))
        ]

    if args.keep_pair_details:
        result["pairwise_nli_details"] = pairwise_debug

    return result





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Semantic Entropy and Semantic Density for reasoning-sampling outputs using NLI over full reasoning chains."
    )
    parser.add_argument("--input-file", required=True, help="Input JSONL/JSON file in reasoning-sampling format")
    parser.add_argument("--output-file", required=True, help="Output JSONL path")
    parser.add_argument("--nli-model", default="microsoft/deberta-large-mnli", help="NLI model path/name")
    parser.add_argument("--hf-endpoint", default=None, help="Optional HF endpoint, e.g. https://hf-mirror.com")
    parser.add_argument("--device", default="auto", help="auto/cuda/cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "fp32", "float16", "fp16", "half", "bfloat16", "bf16"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--nli-max-length", type=int, default=512)

    parser.add_argument("--prob-source", choices=["answer", "reasoning_augmented", "all"], default="all",
                        help="Where to get token logprobs for response probability")
    parser.add_argument("--length-normalization", choices=["sum", "mean"], default="mean",
                        help="Use sum or mean token logprob before softmax. mean is less length-biased for reasoning chains.")
    parser.add_argument("--missing-prob", choices=["error", "uniform"], default="error",
                        help="What to do if any response lacks logprobs. Default error avoids silent approximate computation.")

    parser.add_argument("--similarity-threshold", type=float, default=0.65,
                        help="NLI bidirectional similarity threshold for same semantic cluster")
    parser.add_argument("--max-contradiction-threshold", type=float, default=0.50,
                        help="Maximum allowed contradiction probability for same semantic cluster")

    parser.add_argument("--target-mode", choices=["first", "majority_answer"], default="first")
    parser.add_argument("--primary-uncertainty", choices=["entropy", "density"], default="entropy")
    parser.add_argument("--max-responses", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file instead of resuming")
    parser.add_argument("--keep-responses", action="store_true")
    parser.add_argument("--keep-texts", action="store_true", help="Keep cluster member full texts in output; can be large")
    parser.add_argument("--keep-pair-details", action="store_true", help="Keep pairwise NLI details; can be large")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    safe_print(f"INPUT_FILE = {args.input_file}")
    safe_print(f"OUTPUT_FILE = {args.output_file}")
    safe_print("Correctness source: record['correct answer'] + math_verify")
    safe_print("Semantic unit: full reasoning chain + final answer")
    safe_print(f"NLI model = {args.nli_model}")
    safe_print(f"Probability source = {args.prob_source}")
    safe_print(f"Length normalization = {args.length_normalization}")
    safe_print(f"Primary uncertainty alias = {args.primary_uncertainty}")

    if args.overwrite and os.path.exists(args.output_file):
        os.remove(args.output_file)

    done = load_done_ids(args.output_file)
    if done and args.verbose:
        safe_print(f"existing records: {len(done)}")

    safe_print("Loading NLI model...")
    nli = NLIJudge(
        model_name=args.nli_model,
        device=args.device,
        max_length=args.nli_max_length,
        batch_size=args.batch_size,
        dtype=args.dtype,
    )

    records = load_records(args.input_file)
    if args.limit is not None:
        records = records[:args.limit]

    processed = 0
    skipped = 0
    for idx, record in enumerate(records):
        rid = get_record_id(record, idx)
        if str(rid) in done:
            skipped += 1
            continue

        try:
            result = process_record(record, idx, nli, args)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            result = {
                "id": rid,
                "error": str(e),
                "semantic_entropy": None,
                "semantic_entropy_normalized": None,
                "semantic_density_confidence": None,
                "semantic_density_uncertainty": None,
                "uncertainty": None,
            }

        append_jsonl(args.output_file, result)
        processed += 1

        if args.verbose:
            unc = result.get("uncertainty")
            unc_str = "None" if unc is None else f"{float(unc):.6f}"
            safe_print(
                f"done: {rid} | line={idx+1} | correct={result.get('correct')} | "
                f"clusters={result.get('num_clusters')} | "
                f"se={result.get('semantic_entropy')} | "
                f"sd_unc={result.get('semantic_density_uncertainty')} | unc={unc_str}"
            )

    safe_print(f"Finished. processed={processed}, skipped={skipped}")
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

    _parser = _ap.ArgumentParser(description="semantic_density / qwen / gsm8k / reasoning_augmented (standalone)")
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
    if "semantic_density" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "semantic_density", "family": "qwen",
                           "dataset": "gsm8k", "setting": "reasoning_augmented", "model_loading": False}, indent=2))
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

