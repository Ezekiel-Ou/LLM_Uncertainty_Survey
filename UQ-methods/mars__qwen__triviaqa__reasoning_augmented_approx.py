#!/usr/bin/env python3
# Original paper/author repository: https://github.com/Ybakman/LLM_Uncertainty

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


def extract_token_logprobs_from_unit(unit_obj: Dict[str, Any]) -> List[float]:
    pi = get_prob_info(unit_obj)
    lp_raw = None
    for k in ["token logprob", "token_logprob", "chosen_logprob", "logprob"]:
        if k in pi:
            lp_raw = pi[k]
            break
    return list_floats(lp_raw)


def unit_logprob_score(token_logprobs: List[float], mode: str) -> Optional[float]:
    if not token_logprobs:
        return None

    seq = float(sum(token_logprobs))
    avg = float(seq / len(token_logprobs))

    if mode == "sum":
        return seq
    if mode == "mean":
        return avg

    raise ValueError(f"Unsupported unit-logprob-mode: {mode}")


def split_reasoning_sentences(text: str) -> List[str]:
    text = str(text).strip()
    if not text:
        return []

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    step_like = [ln for ln in lines if re.match(r"^(step\s*\d+|[-*]\s*)", ln, flags=re.IGNORECASE)]
    if step_like:
        return step_like

    parts = re.split(r"(?<=[.!?。！？])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def extract_mars_units(
    response: Dict[str, Any],
    include_answer_unit: bool,
    unit_logprob_mode: str,
    missing_logprob: str,
    max_units: Optional[int],
) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []

    steps = response.get("step-wise info", response.get("step_wise_info", []))
    if isinstance(steps, list):
        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                continue

            text = str(step.get("text", "")).strip()
            if not text:
                continue

            lps = extract_token_logprobs_from_unit(step)
            score = unit_logprob_score(lps, unit_logprob_mode)
            if score is None:
                if missing_logprob == "error":
                    raise ValueError(f"Missing logprob for reasoning step {idx}")
                score = 0.0

            units.append({
                "unit_id": len(units),
                "unit_type": "reasoning_step",
                "step_index": idx,
                "text": text,
                "token_logprobs": lps,
                "num_tokens": len(lps),
                "seq_logprob": float(sum(lps)) if lps else None,
                "avg_logprob": float(sum(lps) / len(lps)) if lps else None,
                "unit_logprob_score": float(score),
            })

    
    if not units:
        for idx, sent in enumerate(split_reasoning_sentences(get_llm_reason(response))):
            units.append({
                "unit_id": len(units),
                "unit_type": "reasoning_sentence_no_logprob",
                "step_index": idx,
                "text": sent,
                "token_logprobs": [],
                "num_tokens": 0,
                "seq_logprob": None,
                "avg_logprob": None,
                "unit_logprob_score": 0.0 if missing_logprob == "zero" else None,
            })
        if any(u["unit_logprob_score"] is None for u in units) and missing_logprob == "error":
            raise ValueError("Missing step-wise logprobs and --missing-logprob error is set.")

    if include_answer_unit:
        answer_info = response.get("answer info", response.get("answer_info", None))
        lps: List[float] = []
        answer_text = ""

        if isinstance(answer_info, dict):
            answer_text = str(answer_info.get("text", "")).strip()
            lps = extract_token_logprobs_from_unit(answer_info)

        
        pred = primary_prediction(response)
        if not answer_text:
            answer_text = pred

        if answer_text.strip():
            score = unit_logprob_score(lps, unit_logprob_mode)
            if score is None:
                if missing_logprob == "error":
                    raise ValueError("Missing logprob for answer unit")
                score = 0.0

            units.append({
                "unit_id": len(units),
                "unit_type": "final_answer",
                "step_index": None,
                "text": answer_text,
                "extracted_answer": pred,
                "token_logprobs": lps,
                "num_tokens": len(lps),
                "seq_logprob": float(sum(lps)) if lps else None,
                "avg_logprob": float(sum(lps) / len(lps)) if lps else None,
                "unit_logprob_score": float(score),
            })

    if max_units is not None and len(units) > max_units:
        answer_units = [u for u in units if u["unit_type"] == "final_answer"]
        reasoning_units = [u for u in units if u["unit_type"] != "final_answer"]

        if answer_units and max_units >= 1:
            units = reasoning_units[: max_units - 1] + answer_units[:1]
        else:
            units = units[:max_units]

        for new_id, unit in enumerate(units):
            unit["unit_id"] = new_id

    return units


def build_reasoning_from_units(units: List[Dict[str, Any]], masked_unit_id: Optional[int], mask_text: str) -> str:
    parts: List[str] = []
    for unit in units:
        if unit["unit_type"] == "final_answer":
            continue

        if masked_unit_id is not None and unit["unit_id"] == masked_unit_id:
            parts.append(mask_text)
        else:
            parts.append(str(unit["text"]).strip())

    return "\n".join(p for p in parts if p.strip())


def build_answer_from_units(
    response: Dict[str, Any],
    units: List[Dict[str, Any]],
    masked_unit_id: Optional[int],
    mask_text: str,
) -> str:
    for unit in units:
        if unit["unit_type"] == "final_answer" and masked_unit_id is not None and unit["unit_id"] == masked_unit_id:
            return mask_text
    return primary_prediction(response)





class LocalJudge:
    def __init__(self, model_path: str, model_family: str, dtype: str = "auto", device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_family = model_family
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        torch_dtype = self.resolve_dtype(dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" and torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        if not (device == "cuda" and torch.cuda.is_available()):
            self.model.to(torch.device("cpu"))

        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def resolve_dtype(self, dtype: str):
        import torch

        if dtype == "auto":
            if torch.cuda.is_available():
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.float32
        if dtype in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype in {"fp16", "float16", "half"}:
            return torch.float16
        if dtype in {"fp32", "float32"}:
            return torch.float32
        raise ValueError(f"Unsupported dtype: {dtype}")

    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        if self.model_family == "qwen":
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            parts = []
            for m in messages:
                parts.append(f"{m['role'].upper()}: {m['content']}")
            parts.append("ASSISTANT:")
            return "\n\n".join(parts)

    def candidate_logprob(self, prompt: str, candidate: str) -> float:
        import torch

        prefix_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        variants = [
            candidate,
            " " + candidate,
            candidate + ".",
            " " + candidate + ".",
        ]

        best = -float("inf")
        for cand in variants:
            suffix_ids = self.tokenizer(cand, add_special_tokens=False)["input_ids"]
            if not suffix_ids:
                continue

            full_ids = prefix_ids + suffix_ids
            input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.model.device)

            with torch.inference_mode():
                logits = self.model(input_ids=input_ids).logits[:, :-1, :].float()
                logprobs = torch.log_softmax(logits, dim=-1)

            total = 0.0
            start = len(prefix_ids)
            valid = True

            for pos, tok_id in enumerate(suffix_ids):
                lp_pos = start + pos - 1
                if lp_pos < 0 or lp_pos >= logprobs.shape[1]:
                    valid = False
                    break
                total += float(logprobs[0, lp_pos, tok_id].item())

            if valid:
                best = max(best, total)

        return best

    def judge_score(self, question: str, reasoning: str, answer: str) -> Dict[str, Any]:
        if question.strip():
            question_part = f"Question:\n{question.strip()}\n"
        else:
            question_part = (
                "Question:\n"
                "[The original TrivialQA question is missing from the input record. "
                "Judge only from the proposed answer and reasoning.]\n"
            )

        user_content = (
            f"{question_part}\n"
            f"Model reasoning:\n{reasoning.strip() if reasoning else '[empty]'}\n\n"
            f"Proposed final answer:\n{answer.strip() if answer else '[empty]'}\n\n"
            "Is the proposed final answer correct for the question?\n"
            "Choose exactly one option:\n"
            "A. Correct\n"
            "B. Incorrect\n"
            "C. Not sure\n\n"
            "Answer with only one letter: A, B, or C."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict factual QA judge. "
                    "Given a question, a model reasoning chain, and a proposed final answer, "
                    "judge whether the final answer is correct. "
                    "Do not use the hidden gold answer. Return only A, B, or C."
                ),
            },
            {"role": "user", "content": user_content},
        ]

        prompt = self.apply_chat_template(messages)

        cand_scores = {
            "A": self.candidate_logprob(prompt, "A"),
            "B": self.candidate_logprob(prompt, "B"),
            "C": self.candidate_logprob(prompt, "C"),
        }

        max_log = max(cand_scores.values())
        exps = {k: math.exp(v - max_log) for k, v in cand_scores.items()}
        denom = sum(exps.values()) or 1.0
        probs = {k: float(v / denom) for k, v in exps.items()}

        score = probs["A"] + 0.5 * probs["C"]

        return {
            "score": float(score),
            "p_a_correct": probs["A"],
            "p_b_incorrect": probs["B"],
            "p_c_not_sure": probs["C"],
            "candidate_logprobs": cand_scores,
            "prompt_len": len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"]),
        }





def normalize_weights(values: List[float], fallback: str, unit_scores: List[float]) -> List[float]:
    if not values:
        return []

    total = sum(values)
    if total > 0:
        return [float(v / total) for v in values]

    if fallback == "uniform":
        return [1.0 / len(values)] * len(values)

    if fallback == "probability":
        m = max(unit_scores)
        exps = [math.exp(u - m) for u in unit_scores]
        denom = sum(exps)
        if denom <= 0:
            return [1.0 / len(values)] * len(values)
        return [float(e / denom) for e in exps]

    if fallback == "zero":
        return [0.0 for _ in values]

    raise ValueError(f"Unsupported zero-importance-fallback: {fallback}")


def process_item(
    record: Dict[str, Any],
    line_idx: int,
    args: argparse.Namespace,
    judge: LocalJudge,
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

    units = extract_mars_units(
        response=target_resp,
        include_answer_unit=args.include_answer_unit,
        unit_logprob_mode=args.unit_logprob_mode,
        missing_logprob=args.missing_logprob,
        max_units=args.max_units,
    )

    if not units:
        return {
            "id": rid,
            "correct": correctness.get("correct"),
            "error": "no_mars_units",
        }

    original_reasoning = build_reasoning_from_units(units, masked_unit_id=None, mask_text=args.mask_text)
    original_answer = build_answer_from_units(target_resp, units, masked_unit_id=None, mask_text=args.mask_text)

    original_judge = judge.judge_score(
        question=question,
        reasoning=original_reasoning,
        answer=original_answer,
    )
    original_score = float(original_judge["score"])

    unit_debug: List[Dict[str, Any]] = []
    importances: List[float] = []
    unit_scores: List[float] = []

    for unit in units:
        uid = int(unit["unit_id"])

        masked_reasoning = build_reasoning_from_units(units, masked_unit_id=uid, mask_text=args.mask_text)
        masked_answer = build_answer_from_units(target_resp, units, masked_unit_id=uid, mask_text=args.mask_text)

        masked_judge = judge.judge_score(
            question=question,
            reasoning=masked_reasoning,
            answer=masked_answer,
        )
        masked_score = float(masked_judge["score"])
        importance = max(0.0, original_score - masked_score)

        unit_score = float(unit["unit_logprob_score"] if unit["unit_logprob_score"] is not None else 0.0)

        importances.append(float(importance))
        unit_scores.append(unit_score)

        dbg = {
            "unit_id": uid,
            "unit_type": unit["unit_type"],
            "step_index": unit.get("step_index"),
            "text": unit.get("text", ""),
            "num_tokens": unit.get("num_tokens"),
            "seq_logprob": unit.get("seq_logprob"),
            "avg_logprob": unit.get("avg_logprob"),
            "unit_logprob_score": unit_score,
            "judge_masked_score": masked_score,
            "importance": float(importance),
        }

        if unit.get("unit_type") == "final_answer":
            dbg["extracted_answer"] = unit.get("extracted_answer")

        if args.keep_judge_details:
            dbg["masked_judge"] = masked_judge

        unit_debug.append(dbg)

    weights = normalize_weights(importances, args.zero_importance_fallback, unit_scores)

    mars_score = 0.0
    for w, unit_score in zip(weights, unit_scores):
        mars_score += w * unit_score

    mars_score = float(mars_score)
    mars_uncertainty = float(-mars_score)

    if args.normalize_uncertainty:
        mars_uncertainty_normalized = float(1.0 - math.exp(-max(0.0, mars_uncertainty)))
        uncertainty = mars_uncertainty_normalized
    else:
        mars_uncertainty_normalized = None
        uncertainty = mars_uncertainty

    result: Dict[str, Any] = {
        "id": rid,
        "correct": correctness.get("correct"),

        "target_response_index": target_idx,
        "target_answer": original_answer,
        "matched_prediction": correctness.get("matched_prediction"),
        "matched_gold": correctness.get("matched_gold"),

        "mars_score": mars_score,
        "mars_uncertainty": mars_uncertainty,
        "mars_uncertainty_normalized": mars_uncertainty_normalized,
        "uncertainty": uncertainty,

        "judge_original_score": original_score,
        "num_responses": len(responses),
        "num_units": len(units),
        "judge_calls": len(units) + 1,

        "unit_importance_sum": float(sum(importances)),
        "zero_importance_fallback_used": bool(sum(importances) <= 0),

        "meta": {
            "method": "MARS",
            "dataset": "trivialqa_reasoning_sampling",
            "correctness_source": "input_correct_answer_value_aliases",
            "correctness_rule": "trivialqa_normalized_alias_match",
            "match_mode": args.match_mode,
            "target_mode": args.target_mode,
            "judge_model": args.model_path,
            "model_family": args.model_family,
            "mask_text": args.mask_text,
            "include_answer_unit": bool(args.include_answer_unit),
            "unit_logprob_mode": args.unit_logprob_mode,
            "missing_logprob": args.missing_logprob,
            "zero_importance_fallback": args.zero_importance_fallback,
            "normalized_uncertainty_alias": bool(args.normalize_uncertainty),
            "judge_score": "P(A Correct) + 0.5 * P(C Not sure)",
        },
    }

    if args.keep_question:
        result["question"] = question
    if args.keep_prediction:
        result["prediction"] = original_answer
    if args.keep_reasoning:
        result["target_reasoning"] = original_reasoning
    if args.keep_gold:
        result["gold_answers"] = correctness.get("gold_answers", [])
    if args.keep_candidates:
        result["prediction_candidates"] = correctness.get("prediction_candidates", [])
    if args.keep_unit_debug:
        for dbg, w in zip(unit_debug, weights):
            dbg["weight"] = float(w)
        result["unit_debug"] = unit_debug
    if args.keep_judge_details:
        result["judge_original"] = original_judge

    return result





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute MARS uncertainty for TrivialQA reasoning-sampling data."
    )

    parser.add_argument("--input-file", required=True, help="Input TrivialQA reasoning JSONL/JSON file")
    parser.add_argument("--output-file", required=True, help="Output JSONL file")

    parser.add_argument("--model-path", required=True, help="Local judge model path")
    parser.add_argument(
        "--model-family",
        choices=["qwen", "llama", "auto"],
        default="auto",
        help="Judge model family. qwen disables thinking when possible.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--hf-endpoint", default=None)

    parser.add_argument(
        "--target-mode",
        choices=["first", "majority_answer"],
        default="first",
        help="Target response to evaluate.",
    )
    parser.add_argument(
        "--match-mode",
        choices=["exact", "exact_or_contains"],
        default="exact",
        help="TrivialQA correctness match mode.",
    )

    parser.add_argument(
        "--mask-text",
        default="[MASK]",
        help="Text used to replace a masked reasoning step or final answer.",
    )
    parser.add_argument(
        "--include-answer-unit",
        action="store_true",
        help="Also mask the final answer unit. Recommended for TrivialQA.",
    )
    parser.add_argument(
        "--max-units",
        type=int,
        default=12,
        help="Maximum MARS units to mask. Keeps final answer unit if included.",
    )
    parser.add_argument(
        "--unit-logprob-mode",
        choices=["mean", "sum"],
        default="mean",
        help="How to score each reasoning/answer unit from token logprobs.",
    )
    parser.add_argument(
        "--missing-logprob",
        choices=["error", "zero"],
        default="error",
        help="What to do when a MARS unit has no stored logprob.",
    )
    parser.add_argument(
        "--zero-importance-fallback",
        choices=["uniform", "probability", "zero"],
        default="uniform",
        help="Weight fallback if all masked-unit importances are zero.",
    )
    parser.add_argument(
        "--normalize-uncertainty",
        action="store_true",
        help="Use normalized uncertainty alias 1-exp(-mars_uncertainty) for the generic uncertainty field.",
    )

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--keep-question", action="store_true")
    parser.add_argument("--keep-prediction", action="store_true")
    parser.add_argument("--keep-reasoning", action="store_true")
    parser.add_argument("--keep-gold", action="store_true")
    parser.add_argument("--keep-candidates", action="store_true")
    parser.add_argument("--keep-unit-debug", action="store_true")
    parser.add_argument("--keep-judge-details", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    if args.model_family == "auto":
        low = args.model_path.lower()
        if "qwen" in low:
            args.model_family = "qwen"
        elif "llama" in low:
            args.model_family = "llama"
        else:
            args.model_family = "auto"

    safe_print(f"INPUT_FILE = {args.input_file}")
    safe_print(f"OUTPUT_FILE = {args.output_file}")
    safe_print("Correctness source: record['correct answer']['value'] + aliases")
    safe_print(f"TrivialQA match mode = {args.match_mode}")
    safe_print(f"Judge model = {args.model_path}")
    safe_print(f"Judge family = {args.model_family}")
    safe_print(f"Target mode = {args.target_mode}")
    safe_print(f"Mask text = {args.mask_text}")
    safe_print(f"Include answer unit = {args.include_answer_unit}")
    safe_print(f"Max units = {args.max_units}")
    safe_print(f"Unit logprob mode = {args.unit_logprob_mode}")

    records = load_records(args.input_file)
    safe_print(f"Loaded records: {len(records)}")

    safe_print("Loading local judge model...")
    judge = LocalJudge(
        model_path=args.model_path,
        model_family=args.model_family,
        dtype=args.dtype,
        device=args.device,
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
                result = process_item(
                    record=record,
                    line_idx=line_idx,
                    args=args,
                    judge=judge,
                )
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                processed += 1

                if args.verbose:
                    safe_print(
                        f"done: {rid} | line={line_idx} | "
                        f"correct={result.get('correct')} | "
                        f"origS={result.get('judge_original_score')} | "
                        f"units={result.get('num_units')} | "
                        f"mars={result.get('mars_score')} | "
                        f"unc={result.get('uncertainty')} | "
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

    _parser = _ap.ArgumentParser(description="mars / qwen / triviaqa / reasoning_augmented (standalone)")
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
    if "mars" not in {"max_seq_prob", "mean_tok_ent", "max_tok_prob", "ppl"}:
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
    if "mars" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "mars", "family": "qwen",
                           "dataset": "triviaqa", "setting": "reasoning_augmented", "model_loading": False}, indent=2))
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

