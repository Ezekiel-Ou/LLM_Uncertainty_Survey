#!/usr/bin/env python3
# Original paper/author repository: https://github.com/ZBox1005/CoT-UQ

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
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer






BOX_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
FINAL_RE = re.compile(r"(?:final\s*answer|answer)\s*[:：]\s*(.+)", re.I)
NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/[1-9]\d*)?")


def normalize_answer(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    m = BOX_RE.search(s)
    if m:
        s = m.group(1)
    s = s.strip()
    s = re.sub(r"\\[a-zA-Z]+\s*", " ", s)
    s = s.replace("{", " ").replace("}", " ")
    s = s.replace("_", " ")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\.\-/]+", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def extract_boxed(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    matches = BOX_RE.findall(text)
    if matches:
        return matches[-1].strip()
    if "####" in text:
        tail = text.split("####")[-1].strip()
        if tail:
            return tail.splitlines()[0].strip().strip(".# ")
    final_matches = FINAL_RE.findall(text)
    if final_matches:
        cand = final_matches[-1].strip().splitlines()[0]
        return cand.strip().strip(".# ")
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line.strip("# ").strip()
    return ""


def extract_number(s: Any) -> Optional[float]:
    if s is None:
        return None
    s = str(s)
    if "####" in s:
        s = s.split("####")[-1]
    boxed = extract_boxed(s)
    if boxed:
        s = boxed
    matches = NUM_RE.findall(s.replace(",", ""))
    if not matches:
        return None
    x = matches[-1]
    try:
        if "/" in x:
            a, b = x.split("/", 1)
            return float(a) / float(b)
        return float(x)
    except Exception:
        return None


def get_gold_and_aliases(item: Dict[str, Any]) -> Tuple[str, List[str]]:
    ca = (
        item.get("correct answer")
        or item.get("correct_answer")
        or item.get("answer")
        or item.get("gold")
        or item.get("label")
    )
    if isinstance(ca, dict):
        value = ca.get("value") or ca.get("answer") or ca.get("text") or ""
        aliases = ca.get("aliases") or ca.get("alias") or []
        if not isinstance(aliases, list):
            aliases = [aliases]
        return str(value), [str(x) for x in aliases]
    if isinstance(ca, list):
        value = str(ca[0]) if ca else ""
        return value, [str(x) for x in ca]
    return str(ca or ""), []


def is_correct(prediction: str, gold: str, aliases: Sequence[str], match_mode: str) -> int:
    if match_mode in ("gsm8k", "numeric"):
        p = extract_number(prediction)
        candidates = [gold] + list(aliases or [])
        for c in candidates:
            g = extract_number(c)
            if p is not None and g is not None and abs(p - g) <= 1e-6 * max(1.0, abs(g)):
                return 1
        return 0

    pred_n = normalize_answer(prediction)
    cand_ns = [normalize_answer(x) for x in [gold] + list(aliases or []) if str(x).strip()]
    if not pred_n or not cand_ns:
        return 0
    if match_mode == "exact":
        return int(pred_n in cand_ns)
    if match_mode == "contains":
        return int(any(pred_n in c or c in pred_n for c in cand_ns))
    if match_mode == "token_f1":
        p_tokens = pred_n.split()
        best = 0.0
        for c in cand_ns:
            c_tokens = c.split()
            common = Counter(p_tokens) & Counter(c_tokens)
            same = sum(common.values())
            if not p_tokens or not c_tokens or same == 0:
                continue
            prec = same / len(p_tokens)
            rec = same / len(c_tokens)
            best = max(best, 2 * prec * rec / (prec + rec))
        return int(best >= 0.8)
    raise ValueError(f"Unknown match_mode: {match_mode}")


QUESTION_KEYS = ["question", "query", "prompt", "input", "problem", "instruction", "original_question", "raw_question"]


def get_question(item: Dict[str, Any]) -> str:
    for k in QUESTION_KEYS:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    msgs = item.get("messages")
    if isinstance(msgs, list):
        parts = []
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "user":
                parts.append(str(m.get("content", "")))
        if parts:
            return "\n".join(parts).strip()
    return ""


def split_reasoning_steps(reason: str) -> List[str]:
    reason = str(reason or "").strip()
    if not reason:
        return []
    parts = re.split(r"(?=Step\s*\d+\s*[:：])", reason)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        return parts
    lines = [x.strip() for x in reason.splitlines() if x.strip()]
    return lines or [reason]


def get_response_answer_text(response: Dict[str, Any]) -> str:
    ai = response.get("answer info") or response.get("answer_info") or {}
    if isinstance(ai, dict) and ai.get("text"):
        return str(ai.get("text"))
    return str(response.get("llm answer") or response.get("answer") or "")


def get_units(response: Dict[str, Any], include_answer_unit: bool, max_units: int) -> List[str]:
    units: List[str] = []
    swi = response.get("step-wise info") or response.get("step_wise_info") or response.get("steps")
    if isinstance(swi, list):
        for st in swi:
            txt = st.get("text") if isinstance(st, dict) else st
            if isinstance(txt, str) and txt.strip():
                units.append(txt.strip())
    if not units:
        units = split_reasoning_steps(response.get("llm reason") or response.get("reasoning_augmented") or "")

    if include_answer_unit:
        pred = extract_boxed(get_response_answer_text(response))
        if pred:
            units.append(f"Final answer: {pred}")

    units = [u for u in units if u.strip()]
    if max_units and max_units > 0:
        units = units[:max_units]
    return units


def build_candidates(responses: List[Dict[str, Any]]) -> List[str]:
    return [extract_boxed(get_response_answer_text(r)) for r in responses]


def choose_target(item: Dict[str, Any], responses: List[Dict[str, Any]], response_index: int, target_mode: str) -> str:
    gold, _aliases = get_gold_and_aliases(item)
    preds = build_candidates(responses)
    if target_mode == "first":
        return preds[0] if preds else ""
    if target_mode in ("self", "prediction"):
        return preds[response_index] if response_index < len(preds) else ""
    if target_mode == "majority":
        cnt = Counter()
        surface: Dict[str, str] = {}
        for p in preds:
            pn = normalize_answer(p)
            if pn:
                cnt[pn] += 1
                surface.setdefault(pn, p)
        return surface[cnt.most_common(1)[0][0]] if cnt else (preds[0] if preds else "")
    if target_mode == "gold":
        return gold
    raise ValueError(f"Unknown target_mode: {target_mode}")







def build_reasoning_context(question: str, units: Sequence[str], response_prompt: bool = False) -> str:
    chunks = [f"Question:\n{question.strip()}" if question else "Question:\n"]
    chunks.append("Reasoning:")
    for i, u in enumerate(units, 1):
        u = str(u).strip()
        if re.match(r"^Step\s*\d+\s*[:：]", u, flags=re.I):
            chunks.append(u)
        else:
            chunks.append(f"Step {i}: {u}")
    if response_prompt:
        chunks.append("Therefore, the concise final answer is:")
    return "\n".join(chunks).rstrip() + "\n"


def build_self_probing_prompt(question: str, possible_answer: str, key_step: str) -> str:
    return (
        f"Question: {question}\n"
        f"Possible Answer: {possible_answer}\n"
        f"The most critical step in reasoning to the possible answer: {key_step}\n"
        "Q: Considering this critical reasoning step as additional information, "
        "how likely is the above answer to be correct? Please first show your reasoning concisely "
        "and then answer with the following format:\n"
        f"```Confidence: [the probability of answer {possible_answer} to be correct, "
        "not the one you think correct, please only include the numerical number]%```\n"
        "Confidence: "
    )


def parse_confidence(text: str) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if m:
        v = float(m.group(1)) / 100.0
        return min(max(v, 0.0), 1.0)
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    try:
        v = float(first_line.strip())
        if v > 1:
            v /= 100.0
        return min(max(v, 0.0), 1.0)
    except Exception:
        pass
    m = re.search(r"(\d+(?:\.\d+)?)", first_line)
    if m:
        v = float(m.group(1))
        if v > 1:
            v /= 100.0
        return min(max(v, 0.0), 1.0)
    return None







@dataclass
class ScoreResult:
    token_logprobs: List[float]
    token_ids: List[int]
    token_texts: List[str]
    agg_logprob: float
    mean_logprob: float
    n_tokens: int


class CausalLMScorer:
    def __init__(
        self,
        model_path: str,
        model_family: str = "auto",
        torch_dtype: str = "auto",
        device_map: str = "auto",
        max_length: Optional[int] = None,
        chat_template_mode: str = "off",
        use_fast_tokenizer: bool = True,
        trust_remote_code: bool = True,
        attn_implementation: Optional[str] = None,
    ) -> None:
        self.model_family = (model_family or "auto").lower()
        self.chat_template_mode = (chat_template_mode or "off").lower()
        if self.chat_template_mode not in {"off", "on", "auto"}:
            raise ValueError(f"Unknown chat_template_mode: {chat_template_mode}")

        
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast_tokenizer,
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        dtype_arg: Any = torch_dtype
        if torch_dtype in ("float16", "fp16"):
            dtype_arg = torch.float16
        elif torch_dtype in ("bfloat16", "bf16"):
            dtype_arg = torch.bfloat16
        elif torch_dtype in ("float32", "fp32"):
            dtype_arg = torch.float32
        elif torch_dtype == "auto":
            dtype_arg = "auto"

        model_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype_arg,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        
        
        if len(self.tokenizer) > getattr(self.model.get_input_embeddings(), "num_embeddings", len(self.tokenizer)):
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

        cfg_max = (
            getattr(self.model.config, "max_position_embeddings", None)
            or getattr(self.model.config, "seq_length", None)
            or getattr(self.tokenizer, "model_max_length", None)
        )
        if cfg_max is None or int(cfg_max) > 10**8:
            cfg_max = 4096
        self.max_length = int(max_length or cfg_max or 4096)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _should_use_chat_template(self) -> bool:
        if self.chat_template_mode == "off":
            return False
        if self.chat_template_mode == "on":
            return hasattr(self.tokenizer, "apply_chat_template") and bool(getattr(self.tokenizer, "chat_template", None))
        
        
        
        return hasattr(self.tokenizer, "apply_chat_template") and bool(getattr(self.tokenizer, "chat_template", None))

    def maybe_chat_prompt(self, prompt: str, add_generation_prompt: bool = True) -> str:
        if not self._should_use_chat_template():
            return prompt
        messages = [{"role": "user", "content": prompt}]
        try:
            
            
            
            
            if self.model_family == "qwen":
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    enable_thinking=False,
                )
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            return prompt

    def _encode_pair(self, prompt: str, target: str) -> Tuple[List[int], List[int], List[int]]:
        prompt = self.maybe_chat_prompt(prompt, add_generation_prompt=True)
        target = str(target or "")
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        if not target_ids:
            return [], [], []
        max_prompt_len = max(1, self.max_length - len(target_ids))
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        return input_ids, labels, target_ids

    def score_pairs(
        self,
        pairs: Sequence[Tuple[str, str]],
        batch_size: int,
        agg_mode: str,
        missing_logprob: str,
    ) -> List[ScoreResult]:
        results: List[Optional[ScoreResult]] = [None] * len(pairs)
        encoded_all: List[Optional[Tuple[List[int], List[int], List[int]]]] = []

        for idx, (prompt, target) in enumerate(pairs):
            input_ids, labels, target_ids = self._encode_pair(prompt, target)
            if not target_ids:
                if missing_logprob == "error":
                    raise ValueError(f"Empty target while scoring pair index={idx}")
                val = 0.0 if missing_logprob == "zero" else float("nan")
                results[idx] = ScoreResult([], [], [], val, val, 0)
                encoded_all.append(None)
            else:
                encoded_all.append((input_ids, labels, target_ids))

        valid_indices = [i for i, x in enumerate(encoded_all) if x is not None]
        pad_id = self.tokenizer.pad_token_id

        for start in range(0, len(valid_indices), batch_size):
            batch_indices = valid_indices[start:start + batch_size]
            batch_encoded = [encoded_all[i] for i in batch_indices]  
            max_len = max(len(x[0]) for x in batch_encoded)  
            input_batch, label_batch, attn_batch = [], [], []
            target_ids_list: List[List[int]] = []

            for input_ids, labels, target_ids in batch_encoded:  
                pad_len = max_len - len(input_ids)
                input_batch.append(input_ids + [pad_id] * pad_len)
                label_batch.append(labels + [-100] * pad_len)
                attn_batch.append([1] * len(input_ids) + [0] * pad_len)
                target_ids_list.append(target_ids)

            input_tensor = torch.tensor(input_batch, dtype=torch.long, device=self.device)
            label_tensor = torch.tensor(label_batch, dtype=torch.long, device=self.device)
            attn_tensor = torch.tensor(attn_batch, dtype=torch.long, device=self.device)

            with torch.no_grad():
                logits = self.model(input_ids=input_tensor, attention_mask=attn_tensor).logits

            shift_logits = logits[:, :-1, :]
            shift_labels = label_tensor[:, 1:]
            mask = shift_labels.ne(-100)
            safe_labels = shift_labels.masked_fill(~mask, 0)
            log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
            gathered = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

            for row, original_idx in enumerate(batch_indices):
                vals = [float(x) for x in gathered[row][mask[row]].detach().cpu().tolist()]
                target_ids = target_ids_list[row]
                
                if len(vals) != len(target_ids):
                    target_ids = target_ids[-len(vals):]
                token_texts = [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in target_ids]
                if not vals:
                    if missing_logprob == "error":
                        raise ValueError(f"No scored tokens for pair index={original_idx}")
                    agg = 0.0 if missing_logprob == "zero" else float("nan")
                    mean_lp = agg
                else:
                    mean_lp = sum(vals) / len(vals)
                    if agg_mode == "mean":
                        agg = mean_lp
                    elif agg_mode == "sum":
                        agg = sum(vals)
                    elif agg_mode == "min":
                        agg = min(vals)
                    elif agg_mode == "max":
                        agg = max(vals)
                    else:
                        raise ValueError(f"Unknown agg_mode: {agg_mode}")
                results[original_idx] = ScoreResult(vals, target_ids, token_texts, float(agg), float(mean_lp), len(vals))

        return [r if r is not None else ScoreResult([], [], [], float("nan"), float("nan"), 0) for r in results]

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
    ) -> str:
        prompt = self.maybe_chat_prompt(prompt, add_generation_prompt=True)
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        new_tokens = out[0, enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()







def safe_exp(x: float) -> float:
    if x is None or math.isnan(float(x)):
        return float("nan")
    return math.exp(max(min(float(x), 0.0), -80.0))


def normalize_weights(raw: Sequence[float], fallback: str) -> List[float]:
    vals = [max(0.0, float(x)) if x is not None and not math.isnan(float(x)) else 0.0 for x in raw]
    s = sum(vals)
    n = len(vals)
    if n == 0:
        return []
    if s > 0:
        return [v / s for v in vals]
    if fallback == "uniform":
        return [1.0 / n] * n
    if fallback == "first":
        return [1.0] + [0.0] * (n - 1)
    if fallback == "last":
        return [0.0] * (n - 1) + [1.0]
    if fallback == "zero":
        return [0.0] * n
    raise ValueError(f"Unknown fallback: {fallback}")


def weighted_sum(weights: Sequence[float], values: Sequence[float]) -> float:
    if not weights or not values:
        return float("nan")
    s = 0.0
    for w, v in zip(weights, values):
        if v is None or math.isnan(float(v)):
            continue
        s += float(w) * float(v)
    return float(s)


def load_sar_model(path: Optional[str]):
    if not path:
        return None
    try:
        from sentence_transformers.cross_encoder import CrossEncoder
        return CrossEncoder(model_name=path, num_labels=1)
    except Exception as e:
        print(f"[WARN] Cannot load SAR CrossEncoder from {path}: {e}. Falling back to self-information SAR.", file=sys.stderr)
        return None


def sar_token_relevance(
    question: str,
    unit_text: str,
    score: ScoreResult,
    tokenizer: Any,
    sar_model: Any,
    mode: str,
) -> List[float]:
    n = len(score.token_logprobs)
    if n == 0:
        return []

    if mode == "uniform":
        return [1.0 / n] * n

    if mode == "self_information" or sar_model is None:
        
        raw = [max(0.0, -lp) for lp in score.token_logprobs]
        if sum(raw) <= 0:
            return [1.0 / n] * n
        return [x / sum(raw) for x in raw]

    if mode == "cross_encoder":
        base = (question or "") + "\n" + unit_text
        pairs = []
        for tok in score.token_texts:
            tok_clean = tok.replace("Ġ", " ").replace("▁", " ")
            ablated_unit = unit_text.replace(tok_clean, "", 1)
            if ablated_unit == unit_text:
                
                ablated_unit = unit_text.replace(tok_clean.strip(), "", 1)
            ablated = (question or "") + "\n" + ablated_unit
            pairs.append([base, ablated])
        try:
            sims = sar_model.predict(pairs)
            raw = [max(0.0, 1.0 - float(x)) for x in sims]
            if sum(raw) <= 0:
                return [1.0 / n] * n
            return [x / sum(raw) for x in raw]
        except Exception as e:
            print(f"[WARN] CrossEncoder SAR failed: {e}. Falling back to self-information SAR.", file=sys.stderr)
            raw = [max(0.0, -lp) for lp in score.token_logprobs]
            if sum(raw) <= 0:
                return [1.0 / n] * n
            return [x / sum(raw) for x in raw]

    raise ValueError(f"Unknown sar_relevance_mode: {mode}")


def token_sar_confidence(
    question: str,
    unit_text: str,
    score: ScoreResult,
    tokenizer: Any,
    sar_model: Any,
    mode: str,
) -> Tuple[float, List[float]]:
    if not score.token_logprobs:
        return float("nan"), []
    rel = sar_token_relevance(question, unit_text, score, tokenizer, sar_model, mode)
    probs = [safe_exp(lp) for lp in score.token_logprobs]
    return weighted_sum(rel, probs), rel


def compute_for_response(
    item: Dict[str, Any],
    responses: List[Dict[str, Any]],
    response_index: int,
    scorer: CausalLMScorer,
    sar_model: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    response = responses[response_index]
    question = get_question(item)
    gold, aliases = get_gold_and_aliases(item)
    answer_text = get_response_answer_text(response)
    prediction = extract_boxed(answer_text)
    target = choose_target(item, responses, response_index, args.target_mode)
    units = get_units(response, include_answer_unit=args.include_answer_unit, max_units=args.max_units)

    out: Dict[str, Any] = {
        "id": item.get("id") or item.get("qid") or item.get("question_id"),
        "response_index": response_index,
        "correct": is_correct(prediction, gold, aliases, args.match_mode),
        "prediction": prediction if args.keep_prediction else None,
        "cotuq_num_units": len(units),
    }
    if not args.keep_prediction:
        out.pop("prediction", None)
    if args.keep_question:
        out["question"] = question
    if args.keep_gold:
        out["gold"] = gold
        out["gold_aliases"] = aliases
    if args.keep_candidates:
        out["candidates"] = build_candidates(responses)
    if args.keep_units:
        out["cotuq_units"] = units

    if not units or not target:
        for name in ["probas_mean", "probas_min", "token_sar", "self_probing"]:
            out[f"cotuq_{name}_confidence"] = None
            out[f"cotuq_{name}_uncertainty"] = None
        out["error"] = "empty_units_or_target"
        return out

    
    unit_pairs: List[Tuple[str, str]] = []
    for i, unit in enumerate(units):
        prefix = units[:i] if args.unit_context_mode == "prefix" else []
        unit_pairs.append((build_reasoning_context(question, prefix, response_prompt=False), unit))

    unit_scores = scorer.score_pairs(
        unit_pairs,
        batch_size=args.batch_size,
        agg_mode="mean",  
        missing_logprob=args.missing_logprob,
    )

    
    target_surface = " " + target if not str(target).startswith(" ") else str(target)
    full_prompt = build_reasoning_context(question, units, response_prompt=True)
    mask_pairs = [(full_prompt, target_surface)]
    for i in range(len(units)):
        masked_units = list(units)
        masked_units[i] = args.mask_text
        mask_pairs.append((build_reasoning_context(question, masked_units, response_prompt=True), target_surface))

    target_scores = scorer.score_pairs(
        mask_pairs,
        batch_size=args.batch_size,
        agg_mode=args.target_logprob_mode,
        missing_logprob=args.missing_logprob,
    )
    full_target_score = target_scores[0].agg_logprob
    masked_scores = [s.agg_logprob for s in target_scores[1:]]
    raw_importance = [
        max(0.0, full_target_score - ms) if not (math.isnan(full_target_score) or math.isnan(ms)) else 0.0
        for ms in masked_scores
    ]
    unit_weights = normalize_weights(raw_importance, args.zero_importance_fallback)

    
    unit_probas_mean: List[float] = []
    unit_probas_min: List[float] = []
    unit_token_sar: List[float] = []
    unit_sar_relevance: List[List[float]] = []

    for unit, score in zip(units, unit_scores):
        probs = [safe_exp(lp) for lp in score.token_logprobs]
        if probs:
            unit_probas_mean.append(sum(probs) / len(probs))
            unit_probas_min.append(min(probs))
        else:
            unit_probas_mean.append(float("nan"))
            unit_probas_min.append(float("nan"))
        sar_conf, sar_rel = token_sar_confidence(question, unit, score, scorer.tokenizer, sar_model, args.sar_relevance_mode)
        unit_token_sar.append(sar_conf)
        unit_sar_relevance.append(sar_rel)

    probas_mean_conf = weighted_sum(unit_weights, unit_probas_mean)
    probas_min_conf = weighted_sum(unit_weights, unit_probas_min)
    token_sar_conf = weighted_sum(unit_weights, unit_token_sar)

    out.update({
        "cotuq_probas_mean_confidence": probas_mean_conf,
        "cotuq_probas_mean_uncertainty": 1.0 - probas_mean_conf if not math.isnan(probas_mean_conf) else None,
        "cotuq_probas_min_confidence": probas_min_conf,
        "cotuq_probas_min_uncertainty": 1.0 - probas_min_conf if not math.isnan(probas_min_conf) else None,
        "cotuq_token_sar_confidence": token_sar_conf,
        "cotuq_token_sar_uncertainty": 1.0 - token_sar_conf if not math.isnan(token_sar_conf) else None,
        "target_logprob_full": full_target_score,
        "unit_importance_raw": raw_importance,
        "unit_importance": unit_weights,
        "unit_masked_target_logprobs": masked_scores,
        "unit_probas_mean": unit_probas_mean,
        "unit_probas_min": unit_probas_min,
        "unit_token_sar": unit_token_sar,
        "unit_mean_logprobs": [s.mean_logprob for s in unit_scores],
        "unit_min_logprobs": [min(s.token_logprobs) if s.token_logprobs else None for s in unit_scores],
        "unit_token_counts": [s.n_tokens for s in unit_scores],
    })

    
    if args.disable_self_probing:
        out["cotuq_self_probing_confidence"] = None
        out["cotuq_self_probing_uncertainty"] = None
    else:
        if unit_weights:
            key_idx = max(range(len(unit_weights)), key=lambda i: unit_weights[i])
        else:
            key_idx = len(units) - 1
        key_step = units[key_idx] if units else ""
        possible_answer = prediction or answer_text
        sp_prompt = build_self_probing_prompt(question, possible_answer, key_step)
        sp_response = ""
        sp_conf: Optional[float] = None
        for _ in range(max(1, args.self_probing_tries)):
            sp_response = scorer.generate_text(
                sp_prompt,
                max_new_tokens=args.self_probing_max_new_tokens,
                do_sample=args.self_probing_do_sample,
                temperature=args.self_probing_temperature,
            )
            sp_conf = parse_confidence(sp_response)
            if sp_conf is not None:
                break
        out["cotuq_self_probing_confidence"] = sp_conf
        out["cotuq_self_probing_uncertainty"] = 1.0 - sp_conf if sp_conf is not None else None
        out["self_probing_key_unit_index"] = key_idx
        if args.keep_self_probing_response:
            out["self_probing_prompt"] = sp_prompt
            out["self_probing_response"] = sp_response

    if args.keep_token_logprobs:
        out["unit_token_logprobs"] = [s.token_logprobs for s in unit_scores]
        out["unit_token_texts"] = [s.token_texts for s in unit_scores]
        out["unit_sar_relevance"] = unit_sar_relevance
        out["target_token_logprobs_full"] = target_scores[0].token_logprobs

    
    out["uncertainties"] = {
        "Probas-mean": out.get("cotuq_probas_mean_uncertainty"),
        "Probas-min": out.get("cotuq_probas_min_uncertainty"),
        "Token-SAR": out.get("cotuq_token_sar_uncertainty"),
        "Self-Probing": out.get("cotuq_self_probing_uncertainty"),
    }
    out["confidences"] = {
        "Probas-mean": out.get("cotuq_probas_mean_confidence"),
        "Probas-min": out.get("cotuq_probas_min_confidence"),
        "Token-SAR": out.get("cotuq_token_sar_confidence"),
        "Self-Probing": out.get("cotuq_self_probing_confidence"),
    }
    return out







def iter_jsonl(path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except Exception as e:
                raise ValueError(f"Failed to parse JSON at line {line_no}: {e}") from e


def json_sanitize(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_sanitize(v) for v in obj]
    return obj


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute four model-scored CoT-UQ methods for reasoning JSONL data.")
    p.add_argument("--input-file", required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-family", default="auto", choices=["auto", "qwen", "llama", "mistral", "deepseek"])
    p.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    p.add_argument("--device-map", default="auto")
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--chat-template-mode", default="off", choices=["off", "on", "auto"],
                   help="Prompt wrapping for tokenizer chat templates. Use off for raw/base-model scoring; auto/on for instruct checkpoints.")
    p.add_argument("--use-chat-template", action="store_true",
                   help="Backward-compatible alias for --chat-template-mode on.")
    p.add_argument("--slow-tokenizer", action="store_true", help="Use the slow tokenizer if the fast tokenizer is problematic.")
    p.add_argument("--attn-implementation", default=None,
                   help="Optional Transformers attention backend, e.g. flash_attention_2 or sdpa.")
    p.add_argument("--no-trust-remote-code", action="store_true")

    
    p.add_argument("--target-mode", default="first", choices=["first", "self", "prediction", "majority", "gold"])
    p.add_argument("--match-mode", default="exact", choices=["exact", "contains", "token_f1", "gsm8k", "numeric"])
    p.add_argument("--mask-text", default="[MASK]")
    p.add_argument("--include-answer-unit", action="store_true")
    p.add_argument("--max-units", type=int, default=12)
    p.add_argument("--target-logprob-mode", default="mean", choices=["mean", "sum", "min", "max"])
    p.add_argument("--missing-logprob", default="error", choices=["error", "zero", "nan"])
    p.add_argument("--zero-importance-fallback", default="uniform", choices=["uniform", "first", "last", "zero"])
    p.add_argument("--unit-context-mode", default="prefix", choices=["prefix", "none"])

    
    p.add_argument("--sar-relevance-mode", default="self_information", choices=["self_information", "cross_encoder", "uniform"])
    p.add_argument("--sar-model-path", default=None, help="Optional CrossEncoder path, e.g. cross-encoder/stsb-roberta-large or local dir.")

    
    p.add_argument("--disable-self-probing", action="store_true")
    p.add_argument("--self-probing-tries", type=int, default=2)
    p.add_argument("--self-probing-max-new-tokens", type=int, default=96)
    p.add_argument("--self-probing-temperature", type=float, default=0.0)
    p.add_argument("--self-probing-do-sample", action="store_true")

    
    p.add_argument("--keep-question", action="store_true")
    p.add_argument("--keep-prediction", action="store_true")
    p.add_argument("--keep-gold", action="store_true")
    p.add_argument("--keep-candidates", action="store_true")
    p.add_argument("--keep-units", action="store_true")
    p.add_argument("--keep-token-logprobs", action="store_true")
    p.add_argument("--keep-self-probing-response", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Limit number of input JSONL items/questions to process.")
    p.add_argument("--response-index", type=int, default=None,
                   help="Only compute one response per item. Use 0 for the first response, 1 for the second, etc. Negative indices are allowed, e.g. -1 for the last response.")
    p.add_argument("--one-response-per-item", action="store_true",
                   help="Backward-compatible shortcut for --response-index 0.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    if args.use_chat_template:
        args.chat_template_mode = "on"

    if args.verbose:
        print(
            f"[load] loading causal LM: family={args.model_family}, "
            f"chat_template_mode={args.chat_template_mode}",
            file=sys.stderr,
        )
    scorer = CausalLMScorer(
        model_path=args.model_path,
        model_family=args.model_family,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        max_length=args.max_length,
        chat_template_mode=args.chat_template_mode,
        use_fast_tokenizer=not args.slow_tokenizer,
        trust_remote_code=not args.no_trust_remote_code,
        attn_implementation=args.attn_implementation,
    )

    sar_model = None
    if args.sar_relevance_mode == "cross_encoder":
        if args.verbose:
            print("[load] loading SAR CrossEncoder...", file=sys.stderr)
        sar_model = load_sar_model(args.sar_model_path or "cross-encoder/stsb-roberta-large")

    n_items, n_rows = 0, 0
    with open(args.output_file, "w", encoding="utf-8") as fout:
        for line_no, item in iter_jsonl(args.input_file):
            n_items += 1
            responses = item.get("responses") or []
            if not isinstance(responses, list):
                responses = []
            if args.verbose:
                print(f"[item {n_items}] line={line_no}, id={item.get('id')}, responses={len(responses)}", file=sys.stderr)
            if args.response_index is not None or args.one_response_per_item:
                requested_idx = 0 if args.one_response_per_item and args.response_index is None else args.response_index
                assert requested_idx is not None
                ridx = requested_idx if requested_idx >= 0 else len(responses) + requested_idx
                if ridx < 0 or ridx >= len(responses):
                    msg = f"response_index={requested_idx} out of range for item id={item.get('id')} with {len(responses)} responses"
                    if args.missing_logprob == "error":
                        raise IndexError(msg)
                    if args.verbose:
                        print(f"[WARN] {msg}; skipped", file=sys.stderr)
                    ridxs = []
                else:
                    ridxs = [ridx]
            else:
                ridxs = range(len(responses))

            for ridx in ridxs:
                try:
                    row = compute_for_response(item, responses, ridx, scorer, sar_model, args)
                except Exception as e:
                    if args.missing_logprob == "error":
                        raise
                    row = {
                        "id": item.get("id") or item.get("qid") or item.get("question_id"),
                        "response_index": ridx,
                        "error": str(e),
                    }
                fout.write(json.dumps(json_sanitize(row), ensure_ascii=False) + "\n")
                n_rows += 1
            if args.limit is not None and n_items >= args.limit:
                break
    if args.verbose:
        print(f"Done. items={n_items}, rows={n_rows}, output={args.output_file}", file=sys.stderr)


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

    _parser = _ap.ArgumentParser(description="cot_uq_sp / llama / triviaqa / reasoning_augmented (standalone)")
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
    if "cot_uq_sp" not in {"max_seq_prob", "mean_tok_ent", "max_tok_prob", "ppl"}:
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
    if "cot_uq_sp" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "cot_uq_sp", "family": "llama",
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

