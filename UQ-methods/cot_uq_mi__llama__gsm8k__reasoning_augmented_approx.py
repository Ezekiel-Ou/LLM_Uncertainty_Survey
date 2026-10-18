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
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from math_verify import parse as mathv_parse
    from math_verify import verify as mathv_verify
except Exception as exc:
    mathv_parse = None
    mathv_verify = None
    _MATH_VERIFY_IMPORT_ERROR = exc
else:
    _MATH_VERIFY_IMPORT_ERROR = None

ID_KEYS = ["id", "question_id", "qid", "task_id", "problem_id", "idx"]


def read_jsonl(path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def first_existing(d: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def get_item_id(item: Dict[str, Any]) -> str:
    val = first_existing(item, ID_KEYS)
    if val is None:
        raise KeyError(f"Cannot find id field. Tried keys={ID_KEYS}")
    return str(val)


def get_question_text(item: Dict[str, Any]) -> str:
    return str(first_existing(item, ["question", "problem", "prompt", "input", "query", "text"], ""))


def get_gold_text(item: Dict[str, Any]) -> str:
    return str(first_existing(item, ["correct answer", "correct_answer", "answer", "gold", "gold_answer", "label", "target"], ""))


def extract_boxed(text: str) -> Optional[str]:
    if not text:
        return None
    starts = [m.start() for m in re.finditer(re.escape(r"\boxed"), text)]
    if not starts:
        return None
    for start in reversed(starts):
        i = start + len(r"\boxed")
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text) or text[i] != "{":
            continue
        depth = 0
        content_start = i + 1
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return text[content_start:j].strip()
    return None


def extract_prediction_text(response: Dict[str, Any], prefer_boxed: bool = False) -> str:
    ans = str(response.get("llm answer", "") or "")
    if not ans and isinstance(response.get("answer info"), dict):
        ans = str(response["answer info"].get("text", "") or "")
    if not ans:
        ans = str(response.get("prediction", "") or response.get("output", "") or "")
    if prefer_boxed:
        boxed = extract_boxed(ans)
        if boxed is not None:
            return boxed
    return ans.strip()


def mathv_correct(ref_text: str, target_text: str) -> bool:
    if mathv_parse is None or mathv_verify is None:
        raise ImportError(
            "math-verify is required for GSM8K correctness. "
            f"Original error: {repr(_MATH_VERIFY_IMPORT_ERROR)}"
        )
    return bool(mathv_verify(mathv_parse(ref_text), mathv_parse(target_text)))


def safe_mathv_correct(ref_text: str, target_text: str, on_error: str = "false") -> Optional[int]:
    try:
        return int(mathv_correct(ref_text, target_text))
    except Exception:
        if on_error == "error":
            raise
        if on_error == "missing":
            return None
        return 0


def split_reasoning_steps(reason: str) -> List[str]:
    if not reason:
        return []
    parts = re.split(r"(?:^|\n)\s*Step\s+\d+\s*:\s*", reason)
    parts = [p.strip() for p in parts if p.strip()]
    cleaned = []
    for p in parts:
        p = re.split(r"\n?\s*Final Answer\s*:", p, maxsplit=1, flags=re.I)[0].strip()
        if p:
            cleaned.append(p)
    if cleaned:
        return cleaned
    lines = [x.strip() for x in reason.splitlines() if x.strip()]
    if len(lines) > 1:
        return lines
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", reason) if s.strip()]


@dataclass
class Unit:
    text: str
    kind: str
    index: int


def extract_units(response: Dict[str, Any], include_answer_unit: bool = False, max_units: int = 12) -> List[Unit]:
    units: List[Unit] = []
    swi = response.get("step-wise info")
    if isinstance(swi, list):
        for idx, step in enumerate(swi):
            if isinstance(step, dict):
                txt = str(step.get("text", "") or "").strip()
                if txt:
                    units.append(Unit(text=txt, kind="reasoning_step", index=idx))
    if not units:
        reason = str(response.get("llm reason", "") or "")
        for idx, txt in enumerate(split_reasoning_steps(reason)):
            units.append(Unit(text=txt, kind="reasoning_step", index=idx))
    if include_answer_unit:
        ans = extract_prediction_text(response, prefer_boxed=False)
        if ans:
            units.append(Unit(text=ans, kind="answer", index=len(units)))
    if max_units > 0:
        units = units[:max_units]
    return units


def get_target_answer(item: Dict[str, Any], responses: List[Dict[str, Any]], current_response: Dict[str, Any], target_mode: str) -> str:
    if target_mode == "self":
        return extract_prediction_text(current_response, prefer_boxed=False)
    if target_mode == "boxed_self":
        boxed = extract_prediction_text(current_response, prefer_boxed=True)
        return f"\\boxed{{{boxed}}}" if boxed else extract_prediction_text(current_response, prefer_boxed=False)
    if target_mode == "first":
        return extract_prediction_text(responses[0], prefer_boxed=False)
    if target_mode == "gold":
        return get_gold_text(item)
    raise ValueError(f"Unsupported target_mode={target_mode}")


def build_context(question: str, units: Sequence[Unit], mask_index: Optional[int] = None, mask_text: str = "...") -> str:
    lines = []
    if question:
        lines += ["Question:", question.strip(), ""]
    lines.append("Reasoning:")
    for i, u in enumerate(units):
        txt = mask_text if (mask_index is not None and i == mask_index) else u.text
        lines.append(f"Step {i + 1}: {txt.strip()}")
    lines += ["", "Answer:"]
    return "\n".join(lines)


def build_unit_context(question: str, prev_units: Sequence[Unit]) -> str:
    lines = []
    if question:
        lines += ["Question:", question.strip(), ""]
    lines.append("Reasoning:")
    for i, u in enumerate(prev_units):
        lines.append(f"Step {i + 1}: {u.text.strip()}")
    lines.append(f"Step {len(prev_units) + 1}:")
    return "\n".join(lines) + " "


def maybe_apply_chat_template(tokenizer, prompt: str, model_family: str, chat_template_mode: str, add_generation_prompt: bool = True) -> str:
    if chat_template_mode == "off":
        return prompt
    use_template = chat_template_mode == "on" or (chat_template_mode == "auto" and getattr(tokenizer, "chat_template", None))
    if not use_template:
        return prompt
    try:
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=add_generation_prompt)
    except Exception:
        if chat_template_mode == "on":
            raise
        return prompt


class LMScorer:
    def __init__(self, model_path: str, model_family: str, chat_template_mode: str, device_map: str, torch_dtype: str, attn_implementation: Optional[str], slow_tokenizer: bool):
        self.model_family = model_family
        self.chat_template_mode = chat_template_mode
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=not slow_tokenizer)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype_arg: Any = torch_dtype
        if torch_dtype == "bf16":
            dtype_arg = torch.bfloat16
        elif torch_dtype == "fp16":
            dtype_arg = torch.float16
        elif torch_dtype == "fp32":
            dtype_arg = torch.float32
        kwargs = dict(trust_remote_code=True, device_map=device_map, torch_dtype=dtype_arg)
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.model.eval()
        try:
            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None
        except Exception:
            pass

    @torch.no_grad()
    def token_logprobs(self, prompt: str, continuation: str) -> List[float]:
        continuation = continuation or ""
        if continuation == "":
            return []
        prompt = maybe_apply_chat_template(self.tokenizer, prompt, self.model_family, self.chat_template_mode, True)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
        cont_ids = self.tokenizer(continuation, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        if cont_ids.numel() == 0:
            return []
        input_ids = torch.cat([prompt_ids[0], cont_ids], dim=0).unsqueeze(0).to(next(self.model.parameters()).device)
        attention_mask = torch.ones_like(input_ids)
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        prompt_len = prompt_ids.shape[1]
        logps = []
        for pos in range(prompt_len, input_ids.shape[1]):
            token_id = input_ids[0, pos]
            lp = F.log_softmax(logits[0, pos - 1].float(), dim=-1)[token_id].item()
            logps.append(float(lp))
        return logps

    @torch.no_grad()
    def generate_text(self, prompt: str, max_new_tokens: int = 32) -> str:
        prompt = maybe_apply_chat_template(self.tokenizer, prompt, self.model_family, self.chat_template_mode, True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(next(self.model.parameters()).device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)


def mean(xs: Sequence[float], default: float = float("nan")) -> float:
    return float(sum(xs) / len(xs)) if xs else default


def logprob_to_prob(lp: float) -> float:
    if not math.isfinite(lp):
        return 0.0
    return float(math.exp(max(min(lp, 0.0), -80.0)))


def aggregate_logprob(logps: Sequence[float], mode: str) -> float:
    if not logps:
        return float("-inf")
    if mode == "mean":
        return mean(logps)
    if mode == "sum":
        return float(sum(logps))
    if mode == "min":
        return float(min(logps))
    raise ValueError(f"Unsupported logprob aggregation mode={mode}")


def normalize_weights(raw: Sequence[float], fallback: str = "uniform") -> List[float]:
    vals = [max(0.0, float(x)) if math.isfinite(float(x)) else 0.0 for x in raw]
    s = sum(vals)
    if s > 0:
        return [x / s for x in vals]
    if not vals:
        return []
    if fallback == "uniform":
        return [1.0 / len(vals)] * len(vals)
    if fallback == "zero":
        return [0.0] * len(vals)
    raise ValueError(f"Unsupported fallback={fallback}")


def compute_token_sar_uncertainty(unit_logps: Sequence[Sequence[float]], unit_weights: Sequence[float], relevance_mode: str) -> float:
    unit_scores = []
    for logps in unit_logps:
        if not logps:
            unit_scores.append(1.0)
            continue
        probs = [logprob_to_prob(lp) for lp in logps]
        token_unc = [1.0 - p for p in probs]
        if relevance_mode == "uniform":
            rel = [1.0 / len(logps)] * len(logps)
        elif relevance_mode == "self_information":
            rel = normalize_weights([-lp if math.isfinite(lp) else 0.0 for lp in logps], fallback="uniform")
        else:
            raise ValueError(f"Unsupported sar_relevance_mode={relevance_mode}")
        unit_scores.append(float(sum(r * u for r, u in zip(rel, token_unc))))
    return float(sum(w * s for w, s in zip(unit_weights, unit_scores))) if unit_scores else float("nan")


def parse_confidence(text: str) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", text)
    if m:
        return min(max(float(m.group(1)) / 100.0, 0.0), 1.0)
    m = re.search(r"(?:confidence|probability|score)[^0-9]{0,20}([01](?:\.[0-9]+)?)", text, flags=re.I)
    if m:
        return min(max(float(m.group(1)), 0.0), 1.0)
    nums = re.findall(r"([0-9]+(?:\.[0-9]+)?)", text)
    if nums:
        val = float(nums[0])
        if val > 1:
            val /= 100.0
        return min(max(val, 0.0), 1.0)
    return None


def build_self_probe_prompt(question: str, reason: str, answer: str) -> str:
    return (
        "You are evaluating a GSM8K math reasoning answer.\n"
        "Return only a confidence score from 0 to 100, where 100 means the final answer is certainly correct.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Reasoning:\n{reason.strip()}\n\n"
        f"Answer:\n{answer.strip()}\n\n"
        "Confidence:"
    )


def compute_for_response(scorer: LMScorer, item: Dict[str, Any], response: Dict[str, Any], responses: List[Dict[str, Any]], response_index: int, args: argparse.Namespace) -> Dict[str, Any]:
    item_id = get_item_id(item)
    question = get_question_text(item)
    gold = get_gold_text(item)
    prediction_text = extract_prediction_text(response, prefer_boxed=False)
    prediction_boxed = extract_prediction_text(response, prefer_boxed=True)
    correct = safe_mathv_correct(gold, prediction_text, on_error=args.mathverify_error)

    units = extract_units(response, include_answer_unit=args.include_answer_unit, max_units=args.max_units)
    if not units:
        if args.missing_logprob == "error":
            raise ValueError(f"No reasoning units found for id={item_id}, response_index={response_index}")
        units = [Unit(text=prediction_text, kind="answer", index=0)]

    target_answer = get_target_answer(item, responses, response, args.target_mode)

    unit_token_logps: List[List[float]] = []
    unit_agg_logps: List[float] = []
    for i, unit in enumerate(units):
        logps = scorer.token_logprobs(build_unit_context(question, units[:i]), unit.text)
        if not logps and args.missing_logprob == "error":
            raise ValueError(f"Empty token logprobs for unit {i}, id={item_id}")
        unit_token_logps.append(logps)
        unit_agg_logps.append(aggregate_logprob(logps, args.unit_logprob_mode))

    unit_mean_probs = [logprob_to_prob(lp) for lp in unit_agg_logps]
    unit_min_probs = [min([logprob_to_prob(lp) for lp in lps], default=0.0) for lps in unit_token_logps]

    full_context = build_context(question, units, mask_index=None, mask_text=args.mask_text)
    target_full_logps = scorer.token_logprobs(full_context, target_answer)
    target_full_score = aggregate_logprob(target_full_logps, args.target_logprob_mode)

    raw_importance, masked_target_scores = [], []
    for i in range(len(units)):
        masked_context = build_context(question, units, mask_index=i, mask_text=args.mask_text)
        masked_score = aggregate_logprob(scorer.token_logprobs(masked_context, target_answer), args.target_logprob_mode)
        masked_target_scores.append(masked_score)
        if not math.isfinite(target_full_score) or not math.isfinite(masked_score):
            raw_importance.append(0.0)
        else:
            raw_importance.append(max(0.0, target_full_score - masked_score))
    unit_weights = normalize_weights(raw_importance, fallback=args.zero_importance_fallback)

    probas_mean_conf = float(sum(w * p for w, p in zip(unit_weights, unit_mean_probs))) if unit_weights else float("nan")
    probas_mean_unc = 1.0 - probas_mean_conf if math.isfinite(probas_mean_conf) else float("nan")
    probas_min_conf = float(sum(w * p for w, p in zip(unit_weights, unit_min_probs))) if unit_weights else float("nan")
    probas_min_unc = 1.0 - probas_min_conf if math.isfinite(probas_min_conf) else float("nan")
    token_sar_unc = compute_token_sar_uncertainty(unit_token_logps, unit_weights, relevance_mode=args.sar_relevance_mode)

    self_probe_raw = self_probe_conf = self_probe_unc = None
    if not args.disable_self_probing:
        self_probe_raw = scorer.generate_text(build_self_probe_prompt(question, str(response.get("llm reason", "") or ""), prediction_text), max_new_tokens=args.self_probe_max_new_tokens)
        self_probe_conf = parse_confidence(self_probe_raw)
        if self_probe_conf is None:
            self_probe_conf = args.self_probe_default_confidence
        self_probe_unc = 1.0 - float(self_probe_conf)

    out: Dict[str, Any] = {
        "id": item_id,
        "question_id": item_id,
        "response_index": response_index,
        "correct": correct,
        "prediction": prediction_boxed if args.prediction_boxed_only else prediction_text,
        "uncertainties": {
            "Probas-mean": probas_mean_unc,
            "Probas-min": probas_min_unc,
            "Token-SAR": token_sar_unc,
            "Self-Probing": self_probe_unc,
        },
        "cotuq_probas_mean_uncertainty": probas_mean_unc,
        "cotuq_probas_min_uncertainty": probas_min_unc,
        "cotuq_token_sar_uncertainty": token_sar_unc,
        "cotuq_self_probing_uncertainty": self_probe_unc,
        "target_logprob_full": target_full_score,
        "target_uncertainty_full": 1.0 - logprob_to_prob(target_full_score) if math.isfinite(target_full_score) else None,
        "unit_importance_raw": raw_importance,
        "unit_importance": unit_weights,
        "unit_mean_logprobs": unit_agg_logps,
        "unit_mean_probs": unit_mean_probs,
        "unit_min_probs": unit_min_probs,
    }
    if args.keep_question:
        out["question"] = question
    if args.keep_gold:
        out["gold"] = gold
    if args.keep_prediction:
        out["llm_answer"] = str(response.get("llm answer", "") or "")
        out["llm_reason"] = str(response.get("llm reason", "") or "")
    if args.keep_candidates:
        out["num_responses"] = len(responses)
    if args.keep_units:
        out["unit_texts"] = [u.text for u in units]
    if args.keep_token_logprobs:
        out["unit_token_logprobs"] = unit_token_logps
        out["target_token_logprobs_full"] = target_full_logps
        out["unit_masked_target_scores"] = masked_target_scores
    if args.keep_self_probe:
        out["self_probe_raw"] = self_probe_raw
        out["self_probe_confidence"] = self_probe_conf
    return {k: v for k, v in out.items() if v is not None}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute CoT-UQ for GSM8K reasoning JSONL.")
    p.add_argument("--input-file", required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-family", choices=["qwen", "llama", "generic"], default="qwen")
    p.add_argument("--chat-template-mode", choices=["off", "on", "auto"], default="off")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--torch-dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--attn-implementation", default=None)
    p.add_argument("--slow-tokenizer", action="store_true")
    p.add_argument("--response-index", type=int, default=0, help="Which response to score. Use -1 to score all responses.")
    p.add_argument("--target-mode", choices=["self", "boxed_self", "first", "gold"], default="self")
    p.add_argument("--mask-text", default="...")
    p.add_argument("--include-answer-unit", action="store_true")
    p.add_argument("--max-units", type=int, default=12)
    p.add_argument("--unit-logprob-mode", choices=["mean", "sum", "min"], default="mean")
    p.add_argument("--target-logprob-mode", choices=["mean", "sum", "min"], default="mean")
    p.add_argument("--missing-logprob", choices=["error", "skip"], default="error")
    p.add_argument("--mathverify-error", choices=["false", "missing", "error"], default="false")
    p.add_argument("--zero-importance-fallback", choices=["uniform", "zero"], default="uniform")
    p.add_argument("--sar-relevance-mode", choices=["self_information", "uniform"], default="self_information")
    p.add_argument("--disable-self-probing", action="store_true")
    p.add_argument("--self-probe-max-new-tokens", type=int, default=32)
    p.add_argument("--self-probe-default-confidence", type=float, default=0.5)
    p.add_argument("--prediction-boxed-only", action="store_true")
    p.add_argument("--keep-question", action="store_true")
    p.add_argument("--keep-prediction", action="store_true")
    p.add_argument("--keep-gold", action="store_true")
    p.add_argument("--keep-candidates", action="store_true")
    p.add_argument("--keep-units", action="store_true")
    p.add_argument("--keep-token-logprobs", action="store_true")
    p.add_argument("--keep-self-probe", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if mathv_parse is None or mathv_verify is None:
        raise ImportError("math-verify is required for GSM8K correctness. Install it first. " f"Original import error: {repr(_MATH_VERIFY_IMPORT_ERROR)}")
    scorer = LMScorer(args.model_path, args.model_family, args.chat_template_mode, args.device_map, args.torch_dtype, args.attn_implementation, args.slow_tokenizer)
    parent = os.path.dirname(args.output_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    n_written = 0
    with open(args.output_file, "w", encoding="utf-8") as fout:
        for line_no, item in tqdm(read_jsonl(args.input_file), desc="GSM8K CoT-UQ"):
            item_id = get_item_id(item)
            responses = item.get("responses", [])
            if not isinstance(responses, list) or not responses:
                if args.missing_logprob == "error":
                    raise ValueError(f"No responses found at line={line_no}, id={item_id}")
                continue
            if args.response_index == -1:
                indices = list(range(len(responses)))
            else:
                if args.response_index >= len(responses):
                    if args.missing_logprob == "error":
                        raise IndexError(f"response_index={args.response_index} out of range for id={item_id}, responses={len(responses)}")
                    continue
                indices = [args.response_index]
            if args.verbose:
                print(f"[item] line={line_no}, id={item_id}, responses={len(responses)}, scoring={indices}", file=sys.stderr)
            for ri in indices:
                row = compute_for_response(scorer, item, responses[ri], responses, ri, args)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                n_written += 1
    if args.verbose:
        print(f"Done. Wrote {n_written} rows to {args.output_file}", file=sys.stderr)


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

    _parser = _ap.ArgumentParser(description="cot_uq_mi / llama / gsm8k / reasoning_augmented (standalone)")
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
    if "cot_uq_mi" not in {"max_seq_prob", "mean_tok_ent", "max_tok_prob", "ppl"}:
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
    if "cot_uq_mi" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "cot_uq_mi", "family": "llama",
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

