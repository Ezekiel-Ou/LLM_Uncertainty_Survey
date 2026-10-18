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
import sys
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from math_verify import parse, verify


print_lock = threading.Lock()


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
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
        except Exception as e:
            safe_print(f"[warn] skip bad json line {line_no}: {e}", file=sys.stderr)
    return records


def write_jsonl(path: str, records: Iterable[Dict[str, Any]], append: bool = False) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
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


def get_record_id(record: Dict[str, Any], line_idx: int = 0) -> str:
    for key in ["id", "question_id", "qid"]:
        if key in record and record[key] is not None:
            return str(record[key])
    return f"line_{line_idx}"


def get_question(record: Dict[str, Any]) -> str:
    for key in ["question", "problem", "prompt", "input"]:
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def get_correct_answer(record: Dict[str, Any]) -> str:
    for key in ["correct answer", "correct_answer", "answer", "gt", "ground_truth"]:
        val = record.get(key)
        if val is None:
            continue
        if isinstance(val, dict):
            if val.get("value") is not None:
                return str(val["value"])
            aliases = val.get("aliases")
            if isinstance(aliases, list) and aliases:
                return str(aliases[0])
        return str(val)
    return ""


def get_responses(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    responses = record.get("responses", [])
    if isinstance(responses, list):
        return [r for r in responses if isinstance(r, dict)]
    return []


def get_answer(response: Dict[str, Any]) -> str:
    for key in ["llm answer", "llm_answer", "answer", "text", "raw_text"]:
        val = response.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def get_reason(response: Dict[str, Any]) -> str:
    for key in ["llm reason", "llm_reason", "reason", "rationale", "full_text"]:
        val = response.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def get_prob_info_token_logprobs(container: Any) -> List[float]:
    if not isinstance(container, dict):
        return []
    prob_info = container.get("prob info") or container.get("prob_info") or {}
    if not isinstance(prob_info, dict):
        return []
    vals = prob_info.get("token logprob")
    if vals is None:
        vals = prob_info.get("token_logprob")
    if vals is None:
        vals = prob_info.get("chosen_logprob")
    if not isinstance(vals, list):
        return []
    out: List[float] = []
    for x in vals:
        try:
            v = float(x)
            if math.isfinite(v):
                out.append(v)
        except Exception:
            continue
    return out


def aggregate_logprob(token_logprobs: List[float], mode: str) -> Optional[float]:
    if not token_logprobs:
        return None
    if mode == "sum":
        return float(sum(token_logprobs))
    if mode == "mean":
        return float(sum(token_logprobs) / len(token_logprobs))
    raise ValueError(f"Unsupported unit logprob mode: {mode}")


def normalize_text_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def extract_units(response: Dict[str, Any], unit_logprob_mode: str = "mean", include_answer_unit: bool = True) -> List[Dict[str, Any]]:
    
    units: List[Dict[str, Any]] = []

    step_infos = response.get("step-wise info")
    if step_infos is None:
        step_infos = response.get("step_wise_info")
    if isinstance(step_infos, list):
        for i, step in enumerate(step_infos):
            if not isinstance(step, dict):
                continue
            text = str(step.get("text", "") or "").strip()
            if not text:
                continue
            token_lps = get_prob_info_token_logprobs(step)
            unit_lp = aggregate_logprob(token_lps, unit_logprob_mode)
            units.append(
                {
                    "unit_index": len(units),
                    "unit_type": "reason_step",
                    "source_index": i,
                    "text": text,
                    "token_logprobs": token_lps,
                    "unit_logprob": unit_lp,
                    "num_tokens": len(token_lps),
                }
            )

    if include_answer_unit:
        answer_info = response.get("answer info")
        if answer_info is None:
            answer_info = response.get("answer_info")
        answer_text = ""
        if isinstance(answer_info, dict):
            answer_text = str(answer_info.get("text", "") or "").strip()
        if not answer_text:
            answer_text = get_answer(response)
        if answer_text:
            token_lps = get_prob_info_token_logprobs(answer_info) if isinstance(answer_info, dict) else []
            unit_lp = aggregate_logprob(token_lps, unit_logprob_mode)
            units.append(
                {
                    "unit_index": len(units),
                    "unit_type": "final_answer",
                    "source_index": None,
                    "text": answer_text,
                    "token_logprobs": token_lps,
                    "unit_logprob": unit_lp,
                    "num_tokens": len(token_lps),
                }
            )

    
    if not units:
        text = get_reason(response)
        answer = get_answer(response)
        full = normalize_text_space((text + "\nFinal Answer: " + answer).strip())
        if full:
            units.append(
                {
                    "unit_index": 0,
                    "unit_type": "full_response_fallback",
                    "source_index": None,
                    "text": full,
                    "token_logprobs": [],
                    "unit_logprob": None,
                    "num_tokens": 0,
                }
            )
    return units


def render_units(units: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    step_no = 1
    for u in units:
        text = str(u.get("text", "")).strip()
        if not text:
            continue
        if u.get("unit_type") == "final_answer":
            lines.append(f"Final Answer: {text}")
        elif re.match(r"(?i)^step\s*\d+", text):
            lines.append(text)
        else:
            lines.append(f"Step {step_no}: {text}")
            step_no += 1
    return "\n".join(lines).strip()


class LocalJudge:
    def __init__(self, model_path: str, model_family: str, dtype: str = "auto", device: str = "cuda") -> None:
        self.model_path = model_path
        self.model_family = model_family.lower()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        torch_dtype = self._resolve_dtype(dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() and device == "cuda" else None,
            trust_remote_code=True,
        )
        if not (torch.cuda.is_available() and device == "cuda"):
            self.model.to(torch.device("cpu"))
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.A_ID = self._get_single_token_id("A")
        self.B_ID = self._get_single_token_id("B")
        self.C_ID = self._get_single_token_id("C")
        if self.A_ID is None or self.B_ID is None or self.C_ID is None:
            raise ValueError("Cannot map A/B/C to single tokens for this judge tokenizer.")

    @staticmethod
    def _resolve_dtype(dtype: str):
        if dtype == "auto":
            if torch.cuda.is_available():
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.float32
        if dtype in {"float16", "fp16", "half"}:
            return torch.float16
        if dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if dtype in {"float32", "fp32"}:
            return torch.float32
        raise ValueError(f"Unsupported dtype: {dtype}")

    def _get_single_token_id(self, token_str: str) -> Optional[int]:
        candidates = [token_str, " " + token_str, f"{token_str}.", f" {token_str}."]
        for cand in candidates:
            ids = self.tokenizer.encode(cand, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        return None

    def build_prompt(self, question: str, proposed_reasoning: str) -> str:
        q_block = question.strip() if question.strip() else "[Question text is not available in the input file.]"
        system_msg = (
            "You are a careful judge for math reasoning tasks. "
            "Your job is to decide whether the proposed reasoning and final answer are correct. "
            "Do not solve from scratch unless needed. Choose one label only."
        )
        user_msg = (
            f"Problem:\n{q_block}\n\n"
            f"Proposed reasoning and answer:\n{proposed_reasoning}\n\n"
            "Question: Is the proposed reasoning and final answer correct?\n"
            "A. Correct\n"
            "B. Incorrect\n"
            "C. Not sure\n\n"
            "Answer with only one letter:"
        )
        if self.model_family == "qwen":
            try:
                return self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        elif self.model_family == "llama":
            try:
                return self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        return f"SYSTEM: {system_msg}\n\nUSER: {user_msg}\n\nASSISTANT:"

    @torch.no_grad()
    def score(self, question: str, proposed_reasoning: str, use_p_a_only: bool = False) -> Dict[str, Any]:
        prompt = self.build_prompt(question, proposed_reasoning)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
        logits = self.model(**inputs).logits[0, -1, :].float()
        abc_logits = torch.tensor(
            [logits[self.A_ID].item(), logits[self.B_ID].item(), logits[self.C_ID].item()],
            dtype=torch.float32,
            device=logits.device,
        )
        probs = F.softmax(abc_logits, dim=-1)
        p_a = float(probs[0].item())
        p_b = float(probs[1].item())
        p_c = float(probs[2].item())
        score = p_a if use_p_a_only else p_a + 0.5 * p_c
        return {
            "p_a": p_a,
            "p_b": p_b,
            "p_c": p_c,
            "score": float(score),
            "prompt_len": int(inputs["input_ids"].shape[1]),
        }


def normalize_importance(raw_importances: List[float], min_uniform_mass: float = 0.5) -> List[float]:
    
    n = len(raw_importances)
    if n <= 0:
        return []
    s = sum(max(0.0, x) for x in raw_importances)
    if s <= 0:
        imp = [1.0 / n] * n
    else:
        imp = [max(0.0, x) / s for x in raw_importances]
    
    uniform = 1.0 / n
    weights = [min_uniform_mass * uniform + (1.0 - min_uniform_mass) * x for x in imp]
    
    z = sum(weights)
    if z > 0:
        weights = [w / z for w in weights]
    return weights


def choose_target_response_index(record: Dict[str, Any], target_mode: str) -> int:
    responses = get_responses(record)
    if not responses:
        return -1
    if target_mode == "first":
        return 0
    if target_mode == "majority_correctness_free":
        
        reps: List[str] = []
        clusters: List[List[int]] = []
        for i, r in enumerate(responses):
            ans = get_answer(r)
            assigned = False
            for cid, rep in enumerate(reps):
                if mathv_correct(rep, ans):
                    clusters[cid].append(i)
                    assigned = True
                    break
            if not assigned:
                reps.append(ans)
                clusters.append([i])
        if not clusters:
            return 0
        largest = max(clusters, key=len)
        return largest[0]
    raise ValueError(f"Unsupported target mode: {target_mode}")


def compute_mars_for_record(record: Dict[str, Any], args: argparse.Namespace, judge: LocalJudge, line_idx: int = 0) -> Dict[str, Any]:
    rid = get_record_id(record, line_idx)
    question = get_question(record)
    correct_answer = get_correct_answer(record)
    responses = get_responses(record)
    if not responses:
        return {
            "id": rid,
            "correct_answer": correct_answer,
            "correct": 0,
            "error": "no_responses",
            "mars_score": None,
            "mars_uncertainty": None,
            "uncertainty": None,
        }

    target_idx = choose_target_response_index(record, args.target_mode)
    if target_idx < 0 or target_idx >= len(responses):
        target_idx = 0
    target = responses[target_idx]
    target_answer = get_answer(target)
    correct = int(mathv_correct(correct_answer, target_answer)) if correct_answer else 0

    units = extract_units(
        target,
        unit_logprob_mode=args.unit_logprob_mode,
        include_answer_unit=not args.exclude_answer_unit,
    )
    if args.max_units is not None and args.max_units > 0 and len(units) > args.max_units:
        units = units[: args.max_units]

    valid_units = [u for u in units if u.get("unit_logprob") is not None]
    if args.missing_logprob == "error" and len(valid_units) != len(units):
        return {
            "id": rid,
            "correct_answer": correct_answer,
            "target_response_index": target_idx,
            "target_answer": target_answer,
            "correct": correct,
            "mars_score": None,
            "mars_uncertainty": None,
            "uncertainty": None,
            "error": f"missing_unit_logprob: {len(units)-len(valid_units)} units missing logprob",
            "num_units": len(units),
            "num_units_with_logprob": len(valid_units),
        }
    if args.missing_logprob == "drop":
        units = valid_units

    if not units:
        return {
            "id": rid,
            "correct_answer": correct_answer,
            "target_response_index": target_idx,
            "target_answer": target_answer,
            "correct": correct,
            "mars_score": None,
            "mars_uncertainty": None,
            "uncertainty": None,
            "error": "no_valid_units",
        }

    full_text = render_units(units)
    orig_score = judge.score(question, full_text, use_p_a_only=args.use_p_a_only)
    p_orig = float(orig_score["score"])
    judge_calls = 1

    raw_importances: List[float] = []
    unit_debug: List[Dict[str, Any]] = []

    for i, u in enumerate(units):
        masked_units = [dict(x) for x in units]
        masked_units[i]["text"] = args.mask_text
        masked_text = render_units(masked_units)
        masked_score = judge.score(question, masked_text, use_p_a_only=args.use_p_a_only)
        judge_calls += 1
        p_mask = float(masked_score["score"])
        importance = max(0.0, p_orig - p_mask)
        if importance < args.min_importance:
            importance = args.min_importance
        raw_importances.append(importance)
        unit_debug.append(
            {
                "unit_index": i,
                "unit_type": u.get("unit_type"),
                "text": u.get("text"),
                "unit_logprob": u.get("unit_logprob"),
                "num_tokens": u.get("num_tokens"),
                "judge_original_score": p_orig,
                "judge_masked_score": p_mask,
                "raw_importance": importance,
                "judge_probs_original": orig_score,
                "judge_probs_masked": masked_score,
            }
        )

    weights = normalize_importance(raw_importances, min_uniform_mass=args.min_uniform_weight_mass)
    mars_score = 0.0
    used = 0
    for i, (u, w) in enumerate(zip(units, weights)):
        lp = u.get("unit_logprob")
        if lp is None:
            if args.missing_logprob == "zero":
                lp = 0.0
            else:
                continue
        mars_score += float(w) * float(lp)
        used += 1
        unit_debug[i]["weight"] = float(w)

    mars_uncertainty = float(-mars_score)
    confidence_proxy = math.exp(mars_score) if mars_score <= 0 and mars_score > -745 else None

    result: Dict[str, Any] = {
        "id": rid,
        "correct_answer": correct_answer,
        "target_response_index": target_idx,
        "target_answer": target_answer,
        "correct": correct,
        "mars_score": float(mars_score),
        "mars_uncertainty": mars_uncertainty,
        "uncertainty": mars_uncertainty,
        "mars_confidence_proxy_exp_score": confidence_proxy,
        "judge_original_score": p_orig,
        "num_responses": len(responses),
        "num_units": len(units),
        "num_units_used": used,
        "judge_calls": judge_calls,
        "meta": {
            "method": "MARS-step-level-ma[REMOVED_CREDENTIAL]",
            "dataset_format": "reasoning_sampling_gsm8k_style",
            "judge_model": args.model_path,
            "model_family": args.model_family,
            "target_mode": args.target_mode,
            "mask_text": args.mask_text,
            "unit_logprob_mode": args.unit_logprob_mode,
            "exclude_answer_unit": args.exclude_answer_unit,
            "correctness_checker": "math_verify",
            "correctness_leakage_into_judge": False,
        },
    }
    if args.keep_question:
        result["question"] = question
    if args.keep_target_reasoning:
        result["target_reasoning"] = get_reason(target)
        result["target_units_text"] = full_text
    if args.keep_unit_debug:
        result["unit_debug"] = unit_debug
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute MARS uncertainty for reasoning-sampling GSM8K-style records.")
    parser.add_argument("--input-file", required=True, help="Input JSONL/JSON file with reasoning responses")
    parser.add_argument("--output-file", required=True, help="Output JSONL file")
    parser.add_argument("--model-path", required=True, help="Local judge model path")
    parser.add_argument("--model-family", choices=["qwen", "llama", "plain"], default="qwen")
    parser.add_argument("--hf-endpoint", default=None, help="Optional HF endpoint mirror, e.g. https://hf-mirror.com")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--target-mode", choices=["first", "majority_correctness_free"], default="first")
    parser.add_argument("--mask-text", default="[MASK]")
    parser.add_argument("--max-units", type=int, default=12, help="Max reasoning units to mask; 0 or negative means no cap")
    parser.add_argument("--unit-logprob-mode", choices=["mean", "sum"], default="mean", help="Aggregate token logprobs inside a reasoning unit")
    parser.add_argument("--missing-logprob", choices=["error", "drop", "zero"], default="error")
    parser.add_argument("--exclude-answer-unit", action="store_true", help="Use only reasoning steps, not final answer unit")
    parser.add_argument("--use-p-a-only", action="store_true", help="Judge score = P(A), otherwise P(A)+0.5P(C)")
    parser.add_argument("--min-importance", type=float, default=1e-8)
    parser.add_argument("--min-uniform-weight-mass", type=float, default=0.5, help="Blend with uniform weights; 0.5 matches prior approximation style")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-question", action="store_true")
    parser.add_argument("--keep-target-reasoning", action="store_true")
    parser.add_argument("--keep-unit-debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if args.max_units is not None and args.max_units <= 0:
        args.max_units = None

    safe_print(f"INPUT_FILE = {args.input_file}")
    safe_print(f"OUTPUT_FILE = {args.output_file}")
    safe_print("Correctness source: record['correct answer'] via math_verify")
    safe_print(f"Judge model: {args.model_path}")
    safe_print(f"Judge family: {args.model_family}")
    safe_print("MARS unit: reasoning step + final answer unit")
    safe_print(f"Unit logprob mode: {args.unit_logprob_mode}")
    safe_print(f"Missing logprob mode: {args.missing_logprob}")

    if os.path.exists(args.output_file) and args.overwrite:
        os.remove(args.output_file)

    done = load_done_ids(args.output_file) if not args.overwrite else set()
    records = load_records(args.input_file)
    if args.limit is not None:
        records = records[: args.limit]

    safe_print(f"Records loaded: {len(records)}")
    safe_print(f"Existing records skipped: {len(done)}")
    safe_print("Loading local judge model...")
    judge = LocalJudge(args.model_path, args.model_family, dtype=args.dtype, device=args.device)
    safe_print("Judge model loaded.")

    processed = 0
    for line_idx, record in enumerate(records, start=1):
        rid = get_record_id(record, line_idx)
        if str(rid) in done:
            continue
        try:
            result = compute_mars_for_record(record, args, judge, line_idx=line_idx)
            write_jsonl(args.output_file, [result], append=True)
            processed += 1
            if args.verbose:
                unc = result.get("mars_uncertainty")
                unc_str = "None" if unc is None else f"{float(unc):.6f}"
                safe_print(
                    f"done: {rid} | line={line_idx} | correct={result.get('correct')} | "
                    f"units={result.get('num_units')} | judge_calls={result.get('judge_calls')} | unc={unc_str}"
                )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            err = {"id": rid, "error": str(e), "mars_score": None, "mars_uncertainty": None, "uncertainty": None}
            write_jsonl(args.output_file, [err], append=True)
            safe_print(f"error: {rid} | line={line_idx} | {e}", file=sys.stderr)

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

    _parser = _ap.ArgumentParser(description="mars / llama / gsm8k / reasoning_augmented (standalone)")
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
    if "mars" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "mars", "family": "llama",
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

