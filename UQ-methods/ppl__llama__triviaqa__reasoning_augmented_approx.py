#!/usr/bin/env python3
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


def clamp_prob(x: float) -> float:
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


def entropy_values_from_raw(raw: Any) -> List[float]:
    
    vals = list_floats(raw)
    out = []
    for v in vals:
        out.append(float(-v if v <= 0 else v))
    return out


def extract_token_stats(
    response: Dict[str, Any],
    scope: str,
    entropy_key: str,
) -> Tuple[List[float], List[float], List[Dict[str, Any]]]:
    token_logprobs: List[float] = []
    token_entropies: List[float] = []
    unit_debug: List[Dict[str, Any]] = []

    def add_unit(name: str, unit: Dict[str, Any], text: str) -> None:
        pi = get_prob_info(unit)

        lp_raw = None
        for k in ["token logprob", "token_logprob", "chosen_logprob", "logprob"]:
            if k in pi:
                lp_raw = pi[k]
                break
        lps = list_floats(lp_raw)

        ent_raw = None
        if entropy_key in pi:
            ent_raw = pi[entropy_key]
        else:
            alt = entropy_key.replace(" ", "_")
            if alt in pi:
                ent_raw = pi[alt]
            elif "entropy" in pi:
                ent_raw = pi["entropy"]
            elif "neg_entropy" in pi:
                ent_raw = pi["neg_entropy"]
            elif "neg entropy" in pi:
                ent_raw = pi["neg entropy"]

        ents = entropy_values_from_raw(ent_raw)

        token_logprobs.extend(lps)
        token_entropies.extend(ents)

        unit_debug.append({
            "unit_name": name,
            "text": text,
            "num_token_logprobs": len(lps),
            "seq_logprob": float(sum(lps)) if lps else None,
            "avg_logprob": float(sum(lps) / len(lps)) if lps else None,
            "num_token_entropies": len(ents),
            "mean_token_entropy": float(sum(ents) / len(ents)) if ents else None,
        })

    if scope in {"all", "reasoning_augmented"}:
        steps = response.get("step-wise info", response.get("step_wise_info", []))
        if isinstance(steps, list):
            for i, step in enumerate(steps):
                if isinstance(step, dict):
                    add_unit(f"step_{i}", step, str(step.get("text", "")))

    if scope in {"all", "answer"}:
        answer_info = response.get("answer info", response.get("answer_info", None))
        if isinstance(answer_info, dict):
            add_unit("answer", answer_info, str(answer_info.get("text", "")))

    return token_logprobs, token_entropies, unit_debug


def compute_simple_metrics(token_logprobs: List[float], token_entropies: List[float]) -> Dict[str, Any]:
    if not token_logprobs:
        return {
            "maximum_sequence_probability": None,
            "maximum_token_probability": None,
            "perplexity": None,
            "mean_token_entropy": None,
            "token_entropy": token_entropies,
            "num_answer_tokens": 0,
            "avg_logprob_answer_only": None,
            "seq_logprob_answer_only": None,
        }

    seq_logprob = float(sum(token_logprobs))
    avg_logprob = float(seq_logprob / len(token_logprobs))

    token_probs = [clamp_prob(math.exp(min(0.0, lp))) for lp in token_logprobs]

    
    try:
        seq_prob = float(math.exp(seq_logprob))
    except OverflowError:
        seq_prob = 0.0
    seq_prob = clamp_prob(seq_prob)

    max_token_prob = max(token_probs) if token_probs else None
    perplexity = float(math.exp(-avg_logprob))

    mean_token_entropy = (
        float(sum(token_entropies) / len(token_entropies))
        if token_entropies
        else None
    )

    return {
        "maximum_sequence_probability": seq_prob,
        "maximum_token_probability": max_token_prob,
        "perplexity": perplexity,
        "mean_token_entropy": mean_token_entropy,
        "token_entropy": token_entropies,
        "num_answer_tokens": len(token_logprobs),
        "avg_logprob_answer_only": avg_logprob,
        "seq_logprob_answer_only": seq_logprob,
    }





class PMICalculator:
    def __init__(self, model_path: str, dtype: str = "auto", device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
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

    def token_logprobs_for_suffix(self, prefix_ids: List[int], suffix_ids: List[int]) -> List[float]:
        import torch

        if not suffix_ids:
            return []

        if not prefix_ids and self.tokenizer.bos_token_id is not None:
            prefix_ids = [self.tokenizer.bos_token_id]

        full_ids = prefix_ids + suffix_ids
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.model.device)

        with torch.inference_mode():
            logits = self.model(input_ids=input_ids).logits[:, :-1, :].float()
            logprobs = torch.log_softmax(logits, dim=-1)

        result: List[float] = []
        start = len(prefix_ids)

        for pos, tok_id in enumerate(suffix_ids):
            lp_pos = start + pos - 1
            if 0 <= lp_pos < logprobs.shape[1]:
                result.append(float(logprobs[0, lp_pos, tok_id].item()))

        return result

    def compute_pmi(self, context_text: str, target_text: str) -> Tuple[List[float], Optional[float]]:
        if not context_text.strip() or not target_text.strip():
            return [], None

        prefix_ids = self.tokenizer(context_text, add_special_tokens=False)["input_ids"]
        suffix_ids = self.tokenizer(target_text, add_special_tokens=False)["input_ids"]

        cond_lps = self.token_logprobs_for_suffix(prefix_ids, suffix_ids)
        base_lps = self.token_logprobs_for_suffix([], suffix_ids)

        if not cond_lps or not base_lps or len(cond_lps) != len(base_lps):
            return [], None

        pmi = [float(c - b) for c, b in zip(cond_lps, base_lps)]
        mean_pmi = float(sum(pmi) / len(pmi)) if pmi else None
        return pmi, mean_pmi


def build_pmi_context(question: str, template: str) -> str:
    if not question.strip():
        return ""
    return template.replace("{question}", question)





def build_prediction_text(response: Dict[str, Any], scope: str, prediction_format: str) -> str:
    reason = get_llm_reason(response)
    answer = get_llm_answer(response)
    extracted = primary_prediction(response)

    if prediction_format == "extracted_answer":
        return extracted

    if prediction_format == "raw":
        if scope == "answer":
            return answer
        if scope == "reasoning_augmented":
            return reason
        if reason and answer:
            return reason.rstrip() + "\nFinal Answer: " + answer
        return reason or answer

    
    if scope == "answer":
        return extracted
    if scope == "reasoning_augmented":
        return reason
    if reason and extracted:
        return reason.rstrip() + "\nFinal Answer: " + extracted
    return reason or extracted





def process_item(
    record: Dict[str, Any],
    line_idx: int,
    args: argparse.Namespace,
    pmi_calc: Optional[PMICalculator],
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

    token_logprobs, token_entropies, unit_debug = extract_token_stats(
        response=target_resp,
        scope=args.scope,
        entropy_key=args.entropy_key,
    )

    metrics = compute_simple_metrics(token_logprobs, token_entropies)

    prediction = build_prediction_text(
        response=target_resp,
        scope=args.scope,
        prediction_format=args.prediction_format,
    )

    target_answer = primary_prediction(target_resp)

    pmi_per_token: List[float] = []
    mean_pmi: Optional[float] = None

    if pmi_calc is not None:
        context_text = build_pmi_context(question, args.pmi_context_template)
        pmi_target = prediction
        if args.pmi_target == "answer":
            pmi_target = target_answer
        elif args.pmi_target == "all":
            pmi_target = prediction
        pmi_per_token, mean_pmi = pmi_calc.compute_pmi(context_text, pmi_target)

    out: Dict[str, Any] = {
        "id": rid,
        "correct": correctness.get("correct"),

        "maximum_sequence_probability": metrics["maximum_sequence_probability"],
        "maximum_token_probability": metrics["maximum_token_probability"],
        "perplexity": metrics["perplexity"],
        "mean_token_entropy": metrics["mean_token_entropy"],
        "token_entropy": metrics["token_entropy"],
        "num_answer_tokens": metrics["num_answer_tokens"],
        "avg_logprob_answer_only": metrics["avg_logprob_answer_only"],
        "seq_logprob_answer_only": metrics["seq_logprob_answer_only"],
        "pointwise_mutual_information": pmi_per_token,
        "mean_pointwise_mutual_information": mean_pmi,

        
        "MaximumSequenceProbability": metrics["maximum_sequence_probability"],
        "MaximumTokenProbability": metrics["maximum_token_probability"],
        "Perplexity": metrics["perplexity"],
        "MeanTokenEntropy": metrics["mean_token_entropy"],
        "TokenEntropy": metrics["token_entropy"],
        "PointwiseMutualInformation": pmi_per_token,
        "MeanPointwiseMutualInformation": mean_pmi,

        "target_response_index": target_idx,
        "target_answer": target_answer,
        "matched_prediction": correctness.get("matched_prediction"),
        "matched_gold": correctness.get("matched_gold"),

        "meta": {
            "method": "simple_uq_reasoning",
            "dataset": "trivialqa_reasoning_sampling",
            "correctness_source": "input_correct_answer_value_aliases",
            "correctness_rule": "trivialqa_normalized_alias_match",
            "match_mode": args.match_mode,
            "target_mode": args.target_mode,
            "scope": args.scope,
            "entropy_key": args.entropy_key,
            "pmi_enabled": pmi_calc is not None,
            "pmi_target": args.pmi_target if pmi_calc is not None else None,
            "prediction_format": args.prediction_format,
        },
    }

    if args.keep_question:
        out["question"] = question
    if args.keep_prediction:
        out["prediction"] = prediction
    if args.keep_reasoning:
        out["target_reasoning"] = get_llm_reason(target_resp)
    if args.keep_gold:
        out["gold_answers"] = correctness.get("gold_answers", [])
    if args.keep_candidates:
        out["prediction_candidates"] = correctness.get("prediction_candidates", [])
    if args.keep_units_debug:
        out["units_debug"] = unit_debug
    if args.keep_token_logprobs:
        out["token_logprobs"] = token_logprobs

    return out





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute simple UQ metrics for TrivialQA reasoning-sampling data."
    )

    parser.add_argument("--input-file", required=True, help="Input TrivialQA reasoning JSONL/JSON file")
    parser.add_argument("--output-file", required=True, help="Output JSONL file")

    parser.add_argument(
        "--target-mode",
        choices=["first", "majority_answer"],
        default="first",
        help="Target response to score.",
    )
    parser.add_argument(
        "--match-mode",
        choices=["exact", "exact_or_contains"],
        default="exact",
        help="TrivialQA correctness match mode.",
    )
    parser.add_argument(
        "--scope",
        choices=["answer", "reasoning_augmented", "all"],
        default="all",
        help="Which part to compute token metrics on. Default all.",
    )
    parser.add_argument(
        "--entropy-key",
        default="neg entropy",
        help="Entropy field in prob info. Usually 'neg entropy' or 'neg entropy nor'.",
    )
    parser.add_argument(
        "--prediction-format",
        choices=["raw", "normalized_short_answer", "extracted_answer"],
        default="raw",
        help="Format for kept prediction and PMI target text.",
    )

    
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional local model path for PMI / Mean PMI. If omitted, PMI outputs [] / null.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--hf-endpoint", default=None)
    parser.add_argument(
        "--pmi-context-template",
        default="Question:\n{question}\n\nAnswer:\n",
        help="Context template for PMI conditional probability.",
    )
    parser.add_argument(
        "--pmi-target",
        choices=["answer", "all"],
        default="answer",
        help="PMI target text. Default answer is more suitable for TrivialQA.",
    )

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--keep-question", action="store_true")
    parser.add_argument("--keep-prediction", action="store_true")
    parser.add_argument("--keep-reasoning", action="store_true")
    parser.add_argument("--keep-gold", action="store_true")
    parser.add_argument("--keep-candidates", action="store_true")
    parser.add_argument("--keep-units-debug", action="store_true")
    parser.add_argument("--keep-token-logprobs", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    safe_print(f"INPUT_FILE = {args.input_file}")
    safe_print(f"OUTPUT_FILE = {args.output_file}")
    safe_print("Correctness source: record['correct answer']['value'] + aliases")
    safe_print(f"TrivialQA match mode = {args.match_mode}")
    safe_print(f"Scope = {args.scope}")
    safe_print(f"Entropy key = {args.entropy_key}")
    safe_print(f"PMI model = {args.model_path if args.model_path else 'disabled'}")
    safe_print(f"PMI target = {args.pmi_target if args.model_path else 'disabled'}")

    records = load_records(args.input_file)
    safe_print(f"Loaded records: {len(records)}")

    pmi_calc = None
    if args.model_path:
        safe_print("Loading local model for PMI...")
        pmi_calc = PMICalculator(
            model_path=args.model_path,
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
                    pmi_calc=pmi_calc,
                )
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                processed += 1

                if args.verbose:
                    safe_print(
                        f"done: {rid} | line={line_idx} | "
                        f"correct={result.get('correct')} | "
                        f"msp={result.get('maximum_sequence_probability')} | "
                        f"mtp={result.get('maximum_token_probability')} | "
                        f"ppl={result.get('perplexity')} | "
                        f"mte={result.get('mean_token_entropy')} | "
                        f"mean_pmi={result.get('mean_pointwise_mutual_information')} | "
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

    _parser = _ap.ArgumentParser(description="ppl / llama / triviaqa / reasoning_augmented (standalone)")
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
    if "ppl" not in {"max_seq_prob", "mean_tok_ent", "max_tok_prob", "ppl"}:
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
    if "ppl" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "ppl", "family": "llama",
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

