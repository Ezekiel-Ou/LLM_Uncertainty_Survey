#!/usr/bin/env python3
# Original paper/author repository: https://github.com/LongchaoDa/LLM-Topology

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
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  
    linear_sum_assignment = None

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
QUESTION_KEYS = ["question", "prompt", "input", "problem", "query", "text"]
GOLD_KEYS = [
    "correct answer", "correct_answer", "answer", "gold", "gold_answer",
    "target", "label", "gt", "ground_truth"
]
CORRECT_ID_KEYS = ["question_id", "id", "task_id", "qid", "problem_id", "idx"]





def read_jsonl(path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                print(f"[skip] empty line={line_no} in {path}", flush=True)
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield line_no, obj
                else:
                    print(f"[skip] non-dict json line={line_no} in {path}", flush=True)
            except json.JSONDecodeError as e:
                print(
                    f"[skip] bad json line={line_no} in {path}: {e}; "
                    f"prefix={raw[:160]!r}",
                    flush=True,
                )
                continue


def get_first(obj: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return default


def normalize_space(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def get_item_id(item: Dict[str, Any], fallback: int) -> str:
    return str(get_first(item, ID_KEYS, fallback))


def get_question(item: Dict[str, Any]) -> str:
    return str(get_first(item, QUESTION_KEYS, "") or "")





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


def extract_gsm8k_final_answer(text: Any) -> str:
    
    text = "" if text is None else str(text).strip()
    if not text:
        return ""

    
    m = re.search(r"####\s*\$?(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")

    
    m = re.search(r"(?i)(?:final\s+answer|answer)\s*:\s*\$?(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")

    boxed = extract_boxed(text)
    if boxed:
        nums = re.findall(r"-?\d+(?:\.\d+)?", boxed.replace(",", ""))
        if nums:
            return nums[-1]
        return boxed.strip()

    
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if nums:
        return nums[-1]
    return text


def normalize_gt_field(gt: Any) -> str:
    if isinstance(gt, dict):
        if gt.get("value") is not None:
            return extract_gsm8k_final_answer(gt.get("value"))
        aliases = gt.get("aliases")
        if isinstance(aliases, list) and aliases:
            return extract_gsm8k_final_answer(aliases[0])
    return extract_gsm8k_final_answer(gt)


def get_gold(item: Dict[str, Any]) -> str:
    for k in GOLD_KEYS:
        if k in item and item[k] is not None:
            return normalize_gt_field(item[k])
    return ""


def mathv_correct(ref_text: Any, target_text: Any) -> bool:
    if mathv_parse is None or mathv_verify is None:
        raise ImportError(
            "math-verify is required for GSM8K correctness. Install it first. "
            f"Original import error: {repr(_MATH_VERIFY_IMPORT_ERROR)}"
        )
    try:
        return bool(mathv_verify(mathv_parse(str(ref_text)), mathv_parse(str(target_text))))
    except Exception:
        return False


def gsm8k_correct(ref_text: Any, target_text: Any) -> int:
    
    if mathv_correct(ref_text, target_text):
        return 1
    ref_ans = extract_gsm8k_final_answer(ref_text)
    pred_ans = extract_gsm8k_final_answer(target_text)
    if ref_ans and pred_ans and ref_ans == pred_ans:
        return 1
    if ref_ans and pred_ans and mathv_correct(ref_ans, pred_ans):
        return 1
    return 0





def extract_text_field(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        for key in ["text", "raw_text", "answer", "response", "completion", "output", "prediction"]:
            if x.get(key) is not None:
                return str(x[key]).strip()
    return str(x).strip()


def get_prediction_text(response: Dict[str, Any]) -> str:
    for key in ["llm answer", "llm_answer", "answer", "final_answer", "prediction", "output", "completion", "response"]:
        if key in response and response[key] is not None:
            return str(response[key]).strip()
    if "answer info" in response and isinstance(response["answer info"], dict):
        txt = response["answer info"].get("text")
        if txt:
            return str(txt).strip()
    return str(response.get("llm reason", response.get("reason", response.get("text", ""))) or "").strip()


def get_reason_text(response: Dict[str, Any]) -> str:
    for key in [
        "llm reason", "llm_reason", "reason", "reasoning_augmented", "rationale",
        "chain_of_thought", "cot", "text", "response", "output"
    ]:
        if key in response and response[key] is not None:
            return str(response[key]).strip()
    return ""


def get_responses(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    
    val = item.get("responses")
    if isinstance(val, list) and val:
        return [x for x in val if isinstance(x, dict)]

    
    for key in ["samples", "sampled_answers", "sample_outputs", "sampled"]:
        val = item.get(key)
        if isinstance(val, list) and val:
            outs: List[Dict[str, Any]] = []
            for s in val:
                txt = extract_text_field(s)
                if txt:
                    outs.append({"llm reason": txt, "llm answer": txt, "source": key})
            if outs:
                return outs

    
    if isinstance(item.get("greedy"), dict):
        txt = extract_text_field(item["greedy"])
        if txt:
            return [{"llm reason": txt, "llm answer": txt, "source": "greedy"}]

    for key in ["greedy_answer", "prediction", "answer", "output", "response"]:
        if item.get(key) is not None:
            txt = str(item[key]).strip()
            if txt:
                return [{"llm reason": txt, "llm answer": txt, "source": key}]

    return []





def split_reason_steps_from_text(text: str) -> List[str]:
    text = str(text or "")
    text = re.sub(r"Final\s+Answer\s*:.*$", "", text, flags=re.I | re.S).strip()
    text = re.sub(r"####\s*.*$", "", text, flags=re.S).strip()
    if not text:
        return []

    pattern = re.compile(r"(?:^|\n)\s*Step\s*\d+\s*[:\.)-]\s*", flags=re.I)
    matches = list(pattern.finditer(text))
    if matches:
        steps: List[str] = []
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            s = normalize_space(text[start:end])
            if s:
                steps.append(s)
        if steps:
            return steps

    lines = [normalize_space(x) for x in text.splitlines() if normalize_space(x)]
    if len(lines) >= 2:
        return lines

    parts = re.split(r"(?<=[.!?。！？])\s+", normalize_space(text))
    return [p for p in parts if p]


def extract_reason_units(response: Dict[str, Any], include_answer_unit: bool = False, max_units: int = 12) -> List[str]:
    units: List[str] = []

    step_info = response.get("step-wise info") or response.get("step_wise_info")
    if isinstance(step_info, list):
        for s in step_info:
            if isinstance(s, dict):
                txt = s.get("text") or s.get("step") or s.get("content")
            else:
                txt = str(s)
            txt = normalize_space(txt)
            if txt:
                units.append(txt)

    if not units:
        units = split_reason_steps_from_text(get_reason_text(response))

    if include_answer_unit:
        ans = normalize_space(get_prediction_text(response))
        if ans:
            units.append("Final answer: " + ans)

    cleaned: List[str] = []
    for u in units:
        u = normalize_space(u)
        if not u:
            continue
        if not cleaned or cleaned[-1] != u:
            cleaned.append(u)

    if max_units and max_units > 0:
        cleaned = cleaned[:max_units]
    return cleaned


def choose_response_indices(n: int, max_responses: int) -> List[int]:
    if n <= 0:
        return []
    idxs = list(range(n))
    if max_responses and max_responses > 0:
        idxs = idxs[:max_responses]
    return idxs





def get_correct_id(row: Dict[str, Any]) -> Optional[str]:
    value = get_first(row, CORRECT_ID_KEYS, None)
    if value is None:
        return None
    return str(value)


def load_correct_map(path: Optional[str]) -> Tuple[Dict[str, int], Dict[Tuple[str, int], int]]:
    q_map: Dict[str, int] = {}
    r_map: Dict[Tuple[str, int], int] = {}
    if not path:
        return q_map, r_map
    for _, row in read_jsonl(path):
        qid = get_correct_id(row)
        if qid is None or "correct" not in row:
            continue
        correct = int(row["correct"])
        ridx = row.get("response_index", row.get("sample_index", row.get("idx", None)))
        if ridx is None:
            q_map[qid] = correct
        else:
            try:
                r_map[(qid, int(ridx))] = correct
            except Exception:
                q_map[qid] = correct
    return q_map, r_map


def lookup_external_correct(
    item_id: str,
    item: Dict[str, Any],
    response_index: int,
    q_map: Dict[str, int],
    r_map: Dict[Tuple[str, int], int],
) -> Optional[int]:
    candidate_ids = [item_id]
    for k in ID_KEYS:
        if k in item and item[k] is not None:
            candidate_ids.append(str(item[k]))
    seen = set()
    candidate_ids = [x for x in candidate_ids if not (x in seen or seen.add(x))]
    for qid in candidate_ids:
        if (qid, response_index) in r_map:
            return int(r_map[(qid, response_index)])
    for qid in candidate_ids:
        if qid in q_map:
            return int(q_map[qid])
    return None


def get_correct_label(
    item_id: str,
    item: Dict[str, Any],
    response: Dict[str, Any],
    response_index: int,
    q_map: Dict[str, int],
    r_map: Dict[Tuple[str, int], int],
    missing_policy: str,
) -> Optional[int]:
    ext = lookup_external_correct(item_id, item, response_index, q_map, r_map)
    if ext is not None:
        return int(ext)

    gold = get_gold(item)
    pred = get_prediction_text(response)
    if gold:
        return int(gsm8k_correct(gold, pred))

    if missing_policy == "error":
        raise KeyError(f"Missing correctness and gold answer for item_id={item_id}, response_index={response_index}")
    if missing_policy == "zero":
        return 0
    return None





@dataclass
class NLIResult:
    entailment: float
    neutral: float
    contradiction: float


class NLIScorer:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_length: int = 256,
        batch_size: int = 16,
        entailment_label: Optional[str] = None,
        contradiction_label: Optional[str] = None,
        use_safetensors: Optional[bool] = None,
    ):
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.max_length = max_length
        self.batch_size = batch_size

        print(f"[info] loading NLI model: {model_path} on {self.device}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        kwargs: Dict[str, Any] = {"torch_dtype": torch.float32}
        if use_safetensors is not None:
            kwargs["use_safetensors"] = use_safetensors
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)
        self.model.to(self.device)
        self.model.float()
        self.model.eval()

        id2label = getattr(self.model.config, "id2label", {}) or {}
        norm = {int(k): str(v).lower() for k, v in id2label.items()}
        self.entailment_id = self._find_label_id(norm, entailment_label, ["entailment", "entails", "entail"])
        self.contradiction_id = self._find_label_id(norm, contradiction_label, ["contradiction", "contradict"])
        all_ids = list(range(int(self.model.config.num_labels)))
        self.neutral_id = next((i for i in all_ids if i not in {self.entailment_id, self.contradiction_id}), None)
        if self.entailment_id is None or self.contradiction_id is None:
            if int(self.model.config.num_labels) == 3:
                self.contradiction_id = 0
                self.neutral_id = 1
                self.entailment_id = 2
            else:
                raise ValueError(f"Cannot infer NLI labels from id2label={id2label}")

    @staticmethod
    def _find_label_id(norm: Dict[int, str], user_label: Optional[str], keywords: Sequence[str]) -> Optional[int]:
        if user_label is not None:
            user_label = user_label.lower()
            for i, label in norm.items():
                if label == user_label or user_label in label:
                    return i
        for i, label in norm.items():
            if any(k in label for k in keywords):
                return i
        return None

    @torch.no_grad()
    def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[NLIResult]:
        results: List[NLIResult] = []
        if not pairs:
            return results
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            premises = [p for p, _ in batch]
            hypotheses = [h for _, h in batch]
            enc = self.tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            self.model.float()
            logits = self.model(**enc).logits.float()
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            for row in probs:
                ent = float(row[self.entailment_id])
                con = float(row[self.contradiction_id])
                neu = float(row[self.neutral_id]) if self.neutral_id is not None else max(0.0, 1.0 - ent - con)
                results.append(NLIResult(entailment=ent, neutral=neu, contradiction=con))
        return results

    def symmetric_cost(self, a: str, b: str, neutral_cost: float = 0.5) -> float:
        a_norm = normalize_space(a).lower()
        b_norm = normalize_space(b).lower()
        if a_norm and a_norm == b_norm:
            return 0.0
        ab, ba = self.score_pairs([(a, b), (b, a)])
        mutual_entail = min(ab.entailment, ba.entailment)
        contradiction = max(ab.contradiction, ba.contradiction)
        cost = (1.0 - mutual_entail) * (1.0 - contradiction) + contradiction
        neutral_strength = max(ab.neutral, ba.neutral)
        cost = (1.0 - neutral_strength) * cost + neutral_strength * neutral_cost
        return float(max(0.0, min(1.0, cost)))





def chain_edit_distance_nli(
    chain_a: Sequence[str],
    chain_b: Sequence[str],
    scorer: NLIScorer,
    length_penalty_weight: float = 0.2,
    edit_penalty: float = 0.75,
) -> float:
    a = list(chain_a)
    b = list(chain_b)
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return float(edit_penalty)

    sub = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            sub[i, j] = scorer.symmetric_cost(a[i], b[j])

    if linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(sub)
        matched_cost = float(sub[row_ind, col_ind].sum())
        unmatched = abs(n - m)
        semantic_cost = (matched_cost + unmatched * edit_penalty) / max(n, m)
    else:
        dp = np.zeros((n + 1, m + 1), dtype=np.float32)
        for i in range(1, n + 1):
            dp[i, 0] = dp[i - 1, 0] + edit_penalty
        for j in range(1, m + 1):
            dp[0, j] = dp[0, j - 1] + edit_penalty
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dp[i, j] = min(
                    dp[i - 1, j] + edit_penalty,
                    dp[i, j - 1] + edit_penalty,
                    dp[i - 1, j - 1] + sub[i - 1, j - 1],
                )
        semantic_cost = float(dp[n, m] / max(n, m))

    structure_cost = abs(n - m) / max(n, m)
    total = (1.0 - length_penalty_weight) * semantic_cost + length_penalty_weight * structure_cost
    return float(max(0.0, min(1.0, total)))


def pairwise_chain_topouq(
    chains: Sequence[Sequence[str]],
    scorer: NLIScorer,
    length_penalty_weight: float = 0.2,
    edit_penalty: float = 0.75,
) -> Dict[str, Any]:
    k = len(chains)
    if k <= 1:
        return {
            "uncertainty": 0.0,
            "chain_topouq_nli_variance": 0.0,
            "chain_topouq_nli_mean_distance": 0.0,
            "chain_topouq_nli_max_distance": 0.0,
            "chain_topouq_nli_min_distance": 0.0,
            "distance_matrix": [[0.0]] if k == 1 else [],
        }

    D = np.zeros((k, k), dtype=np.float32)
    vals: List[float] = []
    for i in range(k):
        for j in range(i + 1, k):
            d = chain_edit_distance_nli(
                chains[i],
                chains[j],
                scorer=scorer,
                length_penalty_weight=length_penalty_weight,
                edit_penalty=edit_penalty,
            )
            D[i, j] = D[j, i] = d
            vals.append(d)

    arr = np.array(vals, dtype=np.float32)
    return {
        "uncertainty": float(np.var(arr)) if len(arr) else 0.0,
        "chain_topouq_nli_variance": float(np.var(arr)) if len(arr) else 0.0,
        "chain_topouq_nli_mean_distance": float(np.mean(arr)) if len(arr) else 0.0,
        "chain_topouq_nli_max_distance": float(np.max(arr)) if len(arr) else 0.0,
        "chain_topouq_nli_min_distance": float(np.min(arr)) if len(arr) else 0.0,
        "distance_matrix": D.tolist(),
    }





def build_output_row(
    item: Dict[str, Any],
    item_id: str,
    response_index: int,
    response: Dict[str, Any],
    selected_indices: Sequence[int],
    chains: Sequence[Sequence[str]],
    uq: Dict[str, Any],
    correct: Optional[int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "id": item_id,
        "question_id": item_id,
        "response_index": response_index,
        "correct": correct,
        "uncertainty": uq["uncertainty"],
        "chain_topouq_nli_variance": uq["chain_topouq_nli_variance"],
        "chain_topouq_nli_mean_distance": uq["chain_topouq_nli_mean_distance"],
        "chain_topouq_nli_max_distance": uq["chain_topouq_nli_max_distance"],
        "chain_topouq_nli_min_distance": uq["chain_topouq_nli_min_distance"],
        "method": "NLI-enhanced Chain-Topo-UQ",
        "dataset": "gsm8k",
        "selected_response_indices_for_topology": list(selected_indices),
        "num_topology_responses": len(chains),
        "meta": {
            "method": "NLI-enhanced Chain-Topo-UQ",
            "dataset": "gsm8k",
            "source_model_family": args.model_family,
            "correctness": "external_correct_file_if_provided_else_math_verify",
            "nli_model_path": args.nli_model_path,
            "nli_max_length": args.nli_max_length,
            "nli_batch_size": args.nli_batch_size,
            "length_penalty_weight": args.length_penalty_weight,
            "edit_penalty": args.edit_penalty,
            "max_responses": args.max_responses,
            "max_units": args.max_units,
        },
    }
    if response_index in selected_indices:
        row["num_units_current_response"] = len(chains[list(selected_indices).index(response_index)])
    else:
        row["num_units_current_response"] = None

    if args.keep_question:
        row["question"] = get_question(item)
    if args.keep_prediction:
        row["prediction"] = get_prediction_text(response)
        row["prediction_final"] = extract_gsm8k_final_answer(get_prediction_text(response))
    if args.keep_gold:
        row["gold"] = get_gold(item)
    if args.keep_candidates:
        responses = get_responses(item)
        row["candidate_predictions"] = [get_prediction_text(responses[i]) for i in selected_indices]
    if args.keep_chains:
        row["reasoning_chains"] = [list(c) for c in chains]
    if args.keep_distance_matrix:
        row["distance_matrix"] = uq["distance_matrix"]
    return row





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NLI-enhanced Chain-Topo-UQ for GSM8K reasoning JSONL generated by Qwen.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)

    
    parser.add_argument("--correct-file", default=None, help="Optional correctness JSONL. If provided, it takes priority over math_verify.")
    parser.add_argument("--missing-correct", default="zero", choices=["error", "zero", "none"])

    
    parser.add_argument("--nli-model-path", default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nli-max-length", type=int, default=256)
    parser.add_argument("--nli-batch-size", type=int, default=16)
    parser.add_argument("--use-safetensors", action="store_true")
    parser.add_argument("--no-safetensors", action="store_true")

    
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-family", default="qwen", choices=["qwen", "llama", "auto", "generic"])
    parser.add_argument("--chat-template-mode", default="off", choices=["off", "on", "auto"])

    
    parser.add_argument("--response-index", type=int, default=0, help="Which response to output. Use -1 to output all selected responses.")
    parser.add_argument("--max-responses", type=int, default=10, help="Responses per problem used for topology distance.")
    parser.add_argument("--include-answer-unit", action="store_true")
    parser.add_argument("--max-units", type=int, default=12)
    parser.add_argument("--length-penalty-weight", type=float, default=0.2)
    parser.add_argument("--edit-penalty", type=float, default=0.75)
    parser.add_argument("--limit", type=int, default=None)

    
    parser.add_argument("--keep-question", action="store_true")
    parser.add_argument("--keep-prediction", action="store_true")
    parser.add_argument("--keep-gold", action="store_true")
    parser.add_argument("--keep-candidates", action="store_true")
    parser.add_argument("--keep-chains", action="store_true")
    parser.add_argument("--keep-distance-matrix", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if mathv_parse is None or mathv_verify is None and args.correct_file is None:
        raise ImportError(
            "math-verify is required when --correct-file is not provided. "
            f"Original import error: {repr(_MATH_VERIFY_IMPORT_ERROR)}"
        )

    if args.use_safetensors and args.no_safetensors:
        raise ValueError("Cannot set both --use-safetensors and --no-safetensors")
    use_safetensors: Optional[bool]
    if args.use_safetensors:
        use_safetensors = True
    elif args.no_safetensors:
        use_safetensors = False
    else:
        use_safetensors = None

    q_correct_map, r_correct_map = load_correct_map(args.correct_file)

    scorer = NLIScorer(
        model_path=args.nli_model_path,
        device=args.device,
        max_length=args.nli_max_length,
        batch_size=args.nli_batch_size,
        use_safetensors=use_safetensors,
    )

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    saved = 0
    skipped_items = 0

    iterator: Iterable[Tuple[int, Dict[str, Any]]] = read_jsonl(args.input_file)
    if args.limit is not None:
        iterator = list(iterator)[: args.limit]

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for item_count, (line_no, item) in enumerate(tqdm(iterator, desc="GSM8K Chain-Topo-UQ"), start=1):
            try:
                item_id = get_item_id(item, fallback=line_no)
                responses = get_responses(item)
                if not responses:
                    print(f"[skip] line={line_no}, id={item_id}: no valid responses", flush=True)
                    skipped_items += 1
                    continue

                selected = choose_response_indices(len(responses), max_responses=args.max_responses)
                if not selected:
                    print(f"[skip] line={line_no}, id={item_id}: no selected responses", flush=True)
                    skipped_items += 1
                    continue

                chains = [
                    extract_reason_units(
                        responses[i],
                        include_answer_unit=args.include_answer_unit,
                        max_units=args.max_units,
                    )
                    for i in selected
                ]

                
                chains = [
                    list(c) if c else [normalize_space(get_reason_text(responses[selected[i]]) or get_prediction_text(responses[selected[i]]))]
                    for i, c in enumerate(chains)
                ]

                uq = pairwise_chain_topouq(
                    chains,
                    scorer=scorer,
                    length_penalty_weight=args.length_penalty_weight,
                    edit_penalty=args.edit_penalty,
                )

                if args.response_index >= 0 and args.response_index < len(responses):
                    output_indices = [args.response_index]
                else:
                    output_indices = selected

                if args.verbose:
                    print(
                        f"[item {item_count}] line={line_no}, id={item_id}, responses={len(responses)}, "
                        f"topology_responses={selected}, uncertainty={uq['uncertainty']:.6f}, "
                        f"mean_distance={uq['chain_topouq_nli_mean_distance']:.6f}",
                        flush=True,
                    )

                for ridx in output_indices:
                    correct = get_correct_label(
                        item_id=item_id,
                        item=item,
                        response=responses[ridx],
                        response_index=ridx,
                        q_map=q_correct_map,
                        r_map=r_correct_map,
                        missing_policy=args.missing_correct,
                    )
                    row = build_output_row(
                        item, item_id, ridx, responses[ridx], selected, chains, uq, correct, args
                    )
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fout.flush()
                    saved += 1

            except Exception as e:
                skipped_items += 1
                print(
                    f"[skip] failed item at input line={line_no}: {type(e).__name__}: {e}",
                    flush=True,
                )
                continue

    print(f"Saved {saved} rows to {args.output_file}; skipped_items={skipped_items}")


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

    _parser = _ap.ArgumentParser(description="topo_uq / qwen / gsm8k / reasoning_augmented (standalone)")
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
    if "topo_uq" not in {"max_seq_prob", "mean_tok_ent", "max_tok_prob", "ppl"}:
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
    if "topo_uq" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "topo_uq", "family": "qwen",
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

