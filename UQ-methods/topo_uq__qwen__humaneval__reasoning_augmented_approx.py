#!/usr/bin/env python3
# Original paper/author repository: https://github.com/LongchaoDa/LLM-Topology

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

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

ID_KEYS = ["id", "question_id", "qid", "task_id", "problem_id"]
QUESTION_KEYS = ["question", "prompt", "input", "problem", "canonical_solution"]
GOLD_KEYS = ["correct answer", "answer", "gold", "gold_answer", "target", "canonical_solution"]
CORRECT_ID_KEYS = ["question_id", "id", "task_id", "qid", "problem_id"]


def read_jsonl(path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    return str(get_first(item, QUESTION_KEYS, ""))


def get_gold(item: Dict[str, Any]) -> str:
    return str(get_first(item, GOLD_KEYS, ""))


def get_prediction_text(response: Dict[str, Any]) -> str:
    for key in ["llm answer", "answer", "prediction", "output", "completion", "response"]:
        if key in response and response[key] is not None:
            return str(response[key])
    if "answer info" in response and isinstance(response["answer info"], dict):
        txt = response["answer info"].get("text")
        if txt:
            return str(txt)
    return str(response.get("llm reason", ""))


def split_reason_steps_from_text(text: str) -> List[str]:
    text = str(text or "")
    text = re.sub(r"Final\s+Answer\s*:.*$", "", text, flags=re.I | re.S).strip()
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
    parts = re.split(r"(?<=[.!?])\s+", normalize_space(text))
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
        units = split_reason_steps_from_text(str(response.get("llm reason", "")))
    if include_answer_unit:
        ans = normalize_space(get_prediction_text(response))
        if ans:
            units.append("Final answer/code: " + ans)
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


def load_correct_map(path: str) -> Tuple[Dict[str, int], Dict[Tuple[str, int], int]]:
    q_map: Dict[str, int] = {}
    r_map: Dict[Tuple[str, int], int] = {}
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


def lookup_correct(
    item_id: str,
    item: Dict[str, Any],
    response_index: int,
    q_map: Dict[str, int],
    r_map: Dict[Tuple[str, int], int],
    missing_policy: str = "error",
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
    if missing_policy == "error":
        raise KeyError(f"Missing correctness for item_id={item_id}, response_index={response_index}. Tried ids={candidate_ids}")
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
    ):
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, torch_dtype=torch.float32)
        self.model.to(self.device)
        self.model.float()
        self.model.eval()
        id2label = getattr(self.model.config, "id2label", {}) or {}
        norm = {int(k): str(v).lower() for k, v in id2label.items()}
        self.entailment_id = self._find_label_id(norm, entailment_label, ["entailment", "entails"])
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
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            premises = [p for p, _ in batch]
            hypotheses = [h for _, h in batch]
            enc = self.tokenizer(premises, hypotheses, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
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
        if normalize_space(a).lower() == normalize_space(b).lower():
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
                dp[i, j] = min(dp[i - 1, j] + edit_penalty, dp[i, j - 1] + edit_penalty, dp[i - 1, j - 1] + sub[i - 1, j - 1])
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
            d = chain_edit_distance_nli(chains[i], chains[j], scorer=scorer, length_penalty_weight=length_penalty_weight, edit_penalty=edit_penalty)
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


def build_output_row(item, item_id, response_index, response, selected_indices, chains, uq, correct, args):
    row = {
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
        "selected_response_indices_for_topology": list(selected_indices),
        "num_topology_responses": len(chains),
    }
    if response_index in selected_indices:
        row["num_units_current_response"] = len(chains[list(selected_indices).index(response_index)])
    else:
        row["num_units_current_response"] = None
    if args.keep_question:
        row["question"] = get_question(item)
    if args.keep_prediction:
        row["prediction"] = get_prediction_text(response)
    if args.keep_gold:
        row["gold"] = get_gold(item)
    if args.keep_candidates:
        responses = item.get("responses", [])
        row["candidate_predictions"] = [get_prediction_text(responses[i]) for i in selected_indices]
    if args.keep_chains:
        row["reasoning_chains"] = [list(c) for c in chains]
    if args.keep_distance_matrix:
        row["distance_matrix"] = uq["distance_matrix"]
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NLI-enhanced Chain-Topo-UQ for HumanEval reasoning JSONL.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--correct-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--nli-model-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nli-max-length", type=int, default=256)
    parser.add_argument("--nli-batch-size", type=int, default=16)
    
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-family", default="auto", choices=["qwen", "llama", "auto"])
    parser.add_argument("--chat-template-mode", default="off", choices=["off", "on", "auto"])
    parser.add_argument("--response-index", type=int, default=0, help="Which response to output. Use -1 to output all selected responses.")
    parser.add_argument("--max-responses", type=int, default=10, help="Responses per problem used for topology distance.")
    parser.add_argument("--include-answer-unit", action="store_true")
    parser.add_argument("--max-units", type=int, default=12)
    parser.add_argument("--length-penalty-weight", type=float, default=0.2)
    parser.add_argument("--edit-penalty", type=float, default=0.75)
    parser.add_argument("--missing-correct", default="error", choices=["error", "zero", "none"])
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
    q_correct_map, r_correct_map = load_correct_map(args.correct_file)
    scorer = NLIScorer(model_path=args.nli_model_path, device=args.device, max_length=args.nli_max_length, batch_size=args.nli_batch_size)
    rows: List[Dict[str, Any]] = []
    iterator = read_jsonl(args.input_file)
    if args.limit is not None:
        iterator = list(iterator)[: args.limit]
    for item_count, (line_no, item) in enumerate(tqdm(iterator, desc="HumanEval Chain-Topo-UQ"), start=1):
        item_id = get_item_id(item, fallback=line_no)
        responses = item.get("responses", [])
        if not isinstance(responses, list) or not responses:
            continue
        selected = choose_response_indices(len(responses), max_responses=args.max_responses)
        chains = [extract_reason_units(responses[i], include_answer_unit=args.include_answer_unit, max_units=args.max_units) for i in selected]
        uq = pairwise_chain_topouq(chains, scorer=scorer, length_penalty_weight=args.length_penalty_weight, edit_penalty=args.edit_penalty)
        output_indices = ([args.response_index] if args.response_index >= 0 and args.response_index < len(responses) else selected)
        if args.verbose:
            print(
                f"[item {item_count}] line={line_no}, id={item_id}, responses={len(responses)}, "
                f"topology_responses={selected}, uncertainty={uq['uncertainty']:.6f}, "
                f"mean_distance={uq['chain_topouq_nli_mean_distance']:.6f}",
                flush=True,
            )
        for ridx in output_indices:
            correct = lookup_correct(item_id=item_id, item=item, response_index=ridx, q_map=q_correct_map, r_map=r_correct_map, missing_policy=args.missing_correct)
            rows.append(build_output_row(item, item_id, ridx, responses[ridx], selected, chains, uq, correct, args))
    write_jsonl(args.output_file, rows)
    print(f"Saved {len(rows)} rows to {args.output_file}")


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

    _parser = _ap.ArgumentParser(description="topo_uq / qwen / humaneval / reasoning_augmented (standalone)")
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

