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
import math
import os
import re
import string
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer






def read_jsonl(path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc


def write_jsonl_record(fp, record: Dict[str, Any]) -> None:
    fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    fp.flush()


def get_first_existing(d: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def get_item_id(item: Dict[str, Any], line_no: int) -> str:
    return str(get_first_existing(item, ["id", "question_id", "qid", "task_id"], f"line_{line_no}"))






def normalize_triviaqa_answer(s: Any) -> str:
    
    if s is None:
        return ""
    s = str(s)
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    table = str.maketrans("", "", string.punctuation)
    s = s.translate(table)
    s = " ".join(s.split())
    return s


def extract_boxed_text(text: str) -> Optional[str]:
    if not text:
        return None
    
    pattern = r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def extract_prediction(response: Dict[str, Any]) -> str:
    ans = response.get("llm answer") or response.get("answer") or response.get("prediction") or ""
    boxed = extract_boxed_text(str(ans))
    if boxed is not None:
        return boxed
    reason = response.get("llm reason") or response.get("reasoning_augmented") or response.get("response") or ""
    boxed = extract_boxed_text(str(reason))
    if boxed is not None:
        return boxed
    return str(ans if ans else reason).strip()


def get_gold_answers(item: Dict[str, Any]) -> List[str]:
    gold = get_first_existing(item, ["correct answer", "gold", "answer", "answers", "reference"], None)
    answers: List[str] = []
    if isinstance(gold, dict):
        
        for key in ["value", "normalized_value", "answer", "text"]:
            if gold.get(key) is not None:
                answers.append(str(gold[key]))
        for key in ["aliases", "normalized_aliases", "human_answers"]:
            vals = gold.get(key)
            if isinstance(vals, list):
                answers.extend(str(x) for x in vals if x is not None)
            elif vals is not None:
                answers.append(str(vals))
    elif isinstance(gold, list):
        answers.extend(str(x) for x in gold if x is not None)
    elif gold is not None:
        answers.append(str(gold))
    
    out, seen = [], set()
    for a in answers:
        na = normalize_triviaqa_answer(a)
        if na and na not in seen:
            seen.add(na)
            out.append(a)
    return out


def triviaqa_correct(prediction: str, gold_answers: Sequence[str], match_mode: str = "exact") -> int:
    pred_norm = normalize_triviaqa_answer(prediction)
    gold_norms = [normalize_triviaqa_answer(g) for g in gold_answers if normalize_triviaqa_answer(g)]
    if not pred_norm or not gold_norms:
        return 0
    if match_mode == "exact":
        return int(any(pred_norm == g for g in gold_norms))
    if match_mode == "contains":
        return int(any(g in pred_norm or pred_norm in g for g in gold_norms))
    raise ValueError(f"Unsupported match_mode: {match_mode}")


def load_correct_file(path: Optional[str]) -> Dict[Tuple[str, Optional[int]], int]:
    
    result: Dict[Tuple[str, Optional[int]], int] = {}
    if not path:
        return result
    for line_no, row in read_jsonl(path):
        qid = get_first_existing(row, ["question_id", "id", "qid", "task_id"], None)
        if qid is None:
            raise ValueError(f"correct-file line {line_no} has no id/question_id/qid/task_id")
        corr = row.get("correct")
        if corr is None:
            raise ValueError(f"correct-file line {line_no} has no correct field")
        ridx = get_first_existing(row, ["response_index", "response_id", "idx"], None)
        result[(str(qid), int(ridx) if ridx is not None else None)] = int(corr)
    return result






STEP_RE = re.compile(
    r"(?:^|\n)\s*(?:Step\s*\d+\s*[:\).\-]|\d+\s*[\).\-])\s*(.*?)(?=(?:\n\s*(?:Step\s*\d+\s*[:\).\-]|\d+\s*[\).\-])\s*)|(?:\n\s*Final\s*Answer\s*:)|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)


def clean_step_text(s: Any) -> str:
    s = "" if s is None else str(s)
    s = re.sub(r"\s+", " ", s).strip()
    
    s = re.sub(r"\bFinal\s+Answer\s*:.*$", "", s, flags=re.IGNORECASE).strip()
    return s


def extract_steps_from_response(response: Dict[str, Any], include_answer_unit: bool = False, max_units: Optional[int] = None) -> List[str]:
    steps: List[str] = []

    swi = response.get("step-wise info") or response.get("step_wise_info") or response.get("steps")
    if isinstance(swi, list):
        for x in swi:
            if isinstance(x, dict):
                txt = clean_step_text(x.get("text") or x.get("step") or x.get("content"))
            else:
                txt = clean_step_text(x)
            if txt:
                steps.append(txt)

    if not steps:
        reason = response.get("llm reason") or response.get("reasoning_augmented") or response.get("response") or ""
        matches = STEP_RE.findall(str(reason))
        for m in matches:
            txt = clean_step_text(m)
            if txt:
                steps.append(txt)
        if not steps and reason:
            
            tmp = re.sub(r"Final\s*Answer\s*:.*$", "", str(reason), flags=re.IGNORECASE | re.DOTALL)
            parts = [p.strip() for p in re.split(r"\n+|(?<=[.!?])\s+", tmp) if p.strip()]
            steps.extend(clean_step_text(p) for p in parts if clean_step_text(p))

    if include_answer_unit:
        ans = response.get("answer info", {})
        ans_text = ""
        if isinstance(ans, dict):
            ans_text = ans.get("text") or ""
        if not ans_text:
            ans_text = response.get("llm answer") or response.get("answer") or ""
        ans_text = clean_step_text(ans_text)
        if ans_text:
            steps.append("Final answer: " + ans_text)

    if max_units is not None and max_units > 0:
        steps = steps[:max_units]
    return steps






@dataclass
class NLIScore:
    entailment: float
    neutral: float
    contradiction: float


class NLIScorer:
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 512,
        fp16: bool = True,
    ) -> None:
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_length = max_length
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        dtype = torch.float16 if (fp16 and self.device.startswith("cuda")) else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        self.label_map = self._infer_label_map()
        self.cache: Dict[Tuple[str, str], NLIScore] = {}

    def _infer_label_map(self) -> Dict[str, int]:
        cfg = self.model.config
        id2label = getattr(cfg, "id2label", None) or {}
        label2id = {str(v).lower(): int(k) for k, v in id2label.items()}

        def find(name: str, fallback: int) -> int:
            for k, v in label2id.items():
                if name in k:
                    return v
            return fallback

        
        
        return {
            "contradiction": find("contradiction", 0),
            "neutral": find("neutral", 1),
            "entailment": find("entail", 2),
        }

    @torch.no_grad()
    def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[NLIScore]:
        out: List[Optional[NLIScore]] = [None] * len(pairs)
        missing: List[Tuple[int, str, str]] = []
        for i, (prem, hyp) in enumerate(pairs):
            key = (prem, hyp)
            if key in self.cache:
                out[i] = self.cache[key]
            else:
                missing.append((i, prem, hyp))

        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            prems = [x[1] for x in batch]
            hyps = [x[2] for x in batch]
            enc = self.tokenizer(
                prems,
                hyps,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits.float()
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            for row, (orig_idx, prem, hyp) in zip(probs, batch):
                score = NLIScore(
                    entailment=float(row[self.label_map["entailment"]]),
                    neutral=float(row[self.label_map["neutral"]]),
                    contradiction=float(row[self.label_map["contradiction"]]),
                )
                self.cache[(prem, hyp)] = score
                out[orig_idx] = score

        return [x for x in out if x is not None]

    def symmetric_cost(self, a: str, b: str, mode: str = "soft", contradiction_weight: float = 0.5) -> float:
        if not a and not b:
            return 0.0
        if not a or not b:
            return 1.0
        if normalize_space(a) == normalize_space(b):
            return 0.0
        ab, ba = self.score_pairs([(a, b), (b, a)])
        if mode == "threshold":
            if ab.entailment >= 0.8 and ba.entailment >= 0.8:
                return 0.1
            if max(ab.contradiction, ba.contradiction) >= 0.7:
                return 1.0
            return 0.5
        
        bi_entail = min(ab.entailment, ba.entailment)
        contra = max(ab.contradiction, ba.contradiction)
        cost = 1.0 - bi_entail + contradiction_weight * contra
        return float(max(0.0, min(1.0, cost)))


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())






def chain_edit_distance_nli(
    chain_a: Sequence[str],
    chain_b: Sequence[str],
    scorer: NLIScorer,
    insertion_cost: float = 1.0,
    deletion_cost: float = 1.0,
    nli_cost_mode: str = "soft",
    contradiction_weight: float = 0.5,
    normalize: bool = True,
) -> float:
    
    n, m = len(chain_a), len(chain_b)
    if n == 0 and m == 0:
        return 0.0
    dp = np.zeros((n + 1, m + 1), dtype=np.float32)
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + deletion_cost
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + insertion_cost

    
    sub = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            sub[i, j] = scorer.symmetric_cost(
                chain_a[i],
                chain_b[j],
                mode=nli_cost_mode,
                contradiction_weight=contradiction_weight,
            )

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = min(
                dp[i - 1, j] + deletion_cost,
                dp[i, j - 1] + insertion_cost,
                dp[i - 1, j - 1] + sub[i - 1, j - 1],
            )
    dist = float(dp[n, m])
    if normalize:
        dist /= float(max(n, m, 1))
    return dist


def pairwise_chain_topouq(
    chains: Sequence[Sequence[str]],
    scorer: NLIScorer,
    insertion_cost: float,
    deletion_cost: float,
    nli_cost_mode: str,
    contradiction_weight: float,
) -> Dict[str, Any]:
    L = len(chains)
    D = np.zeros((L, L), dtype=np.float32)
    if L <= 1:
        return {
            "chain_topouq_nli_variance": 0.0,
            "chain_topouq_nli_mean_distance": 0.0,
            "chain_topouq_nli_max_distance": 0.0,
            "chain_topouq_nli_min_distance": 0.0,
            "chain_topouq_nli_num_chains": L,
            "chain_topouq_nli_distance_matrix": D.tolist(),
        }
    vals: List[float] = []
    for i in range(L):
        for j in range(i + 1, L):
            d = chain_edit_distance_nli(
                chains[i],
                chains[j],
                scorer=scorer,
                insertion_cost=insertion_cost,
                deletion_cost=deletion_cost,
                nli_cost_mode=nli_cost_mode,
                contradiction_weight=contradiction_weight,
                normalize=True,
            )
            D[i, j] = D[j, i] = d
            vals.append(d)
    arr = np.asarray(vals, dtype=np.float32)
    return {
        "chain_topouq_nli_variance": float(np.var(arr)),
        "chain_topouq_nli_mean_distance": float(np.mean(arr)),
        "chain_topouq_nli_max_distance": float(np.max(arr)),
        "chain_topouq_nli_min_distance": float(np.min(arr)),
        "chain_topouq_nli_num_chains": L,
        "chain_topouq_nli_distance_matrix": D.tolist(),
    }






def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NLI-enhanced Chain-Topo-UQ for TriviaQA reasoning data")
    p.add_argument("--input-file", required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--correct-file", default=None, help="Optional external JSONL correctness file")

    
    p.add_argument("--model-path", default=None, help="Generator model path, kept for Qwen/LLaMA script compatibility")
    p.add_argument("--model-family", choices=["qwen", "llama", "auto"], default="auto")
    p.add_argument("--chat-template-mode", choices=["off", "on", "auto"], default="off")

    
    p.add_argument("--nli-model-path", default="microsoft/deberta-v3-large-mnli")
    p.add_argument("--nli-device", default="auto")
    p.add_argument("--nli-batch-size", type=int, default=16)
    p.add_argument("--nli-max-length", type=int, default=512)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--nli-cost-mode", choices=["soft", "threshold"], default="soft")
    p.add_argument("--contradiction-weight", type=float, default=0.5)

    
    p.add_argument("--response-index", type=int, default=0, help="Response used for correctness/prediction output")
    p.add_argument("--max-responses", type=int, default=10, help="Number of responses used to compute Topo-UQ")
    p.add_argument("--min-responses", type=int, default=2)
    p.add_argument("--include-answer-unit", action="store_true")
    p.add_argument("--max-units", type=int, default=12)
    p.add_argument("--insertion-cost", type=float, default=1.0)
    p.add_argument("--deletion-cost", type=float, default=1.0)

    
    p.add_argument("--match-mode", choices=["exact", "contains"], default="exact")
    p.add_argument("--missing-correct", choices=["error", "zero", "none"], default="zero")
    p.add_argument("--keep-question", action="store_true")
    p.add_argument("--keep-prediction", action="store_true")
    p.add_argument("--keep-gold", action="store_true")
    p.add_argument("--keep-candidates", action="store_true")
    p.add_argument("--keep-chains", action="store_true")
    p.add_argument("--keep-distance-matrix", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    correct_map = load_correct_file(args.correct_file)
    scorer = NLIScorer(
        model_path=args.nli_model_path,
        device=args.nli_device,
        batch_size=args.nli_batch_size,
        max_length=args.nli_max_length,
        fp16=not args.no_fp16,
    )

    total = 0
    written = 0
    with open(args.output_file, "w", encoding="utf-8") as out_fp:
        for line_no, item in read_jsonl(args.input_file):
            total += 1
            qid = get_item_id(item, line_no)
            responses = item.get("responses") or item.get("candidates") or []
            if not isinstance(responses, list):
                responses = []

            if args.verbose:
                print(f"[item {total}] line={line_no}, id={qid}, responses={len(responses)}", flush=True)

            selected_response: Dict[str, Any] = {}
            if responses and 0 <= args.response_index < len(responses):
                selected_response = responses[args.response_index]
            elif responses:
                selected_response = responses[0]

            prediction = extract_prediction(selected_response) if selected_response else ""
            gold_answers = get_gold_answers(item)

            
            corr: Optional[int]
            if (qid, args.response_index) in correct_map:
                corr = correct_map[(qid, args.response_index)]
            elif (qid, None) in correct_map:
                corr = correct_map[(qid, None)]
            elif gold_answers:
                corr = triviaqa_correct(prediction, gold_answers, match_mode=args.match_mode)
            else:
                if args.missing_correct == "error":
                    raise ValueError(f"No gold answer/correct entry for id={qid}")
                corr = 0 if args.missing_correct == "zero" else None

            chains: List[List[str]] = []
            used_indices: List[int] = []
            for ridx, resp in enumerate(responses[: args.max_responses]):
                if not isinstance(resp, dict):
                    continue
                chain = extract_steps_from_response(
                    resp,
                    include_answer_unit=args.include_answer_unit,
                    max_units=args.max_units,
                )
                if chain:
                    chains.append(chain)
                    used_indices.append(ridx)

            if len(chains) < args.min_responses:
                
                uq = {
                    "chain_topouq_nli_variance": None,
                    "chain_topouq_nli_mean_distance": None,
                    "chain_topouq_nli_max_distance": None,
                    "chain_topouq_nli_min_distance": None,
                    "chain_topouq_nli_num_chains": len(chains),
                    "chain_topouq_nli_distance_matrix": [],
                }
            else:
                uq = pairwise_chain_topouq(
                    chains,
                    scorer=scorer,
                    insertion_cost=args.insertion_cost,
                    deletion_cost=args.deletion_cost,
                    nli_cost_mode=args.nli_cost_mode,
                    contradiction_weight=args.contradiction_weight,
                )

            record: Dict[str, Any] = {
                "id": qid,
                "question_id": qid,
                "response_index": args.response_index,
                "correct": corr,
                "uncertainty": uq["chain_topouq_nli_variance"],
                "chain_topouq_nli_variance": uq["chain_topouq_nli_variance"],
                "chain_topouq_nli_mean_distance": uq["chain_topouq_nli_mean_distance"],
                "chain_topouq_nli_max_distance": uq["chain_topouq_nli_max_distance"],
                "chain_topouq_nli_min_distance": uq["chain_topouq_nli_min_distance"],
                "chain_topouq_nli_num_chains": uq["chain_topouq_nli_num_chains"],
                "chain_topouq_nli_used_response_indices": used_indices,
                "method": "Chain-Topo-UQ-NLI-GED",
                "nli_model_path": args.nli_model_path,
                "generator_model_path": args.model_path,
                "model_family": args.model_family,
            }
            if args.keep_question:
                record["question"] = get_first_existing(item, ["question", "input", "prompt"], None)
            if args.keep_prediction:
                record["prediction"] = prediction
            if args.keep_gold:
                record["gold"] = gold_answers
            if args.keep_candidates:
                record["candidates"] = [extract_prediction(r) for r in responses if isinstance(r, dict)]
            if args.keep_chains:
                record["reasoning_chains"] = chains
            if args.keep_distance_matrix:
                record["chain_topouq_nli_distance_matrix"] = uq["chain_topouq_nli_distance_matrix"]

            write_jsonl_record(out_fp, record)
            written += 1

            if args.verbose:
                print(
                    f"  -> id={qid} correct={corr} "
                    f"uncertainty={record['chain_topouq_nli_variance']} "
                    f"mean_dist={record['chain_topouq_nli_mean_distance']} "
                    f"chains={record['chain_topouq_nli_num_chains']}",
                    flush=True,
                )

    print(f"Done. processed={total}, written={written}, output={args.output_file}", flush=True)


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

    _parser = _ap.ArgumentParser(description="topo_uq / llama / triviaqa / reasoning_augmented (standalone)")
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
    if "topo_uq" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "topo_uq", "family": "llama",
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

