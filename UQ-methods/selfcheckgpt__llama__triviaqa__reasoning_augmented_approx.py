#!/usr/bin/env python3
# Original paper/author repository: https://github.com/potsawee/selfcheckgpt

from __future__ import annotations

if __name__ == "__main__" and not globals().get("_UQ_EXECUTING", False):
    
    import ast as _uq_ast
    from pathlib import Path as _uq_Path
    _uq_tree = _uq_ast.parse(_uq_Path(__file__).read_text())
    _uq_helper = next(n for n in _uq_tree.body if isinstance(n, _uq_ast.FunctionDef) and n.name == "_uq_standalone")
    exec(compile(_uq_ast.Module(body=[_uq_helper], type_ignores=[]), __file__, "exec"), globals())
    _uq_standalone()
    raise SystemExit(0)

import argparse, json, math, os, re, string, sys
from typing import Any, Dict, List, Optional, Tuple

ARTICLES = {"a", "an", "the"}


def safe_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
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
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            x = json.loads(line)
            if isinstance(x, dict):
                out.append(x)
        except Exception as e:
            safe_print(f"[warn] skip bad json line {i}: {e}", file=sys.stderr)
    return out


def load_done_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                x = json.loads(line)
                if "id" in x:
                    done.add(str(x["id"]))
            except Exception:
                pass
    return done


def get_id(rec: Dict[str, Any], line_idx: int) -> str:
    for k in ["id", "question_id", "qid", "task_id"]:
        if rec.get(k) is not None:
            return str(rec[k])
    return f"line_{line_idx}"


def get_question(rec: Dict[str, Any]) -> str:
    for k in ["question", "prompt", "input", "query", "problem"]:
        if rec.get(k) is not None:
            return str(rec[k])
    return ""


def get_responses(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    rs = rec.get("responses", [])
    return [r for r in rs if isinstance(r, dict)] if isinstance(rs, list) else []


def get_reason(resp: Dict[str, Any]) -> str:
    for k in ["llm reason", "llm_reason", "reason", "rationale", "solution"]:
        if resp.get(k) is not None:
            return str(resp[k]).strip()
    return ""


def get_answer(resp: Dict[str, Any]) -> str:
    for k in ["llm answer", "llm_answer", "answer", "text", "raw_text", "prediction"]:
        if resp.get(k) is not None:
            return str(resp[k]).strip()
    return ""


def normalize_answer(s: str) -> str:
    s = str(s).lower()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("_", " ")
    s = s.translate(str.maketrans({ch: " " for ch in string.punctuation}))
    toks = [t for t in s.split() if t not in ARTICLES]
    return " ".join(toks)


def gold_answers(rec: Dict[str, Any]) -> List[str]:
    ca = rec.get("correct answer", rec.get("correct_answer"))
    ans = []
    if isinstance(ca, dict):
        if ca.get("value") is not None:
            ans.append(str(ca["value"]))
        aliases = ca.get("aliases", [])
        if isinstance(aliases, list):
            ans += [str(a) for a in aliases if a is not None]
        elif aliases is not None:
            ans.append(str(aliases))
    elif ca is not None:
        ans.append(str(ca))
    if not ans:
        for k in ["answer", "gold_answer", "reference"]:
            if rec.get(k) is not None:
                ans.append(str(rec[k]))
        aliases = rec.get("aliases", [])
        if isinstance(aliases, list):
            ans += [str(a) for a in aliases if a is not None]
    seen, out = set(), []
    for a in ans:
        key = normalize_answer(a)
        if key and key not in seen:
            seen.add(key)
            out.append(a)
    return out


def extract_all_boxed(text: str) -> List[str]:
    s, out, start = str(text), [], 0
    marker = r"\boxed"
    while True:
        idx = s.find(marker, start)
        if idx < 0:
            break
        b = s.find("{", idx + len(marker))
        if b < 0:
            start = idx + len(marker)
            continue
        depth = 0
        found = False
        for pos in range(b, len(s)):
            if s[pos] == "{":
                depth += 1
            elif s[pos] == "}":
                depth -= 1
                if depth == 0:
                    out.append(s[b + 1:pos].strip())
                    start = pos + 1
                    found = True
                    break
        if not found:
            out.append(s[b + 1:].strip())
            break
    return out


def clean_pred(text: str) -> str:
    s = str(text).strip()
    m = re.search(r"```(?:text|answer)?\s*(.*?)```", s, re.S | re.I)
    if m:
        s = m.group(1).strip()
    s = re.sub(r"\s*####\s*$", "", s).strip()
    s = re.sub(r"^\s*(?:final\s*answer|answer)\s*:\s*", "", s, flags=re.I).strip()
    lines = [x.strip() for x in s.splitlines() if x.strip()]
    if lines:
        s = lines[0]
    return s.strip(" \t\n\r\"'`")


def pred_candidates(llm_answer: str) -> List[str]:
    raw = str(llm_answer)
    cands = []
    boxes = extract_all_boxed(raw)
    if boxes:
        cands += list(reversed(boxes))
    m = re.search(r"final\s*answer\s*:\s*(.*)", raw, re.I | re.S)
    if m:
        cands.append(m.group(1).strip())
    cands.append(raw)
    out, seen = [], set()
    for c in cands:
        cc = clean_pred(c)
        key = normalize_answer(cc)
        if key and key not in seen:
            seen.add(key)
            out.append(cc)
    return out


def answer_match(pred: str, gold: str, mode: str) -> bool:
    p, g = normalize_answer(pred), normalize_answer(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    if mode == "exact_or_contains":
        return len(g) >= 3 and (g in p or p in g)
    return False


def correct_info(rec: Dict[str, Any], resp: Dict[str, Any], mode: str) -> Dict[str, Any]:
    golds = gold_answers(rec)
    preds = pred_candidates(get_answer(resp))
    matched_pred, matched_gold, correct = (preds[0] if preds else None), None, 0
    for p in preds:
        for g in golds:
            if answer_match(p, g, mode):
                return {
                    "correct": 1,
                    "matched_prediction": p,
                    "matched_gold": g,
                    "prediction_candidates": preds,
                    "gold_answers": golds,
                }
    return {
        "correct": 0 if golds else None,
        "matched_prediction": matched_pred,
        "matched_gold": matched_gold,
        "prediction_candidates": preds,
        "gold_answers": golds,
    }


def pred_key(resp: Dict[str, Any]) -> str:
    c = pred_candidates(get_answer(resp))
    return normalize_answer(c[0]) if c else ""


def choose_target(responses: List[Dict[str, Any]], mode: str) -> Tuple[int, Dict[str, Any]]:
    if not responses:
        raise ValueError("no responses")
    if mode == "first":
        return 0, responses[0]
    if mode == "majority_answer":
        counts, first = {}, {}
        for i, r in enumerate(responses):
            k = pred_key(r)
            if k not in counts:
                counts[k] = 0
                first[k] = i
            counts[k] += 1
        best = max(counts, key=lambda k: (counts[k], -first[k]))
        return first[best], responses[first[best]]
    raise ValueError(f"bad target mode: {mode}")


def split_sentences(text: str) -> List[str]:
    text = str(text).strip()
    if not text:
        return []
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    steps = [x for x in lines if re.match(r"^(step\s*\d+|[-*]\s*)", x, re.I)]
    if steps:
        return steps
    return [x.strip() for x in re.split(r"(?<=[.!?。！？])\s+", text) if x.strip()]


def build_claims(resp: Dict[str, Any], source: str, include_answer: bool) -> List[str]:
    claims = []
    if source in {"steps", "auto"}:
        steps = resp.get("step-wise info", resp.get("step_wise_info", []))
        if isinstance(steps, list):
            for st in steps:
                if isinstance(st, dict) and str(st.get("text", "")).strip():
                    claims.append(str(st["text"]).strip())
    if not claims and source in {"sentences", "auto"}:
        claims += split_sentences(get_reason(resp))
    if source == "full":
        r = get_reason(resp).strip()
        if r:
            claims = [r]
    if include_answer:
        c = pred_candidates(get_answer(resp))
        ans = c[0] if c else clean_pred(get_answer(resp))
        if ans:
            claims.append(f"The final answer is {ans}.")
    seen, out = set(), []
    for c in claims:
        k = re.sub(r"\s+", " ", c).strip()
        if k and k not in seen:
            seen.add(k)
            out.append(c)
    return out


def build_passage(resp: Dict[str, Any], mode: str) -> str:
    reason = get_reason(resp)
    c = pred_candidates(get_answer(resp))
    ans = c[0] if c else clean_pred(get_answer(resp))
    if mode == "reasoning_augmented":
        return reason
    if mode == "answer":
        return f"Final answer: {ans}"
    parts = []
    if reason:
        parts.append("Reasoning:\n" + reason)
    if ans:
        parts.append("Final answer:\n" + ans)
    return "\n\n".join(parts).strip()


class NLIScorer:
    def __init__(self, model_name_or_path: str, device: str = "cuda", dtype: str = "fp32", max_length: int = 512, batch_size: int = 8):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        torch_dtype = self._dtype(dtype)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=True
        )
        self.model.to(torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"))
        self.model.eval()
        self.contradiction_id = self._label_id(["contradiction", "contradictory"], 0)
        self.neutral_id = self._label_id(["neutral"], 1)
        self.entailment_id = self._label_id(["entailment", "entails"], 2)

    def _dtype(self, dtype: str):
        import torch
        if dtype == "auto":
            return torch.float32
        return {
            "fp32": torch.float32, "float32": torch.float32,
            "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        }[dtype]

    def _label_id(self, names: List[str], default: int) -> int:
        label2id = getattr(self.model.config, "label2id", {}) or {}
        low = {str(k).lower(): int(v) for k, v in label2id.items()}
        for k, v in low.items():
            if any(n in k for n in names):
                return v
        return default

    def score_pairs(self, premises: List[str], hypotheses: List[str]) -> List[Dict[str, float]]:
        import torch
        out = []
        for st in range(0, len(premises), self.batch_size):
            p = premises[st:st + self.batch_size]
            h = hypotheses[st:st + self.batch_size]
            inputs = self.tokenizer(
                p, h, truncation=True, max_length=self.max_length,
                padding=True, return_tensors="pt"
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.inference_mode():
                probs = torch.softmax(self.model(**inputs).logits.float(), dim=-1).detach().cpu().tolist()
            for row in probs:
                c = row[self.contradiction_id] if self.contradiction_id < len(row) else row[0]
                e = row[self.entailment_id] if self.entailment_id < len(row) else row[-1]
                n = row[self.neutral_id] if self.neutral_id < len(row) else 1.0 - c - e
                out.append({"contradiction": float(c), "neutral": float(n), "entailment": float(e)})
        return out


def answer_consistency(target: Dict[str, Any], responses: List[Dict[str, Any]], target_idx: int, mode: str) -> Tuple[Optional[float], List[Dict[str, Any]]]:
    tc = pred_candidates(get_answer(target))
    tans = tc[0] if tc else ""
    vals, details = [], []
    for i, r in enumerate(responses):
        if i == target_idx:
            continue
        c = pred_candidates(get_answer(r))
        ans = c[0] if c else ""
        same = answer_match(ans, tans, mode)
        vals.append(1 if same else 0)
        details.append({"response_index": i, "sample_answer": ans, "same_answer_as_target": bool(same)})
    if not vals:
        return None, details
    return float(sum(vals) / len(vals)), details


def process_record(rec: Dict[str, Any], line_idx: int, args, nli: NLIScorer) -> Dict[str, Any]:
    rid = get_id(rec, line_idx)
    q = get_question(rec)
    responses = get_responses(rec)
    if not responses:
        return {"id": rid, "correct": None, "error": "no_responses"}

    target_idx, target = choose_target(responses, args.target_mode)
    corr = correct_info(rec, target, args.match_mode)

    claims = build_claims(target, args.claim_source, args.include_answer_claim)
    sampled = []
    for i, r in enumerate(responses):
        if i == target_idx:
            continue
        passage = build_passage(r, args.passage_mode)
        if passage:
            sampled.append((i, passage))
    if args.max_sampled_passages is not None:
        sampled = sampled[:args.max_sampled_passages]

    premises, hypotheses, meta = [], [], []
    for ci, claim in enumerate(claims):
        hyp = claim
        if args.include_question_in_nli and q:
            hyp = f"Question:\n{q}\n\nClaim:\n{claim}"
        for si, passage in sampled:
            prem = passage
            if args.include_question_in_nli and q:
                prem = f"Question:\n{q}\n\nSampled response:\n{passage}"
            premises.append(prem)
            hypotheses.append(hyp)
            meta.append((ci, si))

    scores = nli.score_pairs(premises, hypotheses) if premises else []
    nli_unc = float(sum(s["contradiction"] for s in scores) / len(scores)) if scores else None

    per_claim = []
    if scores:
        tmp = {}
        for (ci, si), sc in zip(meta, scores):
            tmp.setdefault(ci, []).append(sc["contradiction"])
        for i, c in enumerate(claims):
            vals = tmp.get(i, [])
            per_claim.append({
                "claim_index": i,
                "claim": c,
                "mean_contradiction": float(sum(vals) / len(vals)) if vals else None,
                "num_sampled_passages": len(vals),
            })

    ans_cons, ans_details = answer_consistency(target, responses, target_idx, args.match_mode)
    ans_dis = None if ans_cons is None else float(1.0 - ans_cons)

    if nli_unc is None:
        final_unc = ans_dis
    elif ans_dis is None or args.answer_weight <= 0:
        final_unc = nli_unc
    else:
        final_unc = (1.0 - args.answer_weight) * nli_unc + args.answer_weight * ans_dis

    if final_unc is not None:
        final_unc = max(0.0, min(1.0, float(final_unc)))
    conf = None if final_unc is None else 1.0 - final_unc

    preds = corr.get("prediction_candidates", [])
    prediction = preds[0] if preds else clean_pred(get_answer(target))

    out = {
        "id": rid,
        "correct": corr.get("correct"),
        "selfcheck_confidence": conf,
        "selfcheck_uncertainty": final_unc,
        "uncertainty": final_unc,
        "selfcheck_nli_uncertainty": nli_unc,
        "answer_consistency_with_target": ans_cons,
        "answer_disagreement_uncertainty": ans_dis,
        "target_response_index": target_idx,
        "matched_prediction": corr.get("matched_prediction"),
        "matched_gold": corr.get("matched_gold"),
        "num_responses": len(responses),
        "num_sampled_passages_used": len(sampled),
        "num_claim_units": len(claims),
        "num_nli_pairs": len(scores),
        "meta": {
            "method": "SelfCheckGPT_NLI",
            "dataset": "trivialqa_reasoning_sampling",
            "correctness_source": "input_correct_answer_value_aliases",
            "correctness_rule": "trivialqa_normalized_alias_match",
            "match_mode": args.match_mode,
            "nli_model": args.nli_model,
            "claim_source": args.claim_source,
            "passage_mode": args.passage_mode,
            "include_answer_claim": bool(args.include_answer_claim),
            "include_question_in_nli": bool(args.include_question_in_nli),
            "answer_weight": args.answer_weight,
            "target_mode": args.target_mode,
        },
    }

    if args.keep_question:
        out["question"] = q
    if args.keep_prediction:
        out["prediction"] = prediction
    if args.keep_reasoning:
        out["target_reasoning"] = get_reason(target)
    if args.keep_gold:
        out["gold_answers"] = corr.get("gold_answers", [])
    if args.keep_candidates:
        out["prediction_candidates"] = preds
    if args.keep_claims:
        out["claim_units"] = claims
    if args.keep_per_claim:
        out["per_claim_selfcheck"] = per_claim
    if args.keep_pair_details:
        out["nli_pair_details"] = [
            {"claim_index": ci, "sample_response_index": si, **sc}
            for (ci, si), sc in zip(meta, scores)
        ]
    if args.keep_answer_details:
        out["answer_consistency_details"] = ans_details
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Compute SelfCheckGPT-NLI for TrivialQA reasoning data.")
    p.add_argument("--input-file", required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--nli-model", default="microsoft/deberta-large-mnli")
    p.add_argument("--hf-endpoint", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="fp32")
    p.add_argument("--nli-max-length", type=int, default=512)
    p.add_argument("--nli-batch-size", type=int, default=8)
    p.add_argument("--target-mode", choices=["first", "majority_answer"], default="first")
    p.add_argument("--match-mode", choices=["exact", "exact_or_contains"], default="exact")
    p.add_argument("--claim-source", choices=["auto", "steps", "sentences", "full"], default="auto")
    p.add_argument("--passage-mode", choices=["all", "reasoning_augmented", "answer"], default="all")
    p.add_argument("--include-answer-claim", action="store_true")
    p.add_argument("--include-question-in-nli", action="store_true")
    p.add_argument("--max-sampled-passages", type=int, default=None)
    p.add_argument("--answer-weight", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--keep-question", action="store_true")
    p.add_argument("--keep-prediction", action="store_true")
    p.add_argument("--keep-reasoning", action="store_true")
    p.add_argument("--keep-gold", action="store_true")
    p.add_argument("--keep-candidates", action="store_true")
    p.add_argument("--keep-claims", action="store_true")
    p.add_argument("--keep-per-claim", action="store_true")
    p.add_argument("--keep-pair-details", action="store_true")
    p.add_argument("--keep-answer-details", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    safe_print(f"INPUT_FILE = {args.input_file}")
    safe_print(f"OUTPUT_FILE = {args.output_file}")
    safe_print("Correctness source: record['correct answer']['value'] + aliases")
    safe_print(f"NLI model = {args.nli_model}")
    safe_print(f"dtype = {args.dtype}")
    safe_print(f"Claim source = {args.claim_source}")
    safe_print(f"Passage mode = {args.passage_mode}")
    safe_print(f"Answer weight = {args.answer_weight}")

    records = load_records(args.input_file)
    safe_print(f"Loaded records: {len(records)}")

    safe_print("Loading NLI model...")
    nli = NLIScorer(
        model_name_or_path=args.nli_model,
        device=args.device,
        dtype=args.dtype,
        max_length=args.nli_max_length,
        batch_size=args.nli_batch_size,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    done = set() if args.overwrite else load_done_ids(args.output_file)
    mode = "w" if args.overwrite else "a"
    processed = 0

    with open(args.output_file, mode, encoding="utf-8") as fout:
        for line_idx, rec in enumerate(records, 1):
            if args.limit is not None and processed >= args.limit:
                break
            rid = get_id(rec, line_idx)
            if rid in done:
                continue
            try:
                res = process_record(rec, line_idx, args, nli)
                fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                fout.flush()
                processed += 1
                if args.verbose:
                    safe_print(
                        f"done: {rid} | line={line_idx} | correct={res.get('correct')} | "
                        f"nli_unc={res.get('selfcheck_nli_uncertainty')} | "
                        f"ans_cons={res.get('answer_consistency_with_target')} | "
                        f"conf={res.get('selfcheck_confidence')} | unc={res.get('selfcheck_uncertainty')} | "
                        f"pred={res.get('matched_prediction')} | gold={res.get('matched_gold')}"
                    )
            except KeyboardInterrupt:
                safe_print("KeyboardInterrupt detected. Exiting.")
                raise
            except Exception as e:
                fout.write(json.dumps({"id": rid, "error": str(e)}, ensure_ascii=False) + "\n")
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

    _parser = _ap.ArgumentParser(description="selfcheckgpt / llama / triviaqa / reasoning_augmented (standalone)")
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
    if "selfcheckgpt" not in {"max_seq_prob", "mean_tok_ent", "max_tok_prob", "ppl"}:
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
    if "selfcheckgpt" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "selfcheckgpt", "family": "llama",
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

