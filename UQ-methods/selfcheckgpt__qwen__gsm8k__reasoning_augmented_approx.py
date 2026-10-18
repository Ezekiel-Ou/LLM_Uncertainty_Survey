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


import argparse
import json
import math
import os
import re
import sys
import threading
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from math_verify import parse, verify





print_lock = threading.Lock()
write_lock = threading.Lock()


def safe_print(*args: Any, **kwargs: Any) -> None:
    kwargs.setdefault("flush", True)
    with print_lock:
        print(*args, **kwargs)


def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir_for_file(path)
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
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "id" in obj:
                    done.add(str(obj["id"]))
            except Exception:
                continue
    return done


def iter_json_records(path: str) -> Iterator[Dict[str, Any]]:
    
    
    parsed_any = False
    bad_lines: List[Tuple[int, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    parsed_any = True
                    yield obj
                elif isinstance(obj, list):
                    parsed_any = True
                    for x in obj:
                        if isinstance(x, dict):
                            yield x
            except Exception as e:
                bad_lines.append((line_idx, str(e)))
                continue

    if parsed_any:
        return

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return
    obj = json.loads(content)
    if isinstance(obj, dict):
        yield obj
    elif isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                yield x
    else:
        raise ValueError(f"Unsupported JSON root type: {type(obj)}")





_parse_cache: Dict[str, Any] = {}


def safe_math_parse(text: Any) -> Any:
    s = "" if text is None else str(text)
    if s in _parse_cache:
        return _parse_cache[s]
    try:
        val = parse(s)
    except Exception:
        val = None
    _parse_cache[s] = val
    return val


def mathv_correct(ref_text: str, target_text: str) -> bool:
    try:
        ref = safe_math_parse(ref_text)
        target = safe_math_parse(target_text)
        if ref is None or target is None:
            return False
        return bool(verify(ref, target))
    except Exception:
        return False


def mathv_equivalent(a: str, b: str) -> bool:
    return mathv_correct(a, b)






def get_record_id(record: Dict[str, Any], fallback_idx: int) -> str:
    for k in ["id", "question_id", "qid"]:
        if k in record and record[k] is not None:
            return str(record[k])
    return f"line_{fallback_idx}"


def get_correct_answer(record: Dict[str, Any]) -> str:
    for k in ["correct answer", "correct_answer", "answer", "gt", "ground_truth"]:
        if k in record and record[k] is not None:
            v = record[k]
            if isinstance(v, dict):
                if v.get("value") is not None:
                    return str(v["value"])
                aliases = v.get("aliases")
                if isinstance(aliases, list) and aliases:
                    return str(aliases[0])
            return str(v)
    return ""


def get_question_text(record: Dict[str, Any]) -> str:
    for k in ["question", "problem", "prompt", "input"]:
        if k in record and record[k] is not None:
            return str(record[k])
    return ""


def get_responses(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    xs = record.get("responses", [])
    if isinstance(xs, list):
        return [x for x in xs if isinstance(x, dict)]
    return []


def get_response_answer(resp: Dict[str, Any]) -> str:
    for k in ["llm answer", "llm_answer", "answer", "prediction", "text"]:
        if k in resp and resp[k] is not None:
            return str(resp[k]).strip()
    
    if resp.get("llm reason") is not None:
        return str(resp["llm reason"]).strip()
    return ""


def get_response_reason(resp: Dict[str, Any]) -> str:
    for k in ["llm reason", "llm_reason", "reason", "rationale", "response"]:
        if k in resp and resp[k] is not None:
            return str(resp[k]).strip()
    return ""


def get_step_texts(resp: Dict[str, Any]) -> List[str]:
    steps = resp.get("step-wise info") or resp.get("step_wise_info") or resp.get("steps") or []
    out: List[str] = []
    if isinstance(steps, list):
        for x in steps:
            if isinstance(x, dict):
                t = x.get("text")
                if t is not None and str(t).strip():
                    out.append(str(t).strip())
            elif isinstance(x, str) and x.strip():
                out.append(x.strip())
    return out


def split_reason_into_claims(text: str) -> List[str]:
    text = str(text or "").strip()
    if not text:
        return []

    
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    step_lines = [x for x in lines if re.match(r"(?i)^(step\s*\d+|final\s*answer)\s*[:：]", x)]
    if step_lines:
        return step_lines

    
    parts = re.split(r"(?<=[.!?。！？])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    if parts:
        return parts
    return [text]


def build_claim_units(
    resp: Dict[str, Any],
    include_answer_claim: bool = True,
    max_claims: int = 32,
) -> List[str]:
    claims = get_step_texts(resp)
    if not claims:
        claims = split_reason_into_claims(get_response_reason(resp))

    if include_answer_claim:
        ans = get_response_answer(resp)
        if ans:
            claims.append(f"Final Answer: {ans}")

    
    cleaned: List[str] = []
    seen = set()
    for c in claims:
        c = re.sub(r"\s+", " ", str(c)).strip()
        if not c:
            continue
        if c not in seen:
            cleaned.append(c)
            seen.add(c)
        if len(cleaned) >= max_claims:
            break
    return cleaned


def build_sample_passage(resp: Dict[str, Any], max_chars: int = 4000) -> str:
    reason = get_response_reason(resp)
    if not reason:
        step_texts = get_step_texts(resp)
        reason = "\n".join(step_texts)
    ans = get_response_answer(resp)
    passage = reason.strip()
    if ans:
        if passage:
            passage += f"\nFinal Answer: {ans}"
        else:
            passage = f"Final Answer: {ans}"
    passage = passage.strip()
    if len(passage) > max_chars:
        passage = passage[:max_chars]
    return passage






def choose_target_index(responses: List[Dict[str, Any]], mode: str = "first") -> int:
    if not responses:
        return -1
    if mode == "first":
        return 0
    if mode != "majority":
        return 0

    reps: List[str] = []
    clusters: List[List[int]] = []
    for i, resp in enumerate(responses):
        ans = get_response_answer(resp)
        assigned = False
        for cid, rep in enumerate(reps):
            if mathv_equivalent(ans, rep):
                clusters[cid].append(i)
                assigned = True
                break
        if not assigned:
            reps.append(ans)
            clusters.append([i])

    best = max(clusters, key=lambda xs: len(xs))
    return int(best[0]) if best else 0


def answer_consistency(target_answer: str, aux_answers: List[str]) -> Tuple[float, List[Dict[str, Any]]]:
    if not aux_answers:
        return 0.5, []
    vals = []
    details = []
    for i, ans in enumerate(aux_answers):
        same = mathv_equivalent(target_answer, ans)
        vals.append(1.0 if same else 0.0)
        details.append({
            "sample_index": i,
            "sample_answer": ans,
            "same_as_target_by_math_verify": bool(same),
        })
    return float(sum(vals) / len(vals)), details





class TransformerNLI:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "float32",
        trust_remote_code: bool = False,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        torch_dtype = torch.float32
        if dtype in {"float16", "fp16", "half"}:
            torch_dtype = torch.float16
        elif dtype in {"bfloat16", "bf16"}:
            torch_dtype = torch.bfloat16
        elif dtype in {"float32", "fp32"}:
            torch_dtype = torch.float32

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

        labels = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.contra_idx = None
        self.entail_idx = None
        self.neutral_idx = None
        for idx, label in labels.items():
            if "contrad" in label:
                self.contra_idx = idx
            elif "entail" in label:
                self.entail_idx = idx
            elif "neutral" in label:
                self.neutral_idx = idx
        if self.contra_idx is None:
            raise ValueError(f"Cannot find contradiction label in id2label={labels}")

    @torch.inference_mode()
    def score_pairs(
        self,
        pairs: Sequence[Tuple[str, str]],
        batch_size: int = 16,
        max_length: int = 512,
    ) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            premises = [p for p, _ in batch]
            hypotheses = [h for _, h in batch]
            inputs = self.tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logits = self.model(**inputs).logits.float()
            probs = F.softmax(logits, dim=-1)
            for row in probs.detach().cpu():
                rec: Dict[str, float] = {
                    "contradiction": float(row[self.contra_idx].item()),
                }
                if self.entail_idx is not None:
                    rec["entailment"] = float(row[self.entail_idx].item())
                if self.neutral_idx is not None:
                    rec["neutral"] = float(row[self.neutral_idx].item())
                out.append(rec)
        return out






def compute_selfcheck_nli_score(
    nli: TransformerNLI,
    claims: List[str],
    sample_passages: List[str],
    batch_size: int,
    max_length: int,
    keep_pair_scores: bool = False,
) -> Tuple[float, List[Dict[str, Any]]]:
    
    if not claims or not sample_passages:
        return 0.5, []

    pairs: List[Tuple[str, str]] = []
    index_pairs: List[Tuple[int, int]] = []
    for ci, claim in enumerate(claims):
        for si, passage in enumerate(sample_passages):
            
            pairs.append((passage, claim))
            index_pairs.append((ci, si))

    scores = nli.score_pairs(pairs, batch_size=batch_size, max_length=max_length)

    by_claim: Dict[int, List[Tuple[int, Dict[str, float]]]] = {i: [] for i in range(len(claims))}
    for (ci, si), score in zip(index_pairs, scores):
        by_claim[ci].append((si, score))

    claim_details: List[Dict[str, Any]] = []
    claim_uncs: List[float] = []
    for ci, claim in enumerate(claims):
        vals = [float(s["contradiction"]) for _, s in by_claim.get(ci, [])]
        if vals:
            mean_contra = float(sum(vals) / len(vals))
        else:
            mean_contra = 0.5
        claim_uncs.append(mean_contra)
        item: Dict[str, Any] = {
            "claim_index": ci,
            "claim": claim,
            "selfcheck_claim_uncertainty": mean_contra,
            "num_passages_compared": len(vals),
        }
        if keep_pair_scores:
            item["pair_scores"] = [
                {
                    "sample_passage_index": si,
                    **score,
                }
                for si, score in by_claim.get(ci, [])
            ]
        claim_details.append(item)

    overall = float(sum(claim_uncs) / len(claim_uncs)) if claim_uncs else 0.5
    return overall, claim_details


def compute_one_record(
    record: Dict[str, Any],
    record_idx: int,
    nli: TransformerNLI,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    rid = get_record_id(record, record_idx)
    correct_answer = get_correct_answer(record)
    question = get_question_text(record)
    responses = get_responses(record)

    if not responses:
        return {
            "id": rid,
            "correct_answer": correct_answer,
            "correct": 0,
            "error": "no_responses",
            "selfcheck_confidence": None,
            "selfcheck_uncertainty": None,
            "uncertainty": None,
        }

    target_idx = choose_target_index(responses, args.target_mode)
    target_resp = responses[target_idx]
    target_answer = get_response_answer(target_resp)

    correct = int(mathv_correct(correct_answer, target_answer))

    aux_indices = [i for i in range(len(responses)) if i != target_idx]
    if args.max_sampled_passages is not None and args.max_sampled_passages > 0:
        aux_indices = aux_indices[:args.max_sampled_passages]

    aux_responses = [responses[i] for i in aux_indices]
    aux_answers = [get_response_answer(r) for r in aux_responses]
    sample_passages = [build_sample_passage(r, max_chars=args.max_passage_chars) for r in aux_responses]
    sample_passages = [p for p in sample_passages if p.strip()]

    claims = build_claim_units(
        target_resp,
        include_answer_claim=not args.exclude_answer_claim,
        max_claims=args.max_claims,
    )

    nli_unc, claim_details = compute_selfcheck_nli_score(
        nli=nli,
        claims=claims,
        sample_passages=sample_passages,
        batch_size=args.nli_batch_size,
        max_length=args.nli_max_length,
        keep_pair_scores=args.keep_pair_scores,
    )
    nli_conf = 1.0 - nli_unc

    ans_consistency, ans_details = answer_consistency(target_answer, aux_answers)
    ans_unc = 1.0 - ans_consistency

    answer_weight = max(0.0, min(1.0, float(args.answer_weight)))
    final_unc = (1.0 - answer_weight) * nli_unc + answer_weight * ans_unc
    final_conf = 1.0 - final_unc

    out: Dict[str, Any] = {
        "id": rid,
        "correct_answer": correct_answer,
        "target_response_index": target_idx,
        "target_answer": target_answer,
        "correct": correct,

        "selfcheck_confidence": float(final_conf),
        "selfcheck_uncertainty": float(final_unc),
        "uncertainty": float(final_unc),

        "selfcheck_nli_confidence": float(nli_conf),
        "selfcheck_nli_uncertainty": float(nli_unc),
        "answer_consistency_with_target": float(ans_consistency),
        "answer_disagreement_uncertainty": float(ans_unc),

        "num_responses": len(responses),
        "num_sampled_passages_used": len(sample_passages),
        "num_claim_units": len(claims),
        "num_correct_responses": int(sum(mathv_correct(correct_answer, get_response_answer(r)) for r in responses)),

        "meta": {
            "method": "SelfCheckGPT-style NLI, implemented with transformers",
            "dataset_format": "reasoning_sampling",
            "nli_model": args.nli_model,
            "correctness_checker": "math_verify",
            "answer_equivalence_checker": "math_verify",
            "target_mode": args.target_mode,
            "answer_weight": answer_weight,
            "score_definition": "uncertainty=(1-answer_weight)*mean_NLI_contradiction + answer_weight*(1-answer_consistency)",
        },
        "claim_details": claim_details,
        "answer_consistency_details": ans_details,
    }

    if args.keep_question:
        out["question"] = question
    if args.keep_target_reason:
        out["target_reason"] = get_response_reason(target_resp)
    if args.keep_sample_passages:
        out["sample_passages"] = sample_passages

    return out






def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute SelfCheckGPT-style uncertainty for reasoning-sampling JSONL.")
    p.add_argument("--input-file", required=True, help="Input reasoning-sampling JSONL")
    p.add_argument("--output-file", required=True, help="Output JSONL")
    p.add_argument("--nli-model", default="potsawee/deberta-v3-large-mnli", help="NLI model loaded by transformers")
    p.add_argument("--hf-endpoint", default=None, help="Optional HF_ENDPOINT, e.g. https://hf-mirror.com")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="NLI device")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16", "fp32", "fp16", "bf16"], help="NLI dtype")
    p.add_argument("--trust-remote-code", action="store_true")

    p.add_argument("--target-mode", default="first", choices=["first", "majority"], help="Target response selection")
    p.add_argument("--answer-weight", type=float, default=0.2, help="Weight of math_verify answer disagreement in final uncertainty. Use 0 for pure SelfCheckNLI.")
    p.add_argument("--max-claims", type=int, default=32)
    p.add_argument("--max-sampled-passages", type=int, default=10)
    p.add_argument("--max-passage-chars", type=int, default=4000)
    p.add_argument("--exclude-answer-claim", action="store_true", help="Do not include Final Answer as a claim unit for NLI")
    p.add_argument("--nli-batch-size", type=int, default=8)
    p.add_argument("--nli-max-length", type=int, default=512)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output instead of resume")

    p.add_argument("--keep-question", action="store_true")
    p.add_argument("--keep-target-reason", action="store_true")
    p.add_argument("--keep-sample-passages", action="store_true")
    p.add_argument("--keep-pair-scores", action="store_true", help="Store every claim-passage NLI score; large output")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    ensure_dir_for_file(args.output_file)
    if args.overwrite and os.path.exists(args.output_file):
        os.remove(args.output_file)

    safe_print(f"INPUT_FILE = {args.input_file}")
    safe_print(f"OUTPUT_FILE = {args.output_file}")
    safe_print("Correctness source: record['correct answer'] + math_verify")
    safe_print("SelfCheckGPT backend: transformers NLI implemented locally, no selfcheckgpt import")
    safe_print(f"NLI model: {args.nli_model}")
    safe_print(f"Target mode: {args.target_mode}")
    safe_print(f"Answer-weight: {args.answer_weight}")

    safe_print("Loading NLI model...")
    nli = TransformerNLI(
        model_name=args.nli_model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )

    done = load_done_ids(args.output_file) if not args.overwrite else set()
    safe_print(f"existing records: {len(done)}")

    processed = 0
    for idx, record in enumerate(iter_json_records(args.input_file), start=1):
        if args.limit is not None and processed >= args.limit:
            break

        rid = get_record_id(record, idx)
        if str(rid) in done:
            continue

        try:
            result = compute_one_record(record, idx, nli, args)
            append_jsonl(args.output_file, result)
            processed += 1
            if args.verbose:
                unc = result.get("selfcheck_uncertainty")
                unc_s = "None" if unc is None else f"{float(unc):.6f}"
                safe_print(
                    f"done: {rid} | correct={result.get('correct')} | "
                    f"nli_unc={result.get('selfcheck_nli_uncertainty')} | "
                    f"ans_cons={result.get('answer_consistency_with_target')} | "
                    f"unc={unc_s}"
                )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            err = {
                "id": rid,
                "error": str(e),
                "selfcheck_confidence": None,
                "selfcheck_uncertainty": None,
                "uncertainty": None,
            }
            append_jsonl(args.output_file, err)
            safe_print(f"error: {rid} | {e}")

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

    _parser = _ap.ArgumentParser(description="selfcheckgpt / qwen / gsm8k / reasoning_augmented (standalone)")
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
    if "selfcheckgpt" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "selfcheckgpt", "family": "qwen",
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

