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
import ast
import json
import math
import os
import re
import sys
import textwrap
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def load_correct_map(path: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for obj in load_records(path):
        qid = None
        for key in ["question_id", "id", "task_id", "qid"]:
            if obj.get(key) is not None:
                qid = str(obj[key])
                break
        if qid is None:
            continue

        val = obj.get("correct", obj.get("is_correct", obj.get("passed", None)))
        try:
            out[qid] = int(val)
        except Exception:
            out[qid] = 1 if bool(val) else 0
    return out


def load_problem_map(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"problem file does not exist: {path}")

    out: Dict[str, Dict[str, Any]] = {}
    for obj in load_records(path):
        qid = None
        for key in ["question_id", "task_id", "id", "qid"]:
            if obj.get(key) is not None:
                qid = str(obj[key])
                break
        if qid is not None:
            out[qid] = obj
    return out





def get_record_id(record: Dict[str, Any], line_idx: int) -> str:
    for key in ["id", "question_id", "task_id", "qid"]:
        if record.get(key) is not None:
            return str(record[key])
    return f"line_{line_idx}"


def get_responses(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    responses = record.get("responses", [])
    if isinstance(responses, list):
        return [r for r in responses if isinstance(r, dict)]
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


def get_question_from_record_or_problem(
    record: Dict[str, Any],
    qid: str,
    problem_map: Dict[str, Dict[str, Any]],
) -> str:
    for key in ["question", "prompt", "problem", "input", "query"]:
        if record.get(key) is not None:
            return str(record[key])

    prob = problem_map.get(qid, {})
    for key in ["question", "prompt", "problem", "input", "query"]:
        if prob.get(key) is not None:
            return str(prob[key])

    return ""





def extract_boxed_content(text: str) -> str:
    s = str(text)
    marker = r"\boxed"
    idx = s.find(marker)
    if idx < 0:
        return s

    brace_start = s.find("{", idx + len(marker))
    if brace_start < 0:
        return s

    depth = 0
    for pos in range(brace_start, len(s)):
        ch = s[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[brace_start + 1:pos]
    return s[brace_start + 1:]


def strip_code_wrappers(text: str) -> str:
    s = extract_boxed_content(str(text)).strip()

    m = re.search(r"<CODE>\s*(.*?)\s*</CODE>", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1).strip()

    m = re.search(r"```(?:python|py)?\s*(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1).strip()

    s = re.sub(r"^\s*(?:answer|output|code)\s*:\s*", "", s, flags=re.IGNORECASE)
    return s.strip("\n")


def find_function_header(prompt: str) -> Optional[str]:
    if not prompt:
        return None

    pattern = re.compile(r"^def\s+\w+\s*\(.*?\)\s*(?:->\s*[^:]+)?\s*:", re.MULTILINE)
    matches = list(pattern.finditer(prompt))
    if not matches:
        return None
    return matches[-1].group(0)


def looks_like_full_function(code: str) -> bool:
    return bool(re.search(r"^\s*def\s+\w+\s*\(", code, flags=re.MULTILINE))


def wrap_body_with_header(code: str, prompt: str) -> str:
    code = strip_code_wrappers(code)
    if looks_like_full_function(code):
        return textwrap.dedent(code).strip()

    header = find_function_header(prompt)
    if not header:
        return textwrap.dedent(code).strip()

    body = textwrap.dedent(code).strip("\n")
    if not body.strip():
        body = "pass"

    return header.rstrip() + "\n" + textwrap.indent(body, "    ")


def normalize_text_code(code: str) -> str:
    s = strip_code_wrappers(code)
    s = textwrap.dedent(s).strip()
    lines = [ln.rstrip() for ln in s.splitlines()]
    s = "\n".join([ln for ln in lines if ln.strip()])
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def ast_key(code: str, prompt: str) -> Optional[str]:
    full = wrap_body_with_header(code, prompt)
    try:
        tree = ast.parse(full)
        return ast.dump(tree, annotate_fields=True, include_attributes=False)
    except Exception:
        return None


def code_equivalent(a: str, b: str, prompt: str, mode: str = "ast_or_text") -> bool:
    if mode in {"ast", "ast_or_text"}:
        ka = ast_key(a, prompt)
        kb = ast_key(b, prompt)
        if ka is not None and kb is not None:
            return ka == kb
        if mode == "ast":
            return False

    return normalize_text_code(a) == normalize_text_code(b)


def normalize_code_for_output(code: str, prompt: str) -> str:
    return wrap_body_with_header(code, prompt)





def cluster_majority_index(answers: List[str], prompt: str, equivalence_mode: str) -> int:
    if not answers:
        return -1

    reps: List[str] = []
    counts: Dict[int, int] = {}
    first_idx: Dict[int, int] = {}

    for idx, ans in enumerate(answers):
        assigned = False
        for cid, rep in enumerate(reps):
            if code_equivalent(ans, rep, prompt, equivalence_mode):
                counts[cid] += 1
                assigned = True
                break

        if not assigned:
            cid = len(reps)
            reps.append(ans)
            counts[cid] = 1
            first_idx[cid] = idx

    best_cid = max(counts.keys(), key=lambda c: (counts[c], -first_idx[c]))
    return first_idx[best_cid]


def choose_target_response(
    responses: List[Dict[str, Any]],
    prompt: str,
    target_mode: str,
    equivalence_mode: str,
) -> Tuple[int, Dict[str, Any]]:
    if not responses:
        raise ValueError("No responses found.")

    if target_mode == "first":
        return 0, responses[0]

    if target_mode == "majority":
        answers = [get_llm_answer(r) for r in responses]
        idx = cluster_majority_index(answers, prompt, equivalence_mode)
        if idx < 0:
            idx = 0
        return idx, responses[idx]

    raise ValueError(f"Unsupported target_mode: {target_mode}")





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


def build_claim_units(
    target_response: Dict[str, Any],
    prompt: str,
    claim_source: str,
    include_answer_claim: bool,
) -> List[str]:
    claims: List[str] = []

    if claim_source in {"steps", "auto"}:
        steps = target_response.get("step-wise info", target_response.get("step_wise_info", []))
        if isinstance(steps, list):
            for step in steps:
                if isinstance(step, dict) and step.get("text") is not None:
                    txt = str(step["text"]).strip()
                    if txt:
                        claims.append(txt)

    if not claims and claim_source in {"sentences", "auto"}:
        claims.extend(split_reasoning_sentences(get_llm_reason(target_response)))

    if claim_source == "full":
        full_reason = get_llm_reason(target_response)
        if full_reason.strip():
            claims = [full_reason.strip()]
        else:
            claims = []

    if include_answer_claim:
        code = normalize_code_for_output(get_llm_answer(target_response), prompt)
        if code.strip():
            claims.append("The proposed Python solution is:\n```python\n" + code.strip() + "\n```")

    
    seen = set()
    uniq: List[str] = []
    for c in claims:
        key = re.sub(r"\s+", " ", c).strip()
        if key and key not in seen:
            seen.add(key)
            uniq.append(c.strip())

    return uniq


def build_sampled_passage(response: Dict[str, Any], prompt: str, passage_mode: str) -> str:
    reason = get_llm_reason(response)
    code = normalize_code_for_output(get_llm_answer(response), prompt)

    if passage_mode == "reasoning_augmented":
        return reason.strip()
    if passage_mode == "answer":
        return "The proposed Python solution is:\n```python\n" + code.strip() + "\n```"

    
    parts = []
    if reason.strip():
        parts.append("Reasoning:\n" + reason.strip())
    if code.strip():
        parts.append("Final Python solution:\n```python\n" + code.strip() + "\n```")
    return "\n\n".join(parts).strip()





class NLIScorer:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: str = "auto",
        max_length: int = 512,
        batch_size: int = 8,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        torch_dtype = self.resolve_dtype(dtype)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        if device == "cuda" and torch.cuda.is_available():
            self.model.to(torch.device("cuda"))
        else:
            self.model.to(torch.device("cpu"))

        self.model.eval()

        self.contradiction_id = self.find_label_id(["contradiction", "contradictory"], default=0)
        self.neutral_id = self.find_label_id(["neutral"], default=1)
        self.entailment_id = self.find_label_id(["entailment", "entails"], default=2)

    def resolve_dtype(self, dtype: str):
        import torch
        if dtype == "auto":
            if torch.cuda.is_available():
                return torch.float16
            return torch.float32
        if dtype in {"fp16", "float16", "half"}:
            return torch.float16
        if dtype in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype in {"fp32", "float32"}:
            return torch.float32
        raise ValueError(f"Unsupported dtype: {dtype}")

    def find_label_id(self, names: List[str], default: int) -> int:
        label2id = getattr(self.model.config, "label2id", {}) or {}
        lower = {str(k).lower(): int(v) for k, v in label2id.items()}
        for k, v in lower.items():
            if any(name in k for name in names):
                return v
        return default

    def score_pairs(self, premises: List[str], hypotheses: List[str]) -> List[Dict[str, float]]:
        import torch

        assert len(premises) == len(hypotheses)
        results: List[Dict[str, float]] = []

        for start in range(0, len(premises), self.batch_size):
            batch_p = premises[start:start + self.batch_size]
            batch_h = hypotheses[start:start + self.batch_size]

            inputs = self.tokenizer(
                batch_p,
                batch_h,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.inference_mode():
                logits = self.model(**inputs).logits.float()
                probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()

            for row in probs:
                contradiction = row[self.contradiction_id] if self.contradiction_id < len(row) else row[0]
                entailment = row[self.entailment_id] if self.entailment_id < len(row) else row[-1]
                neutral = row[self.neutral_id] if self.neutral_id < len(row) else (
                    1.0 - contradiction - entailment
                )
                results.append(
                    {
                        "contradiction": float(contradiction),
                        "neutral": float(neutral),
                        "entailment": float(entailment),
                    }
                )

        return results





def compute_answer_consistency(
    target_answer: str,
    responses: List[Dict[str, Any]],
    target_idx: int,
    prompt: str,
    equivalence_mode: str,
) -> Tuple[Optional[float], List[Dict[str, Any]]]:
    vals: List[int] = []
    details: List[Dict[str, Any]] = []

    for idx, resp in enumerate(responses):
        if idx == target_idx:
            continue
        ans = get_llm_answer(resp)
        same = code_equivalent(target_answer, ans, prompt, equivalence_mode)
        vals.append(1 if same else 0)
        details.append(
            {
                "response_index": idx,
                "same_answer_semantics": bool(same),
                "sample_answer": strip_code_wrappers(ans),
            }
        )

    if not vals:
        return None, details

    return float(sum(vals) / len(vals)), details


def compute_selfcheck_item(
    record: Dict[str, Any],
    line_idx: int,
    args: argparse.Namespace,
    correct_map: Dict[str, int],
    problem_map: Dict[str, Dict[str, Any]],
    nli: NLIScorer,
) -> Dict[str, Any]:
    qid = get_record_id(record, line_idx)
    responses = get_responses(record)
    question = get_question_from_record_or_problem(record, qid, problem_map)

    if not responses:
        return {
            "id": qid,
            "correct": correct_map.get(qid, None),
            "error": "no_responses",
        }

    target_idx, target_resp = choose_target_response(
        responses=responses,
        prompt=question,
        target_mode=args.target_mode,
        equivalence_mode=args.equivalence_mode,
    )

    target_answer_raw = get_llm_answer(target_resp)
    target_code = normalize_code_for_output(target_answer_raw, question)

    correct = correct_map.get(qid, None)
    if correct is not None:
        correct = int(correct)

    claims = build_claim_units(
        target_response=target_resp,
        prompt=question,
        claim_source=args.claim_source,
        include_answer_claim=args.include_answer_claim,
    )

    sampled_passages: List[Tuple[int, str]] = []
    for idx, resp in enumerate(responses):
        if idx == target_idx:
            continue
        passage = build_sampled_passage(resp, question, args.passage_mode)
        if passage.strip():
            sampled_passages.append((idx, passage.strip()))

    if args.max_sampled_passages is not None:
        sampled_passages = sampled_passages[:args.max_sampled_passages]

    pair_premises: List[str] = []
    pair_hypotheses: List[str] = []
    pair_meta: List[Tuple[int, int]] = []

    
    for claim_i, claim in enumerate(claims):
        for sample_idx, passage in sampled_passages:
            pair_premises.append(passage)
            pair_hypotheses.append(claim)
            pair_meta.append((claim_i, sample_idx))

    pair_scores = nli.score_pairs(pair_premises, pair_hypotheses) if pair_premises else []

    
    if pair_scores:
        contradiction_values = [s["contradiction"] for s in pair_scores]
        nli_uncertainty = float(sum(contradiction_values) / len(contradiction_values))
    else:
        nli_uncertainty = None

    
    per_claim: List[Dict[str, Any]] = []
    if pair_scores:
        tmp: Dict[int, List[float]] = {}
        for (claim_i, sample_idx), score in zip(pair_meta, pair_scores):
            tmp.setdefault(claim_i, []).append(score["contradiction"])
        for i, claim in enumerate(claims):
            vals = tmp.get(i, [])
            per_claim.append(
                {
                    "claim_index": i,
                    "claim": claim,
                    "mean_contradiction": float(sum(vals) / len(vals)) if vals else None,
                    "num_sampled_passages": len(vals),
                }
            )

    answer_consistency, answer_details = compute_answer_consistency(
        target_answer=target_answer_raw,
        responses=responses,
        target_idx=target_idx,
        prompt=question,
        equivalence_mode=args.equivalence_mode,
    )

    answer_disagreement = None
    if answer_consistency is not None:
        answer_disagreement = float(1.0 - answer_consistency)

    if nli_uncertainty is None:
        combined_uncertainty = answer_disagreement
    elif answer_disagreement is None or args.answer_weight <= 0:
        combined_uncertainty = nli_uncertainty
    else:
        w = float(args.answer_weight)
        combined_uncertainty = (1.0 - w) * nli_uncertainty + w * answer_disagreement

    if combined_uncertainty is None:
        confidence = None
    else:
        combined_uncertainty = max(0.0, min(1.0, float(combined_uncertainty)))
        confidence = float(1.0 - combined_uncertainty)

    result: Dict[str, Any] = {
        "id": qid,
        "correct": correct,

        "target_response_index": target_idx,
        "target_answer": strip_code_wrappers(target_answer_raw),

        "selfcheck_confidence": confidence,
        "selfcheck_uncertainty": combined_uncertainty,
        "uncertainty": combined_uncertainty,

        "selfcheck_nli_uncertainty": nli_uncertainty,
        "answer_consistency_with_target": answer_consistency,
        "answer_disagreement_uncertainty": answer_disagreement,

        "num_responses": len(responses),
        "num_sampled_passages_used": len(sampled_passages),
        "num_claim_units": len(claims),
        "num_nli_pairs": len(pair_scores),

        "meta": {
            "method": "SelfCheckGPT_NLI",
            "dataset": "humaneval_reasoning_sampling",
            "correctness_source": "external_correct_file_only_by_question_id",
            "question_source": "input_question_or_problem_file_by_question_id",
            "ignored_field": "correct answer",
            "nli_model": args.nli_model,
            "claim_source": args.claim_source,
            "passage_mode": args.passage_mode,
            "include_answer_claim": bool(args.include_answer_claim),
            "answer_weight": args.answer_weight,
            "target_mode": args.target_mode,
            "equivalence_mode": args.equivalence_mode,
        },
    }

    if args.keep_question:
        result["question"] = question
    if args.keep_prediction:
        result["prediction"] = target_code
    if args.keep_reasoning:
        result["target_reasoning"] = get_llm_reason(target_resp)
    if args.keep_claims:
        result["claim_units"] = claims
    if args.keep_per_claim:
        result["per_claim_selfcheck"] = per_claim
    if args.keep_pair_details:
        details = []
        for (claim_i, sample_idx), score in zip(pair_meta, pair_scores):
            details.append(
                {
                    "claim_index": claim_i,
                    "sample_response_index": sample_idx,
                    **score,
                }
            )
        result["nli_pair_details"] = details
    if args.keep_answer_details:
        result["answer_consistency_details"] = answer_details

    return result





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SelfCheckGPT uncertainty for HumanEval reasoning-sampling data."
    )

    parser.add_argument("--input-file", required=True, help="Input HumanEval reasoning JSONL/JSON file")
    parser.add_argument("--correct-file", required=True, help="External correct JSONL/JSON file")
    parser.add_argument("--output-file", required=True, help="Output JSONL file")
    parser.add_argument(
        "--problem-file",
        default=None,
        help="Optional HumanEval question file with question_id/question/ground_truth.",
    )

    parser.add_argument(
        "--nli-model",
        default="microsoft/deberta-large-mnli",
        help="NLI model path/name. Examples: microsoft/deberta-large-mnli, potsawee/deberta-v3-large-mnli",
    )
    parser.add_argument("--hf-endpoint", default=None, help="Optional HF mirror, e.g. https://hf-mirror.com")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--nli-max-length", type=int, default=512)
    parser.add_argument("--nli-batch-size", type=int, default=8)

    parser.add_argument(
        "--target-mode",
        choices=["first", "majority"],
        default="first",
        help="Target response to evaluate.",
    )
    parser.add_argument(
        "--equivalence-mode",
        choices=["ast", "text", "ast_or_text"],
        default="ast_or_text",
        help="Code equivalence mode for target selection and auxiliary answer consistency.",
    )

    parser.add_argument(
        "--claim-source",
        choices=["auto", "steps", "sentences", "full"],
        default="auto",
        help="How to build target claim units.",
    )
    parser.add_argument(
        "--passage-mode",
        choices=["all", "reasoning_augmented", "answer"],
        default="all",
        help="What to use as sampled passage for NLI.",
    )
    parser.add_argument(
        "--include-answer-claim",
        action="store_true",
        help="Also include target final code answer as a claim unit. Default: off.",
    )
    parser.add_argument(
        "--max-sampled-passages",
        type=int,
        default=None,
        help="Optionally limit number of sampled passages per item.",
    )
    parser.add_argument(
        "--answer-weight",
        type=float,
        default=0.0,
        help=(
            "Optional weight for answer disagreement in final uncertainty. "
            "Default 0.0 means pure NLI SelfCheckGPT."
        ),
    )

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--keep-question", action="store_true")
    parser.add_argument("--keep-prediction", action="store_true")
    parser.add_argument("--keep-reasoning", action="store_true")
    parser.add_argument("--keep-claims", action="store_true")
    parser.add_argument("--keep-per-claim", action="store_true")
    parser.add_argument("--keep-pair-details", action="store_true")
    parser.add_argument("--keep-answer-details", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    safe_print(f"INPUT_FILE = {args.input_file}")
    safe_print(f"CORRECT_FILE = {args.correct_file}")
    safe_print(f"PROBLEM_FILE = {args.problem_file}")
    safe_print(f"OUTPUT_FILE = {args.output_file}")
    safe_print("Correctness source: external correct file only, matched by input id -> correct_file.question_id")
    safe_print("Question source: input question/prompt first; otherwise --problem-file matched by input id -> problem_file.question_id")
    safe_print("Ignored field: record['correct answer']")
    safe_print(f"NLI model = {args.nli_model}")
    safe_print(f"Claim source = {args.claim_source}")
    safe_print(f"Passage mode = {args.passage_mode}")
    safe_print(f"Answer weight = {args.answer_weight}")

    correct_map = load_correct_map(args.correct_file)
    problem_map = load_problem_map(args.problem_file)
    records = load_records(args.input_file)

    if not correct_map:
        raise ValueError("No correctness labels loaded from correct file.")

    safe_print(f"Loaded records: {len(records)}")
    safe_print(f"Loaded correctness labels: {len(correct_map)}")
    if problem_map:
        safe_print(f"Loaded problem prompts: {len(problem_map)}")

    safe_print("Loading NLI model...")
    nli = NLIScorer(
        model_name_or_path=args.nli_model,
        device=args.device,
        dtype=args.dtype,
        max_length=args.nli_max_length,
        batch_size=args.nli_batch_size,
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

            qid = get_record_id(record, line_idx)
            if qid in done_ids:
                continue

            try:
                result = compute_selfcheck_item(
                    record=record,
                    line_idx=line_idx,
                    args=args,
                    correct_map=correct_map,
                    problem_map=problem_map,
                    nli=nli,
                )
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                processed += 1

                if args.verbose:
                    safe_print(
                        f"done: {qid} | line={line_idx} | "
                        f"correct={result.get('correct')} | "
                        f"nli_unc={result.get('selfcheck_nli_uncertainty')} | "
                        f"ans_cons={result.get('answer_consistency_with_target')} | "
                        f"conf={result.get('selfcheck_confidence')} | "
                        f"unc={result.get('selfcheck_uncertainty')}"
                    )

            except KeyboardInterrupt:
                safe_print("KeyboardInterrupt detected. Exiting.")
                raise
            except Exception as e:
                err = {"id": qid, "error": str(e)}
                fout.write(json.dumps(err, ensure_ascii=False) + "\n")
                fout.flush()
                if args.verbose:
                    safe_print(f"error: {qid} | line={line_idx} | {e}")

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

    _parser = _ap.ArgumentParser(description="selfcheckgpt / llama / humaneval / reasoning_augmented (standalone)")
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
                           "dataset": "humaneval", "setting": "reasoning_augmented", "model_loading": False}, indent=2))
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

