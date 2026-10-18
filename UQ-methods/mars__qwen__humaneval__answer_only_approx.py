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






write_lock = threading.Lock()
print_lock = threading.Lock()


def safe_print(*args, **kwargs) -> None:
    with print_lock:
        print(*args, **kwargs, flush=True)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception as e:
                print(f"[warn] skip bad json line {line_no}: {e}", file=sys.stderr)
    return records


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with write_lock:
        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f.flush()


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


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


def build_correct_map(records: List[Dict[str, Any]], question_id_key: str, correct_key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for rec in records:
        qid = rec.get(question_id_key)
        if qid is None and question_id_key != "id":
            qid = rec.get("id")
        if qid is None:
            qid = rec.get("question_id")
        if qid is None:
            continue
        try:
            correct = int(rec.get(correct_key, 0))
        except Exception:
            correct = 0
        out[str(qid)] = 1 if correct else 0
    return out


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







def get_question_id(obj: Dict[str, Any]) -> str:
    for key in ["id", "question_id", "qid"]:
        if key in obj and obj[key] is not None:
            return str(obj[key])
    return "unknown_id"


def get_question(obj: Dict[str, Any]) -> str:
    return str(obj.get("question", ""))


def get_entry_point(obj: Dict[str, Any]) -> str:
    gt = obj.get("gt", {})
    if isinstance(gt, dict) and gt.get("entry_point") is not None:
        return str(gt.get("entry_point"))
    if obj.get("entry_point") is not None:
        return str(obj.get("entry_point"))
    return ""


def extract_text_field(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        for key in ["text", "raw_text", "answer", "response", "completion", "code"]:
            if key in x and x[key] is not None:
                t = str(x[key]).strip()
                if t:
                    return t
    return str(x).strip()


def get_greedy_answer(obj: Dict[str, Any]) -> str:
    if obj.get("greedy_answer") is not None:
        return str(obj["greedy_answer"]).strip()
    greedy = obj.get("greedy")
    if greedy is not None:
        return extract_text_field(greedy)
    return ""


def get_greedy_steps(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    greedy = obj.get("greedy")
    if isinstance(greedy, dict):
        for key in ["steps", "raw_steps", "tokens"]:
            steps = greedy.get(key)
            if isinstance(steps, list):
                return steps
    return []


_CODE_TAG_RE = re.compile(r"<CODE>(.*?)</CODE>", flags=re.IGNORECASE | re.DOTALL)
_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)


def clean_code_answer(text: str) -> str:
    
    text = "" if text is None else str(text).strip()
    if not text:
        return ""

    m = _CODE_TAG_RE.search(text)
    if m:
        text = m.group(1).strip()

    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()

    
    text = re.sub(r"</?CODE>", "", text, flags=re.IGNORECASE).strip()

    
    return text


def get_function_header(question: str, entry_point: str = "") -> str:
    
    if entry_point:
        pat = re.compile(rf"^\s*def\s+{re.escape(entry_point)}\s*\([^\n]*\)\s*(?:->\s*[^:]+)?\s*:", re.M)
        m = pat.search(question)
        if m:
            return m.group(0).strip()
    m = re.search(r"^\s*def\s+\w+\s*\([^\n]*\)\s*(?:->\s*[^:]+)?\s*:", question, flags=re.M)
    return m.group(0).strip() if m else ""


def format_candidate_for_judge(question: str, answer: str, entry_point: str = "") -> str:
    
    answer = clean_code_answer(answer)
    if not answer:
        return ""

    if re.search(r"^\s*def\s+\w+\s*\(", answer, flags=re.M):
        return answer

    header = get_function_header(question, entry_point)
    if not header:
        return answer

    lines = answer.splitlines() or [answer]
    indented = []
    for line in lines:
        if line.strip() == "":
            indented.append("")
        elif line.startswith("    ") or line.startswith("\t"):
            indented.append(line)
        else:
            indented.append("    " + line)
    return header + "\n" + "\n".join(indented)






class LocalCodeJudge:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "auto",
        judge_family: str = "auto",
        use_p_a_only: bool = False,
    ) -> None:
        self.model_path = model_path
        self.judge_family = judge_family.lower()
        self.use_p_a_only = use_p_a_only

        torch_dtype = self._resolve_dtype(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model_kwargs = dict(trust_remote_code=True)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if device == "cuda" and torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if not (device == "cuda" and torch.cuda.is_available()):
            self.model.to("cpu")
        self.model.eval()

        self.A_ID = self._get_single_token_id("A")
        self.B_ID = self._get_single_token_id("B")
        self.C_ID = self._get_single_token_id("C")
        if self.A_ID is None or self.B_ID is None or self.C_ID is None:
            raise ValueError("Cannot map A/B/C to single tokens in judge tokenizer.")

    def _resolve_dtype(self, dtype: str):
        dtype = (dtype or "auto").lower()
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

    def _get_single_token_id(self, token_str: str) -> Optional[int]:
        candidates = [token_str, " " + token_str, "\n" + token_str]
        for cand in candidates:
            ids = self.tokenizer.encode(cand, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        return None

    def _build_messages(self, question: str, candidate_code: str, entry_point: str = "") -> List[Dict[str, str]]:
        ep_line = f"\nTarget function name: {entry_point}" if entry_point else ""
        return [
            {
                "role": "system",
                "content": (
                    "You are a strict judge for Python HumanEval programming tasks. "
                    "Decide whether the proposed code correctly solves the task. "
                    "Consider edge cases and the docstring specification."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task prompt:\n{question}{ep_line}\n\n"
                    f"Proposed code:\n```python\n{candidate_code}\n```\n\n"
                    "Choose exactly one option:\n"
                    "A. Correct\n"
                    "B. Incorrect\n"
                    "C. Not sure\n\n"
                    "Answer with only one letter."
                ),
            },
        ]

    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        
        if self.judge_family in {"qwen", "auto"}:
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                pass
            except Exception:
                if self.judge_family == "qwen":
                    raise

        if self.judge_family in {"llama", "auto", "chat"}:
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        chunks = []
        for m in messages:
            role = str(m.get("role", "user")).upper()
            content = str(m.get("content", ""))
            chunks.append(f"{role}: {content}")
        chunks.append("ASSISTANT:")
        return "\n\n".join(chunks)

    @torch.no_grad()
    def score(self, question: str, candidate_code: str, entry_point: str = "") -> Dict[str, Any]:
        messages = self._build_messages(question, candidate_code, entry_point)
        prompt = self._format_prompt(messages)
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
        score = p_a if self.use_p_a_only else (p_a + 0.5 * p_c)

        return {
            "p_a": p_a,
            "p_b": p_b,
            "p_c": p_c,
            "score": float(score),
            "prompt_len": int(inputs["input_ids"].shape[1]),
        }






SPECIAL_TOKEN_PATTERNS = (
    re.compile(r"^<\|.*\|>$"),
    re.compile(r"^</?think>$"),
    re.compile(r"^<tool_.*>$"),
    re.compile(r"^<\|endoftext\|>$"),
)


def is_special_token_text(token_text: str) -> bool:
    if not token_text:
        return False
    return any(p.match(token_text) for p in SPECIAL_TOKEN_PATTERNS)


def extract_token_and_logprob(step: Dict[str, Any]) -> Dict[str, Any]:
    text_piece = ""
    logprob: Optional[float] = None

    for k in [
        "token_text",
        "text_piece",
        "piece",
        "text",
        "token",
        "generated_token",
    ]:
        if k in step and step[k] is not None:
            text_piece = str(step[k])
            break

    for k in ["chosen_logprob", "logprob", "token_logprob", "selected_logprob"]:
        if k in step and step[k] is not None:
            v = safe_float(step[k], None)
            if v is not None:
                logprob = v
                break

    if logprob is None:
        for k in ["chosen_prob", "prob", "token_prob", "selected_prob"]:
            if k in step and step[k] is not None:
                p = safe_float(step[k], None)
                if p is not None and p > 0:
                    logprob = math.log(max(p, 1e-12))
                    break

    return {"text_piece": text_piece, "logprob": logprob}


def rebuild_answer_from_pieces(pieces: List[str]) -> str:
    
    text = "".join(pieces)
    return clean_code_answer(text)


def normalize_importance(raw_importances: List[float], seq_len: int) -> List[float]:
    if seq_len <= 0:
        return []
    s = sum(raw_importances)
    if s <= 0:
        u = [1.0 / seq_len] * seq_len
    else:
        u = [x / s for x in raw_importances]
    
    return [(1.0 / (2.0 * seq_len)) + (ui / 2.0) for ui in u]


def compute_mars_approx(
    obj: Dict[str, Any],
    correct_map: Dict[str, int],
    judge: LocalCodeJudge,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    qid = get_question_id(obj)
    question = get_question(obj)
    entry_point = get_entry_point(obj)
    greedy_answer_raw = get_greedy_answer(obj)
    greedy_answer_clean = clean_code_answer(greedy_answer_raw)
    steps = get_greedy_steps(obj)

    correct = correct_map.get(qid, 0)

    parsed_steps = [extract_token_and_logprob(s) for s in steps]
    parsed_steps = [
        x for x in parsed_steps
        if x["logprob"] is not None
        and str(x["text_piece"]) != ""
        and not is_special_token_text(str(x["text_piece"]))
    ]

    
    if not parsed_steps:
        avg_lp = None
        greedy = obj.get("greedy")
        if isinstance(greedy, dict):
            for key in ["avg_logprob", "avg_logprob_answer_only", "mean_logprob"]:
                avg_lp = safe_float(greedy.get(key), None)
                if avg_lp is not None:
                    break

        result: Dict[str, Any] = {
            "id": qid,
            "correct": int(correct),
            "mars_score": None,
            "mars_uncertainty": None,
            "uncertainty": None,
            "judge_original_score": None,
            "num_tokens_used": 0,
            "judge_calls": 0,
            "meta": {
                "method": "MARS-approx-ma[REMOVED_CREDENTIAL]",
                "dataset": "humaneval",
                "judge_model": args.judge_model_path,
                "fallback": "no_valid_steps_and_no_avg_logprob",
                "correctness_source": "external_correct_file_only",
            },
            "token_debug": [],
        }
        if avg_lp is not None:
            result["mars_score"] = float(avg_lp)
            result["mars_uncertainty"] = float(-avg_lp)
            result["uncertainty"] = float(-avg_lp)
            result["meta"]["fallback"] = "avg_logprob_only"
        if args.keep_question:
            result["question"] = question
        if args.keep_prediction:
            result["greedy_answer"] = greedy_answer_raw
            result["candidate_code_for_judge"] = format_candidate_for_judge(question, greedy_answer_raw, entry_point)
        return result

    pieces = [str(x["text_piece"]) for x in parsed_steps]
    logprobs = [float(x["logprob"]) for x in parsed_steps]

    if args.max_mask_tokens is not None and len(pieces) > args.max_mask_tokens:
        if args.token_selection == "head":
            selected_indices = list(range(args.max_mask_tokens))
        elif args.token_selection == "tail":
            selected_indices = list(range(len(pieces) - args.max_mask_tokens, len(pieces)))
        else:
            
            selected_indices = sorted(range(len(logprobs)), key=lambda i: logprobs[i])[:args.max_mask_tokens]
            selected_indices = sorted(selected_indices)
        pieces = [pieces[i] for i in selected_indices]
        logprobs = [logprobs[i] for i in selected_indices]
    else:
        selected_indices = list(range(len(pieces)))

    answer_text_from_steps = rebuild_answer_from_pieces(pieces)
    if not answer_text_from_steps.strip():
        answer_text_from_steps = greedy_answer_clean

    candidate_code = format_candidate_for_judge(question, answer_text_from_steps, entry_point)
    if not candidate_code.strip():
        candidate_code = format_candidate_for_judge(question, greedy_answer_clean, entry_point)

    judge_calls = 0
    orig_judge = judge.score(question, candidate_code, entry_point)
    p_orig = float(orig_judge["score"])
    judge_calls += 1

    raw_importances: List[float] = []
    token_debug: List[Dict[str, Any]] = []

    for local_i in range(len(pieces)):
        masked_pieces = pieces.copy()
        masked_pieces[local_i] = args.mask_token
        masked_answer = rebuild_answer_from_pieces(masked_pieces)
        masked_candidate_code = format_candidate_for_judge(question, masked_answer, entry_point)

        masked_judge = judge.score(question, masked_candidate_code, entry_point)
        p_mask = float(masked_judge["score"])
        judge_calls += 1

        importance = max(0.0, p_orig - p_mask)
        if importance < args.min_importance:
            importance = args.min_importance

        raw_importances.append(importance)
        token_debug.append(
            {
                "token_index": int(selected_indices[local_i]),
                "local_token_index": int(local_i),
                "text_piece": pieces[local_i],
                "logprob": logprobs[local_i],
                "masked_answer": masked_answer,
                "masked_candidate_code": masked_candidate_code if args.keep_masked_code else None,
                "judge_original_score": p_orig,
                "judge_masked_score": p_mask,
                "raw_importance": importance,
                "judge_probs_original": orig_judge,
                "judge_probs_masked": masked_judge,
            }
        )

    if sum(raw_importances) <= 0:
        raw_importances = [1.0] * len(pieces)

    weights = normalize_importance(raw_importances, len(pieces))
    for i in range(len(token_debug)):
        token_debug[i]["weight"] = weights[i]

    mars_score = sum(w * lp for w, lp in zip(weights, logprobs))
    mars_uncertainty = float(-mars_score)

    result = {
        "id": qid,
        "correct": int(correct),
        "mars_score": float(mars_score),
        "mars_uncertainty": mars_uncertainty,
        "uncertainty": mars_uncertainty,
        "judge_original_score": p_orig,
        "num_tokens_used": len(pieces),
        "num_original_steps": len(parsed_steps),
        "judge_calls": judge_calls,
        "meta": {
            "method": "MARS-approx-ma[REMOVED_CREDENTIAL]",
            "dataset": "humaneval",
            "judge_model": args.judge_model_path,
            "judge_family": args.judge_family,
            "mask_token": args.mask_token,
            "max_mask_tokens": args.max_mask_tokens,
            "token_selection": args.token_selection,
            "use_p_a_only": args.use_p_a_only,
            "correctness_source": "external_correct_file_only",
        },
        "token_debug": token_debug,
    }

    if args.keep_question:
        result["question"] = question
    if args.keep_prediction:
        result["greedy_answer"] = greedy_answer_raw
        result["candidate_code_for_judge"] = candidate_code

    return result







def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute MARS-style uncertainty for HumanEval.")

    parser.add_argument("--input-file", "--input_file", dest="input_file", required=True)
    parser.add_argument("--correct-file", "--correct_file", dest="correct_file", required=True)
    parser.add_argument("--output-file", "--output_file", dest="output_file", required=True)
    parser.add_argument("--judge-model-path", "--model-path", dest="judge_model_path", required=True)

    parser.add_argument("--question-id-key", default="question_id")
    parser.add_argument("--correct-key", default="correct")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--judge-family", default="auto", choices=["auto", "qwen", "llama", "chat", "plain"])

    parser.add_argument("--mask-token", default="[MASK]")
    parser.add_argument("--max-mask-tokens", type=int, default=12)
    parser.add_argument("--token-selection", default="head", choices=["head", "tail", "lowest_prob"])
    parser.add_argument("--use-p-a-only", action="store_true")
    parser.add_argument("--min-importance", type=float, default=1e-8)

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-question", action="store_true")
    parser.add_argument("--keep-prediction", action="store_true")
    parser.add_argument("--keep-masked-code", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--hf-endpoint", default=None, help="Optional HF_ENDPOINT, e.g. https://hf-mirror.com")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    if args.overwrite and os.path.exists(args.output_file):
        os.remove(args.output_file)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    correct_records = load_jsonl(args.correct_file)
    correct_map = build_correct_map(correct_records, args.question_id_key, args.correct_key)

    safe_print(f"INPUT_FILE = {args.input_file}")
    safe_print(f"CORRECT_FILE = {args.correct_file}")
    safe_print(f"OUTPUT_FILE = {args.output_file}")
    safe_print("Correctness source: external correct file only")
    safe_print(f"Judge model: {args.judge_model_path}")
    safe_print(f"Judge family: {args.judge_family}")
    safe_print(f"Mask token: {args.mask_token}")
    safe_print(f"Max mask tokens: {args.max_mask_tokens}")

    safe_print("Loading local judge model...", flush=True)
    judge = LocalCodeJudge(
        model_path=args.judge_model_path,
        device=args.device,
        dtype=args.dtype,
        judge_family=args.judge_family,
        use_p_a_only=args.use_p_a_only,
    )

    done = load_done_ids(args.output_file)
    if args.verbose:
        safe_print(f"existing records: {len(done)}")

    processed = 0
    missing_correct = 0
    with open(args.input_file, "r", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin, start=1):
            if args.limit is not None and processed >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                safe_print(f"skip line {line_idx}: json parse error: {e}")
                continue

            qid = get_question_id(obj)
            if qid in done:
                continue
            if qid not in correct_map:
                missing_correct += 1

            try:
                result = compute_mars_approx(obj, correct_map, judge, args)
                append_jsonl(args.output_file, result)
                processed += 1

                if args.verbose:
                    unc = result.get("mars_uncertainty")
                    unc_str = "None" if unc is None else f"{float(unc):.6f}"
                    safe_print(
                        f"done: {qid} | line={line_idx} | "
                        f"correct={result.get('correct')} | "
                        f"num_tokens={result.get('num_tokens_used')} | "
                        f"judge_calls={result.get('judge_calls')} | "
                        f"unc={unc_str}"
                    )
            except KeyboardInterrupt:
                safe_print("KeyboardInterrupt detected. Exiting gracefully.")
                raise
            except Exception as e:
                err = {
                    "id": qid,
                    "correct": correct_map.get(qid, 0),
                    "error": str(e),
                    "meta": {
                        "method": "MARS-approx-ma[REMOVED_CREDENTIAL]",
                        "dataset": "humaneval",
                        "correctness_source": "external_correct_file_only",
                    },
                }
                if args.keep_question:
                    err["question"] = obj.get("question", "")
                append_jsonl(args.output_file, err)
                safe_print(f"error: {qid} | line={line_idx} | {e}")

    safe_print(f"Finished. processed={processed}")
    safe_print(f"Saved to: {args.output_file}")
    if missing_correct > 0:
        safe_print(f"[warn] {missing_correct} records were missing in correct file and set to correct=0")


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

    _parser = _ap.ArgumentParser(description="mars / qwen / humaneval / answer_only (standalone)")
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
    if "mars" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "mars", "family": "qwen",
                           "dataset": "humaneval", "setting": "answer_only", "model_loading": False}, indent=2))
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

