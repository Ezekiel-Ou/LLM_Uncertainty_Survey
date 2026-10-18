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
import sys
import textwrap
import threading
from typing import Any, Dict, List, Optional, Tuple


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
        reps: List[str] = []
        counts: Dict[int, int] = {}
        first_idx: Dict[int, int] = {}

        for idx, resp in enumerate(responses):
            ans = get_llm_answer(resp)
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
        idx = first_idx[best_cid]
        return idx, responses[idx]

    raise ValueError(f"Unsupported target_mode: {target_mode}")





class LocalCandidateScorer:
    def __init__(self, model_path: str, model_family: str, dtype: str = "auto", device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_family = model_family
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

    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        if self.model_family == "qwen":
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            parts = []
            for m in messages:
                parts.append(f"{m['role'].upper()}: {m['content']}")
            parts.append("ASSISTANT:")
            return "\n\n".join(parts)

    def sequence_logprob(self, prompt: str, candidate: str) -> float:
        import torch

        prefix_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        
        variants = [
            candidate,
            " " + candidate,
            candidate + ".",
            " " + candidate + ".",
        ]

        best = -float("inf")
        for cand in variants:
            suffix_ids = self.tokenizer(cand, add_special_tokens=False)["input_ids"]
            if not suffix_ids:
                continue

            full_ids = prefix_ids + suffix_ids
            input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.model.device)

            with torch.inference_mode():
                logits = self.model(input_ids=input_ids).logits[:, :-1, :].float()
                logprobs = torch.log_softmax(logits, dim=-1)

            total = 0.0
            start = len(prefix_ids)
            valid = True

            for pos, tok_id in enumerate(suffix_ids):
                lp_pos = start + pos - 1
                if lp_pos < 0 or lp_pos >= logprobs.shape[1]:
                    valid = False
                    break
                total += float(logprobs[0, lp_pos, tok_id].item())

            if valid:
                best = max(best, total)

        return best





def build_ptrue_prompt(
    scorer: LocalCandidateScorer,
    question: str,
    reasoning: str,
    code: str,
    prompt_style: str,
) -> str:
    if question.strip():
        task_part = f"HumanEval Task:\n{question.strip()}\n"
    else:
        task_part = (
            "HumanEval Task:\n"
            "[The original prompt is missing from the input file. Judge only from the proposed code and reasoning.]\n"
        )

    if prompt_style == "direct":
        ask = "Is the proposed Python solution correct for this HumanEval task?"
    else:
        ask = "Will the proposed Python solution pass all hidden HumanEval tests?"

    user_content = (
        f"{task_part}\n"
        f"Model reasoning:\n{reasoning.strip() if reasoning else '[empty]'}\n\n"
        f"Proposed Python solution:\n```python\n{code.strip()}\n```\n\n"
        f"{ask}\n"
        "Answer exactly one word: True or False."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict Python programming judge for HumanEval tasks. "
                "Evaluate whether the proposed solution is correct. "
                "Do not execute code. Answer only True or False."
            ),
        },
        {"role": "user", "content": user_content},
    ]

    return scorer.apply_chat_template(messages)


def compute_ptrue(
    scorer: LocalCandidateScorer,
    question: str,
    reasoning: str,
    code: str,
    prompt_style: str,
) -> Dict[str, Any]:
    prompt = build_ptrue_prompt(
        scorer=scorer,
        question=question,
        reasoning=reasoning,
        code=code,
        prompt_style=prompt_style,
    )

    logp_true = scorer.sequence_logprob(prompt, "True")
    logp_false = scorer.sequence_logprob(prompt, "False")

    max_log = max(logp_true, logp_false)
    true_exp = math.exp(logp_true - max_log)
    false_exp = math.exp(logp_false - max_log)
    denom = true_exp + false_exp

    if denom <= 0:
        p_true = 0.5
        p_false = 0.5
    else:
        p_true = float(true_exp / denom)
        p_false = float(false_exp / denom)

    p_true = max(0.0, min(1.0, p_true))
    p_false = max(0.0, min(1.0, p_false))

    return {
        "p_true": p_true,
        "p_false": p_false,
        "p_true_confidence": p_true,
        "p_true_uncertainty": float(1.0 - p_true),
        "uncertainty": float(1.0 - p_true),
        "true_logprob": float(logp_true),
        "false_logprob": float(logp_false),
        "prompt_len": len(scorer.tokenizer(prompt, add_special_tokens=False)["input_ids"]),
    }





def process_item(
    record: Dict[str, Any],
    line_idx: int,
    args: argparse.Namespace,
    correct_map: Dict[str, int],
    problem_map: Dict[str, Dict[str, Any]],
    scorer: LocalCandidateScorer,
) -> Dict[str, Any]:
    qid = get_record_id(record, line_idx)
    question = get_question_from_record_or_problem(record, qid, problem_map)
    responses = get_responses(record)

    correct = correct_map.get(qid, None)
    if correct is not None:
        correct = int(correct)

    if not responses:
        return {
            "id": qid,
            "correct": correct,
            "error": "no_responses",
        }

    target_idx, target_resp = choose_target_response(
        responses=responses,
        prompt=question,
        target_mode=args.target_mode,
        equivalence_mode=args.equivalence_mode,
    )

    target_answer = get_llm_answer(target_resp)
    target_reasoning = get_llm_reason(target_resp)
    target_code = normalize_code_for_output(target_answer, question)

    ptrue = compute_ptrue(
        scorer=scorer,
        question=question,
        reasoning=target_reasoning,
        code=target_code,
        prompt_style=args.prompt_style,
    )

    result: Dict[str, Any] = {
        "id": qid,
        "correct": correct,

        "target_response_index": target_idx,
        "target_answer": strip_code_wrappers(target_answer),

        "p_true": ptrue["p_true"],
        "p_false": ptrue["p_false"],
        "p_true_confidence": ptrue["p_true_confidence"],
        "p_true_uncertainty": ptrue["p_true_uncertainty"],
        "uncertainty": ptrue["uncertainty"],

        "num_responses": len(responses),

        "meta": {
            "method": "P(True)",
            "dataset": "humaneval_reasoning_sampling",
            "correctness_source": "external_correct_file_only_by_question_id",
            "question_source": "input_question_or_problem_file_by_question_id",
            "ignored_field": "correct answer",
            "model_path": args.model_path,
            "model_family": args.model_family,
            "target_mode": args.target_mode,
            "equivalence_mode": args.equivalence_mode if args.target_mode == "majority" else None,
            "prompt_style": args.prompt_style,
            "scoring": "candidate_sequence_logprob_softmax_over_True_False",
        },
    }

    if args.keep_candidate_logprobs:
        result["candidate_logprobs"] = {
            "True": ptrue["true_logprob"],
            "False": ptrue["false_logprob"],
        }
    if args.keep_prompt_len:
        result["prompt_len"] = ptrue["prompt_len"]
    if args.keep_question:
        result["question"] = question
    if args.keep_prediction:
        result["prediction"] = target_code
    if args.keep_target_reasoning:
        result["target_reasoning"] = target_reasoning

    return result





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute P(True) uncertainty for HumanEval reasoning-sampling data."
    )

    parser.add_argument("--input-file", required=True, help="Input HumanEval reasoning JSONL/JSON file")
    parser.add_argument("--correct-file", required=True, help="External correct JSONL/JSON file")
    parser.add_argument("--output-file", required=True, help="Output JSONL file")
    parser.add_argument(
        "--problem-file",
        default=None,
        help="Optional HumanEval question file with question_id/question/ground_truth.",
    )

    parser.add_argument("--model-path", required=True, help="Local model path used for True/False scoring")
    parser.add_argument(
        "--model-family",
        choices=["qwen", "llama", "auto"],
        default="auto",
        help="Model family. qwen disables thinking when possible.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--hf-endpoint", default=None)

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
        help="Used only when --target-mode majority.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["direct", "hidden_tests"],
        default="hidden_tests",
        help="Prompt wording for P(True).",
    )

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--keep-question", action="store_true")
    parser.add_argument("--keep-prediction", action="store_true")
    parser.add_argument("--keep-target-reasoning", action="store_true")
    parser.add_argument("--keep-candidate-logprobs", action="store_true")
    parser.add_argument("--keep-prompt-len", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    if args.model_family == "auto":
        low = args.model_path.lower()
        if "qwen" in low:
            args.model_family = "qwen"
        elif "llama" in low:
            args.model_family = "llama"
        else:
            args.model_family = "auto"

    safe_print(f"INPUT_FILE = {args.input_file}")
    safe_print(f"CORRECT_FILE = {args.correct_file}")
    safe_print(f"PROBLEM_FILE = {args.problem_file}")
    safe_print(f"OUTPUT_FILE = {args.output_file}")
    safe_print("Correctness source: external correct file only, matched by input id -> correct_file.question_id")
    safe_print("Question source: input question/prompt first; otherwise --problem-file matched by input id -> problem_file.question_id")
    safe_print("Ignored field: record['correct answer']")
    safe_print(f"Model path = {args.model_path}")
    safe_print(f"Model family = {args.model_family}")
    safe_print("Scoring = P(True) via candidate sequence logprobs")
    safe_print(f"Target mode = {args.target_mode}")

    records = load_records(args.input_file)
    correct_map = load_correct_map(args.correct_file)
    problem_map = load_problem_map(args.problem_file)

    if not correct_map:
        raise ValueError("No correctness labels loaded from correct file.")

    safe_print(f"Loaded records: {len(records)}")
    safe_print(f"Loaded correctness labels: {len(correct_map)}")
    if problem_map:
        safe_print(f"Loaded problem prompts: {len(problem_map)}")

    safe_print("Loading local model...")
    scorer = LocalCandidateScorer(
        model_path=args.model_path,
        model_family=args.model_family,
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

            qid = get_record_id(record, line_idx)
            if qid in done_ids:
                continue

            try:
                result = process_item(
                    record=record,
                    line_idx=line_idx,
                    args=args,
                    correct_map=correct_map,
                    problem_map=problem_map,
                    scorer=scorer,
                )
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                processed += 1

                if args.verbose:
                    safe_print(
                        f"done: {qid} | line={line_idx} | "
                        f"correct={result.get('correct')} | "
                        f"p_true={result.get('p_true')} | "
                        f"p_false={result.get('p_false')} | "
                        f"unc={result.get('p_true_uncertainty')}"
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

    _parser = _ap.ArgumentParser(description="p_true / llama / humaneval / reasoning_augmented (standalone)")
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
    if "p_true" not in {"max_seq_prob", "mean_tok_ent", "max_tok_prob", "ppl"}:
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
    if "p_true" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "p_true", "family": "llama",
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

