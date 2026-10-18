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


def safe_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    with print_lock:
        print(*args, **kwargs)


def clamp01(x: float) -> float:
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

    out = []
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception as e:
            safe_print(f"[warn] skip invalid json line {line_no}: {e}", file=sys.stderr)
    return out


def load_done_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    done = set()
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
    out = {}
    for obj in load_records(path):
        qid = None
        for k in ["question_id", "id", "task_id", "qid"]:
            if obj.get(k) is not None:
                qid = str(obj[k])
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
    out = {}
    for obj in load_records(path):
        qid = None
        for k in ["question_id", "task_id", "id", "qid"]:
            if obj.get(k) is not None:
                qid = str(obj[k])
                break
        if qid is not None:
            out[qid] = obj
    return out





def get_record_id(record: Dict[str, Any], line_idx: int) -> str:
    for k in ["id", "question_id", "task_id", "qid"]:
        if record.get(k) is not None:
            return str(record[k])
    return f"line_{line_idx}"


def get_responses(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    rs = record.get("responses", [])
    return [r for r in rs if isinstance(r, dict)] if isinstance(rs, list) else []


def get_llm_reason(resp: Dict[str, Any]) -> str:
    for k in ["llm reason", "llm_reason", "reason", "rationale", "solution"]:
        if resp.get(k) is not None:
            return str(resp[k]).strip()
    return ""


def get_llm_answer(resp: Dict[str, Any]) -> str:
    for k in ["llm answer", "llm_answer", "answer", "text", "raw_text", "prediction"]:
        if resp.get(k) is not None:
            return str(resp[k]).strip()
    return ""


def get_question(record: Dict[str, Any], qid: str, problem_map: Dict[str, Dict[str, Any]]) -> str:
    for k in ["question", "prompt", "problem", "input", "query"]:
        if record.get(k) is not None:
            return str(record[k])
    p = problem_map.get(qid, {})
    for k in ["question", "prompt", "problem", "input", "query"]:
        if p.get(k) is not None:
            return str(p[k])
    return ""





def extract_boxed_content(text: str) -> str:
    s = str(text)
    idx = s.find(r"\boxed")
    if idx < 0:
        return s
    st = s.find("{", idx + len(r"\boxed"))
    if st < 0:
        return s
    depth = 0
    for pos in range(st, len(s)):
        if s[pos] == "{":
            depth += 1
        elif s[pos] == "}":
            depth -= 1
            if depth == 0:
                return s[st + 1:pos]
    return s[st + 1:]


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
    pat = re.compile(r"^def\s+\w+\s*\(.*?\)\s*(?:->\s*[^:]+)?\s*:", re.MULTILINE)
    ms = list(pat.finditer(prompt))
    return ms[-1].group(0) if ms else None


def looks_like_full_function(code: str) -> bool:
    return bool(re.search(r"^\s*def\s+\w+\s*\(", code, flags=re.MULTILINE))


def wrap_body_with_header(code: str, prompt: str) -> str:
    code = strip_code_wrappers(code)
    if looks_like_full_function(code):
        return textwrap.dedent(code).strip()
    header = find_function_header(prompt)
    if not header:
        return textwrap.dedent(code).strip()
    body = textwrap.dedent(code).strip("\n") or "pass"
    return header.rstrip() + "\n" + textwrap.indent(body, "    ")


def normalize_text_code(code: str) -> str:
    s = textwrap.dedent(strip_code_wrappers(code)).strip()
    lines = [ln.rstrip() for ln in s.splitlines() if ln.strip()]
    return re.sub(r"\s+", " ", "\n".join(lines)).strip()


def ast_key(code: str, prompt: str) -> Optional[str]:
    try:
        tree = ast.parse(wrap_body_with_header(code, prompt))
        return ast.dump(tree, annotate_fields=True, include_attributes=False)
    except Exception:
        return None


def code_equivalent(a: str, b: str, prompt: str, mode: str = "ast_or_text") -> bool:
    if mode in {"ast", "ast_or_text"}:
        ka, kb = ast_key(a, prompt), ast_key(b, prompt)
        if ka is not None and kb is not None:
            return ka == kb
        if mode == "ast":
            return False
    return normalize_text_code(a) == normalize_text_code(b)


def choose_target_response(responses: List[Dict[str, Any]], prompt: str, target_mode: str, eq_mode: str) -> Tuple[int, Dict[str, Any]]:
    if not responses:
        raise ValueError("No responses found.")
    if target_mode == "first":
        return 0, responses[0]
    if target_mode != "majority":
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    reps, counts, first_idx = [], {}, {}
    for idx, r in enumerate(responses):
        ans = get_llm_answer(r)
        assigned = False
        for cid, rep in enumerate(reps):
            if code_equivalent(ans, rep, prompt, eq_mode):
                counts[cid] += 1
                assigned = True
                break
        if not assigned:
            cid = len(reps)
            reps.append(ans)
            counts[cid] = 1
            first_idx[cid] = idx
    best = max(counts, key=lambda c: (counts[c], -first_idx[c]))
    return first_idx[best], responses[first_idx[best]]





class LocalOptionScorer:
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
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            parts = [f"{m['role'].upper()}: {m['content']}" for m in messages]
            parts.append("ASSISTANT:")
            return "\n\n".join(parts)

    def candidate_logprob(self, prompt: str, candidate: str) -> float:
        import torch
        prefix_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        variants = [candidate, " " + candidate, candidate + ".", " " + candidate + "."]
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
            total, start, valid = 0.0, len(prefix_ids), True
            for pos, tok_id in enumerate(suffix_ids):
                lp_pos = start + pos - 1
                if lp_pos < 0 or lp_pos >= logprobs.shape[1]:
                    valid = False
                    break
                total += float(logprobs[0, lp_pos, tok_id].item())
            if valid:
                best = max(best, total)
        return best

    def score_options(self, prompt: str, option_letters: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
        logps = {letter: self.candidate_logprob(prompt, letter) for letter in option_letters}
        max_log = max(logps.values())
        exps = {k: math.exp(v - max_log) for k, v in logps.items()}
        denom = sum(exps.values()) or 1.0
        probs = {k: float(v / denom) for k, v in exps.items()}
        return probs, logps


OPTION_VALUES = {chr(ord("A") + i): i / 10.0 for i in range(11)}


def build_vp_prompt(scorer: LocalOptionScorer, question: str, reasoning: str, code: str, prompt_style: str) -> str:
    options = "\n".join([f"{letter}. {int(value * 100)}%" for letter, value in OPTION_VALUES.items()])
    task = f"HumanEval Task:\n{question.strip()}\n" if question.strip() else "HumanEval Task:\n[missing prompt]\n"
    ask = "How likely is this proposed Python solution to pass the hidden HumanEval tests?" if prompt_style == "direct" else "Estimate the probability that this proposed Python solution is correct and will pass all hidden HumanEval tests."
    user = (
        f"{task}\n"
        f"Model reasoning:\n{reasoning.strip() if reasoning else '[empty]'}\n\n"
        f"Proposed Python solution:\n```python\n{code.strip()}\n```\n\n"
        f"{ask}\n"
        "Choose exactly one confidence level:\n"
        f"{options}\n\n"
        "Answer with only one letter from A to K."
    )
    messages = [
        {"role": "system", "content": "You are a strict Python programming judge for HumanEval tasks. Estimate the correctness probability of a proposed solution. Do not execute code. Return only the option letter."},
        {"role": "user", "content": user},
    ]
    return scorer.apply_chat_template(messages)


def compute_vp(scorer: LocalOptionScorer, question: str, reasoning: str, code: str, prompt_style: str) -> Dict[str, Any]:
    prompt = build_vp_prompt(scorer, question, reasoning, code, prompt_style)
    letters = list(OPTION_VALUES.keys())
    option_probs, option_logprobs = scorer.score_options(prompt, letters)
    conf = clamp01(sum(option_probs.get(l, 0.0) * OPTION_VALUES[l] for l in letters))
    unc = 1.0 - conf
    best = max(letters, key=lambda x: option_probs.get(x, 0.0))
    return {
        "verbalized_confidence_0_1": conf,
        "verbalized_probability_uncertainty": unc,
        "uncertainty": unc,
        "parse_mode": "scored_bins_expected_value",
        "best_option": best,
        "best_option_value": OPTION_VALUES[best],
        "best_option_prob": option_probs.get(best, 0.0),
        "option_probs": {l: {"value": OPTION_VALUES[l], "prob": option_probs[l], "logprob": option_logprobs[l]} for l in letters},
        "prompt_len": len(scorer.tokenizer(prompt, add_special_tokens=False)["input_ids"]),
    }


def process_item(record, line_idx, args, correct_map, problem_map, scorer):
    qid = get_record_id(record, line_idx)
    question = get_question(record, qid, problem_map)
    responses = get_responses(record)
    correct = correct_map.get(qid, None)
    correct = int(correct) if correct is not None else None
    if not responses:
        return {"id": qid, "correct": correct, "error": "no_responses"}

    target_idx, target_resp = choose_target_response(responses, question, args.target_mode, args.equivalence_mode)
    target_answer = get_llm_answer(target_resp)
    target_reasoning = get_llm_reason(target_resp)
    target_code = wrap_body_with_header(target_answer, question)
    vp = compute_vp(scorer, question, target_reasoning, target_code, args.prompt_style)

    out = {
        "id": qid,
        "correct": correct,
        "target_response_index": target_idx,
        "target_answer": strip_code_wrappers(target_answer),
        "verbalized_confidence_raw": vp["verbalized_confidence_0_1"],
        "verbalized_confidence_0_1": vp["verbalized_confidence_0_1"],
        "verbalized_probability_uncertainty": vp["verbalized_probability_uncertainty"],
        "uncertainty": vp["uncertainty"],
        "parse_mode": vp["parse_mode"],
        "best_option": vp["best_option"],
        "best_option_value": vp["best_option_value"],
        "best_option_prob": vp["best_option_prob"],
        "num_responses": len(responses),
        "meta": {
            "method": "Verbalized_Probability_Scored_Bins",
            "dataset": "humaneval_reasoning_sampling",
            "correctness_source": "external_correct_file_only_by_question_id",
            "question_source": "input_question_or_problem_file_by_question_id",
            "ignored_field": "correct answer",
            "model_path": args.model_path,
            "model_family": args.model_family,
            "target_mode": args.target_mode,
            "equivalence_mode": args.equivalence_mode if args.target_mode == "majority" else None,
            "prompt_style": args.prompt_style,
            "scoring": "candidate_sequence_logprob_softmax_over_A_to_K",
        },
    }
    if args.keep_option_probs:
        out["option_probs"] = vp["option_probs"]
    if args.keep_question:
        out["question"] = question
    if args.keep_prediction:
        out["prediction"] = target_code
    if args.keep_target_reasoning:
        out["target_reasoning"] = target_reasoning
    if args.keep_prompt_len:
        out["prompt_len"] = vp["prompt_len"]
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Compute scored-bin Verbalized Probability for HumanEval reasoning data.")
    p.add_argument("--input-file", required=True)
    p.add_argument("--correct-file", required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--problem-file", default=None)
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-family", choices=["qwen", "llama", "auto"], default="auto")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--hf-endpoint", default=None)
    p.add_argument("--target-mode", choices=["first", "majority"], default="first")
    p.add_argument("--equivalence-mode", choices=["ast", "text", "ast_or_text"], default="ast_or_text")
    p.add_argument("--prompt-style", choices=["direct", "estimate"], default="estimate")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--keep-question", action="store_true")
    p.add_argument("--keep-prediction", action="store_true")
    p.add_argument("--keep-target-reasoning", action="store_true")
    p.add_argument("--keep-option-probs", action="store_true")
    p.add_argument("--keep-prompt-len", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if args.model_family == "auto":
        low = args.model_path.lower()
        args.model_family = "qwen" if "qwen" in low else ("llama" if "llama" in low else "auto")

    safe_print(f"INPUT_FILE = {args.input_file}")
    safe_print(f"CORRECT_FILE = {args.correct_file}")
    safe_print(f"PROBLEM_FILE = {args.problem_file}")
    safe_print(f"OUTPUT_FILE = {args.output_file}")
    safe_print("Correctness source: external correct file only, matched by input id -> correct_file.question_id")
    safe_print("Question source: input question/prompt first; otherwise --problem-file matched by input id -> problem_file.question_id")
    safe_print("Ignored field: record['correct answer']")
    safe_print(f"Model path = {args.model_path}")
    safe_print(f"Model family = {args.model_family}")
    safe_print("Elicitation mode = scored bins A-K")

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
    scorer = LocalOptionScorer(args.model_path, args.model_family, args.dtype, args.device)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    done_ids = set() if args.overwrite else load_done_ids(args.output_file)
    mode = "w" if args.overwrite else "a"
    processed = 0
    with open(args.output_file, mode, encoding="utf-8") as fout:
        for line_idx, record in enumerate(records, 1):
            if args.limit is not None and processed >= args.limit:
                break
            qid = get_record_id(record, line_idx)
            if qid in done_ids:
                continue
            try:
                res = process_item(record, line_idx, args, correct_map, problem_map, scorer)
                fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                fout.flush()
                processed += 1
                if args.verbose:
                    safe_print(f"done: {qid} | line={line_idx} | correct={res.get('correct')} | conf={res.get('verbalized_confidence_0_1')} | unc={res.get('verbalized_probability_uncertainty')} | best={res.get('best_option')}:{res.get('best_option_value')}")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                fout.write(json.dumps({"id": qid, "error": str(e)}, ensure_ascii=False) + "\n")
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

    _parser = _ap.ArgumentParser(description="verbalized_probability / llama / humaneval / reasoning_augmented (standalone)")
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
    if "verbalized_probability" not in {"max_seq_prob", "mean_tok_ent", "max_tok_prob", "ppl"}:
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
    if "verbalized_probability" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "verbalized_probability", "family": "llama",
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

