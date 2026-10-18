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
import json
import math
import os
import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

try:
    from math_verify import parse, verify
except Exception as e:  
    raise RuntimeError("Please install math-verify: pip install math-verify") from e


print_lock = threading.Lock()


def safe_print(*args: Any, **kwargs: Any) -> None:
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
            safe_print(f"[warn] skip bad json line {line_no}: {e}", file=sys.stderr)
    return records


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


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





def mathv_correct(ref_text: Any, target_text: Any) -> bool:
    try:
        ref = parse(str(ref_text))
        target = parse(str(target_text))
        return bool(verify(ref, target))
    except Exception:
        return False


def get_record_id(record: Dict[str, Any], fallback_idx: int) -> str:
    for key in ["id", "question_id", "qid"]:
        if key in record and record[key] is not None:
            return str(record[key])
    return f"line_{fallback_idx}"


def get_correct_answer(record: Dict[str, Any]) -> str:
    for key in ["correct answer", "correct_answer", "answer", "gt", "ground_truth"]:
        if key in record and record[key] is not None:
            v = record[key]
            if isinstance(v, dict):
                if v.get("value") is not None:
                    return str(v["value"])
                aliases = v.get("aliases")
                if isinstance(aliases, list) and aliases:
                    return str(aliases[0])
            return str(v)
    return ""


def get_question(record: Dict[str, Any]) -> str:
    for key in ["question", "problem", "prompt", "input"]:
        if key in record and record[key] is not None:
            return str(record[key])
    return ""


def get_responses(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    responses = record.get("responses", [])
    if isinstance(responses, list):
        return [r for r in responses if isinstance(r, dict)]
    return []


def get_response_reason(resp: Dict[str, Any]) -> str:
    for key in ["llm reason", "llm_reason", "reason", "reasoning_augmented", "text", "response"]:
        if key in resp and resp[key] is not None:
            return str(resp[key]).strip()
    return ""


def get_response_answer(resp: Dict[str, Any]) -> str:
    for key in ["llm answer", "llm_answer", "answer", "final_answer"]:
        if key in resp and resp[key] is not None:
            return str(resp[key]).strip()
    
    return get_response_reason(resp)


def build_response_text(resp: Dict[str, Any]) -> str:
    reason = get_response_reason(resp)
    answer = get_response_answer(resp)
    if reason and answer:
        if answer in reason:
            return reason.strip()
        return f"{reason.strip()}\nFinal Answer: {answer.strip()}"
    return reason or answer


def cluster_answers_by_mathverify(answers: Sequence[str]) -> Tuple[List[int], List[str]]:
    reps: List[str] = []
    cluster_ids: List[int] = []
    for ans in answers:
        assigned = False
        for cid, rep in enumerate(reps):
            if mathv_correct(rep, ans):
                cluster_ids.append(cid)
                assigned = True
                break
        if not assigned:
            cid = len(reps)
            reps.append(ans)
            cluster_ids.append(cid)
    return cluster_ids, reps


def choose_target_response_index(
    responses: List[Dict[str, Any]],
    mode: str,
) -> int:
    if not responses:
        return -1
    if mode == "first":
        return 0
    if mode == "majority":
        answers = [get_response_answer(r) for r in responses]
        cluster_ids, _ = cluster_answers_by_mathverify(answers)
        counts: Dict[int, int] = {}
        for cid in cluster_ids:
            counts[cid] = counts.get(cid, 0) + 1
        if not counts:
            return 0
        best_cid = max(sorted(counts.keys()), key=lambda cid: counts[cid])
        for i, cid in enumerate(cluster_ids):
            if cid == best_cid:
                return i
        return 0
    raise ValueError(f"Unsupported target mode: {mode}")


def answer_correct_stats(correct_answer: str, responses: List[Dict[str, Any]]) -> Tuple[int, float]:
    if not responses:
        return 0, 0.0
    vals = [1 if mathv_correct(correct_answer, get_response_answer(r)) else 0 for r in responses]
    return int(sum(vals)), float(sum(vals) / len(vals))





def resolve_dtype(dtype_name: str):
    if dtype_name == "auto":
        return None
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_name in {"fp16", "float16", "half"}:
        return torch.float16
    if dtype_name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


@dataclass
class PTrueScore:
    p_true: float
    p_false: float
    logp_true: float
    logp_false: float
    true_variant_logprobs: Dict[str, float]
    false_variant_logprobs: Dict[str, float]
    prompt_len: int


class LocalPTrueJudge:
    def __init__(
        self,
        model_path: str,
        model_family: str = "auto",
        device: str = "cuda",
        dtype: str = "auto",
        max_input_tokens: int = 4096,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = model_path
        self.model_family = model_family.lower()
        self.max_input_tokens = max_input_tokens
        torch_dtype = resolve_dtype(dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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

    def _chat_prompt(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        
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
        elif self.model_family in {"llama", "auto"}:
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        
        return (
            f"System: {system}\n\n"
            f"User: {user}\n\n"
            "Assistant:"
        )

    def build_prompt(
        self,
        question: str,
        target_reasoning: str,
        target_answer: str,
        include_reasoning: bool = True,
    ) -> str:
        system = (
            "You are evaluating whether a proposed answer to a math reasoning problem is correct. "
            "Respond only with True or False."
        )

        parts: List[str] = []
        if question.strip():
            parts.append(f"Problem:\n{question.strip()}")
        else:
            parts.append("Problem:\nThe original problem statement is not available. Evaluate the proposed reasoning and final answer for internal mathematical correctness.")

        if include_reasoning and target_reasoning.strip():
            parts.append(f"Proposed reasoning:\n{target_reasoning.strip()}")
        parts.append(f"Proposed final answer:\n{target_answer.strip()}")
        parts.append(
            "Question: Is the proposed final answer correct?\n"
            "Answer with exactly one word: True or False."
        )
        user = "\n\n".join(parts)
        return self._chat_prompt(system, user)

    @staticmethod
    def _logsumexp(values: List[float]) -> float:
        finite = [v for v in values if math.isfinite(v)]
        if not finite:
            return float("-inf")
        m = max(finite)
        return m + math.log(sum(math.exp(v - m) for v in finite))

    @torch.no_grad()
    def _candidate_logprob(self, prefix_text: str, candidate_text: str) -> Tuple[float, int]:
        
        prefix_ids = self.tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        cand_ids = self.tokenizer(candidate_text, add_special_tokens=False)["input_ids"]
        if not cand_ids:
            return float("-inf"), len(prefix_ids)

        max_prefix_len = max(1, self.max_input_tokens - len(cand_ids))
        if len(prefix_ids) > max_prefix_len:
            prefix_ids = prefix_ids[-max_prefix_len:]

        input_ids_list = prefix_ids + cand_ids
        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=self.model.device)
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits[:, :-1, :].float()
        logprobs = torch.log_softmax(logits, dim=-1)

        total = 0.0
        start = len(prefix_ids)
        
        for k, tok_id in enumerate(cand_ids):
            pos = start + k - 1
            if pos < 0 or pos >= logprobs.shape[1]:
                return float("-inf"), len(prefix_ids)
            total += float(logprobs[0, pos, tok_id].item())
        return total, len(prefix_ids)

    def score_ptrue(
        self,
        prompt_text: str,
        true_variants: Optional[List[str]] = None,
        false_variants: Optional[List[str]] = None,
    ) -> PTrueScore:
        if true_variants is None:
            true_variants = ["True", " True", "true", " true"]
        if false_variants is None:
            false_variants = ["False", " False", "false", " false"]

        true_lps: Dict[str, float] = {}
        false_lps: Dict[str, float] = {}
        prompt_len = 0

        for cand in true_variants:
            lp, plen = self._candidate_logprob(prompt_text, cand)
            true_lps[cand] = lp
            prompt_len = max(prompt_len, plen)
        for cand in false_variants:
            lp, plen = self._candidate_logprob(prompt_text, cand)
            false_lps[cand] = lp
            prompt_len = max(prompt_len, plen)

        logp_true = self._logsumexp(list(true_lps.values()))
        logp_false = self._logsumexp(list(false_lps.values()))

        denom = self._logsumexp([logp_true, logp_false])
        if not math.isfinite(denom):
            p_true = 0.5
            p_false = 0.5
        else:
            p_true = math.exp(logp_true - denom)
            p_false = math.exp(logp_false - denom)

        return PTrueScore(
            p_true=float(p_true),
            p_false=float(p_false),
            logp_true=float(logp_true),
            logp_false=float(logp_false),
            true_variant_logprobs={k: float(v) for k, v in true_lps.items()},
            false_variant_logprobs={k: float(v) for k, v in false_lps.items()},
            prompt_len=int(prompt_len),
        )





def compute_one_record(
    record: Dict[str, Any],
    rec_idx: int,
    judge: LocalPTrueJudge,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    qid = get_record_id(record, rec_idx)
    question = get_question(record)
    correct_answer = get_correct_answer(record)
    responses = get_responses(record)

    if not responses:
        return {
            "id": qid,
            "correct_answer": correct_answer,
            "correct": 0,
            "error": "no_responses",
            "p_true_confidence": None,
            "p_true_uncertainty": None,
            "uncertainty": None,
        }

    target_idx = choose_target_response_index(responses, args.target_mode)
    target_resp = responses[target_idx]
    target_answer = get_response_answer(target_resp)
    target_reasoning = get_response_reason(target_resp)
    target_full_text = build_response_text(target_resp)

    correct = 1 if mathv_correct(correct_answer, target_answer) else 0
    num_correct_responses, answer_correct_rate = answer_correct_stats(correct_answer, responses)

    prompt_text = judge.build_prompt(
        question=question,
        target_reasoning=target_full_text if args.include_full_response else target_reasoning,
        target_answer=target_answer,
        include_reasoning=not args.answer_only,
    )
    score = judge.score_ptrue(prompt_text)

    uncertainty = float(1.0 - score.p_true)
    result: Dict[str, Any] = {
        "id": qid,
        "correct_answer": correct_answer,
        "target_response_index": target_idx,
        "target_answer": target_answer,
        "correct": int(correct),
        "p_true": score.p_true,
        "p_false": score.p_false,
        "p_true_confidence": score.p_true,
        "p_true_uncertainty": uncertainty,
        "uncertainty": uncertainty,
        "logp_true": score.logp_true,
        "logp_false": score.logp_false,
        "num_responses": len(responses),
        "num_correct_responses": num_correct_responses,
        "answer_correct_rate": answer_correct_rate,
        "meta": {
            "method": "P(True)",
            "dataset_format": "reasoning_sampling_gsm8k_style",
            "correctness_checker": "math_verify",
            "judge_model": args.model_path,
            "model_family": args.model_family,
            "target_mode": args.target_mode,
            "answer_only": args.answer_only,
            "candidate_scoring": "sequence_logprob_true_false_variants",
            "max_input_tokens": args.max_input_tokens,
        },
    }

    if args.keep_candidate_logprobs:
        result["candidate_logprobs"] = {
            "true_variants": score.true_variant_logprobs,
            "false_variants": score.false_variant_logprobs,
        }
    if args.keep_question:
        result["question"] = question
    if args.keep_target_reasoning:
        result["target_reasoning"] = target_reasoning
    if args.keep_prompt:
        result["judge_prompt"] = prompt_text
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute P(True) uncertainty for reasoning sampling records.")
    parser.add_argument("--input-file", required=True, help="Input JSON/JSONL reasoning sampling file")
    parser.add_argument("--output-file", required=True, help="Output JSONL file")
    parser.add_argument("--model-path", required=True, help="Local judge model path")
    parser.add_argument("--model-family", default="auto", choices=["auto", "qwen", "llama", "plain"], help="Prompt formatting family")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"], help="Model dtype")
    parser.add_argument("--hf-endpoint", default=None, help="Optional HuggingFace mirror endpoint")
    parser.add_argument("--target-mode", default="first", choices=["first", "majority"], help="Which response to evaluate")
    parser.add_argument("--answer-only", action="store_true", help="Judge only final answer without reasoning trace")
    parser.add_argument("--include-full-response", action="store_true", help="Use llm reason + final answer as reasoning text")
    parser.add_argument("--max-input-tokens", type=int, default=4096, help="Maximum input tokens for judge scoring")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N records")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output instead of resuming")
    parser.add_argument("--keep-question", action="store_true")
    parser.add_argument("--keep-target-reasoning", action="store_true")
    parser.add_argument("--keep-prompt", action="store_true")
    parser.add_argument("--keep-candidate-logprobs", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    if args.overwrite and os.path.exists(args.output_file):
        os.remove(args.output_file)

    safe_print(f"INPUT_FILE = {args.input_file}")
    safe_print(f"OUTPUT_FILE = {args.output_file}")
    safe_print("Correctness source: record['correct answer'] with math_verify")
    safe_print(f"Judge model: {args.model_path}")
    safe_print(f"Judge family: {args.model_family}")
    safe_print("Method: P(True) via True/False candidate sequence scoring")

    records = load_records(args.input_file)
    if args.limit is not None:
        records = records[: args.limit]

    done = set() if args.overwrite else load_done_ids(args.output_file)
    safe_print(f"records loaded: {len(records)} | existing output records: {len(done)}")

    safe_print("Loading local judge model...")
    judge = LocalPTrueJudge(
        model_path=args.model_path,
        model_family=args.model_family,
        device=args.device,
        dtype=args.dtype,
        max_input_tokens=args.max_input_tokens,
    )

    processed = 0
    skipped = 0
    for idx, record in enumerate(records, start=1):
        qid = get_record_id(record, idx)
        if str(qid) in done:
            skipped += 1
            continue
        try:
            result = compute_one_record(record, idx, judge, args)
            append_jsonl(args.output_file, result)
            processed += 1
            if args.verbose:
                conf = result.get("p_true_confidence")
                unc = result.get("p_true_uncertainty")
                conf_s = "None" if conf is None else f"{float(conf):.4f}"
                unc_s = "None" if unc is None else f"{float(unc):.4f}"
                safe_print(
                    f"done: {qid} | line={idx} | correct={result.get('correct')} | "
                    f"p_true={conf_s} | unc={unc_s}"
                )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            err = {"id": qid, "error": str(e)}
            append_jsonl(args.output_file, err)
            safe_print(f"error: {qid} | line={idx} | {e}")

    safe_print(f"Finished. processed={processed}, skipped={skipped}")
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

    _parser = _ap.ArgumentParser(description="p_true / qwen / gsm8k / reasoning_augmented (standalone)")
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
    if "p_true" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "p_true", "family": "qwen",
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

