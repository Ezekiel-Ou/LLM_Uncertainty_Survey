#!/usr/bin/env python3
# Reference repository: https://github.com/IINemo/lm-polygraph

if __name__ == "__main__" and not globals().get("_UQ_EXECUTING", False):
    
    import ast as _uq_ast
    from pathlib import Path as _uq_Path
    _uq_tree = _uq_ast.parse(_uq_Path(__file__).read_text())
    _uq_helper = next(n for n in _uq_tree.body if isinstance(n, _uq_ast.FunctionDef) and n.name == "_uq_standalone")
    exec(compile(_uq_ast.Module(body=[_uq_helper], type_ignores=[]), __file__, "exec"), globals())
    _uq_standalone()
    raise SystemExit(0)




import os
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

import json
import math
import re
import string
import threading
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer





INPUT_FILE = __import__("os").environ.get("UQ_INPUT_FILE", "")
OUTPUT_FILE = __import__("os").environ.get("UQ_OUTPUT_FILE", "")

MODEL_NAME = __import__("os").environ.get("UQ_MODEL_PATH", "")



TRUE_CANDIDATES = ["A", " A", "True", " True", "true", " true", "正确", " 是"]
FALSE_CANDIDATES = ["B", " B", "False", " False", "false", " false", "错误", " 否"]

MAX_PROMPT_TOKENS = 2048

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        DTYPE = torch.bfloat16
    else:
        DTYPE = torch.float16
else:
    DTYPE = torch.float32





print("Loading Qwen model...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True,
)
if not torch.cuda.is_available():
    model.to(DEVICE)

model.eval()
model.generation_config.use_cache = True
MODEL_INPUT_DEVICE = next(model.parameters()).device

write_lock = threading.Lock()
print_lock = threading.Lock()





def safe_print(*args, **kwargs) -> None:
    with print_lock:
        print(*args, **kwargs, flush=True)


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with write_lock:
        out_dir = os.path.dirname(path)
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
                if "id" in obj and "error" not in obj:
                    done.add(str(obj["id"]))
            except Exception:
                continue
    return done





def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def normalize_ground_truth(gt_field: Any) -> Dict[str, Any]:
    if isinstance(gt_field, str):
        value = gt_field.strip()
        return {"value": value, "aliases": [value]}

    if isinstance(gt_field, dict):
        value = str(gt_field.get("value", "")).strip()
        aliases = [str(x).strip() for x in gt_field.get("aliases", []) if str(x).strip()]
        if value and value not in aliases:
            aliases = [value] + aliases
        return {"value": value, "aliases": aliases}

    value = str(gt_field).strip()
    return {"value": value, "aliases": [value]}


def triviaqa_is_correct(answer: str, gt: Dict[str, Any]) -> int:
    pred = normalize_answer(answer)
    aliases = [normalize_answer(x) for x in gt.get("aliases", [])]
    return int(pred in aliases)





def get_question_id(obj: Dict[str, Any], line_idx: int) -> str:
    for key in ["id", "question_id", "qid"]:
        if key in obj and obj[key] is not None:
            return str(obj[key])
    return f"line_{line_idx}"


def get_question(obj: Dict[str, Any]) -> str:
    return str(obj.get("question", "")).strip()


def get_gt(obj: Dict[str, Any]) -> Dict[str, Any]:
    if "gt" in obj:
        return normalize_ground_truth(obj["gt"])
    if "ground_truth" in obj:
        return normalize_ground_truth(obj["ground_truth"])
    if "answer" in obj:
        return normalize_ground_truth(obj["answer"])
    return {"value": "", "aliases": [""]}


def extract_text_field(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        for key in ["text", "raw_text", "answer", "response", "output"]:
            if key in x and x[key] is not None:
                return str(x[key]).strip()
    return str(x).strip()


def get_greedy_answer(obj: Dict[str, Any]) -> str:
    if "greedy_answer" in obj and obj["greedy_answer"] is not None:
        return str(obj["greedy_answer"]).strip()

    if "greedy" in obj:
        text = extract_text_field(obj["greedy"])
        if text:
            return text

    if "prediction" in obj:
        return extract_text_field(obj["prediction"])

    return ""





def build_ptrue_messages(question: str, proposed_answer: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are judging whether a proposed answer to a factoid question is correct. "
                "Respond only with one letter."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Proposed answer:\n{proposed_answer}\n\n"
                "Is the proposed answer correct?\n"
                "A. True\n"
                "B. False\n\n"
                "answer:"
            ),
        },
    ]


def format_chat_prompt(messages: List[Dict[str, str]]) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if os.environ.get("UQ_FAMILY") == "qwen":
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _candidate_token_ids(candidate: str) -> List[int]:
    ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
    return [int(x) for x in ids]


def score_candidate_continuations(
    prompt: str,
    candidates: List[str],
) -> Dict[str, float]:
    
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_PROMPT_TOKENS,
    )["input_ids"]

    if len(prompt_ids) == 0:
        raise ValueError("Empty prompt after tokenization.")

    cand_id_list = [_candidate_token_ids(c) for c in candidates]
    if any(len(ids) == 0 for ids in cand_id_list):
        raise ValueError("Some candidate is tokenized into empty ids.")

    sequences = [prompt_ids + cand_ids for cand_ids in cand_id_list]
    max_len = max(len(x) for x in sequences)
    pad_id = tokenizer.pad_token_id

    input_ids = []
    attention_mask = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        input_ids.append(seq + [pad_id] * pad_len)
        attention_mask.append([1] * len(seq) + [0] * pad_len)

    input_ids = torch.tensor(input_ids, dtype=torch.long, device=MODEL_INPUT_DEVICE)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=MODEL_INPUT_DEVICE)

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        log_probs = F.log_softmax(logits, dim=-1)

    scores = {}
    prompt_len = len(prompt_ids)

    for row_idx, (candidate, cand_ids) in enumerate(zip(candidates, cand_id_list)):
        total = 0.0
        for k, token_id in enumerate(cand_ids):
            target_pos = prompt_len + k
            logit_pos = target_pos - 1
            total += float(log_probs[row_idx, logit_pos, token_id].detach().cpu())
        scores[candidate] = total

    return scores


def logsumexp(values: List[float]) -> float:
    if not values:
        return -math.inf
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def compute_ptrue(question: str, proposed_answer: str) -> Dict[str, Any]:
    messages = build_ptrue_messages(question, proposed_answer)
    prompt = format_chat_prompt(messages)

    all_candidates = TRUE_CANDIDATES + FALSE_CANDIDATES
    candidate_scores = score_candidate_continuations(prompt, all_candidates)

    true_logp = logsumexp([candidate_scores[c] for c in TRUE_CANDIDATES])
    false_logp = logsumexp([candidate_scores[c] for c in FALSE_CANDIDATES])

    denom = logsumexp([true_logp, false_logp])
    p_true = math.exp(true_logp - denom) if math.isfinite(denom) else 0.5
    p_false = math.exp(false_logp - denom) if math.isfinite(denom) else 0.5

    return {
        "ptrue_confidence": float(p_true),
        "ptrue_uncertainty": float(1.0 - p_true),
        "pfalse": float(p_false),
        "true_logprob_group": float(true_logp),
        "false_logprob_group": float(false_logp),
        "candidate_logprobs": candidate_scores,
        "prompt_messages": messages,
    }





def compute_ptrue_for_item(obj: Dict[str, Any], line_idx: int) -> Dict[str, Any]:
    qid = get_question_id(obj, line_idx)
    question = get_question(obj)
    gt = get_gt(obj)
    greedy_answer = get_greedy_answer(obj)

    correct = triviaqa_is_correct(greedy_answer, gt)
    ptrue_info = compute_ptrue(question, greedy_answer)

    return {
        "id": qid,
        "question": question,
        "gt": gt,
        "greedy_answer": greedy_answer,
        "correct": correct,
        "ptrue_confidence": ptrue_info["ptrue_confidence"],
        "ptrue_uncertainty": ptrue_info["ptrue_uncertainty"],
        "pfalse": ptrue_info["pfalse"],
        "ptrue_details": {
            "true_logprob_group": ptrue_info["true_logprob_group"],
            "false_logprob_group": ptrue_info["false_logprob_group"],
            "candidate_logprobs": ptrue_info["candidate_logprobs"],
            "prompt_messages": ptrue_info["prompt_messages"],
        },
        "meta": {
            "method": "Ptrue",
            "dataset": "triviaqa",
            "model": MODEL_NAME,
            "ptrue_form": "choice_logprob",
            "true_choice": "A",
            "false_choice": "B",
            "correctness_checker": "triviaqa_normalized_exact_match",
            "qwen_enable_thinking": os.environ.get("UQ_FAMILY") == "qwen",
        },
    }





def main() -> None:
    done = load_done_ids(OUTPUT_FILE)
    safe_print(f"existing successful records: {len(done)}")

    with open(INPUT_FILE, "r", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as e:
                safe_print(f"skip invalid json line: {line_idx} | {e}")
                continue

            qid = get_question_id(obj, line_idx)
            if qid in done:
                continue

            try:
                result = compute_ptrue_for_item(obj, line_idx)
                append_jsonl(OUTPUT_FILE, result)
                safe_print(
                    f"done: {qid} | line={line_idx} | "
                    f"correct={result['correct']} | "
                    f"ptrue={result['ptrue_confidence']:.4f} | "
                    f"unc={result['ptrue_uncertainty']:.4f}"
                )
            except KeyboardInterrupt:
                safe_print("KeyboardInterrupt detected. Exiting gracefully.")
                raise
            except Exception as e:
                err = {
                    "id": qid,
                    "question": obj.get("question", ""),
                    "error": str(e),
                }
                append_jsonl(OUTPUT_FILE, err)
                safe_print(f"error: {qid} | line={line_idx} | {e}")


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

    _parser = _ap.ArgumentParser(description="p_true / llama / triviaqa / answer_only (standalone)")
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
                           "dataset": "triviaqa", "setting": "answer_only", "model_loading": False}, indent=2))
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

