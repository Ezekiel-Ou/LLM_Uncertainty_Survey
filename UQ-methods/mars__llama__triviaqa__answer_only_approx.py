#!/usr/bin/env python3
# Original paper/author repository: https://github.com/Ybakman/LLM_Uncertainty

if __name__ == "__main__" and not globals().get("_UQ_EXECUTING", False):
    
    import ast as _uq_ast
    from pathlib import Path as _uq_Path
    _uq_tree = _uq_ast.parse(_uq_Path(__file__).read_text())
    _uq_helper = next(n for n in _uq_tree.body if isinstance(n, _uq_ast.FunctionDef) and n.name == "_uq_standalone")
    exec(compile(_uq_ast.Module(body=[_uq_helper], type_ignores=[]), __file__, "exec"), globals())
    _uq_standalone()
    raise SystemExit(0)





import json
import math
import os
import re
import string
import argparse
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer





INPUT_FILE = __import__("os").environ.get("UQ_INPUT_FILE", "")
OUTPUT_FILE = __import__("os").environ.get("UQ_OUTPUT_FILE", "")

JUDGE_MODEL_NAME = __import__("os").environ.get("UQ_MODEL_PATH", "")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

OVERWRITE_OUTPUT = True
LIMIT = None   


MASK_TOKEN = " ..."
MAX_MASK_TOKENS = 12
USE_P_A_ONLY = False
MIN_IMPORTANCE = 1e-8





print("Loading Llama judge model...", flush=True)
judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME)
judge_model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL_NAME,
    torch_dtype=DTYPE,
    device_map="auto" if torch.cuda.is_available() else None,
)

if not torch.cuda.is_available():
    judge_model.to(DEVICE)

judge_model.eval()

if judge_tokenizer.pad_token_id is None:
    judge_tokenizer.pad_token_id = judge_tokenizer.eos_token_id





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
        aliases = [str(x).strip() for x in gt_field.get("aliases", [])]
        if value and value not in aliases:
            aliases = [value] + aliases
        return {"value": value, "aliases": aliases}

    value = str(gt_field).strip()
    return {"value": value, "aliases": [value]}


def get_gt(obj: Dict[str, Any]) -> Dict[str, Any]:
    if "gt" in obj:
        return normalize_ground_truth(obj["gt"])
    if "ground_truth" in obj:
        return normalize_ground_truth(obj["ground_truth"])
    return {"value": "", "aliases": [""]}


def is_correct(answer: str, gt: Dict[str, Any]) -> int:
    pred = normalize_answer(answer)
    aliases = [normalize_answer(x) for x in gt.get("aliases", [])]
    return int(pred in aliases)





def get_question_id(obj: Dict[str, Any]) -> str:
    if "id" in obj:
        return str(obj["id"])
    if "question_id" in obj:
        return str(obj["question_id"])
    return "unknown_id"


def get_greedy_answer(obj: Dict[str, Any]) -> str:
    if "greedy_answer" in obj:
        return str(obj["greedy_answer"]).strip()
    if "greedy" in obj and isinstance(obj["greedy"], dict):
        return str(obj["greedy"].get("text", "")).strip()
    return ""


def get_greedy_steps(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "greedy" in obj and isinstance(obj["greedy"], dict):
        steps = obj["greedy"].get("steps", [])
        if isinstance(steps, list):
            return steps
    return []





def build_prompt(question: str, proposed_answer: str) -> str:
    return (
        "You are a factuality judge.\n\n"
        f"Question: {question}\n"
        f"Proposed Answer: {proposed_answer}\n\n"
        "Choose exactly one option:\n"
        "A. Correct\n"
        "B. Incorrect\n"
        "C. Not sure\n\n"
        "Answer:"
    )


def get_single_token_id(token_str: str) -> Optional[int]:
    
    for candidate in [f" {token_str}", token_str]:
        ids = judge_tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return None


A_ID = get_single_token_id("A")
B_ID = get_single_token_id("B")
C_ID = get_single_token_id("C")

if A_ID is None or B_ID is None or C_ID is None:
    raise ValueError("Cannot map A/B/C to single tokens in Llama tokenizer.")


@torch.no_grad()
def judge_answer_score(question: str, proposed_answer: str) -> Dict[str, Any]:
    prompt = build_prompt(question, proposed_answer)
    inputs = judge_tokenizer(prompt, return_tensors="pt").to(judge_model.device)

    logits = judge_model(**inputs).logits[0, -1, :].float()
    abc_logits = torch.tensor(
        [logits[A_ID].item(), logits[B_ID].item(), logits[C_ID].item()],
        dtype=torch.float32,
        device=logits.device,
    )
    probs = F.softmax(abc_logits, dim=-1)

    p_a = float(probs[0].item())
    p_b = float(probs[1].item())
    p_c = float(probs[2].item())

    score = p_a if USE_P_A_ONLY else (p_a + 0.5 * p_c)

    return {
        "p_a": p_a,
        "p_b": p_b,
        "p_c": p_c,
        "score": float(score),
        "prompt_len": int(inputs["input_ids"].shape[1]),
    }





def extract_token_and_logprob(step: Dict[str, Any]) -> Dict[str, Any]:
    
    text_piece = ""
    logprob = None

    for k in ["text_piece", "piece", "text", "token", "generated_token"]:
        if k in step and step[k] is not None:
            text_piece = str(step[k])
            break

    for k in ["chosen_logprob", "logprob", "token_logprob", "selected_logprob"]:
        if k in step and step[k] is not None:
            try:
                logprob = float(step[k])
                break
            except Exception:
                pass

    if logprob is None:
        for k in ["chosen_prob", "prob", "token_prob", "selected_prob"]:
            if k in step and step[k] is not None:
                try:
                    p = float(step[k])
                    if p > 0:
                        logprob = math.log(p)
                        break
                except Exception:
                    pass

    return {
        "text_piece": text_piece,
        "logprob": logprob,
    }


def rebuild_answer_from_pieces(pieces: List[str]) -> str:
    text = "".join(pieces)
    text = re.sub(r"\s+", " ", text).strip()
    return text





def normalize_importance(raw_importances: List[float], seq_len: int) -> List[float]:
    if seq_len <= 0:
        return []

    s = sum(raw_importances)
    if s <= 0:
        u = [1.0 / seq_len] * seq_len
    else:
        u = [x / s for x in raw_importances]

    return [(1.0 / (2.0 * seq_len)) + (ui / 2.0) for ui in u]


def compute_mars_approx(obj: Dict[str, Any]) -> Dict[str, Any]:
    qid = get_question_id(obj)
    question = str(obj.get("question", ""))
    gt = get_gt(obj)
    greedy_answer = get_greedy_answer(obj)
    steps = get_greedy_steps(obj)

    correct = is_correct(greedy_answer, gt)

    parsed_steps = [extract_token_and_logprob(s) for s in steps]
    parsed_steps = [
        x for x in parsed_steps
        if x["logprob"] is not None and str(x["text_piece"]) != ""
    ]

    
    if not parsed_steps:
        avg_lp = None
        if "greedy" in obj and isinstance(obj["greedy"], dict):
            try:
                avg_lp = float(obj["greedy"].get("avg_logprob", None))
            except Exception:
                avg_lp = None

        if avg_lp is None:
            return {
                "id": qid,
                "question": question,
                "gt": gt,
                "greedy_answer": greedy_answer,
                "correct": correct,
                "mars_score": None,
                "mars_uncertainty": None,
                "judge_original_score": None,
                "num_tokens_used": 0,
                "judge_calls": 0,
                "meta": {
                    "method": "MARS-approx-mask-judge-llama",
                    "fallback": "no_valid_steps_and_no_avg_logprob",
                },
                "token_debug": [],
            }

        return {
            "id": qid,
            "question": question,
            "gt": gt,
            "greedy_answer": greedy_answer,
            "correct": correct,
            "mars_score": float(avg_lp),
            "mars_uncertainty": float(-avg_lp),
            "judge_original_score": None,
            "num_tokens_used": 0,
            "judge_calls": 0,
            "meta": {
                "method": "MARS-approx-mask-judge-llama",
                "fallback": "avg_logprob_only",
            },
            "token_debug": [],
        }

    pieces = [x["text_piece"] for x in parsed_steps]
    logprobs = [float(x["logprob"]) for x in parsed_steps]

    if len(pieces) > MAX_MASK_TOKENS:
        pieces = pieces[:MAX_MASK_TOKENS]
        logprobs = logprobs[:MAX_MASK_TOKENS]

    answer_text = rebuild_answer_from_pieces(pieces)

    judge_calls = 0

    orig_judge = judge_answer_score(question, answer_text)
    p_orig = float(orig_judge["score"])
    judge_calls += 1

    raw_importances = []
    token_debug = []

    for i in range(len(pieces)):
        masked_pieces = pieces.copy()
        masked_pieces[i] = MASK_TOKEN
        masked_answer = rebuild_answer_from_pieces(masked_pieces)

        masked_judge = judge_answer_score(question, masked_answer)
        p_mask = float(masked_judge["score"])
        judge_calls += 1

        importance = max(0.0, p_orig - p_mask)
        if importance < MIN_IMPORTANCE:
            importance = MIN_IMPORTANCE

        raw_importances.append(importance)
        token_debug.append(
            {
                "token_index": i,
                "text_piece": pieces[i],
                "logprob": logprobs[i],
                "masked_answer": masked_answer,
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

    return {
        "id": qid,
        "question": question,
        "gt": gt,
        "greedy_answer": greedy_answer,
        "correct": correct,

        "mars_score": float(mars_score),
        "mars_uncertainty": mars_uncertainty,
        "judge_original_score": p_orig,
        "num_tokens_used": len(pieces),
        "judge_calls": judge_calls,

        "meta": {
            "method": "MARS-approx-mask-judge-llama",
            "judge_model": JUDGE_MODEL_NAME,
            "mask_token": MASK_TOKEN,
            "max_mask_tokens": MAX_MASK_TOKENS,
            "use_p_a_only": USE_P_A_ONLY,
        },
        "token_debug": token_debug,
    }





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=INPUT_FILE)
    parser.add_argument("--output_file", type=str, default=OUTPUT_FILE)
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()

    input_file = args.input_file
    output_file = args.output_file
    limit = args.limit
    overwrite = args.overwrite

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if os.path.exists(output_file) and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_file}. "
            f"Please set OVERWRITE_OUTPUT=True or use --overwrite."
        )

    print(f"INPUT_FILE = {input_file}", flush=True)
    print(f"OUTPUT_FILE = {output_file}", flush=True)
    print(f"LIMIT = {limit}", flush=True)
    print("Running Llama-specific MARS approximate on TriviaQA...", flush=True)

    processed = 0
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line_idx, line in enumerate(fin, start=1):
            if limit is not None and processed >= limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"skip line {line_idx}: json parse error: {e}", flush=True)
                continue

            try:
                result = compute_mars_approx(obj)
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                processed += 1

                qid = result["id"]
                unc = result["mars_uncertainty"]
                unc_str = "None" if unc is None else f"{unc:.6f}"

                if result["num_tokens_used"] == 0:
                    print(
                        f"done: {qid} | line={line_idx} | "
                        f"correct={result['correct']} | "
                        f"num_tokens=0 | judge_calls=0 | "
                        f"unc={unc_str} | fallback=1",
                        flush=True,
                    )
                else:
                    print(
                        f"done: {qid} | line={line_idx} | "
                        f"correct={result['correct']} | "
                        f"num_tokens={result['num_tokens_used']} | "
                        f"judge_calls={result['judge_calls']} | "
                        f"unc={unc_str}",
                        flush=True,
                    )

            except Exception as e:
                print(f"skip line {line_idx}: processing error: {e}", flush=True)

    print(f"Finished. processed={processed}", flush=True)
    print(f"Saved to: {output_file}", flush=True)


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

    _parser = _ap.ArgumentParser(description="mars / llama / triviaqa / answer_only (standalone)")
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
    if "mars" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "mars", "family": "llama",
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

