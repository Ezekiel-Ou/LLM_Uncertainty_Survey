#!/usr/bin/env python3
# Original paper/author repository: https://github.com/potsawee/selfcheckgpt

if __name__ == "__main__" and not globals().get("_UQ_EXECUTING", False):
    
    import ast as _uq_ast
    from pathlib import Path as _uq_Path
    _uq_tree = _uq_ast.parse(_uq_Path(__file__).read_text())
    _uq_helper = next(n for n in _uq_tree.body if isinstance(n, _uq_ast.FunctionDef) and n.name == "_uq_standalone")
    exec(compile(_uq_ast.Module(body=[_uq_helper], type_ignores=[]), __file__, "exec"), globals())
    _uq_standalone()
    raise SystemExit(0)



import json
import os
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
import re
import string
import threading
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer




INPUT_FILE = __import__("os").environ.get("UQ_INPUT_FILE", "")
OUTPUT_FILE = __import__("os").environ.get("UQ_OUTPUT_FILE", "")

NLI_MODEL_NAME = "microsoft/deberta-large-mnli"


MAX_STAGE1_SAMPLES = 10


USE_EXACT_MATCH_MIX = True
ALPHA = 0.8   


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"




print("Loading NLI model...", flush=True)
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
nli_model = AutoModelForSequenceClassification.from_pretrained(
    NLI_MODEL_NAME,
    dtype=torch.float32,
    use_safetensors=False,
)
nli_model.to(DEVICE)
nli_model.eval()

id2label = {int(k): v.lower() for k, v in nli_model.config.id2label.items()}
ENTAIL_IDX = None
CONTRA_IDX = None
for idx, label in id2label.items():
    if "entail" in label:
        ENTAIL_IDX = idx
    if "contradiction" in label:
        CONTRA_IDX = idx

if ENTAIL_IDX is None or CONTRA_IDX is None:
    raise ValueError(f"Cannot find entailment/contradiction labels in {id2label}")

write_lock = threading.Lock()
print_lock = threading.Lock()




def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs, flush=True)


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
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
            try:
                obj = json.loads(line)
                if "id" in obj:
                    done.add(obj["id"])
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
        aliases = [str(x).strip() for x in gt_field.get("aliases", [])]
        if value and value not in aliases:
            aliases = [value] + aliases
        return {"value": value, "aliases": aliases}

    value = str(gt_field).strip()
    return {"value": value, "aliases": [value]}


def triviaqa_is_correct(answer: str, gt: Dict[str, Any]) -> int:
    pred = normalize_answer(answer)
    aliases = [normalize_answer(x) for x in gt.get("aliases", [])]
    return int(pred in aliases)





def get_question_id(obj: Dict[str, Any]) -> str:
    if "id" in obj:
        return str(obj["id"])
    if "question_id" in obj:
        return str(obj["question_id"])
    return ""


def get_gt(obj: Dict[str, Any]) -> Dict[str, Any]:
    if "gt" in obj:
        return normalize_ground_truth(obj["gt"])
    if "ground_truth" in obj:
        return normalize_ground_truth(obj["ground_truth"])
    return {"value": "", "aliases": [""]}


def get_greedy_answer(obj: Dict[str, Any]) -> str:
    if "greedy_answer" in obj:
        return str(obj["greedy_answer"]).strip()
    if "greedy" in obj and isinstance(obj["greedy"], dict):
        return str(obj["greedy"].get("text", "")).strip()
    return ""


def get_sampled_answers(obj: Dict[str, Any]) -> List[str]:
    if "sampled_answers" in obj and isinstance(obj["sampled_answers"], list):
        return [str(x).strip() for x in obj["sampled_answers"] if str(x).strip()]

    if "samples" in obj and isinstance(obj["samples"], list):
        outs = []
        for s in obj["samples"]:
            if isinstance(s, dict):
                t = str(s.get("text", "")).strip()
            else:
                t = str(s).strip()
            if t:
                outs.append(t)
        return outs

    return []





def nli_probs_batch(pairs: List[Tuple[str, str]], batch_size: int = 32):
    entail_probs = []
    contra_probs = []

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        premises = [x[0] for x in batch]
        hypotheses = [x[1] for x in batch]

        inputs = nli_tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(nli_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = nli_model(**inputs).logits.float()
            probs = F.softmax(logits, dim=-1)

        entail_probs.extend(probs[:, ENTAIL_IDX].detach().cpu().tolist())
        contra_probs.extend(probs[:, CONTRA_IDX].detach().cpu().tolist())

    return [float(x) for x in entail_probs], [float(x) for x in contra_probs]


def compute_selfcheck_score(greedy_answer: str, sampled_answers: List[str]) -> Tuple[float, List[Dict[str, Any]]]:
    if not greedy_answer.strip():
        return 0.5, []

    used_samples = [x for x in sampled_answers if str(x).strip()][:MAX_STAGE1_SAMPLES]
    if not used_samples:
        return 0.5, []

    
    forward_pairs = [(s, greedy_answer) for s in used_samples]
    backward_pairs = [(greedy_answer, s) for s in used_samples]

    p_ent_f, p_con_f = nli_probs_batch(forward_pairs)
    p_ent_b, p_con_b = nli_probs_batch(backward_pairs)

    details = []
    vals = []

    greedy_norm = normalize_answer(greedy_answer)

    for s, ef, cf, eb, cb in zip(used_samples, p_ent_f, p_con_f, p_ent_b, p_con_b):
        sample_norm = normalize_answer(s)

        
        
        support_forward = 0.5 * (ef + (1.0 - cf))
        support_backward = 0.5 * (eb + (1.0 - cb))
        nli_support = 0.5 * (support_forward + support_backward)

        exact_match = float(sample_norm == greedy_norm)

        if USE_EXACT_MATCH_MIX:
            support = ALPHA * nli_support + (1.0 - ALPHA) * exact_match
        else:
            support = nli_support

        details.append(
            {
                "sample_answer": s,
                "entail_forward": ef,
                "contradiction_forward": cf,
                "entail_backward": eb,
                "contradiction_backward": cb,
                "nli_support": nli_support,
                "exact_match": exact_match,
                "single_support": support,
            }
        )
        vals.append(support)

    confidence = float(sum(vals) / len(vals))
    confidence = max(0.0, min(1.0, confidence))
    return confidence, details





def compute_selfcheck_for_item(obj: Dict[str, Any]) -> Dict[str, Any]:
    qid = get_question_id(obj)
    question = str(obj.get("question", ""))
    gt = get_gt(obj)

    greedy_answer = get_greedy_answer(obj)
    sampled_answers = get_sampled_answers(obj)

    correct = triviaqa_is_correct(greedy_answer, gt)
    confidence, details = compute_selfcheck_score(greedy_answer, sampled_answers)
    uncertainty = float(1.0 - confidence)

    return {
        "id": qid,
        "question": question,
        "gt": gt,
        "greedy_answer": greedy_answer,
        "sampled_answers": sampled_answers[:MAX_STAGE1_SAMPLES],
        "correct": correct,
        "selfcheck_confidence": confidence,
        "selfcheck_uncertainty": uncertainty,
        "selfcheck_details": details,
        "meta": {
            "method": "SelfCheckGPT-style",
            "dataset": "triviaqa",
            "nli_model": NLI_MODEL_NAME,
            "num_samples_used": min(len(sampled_answers), MAX_STAGE1_SAMPLES),
            "use_exact_match_mix": USE_EXACT_MATCH_MIX,
            "alpha": ALPHA if USE_EXACT_MATCH_MIX else None,
            "score_definition": "avg sample support to greedy answer",
        },
    }





def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    done = load_done_ids(OUTPUT_FILE)
    safe_print(f"existing records: {len(done)}")

    with open(INPUT_FILE, "r", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                safe_print(f"skip invalid json line: {line_idx}")
                continue

            qid = get_question_id(obj)

            if qid in done:
                continue

            try:
                result = compute_selfcheck_for_item(obj)
                append_jsonl(OUTPUT_FILE, result)
                safe_print(
                    f"done: {qid} | line={line_idx} | "
                    f"correct={result['correct']} | "
                    f"conf={result['selfcheck_confidence']:.4f} | "
                    f"unc={result['selfcheck_uncertainty']:.4f}"
                )
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

    _parser = _ap.ArgumentParser(description="selfcheckgpt / qwen / triviaqa / answer_only (standalone)")
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
                           "dataset": "triviaqa", "setting": "answer_only", "model_loading": False}, indent=2))
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

