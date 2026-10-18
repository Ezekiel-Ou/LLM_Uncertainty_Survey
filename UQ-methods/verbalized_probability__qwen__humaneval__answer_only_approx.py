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
import re
import sys
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

print_lock = threading.Lock()


def safe_print(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    with print_lock:
        print(*args, **kwargs)






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






def build_correct_map(
    records: List[Dict[str, Any]],
    question_id_key: str = "question_id",
    correct_key: str = "correct",
) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for rec in records:
        qid = rec.get(question_id_key)
        if qid is None and question_id_key != "id":
            qid = rec.get("id")
        if qid is None:
            qid = rec.get("question_id") or rec.get("id") or rec.get("qid")
        if qid is None:
            continue
        try:
            correct = int(rec.get(correct_key, 0))
        except Exception:
            correct = 0
        out[str(qid)] = 1 if correct else 0
    return out






def resolve_record_id(record: Dict[str, Any], idx: int = 0) -> str:
    for key in ["id", "question_id", "qid"]:
        if key in record and record[key] is not None:
            return str(record[key])
    return f"line_{idx}"


def extract_text_field(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        for key in ["text", "raw_text", "answer", "response", "prediction", "code"]:
            if key in x and x[key] is not None:
                t = str(x[key]).strip()
                if t:
                    return t
    return str(x).strip()


def get_greedy_prediction(record: Dict[str, Any]) -> str:
    if "greedy_answer" in record and record["greedy_answer"] is not None:
        return str(record["greedy_answer"]).strip()
    if "prediction" in record and record["prediction"] is not None:
        return str(record["prediction"]).strip()
    if "greedy" in record:
        text = extract_text_field(record.get("greedy"))
        if text:
            return text
    return ""


def get_entry_point(record: Dict[str, Any]) -> str:
    gt = record.get("gt") or {}
    if isinstance(gt, dict):
        ep = gt.get("entry_point")
        if ep:
            return str(ep)
    return str(record.get("entry_point", "") or "")


def clean_code_answer(text: str) -> str:
    text = "" if text is None else str(text).strip()
    m = re.search(r"<CODE>\s*(.*?)\s*</CODE>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    text = re.sub(r"^```(?:python|py)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


def find_function_header(question: str, entry_point: str = "") -> Optional[str]:
    if entry_point:
        pat = rf"^\s*def\s+{re.escape(entry_point)}\s*\([^\n]*\)\s*(?:->\s*[^:\n]+)?\s*:"
        for line in str(question).splitlines():
            if re.match(pat, line):
                return line.strip()
    for line in str(question).splitlines():
        if re.match(r"^\s*def\s+\w+\s*\([^\n]*\)\s*(?:->\s*[^:\n]+)?\s*:", line):
            return line.strip()
    return None


def indent_body(body: str, spaces: int = 4) -> str:
    prefix = " " * spaces
    lines = str(body).splitlines()
    if not lines:
        return prefix + "pass"
    out = []
    for line in lines:
        if line.strip() == "":
            out.append("")
        else:
            out.append(prefix + line.lstrip())
    return "\n".join(out)


def build_full_candidate_code(question: str, answer: str, entry_point: str = "") -> str:
    ans = clean_code_answer(answer)
    if re.search(r"^\s*def\s+\w+\s*\(", ans, flags=re.MULTILINE):
        return ans
    header = find_function_header(question, entry_point)
    if header:
        return header + "\n" + indent_body(ans if ans else "pass")
    return ans






def normalize_confidence_value(x: Any) -> Tuple[Optional[float], Optional[float]]:
    try:
        val = float(x)
    except Exception:
        return None, None
    if math.isnan(val) or math.isinf(val):
        return val, None
    if 0.0 <= val <= 1.0:
        return val, val
    if 1.0 < val <= 100.0:
        return val, val / 100.0
    return val, None


def parse_confidence_from_text(text: str) -> Tuple[Optional[float], Optional[float], str]:
    text = "" if text is None else str(text).strip()
    if not text:
        return None, None, "empty"

    
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(text[start:end + 1])
            for key in [
                "confidence", "probability", "verbalized_probability",
                "correctness_probability", "p_correct", "score",
            ]:
                if key in obj:
                    raw, conf = normalize_confidence_value(obj[key])
                    if conf is not None:
                        return raw, conf, f"json:{key}"
    except Exception:
        pass

    
    patterns = [
        r"(?i)(?:confidence|probability|p\s*\(\s*correct\s*\)|p_correct|score|chance)\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        r"(?i)(?:confidence|probability|p\s*\(\s*correct\s*\)|p_correct|score|chance)\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*100",
        r"(?i)(?:confidence|probability|p\s*\(\s*correct\s*\)|p_correct|score|chance)\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)",
        r"([0-9]+(?:\.[0-9]+)?)\s*%",
        r"\b([01](?:\.[0-9]+)?)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            raw, conf = normalize_confidence_value(m.group(1))
            if conf is not None:
                return raw, conf, f"regex:{pat}"

    return None, None, "failed"






class LocalConfidenceModel:
    def __init__(
        self,
        model_path: str,
        model_family: str = "auto",
        device: str = "cuda",
        dtype: str = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_path = model_path
        self.model_family = self._infer_family(model_path, model_family)
        self.torch_dtype = self._resolve_dtype(dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        if device == "cuda" and torch.cuda.is_available():
            kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if not (device == "cuda" and torch.cuda.is_available()):
            self.model.to(torch.device("cpu"))
        self.model.eval()

    def _infer_family(self, model_path: str, model_family: str) -> str:
        if model_family and model_family != "auto":
            return model_family.lower()
        lower = str(model_path).lower()
        if "qwen" in lower:
            return "qwen"
        if "llama" in lower or "meta-llama" in lower:
            return "llama"
        return "generic"

    def _resolve_dtype(self, dtype: str):
        import torch
        dtype = str(dtype).lower()
        if dtype == "auto":
            return None
        if dtype in {"float16", "fp16", "half"}:
            return torch.float16
        if dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if dtype in {"float32", "fp32"}:
            return torch.float32
        raise ValueError(f"Unsupported dtype: {dtype}")

    def build_prompt(self, messages: List[Dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                if self.model_family == "qwen":
                    return self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except TypeError:
                try:
                    return self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    pass
            except Exception:
                pass

        chunks = []
        for m in messages:
            role = str(m.get("role", "user")).upper()
            content = str(m.get("content", ""))
            chunks.append(f"{role}: {content}")
        chunks.append("ASSISTANT:")
        return "\n\n".join(chunks)

    def candidate_token_ids(self, label: str) -> List[int]:
        
        variants = [label, " " + label, "\n" + label, f"({label}", f"({label})"]
        ids: List[int] = []
        for v in variants:
            try:
                toks = self.tokenizer.encode(v, add_special_tokens=False)
                if len(toks) == 1:
                    ids.append(int(toks[0]))
            except Exception:
                continue
        
        seen = set()
        out = []
        for tid in ids:
            if tid not in seen:
                out.append(tid)
                seen.add(tid)
        return out

    def score_labels(self, messages: List[Dict[str, str]], labels: Sequence[str]) -> Dict[str, Any]:
        import torch
        import torch.nn.functional as F

        prompt = self.build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1, :].float()

        label_logits: List[float] = []
        label_token_ids: Dict[str, List[int]] = {}
        for label in labels:
            tids = self.candidate_token_ids(label)
            label_token_ids[label] = tids
            if not tids:
                label_logits.append(float("-inf"))
                continue
            vals = torch.stack([logits[tid] for tid in tids])
            label_logits.append(float(torch.max(vals).item()))

        logit_tensor = torch.tensor(label_logits, dtype=torch.float32, device=logits.device)
        if torch.isneginf(logit_tensor).all():
            probs = torch.full_like(logit_tensor, 1.0 / len(labels))
        else:
            probs = F.softmax(logit_tensor, dim=-1)

        return {
            "prompt": prompt,
            "labels": list(labels),
            "label_logits": [float(x) for x in label_logits],
            "label_probs": [float(x) for x in probs.detach().cpu().tolist()],
            "label_token_ids": label_token_ids,
        }

    def generate_response(self, messages: List[Dict[str, str]], max_new_tokens: int, temperature: float, top_p: float) -> Dict[str, Any]:
        import torch
        prompt = self.build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = int(inputs["input_ids"].shape[1])
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        if temperature and temperature > 0:
            gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p))
        else:
            gen_kwargs.update(dict(do_sample=False))
        with torch.inference_mode():
            outputs = self.model.generate(**gen_kwargs)
        gen_ids = outputs[0, prompt_len:]
        raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        return {
            "prompt": prompt,
            "raw_text": raw_text,
            "prompt_len": prompt_len,
            "num_generated_tokens": int(gen_ids.shape[0]),
        }






BIN_LABELS = list("ABCDEFGHIJK")
BIN_VALUES = [i / 10.0 for i in range(11)]
BIN_TEXT = "\n".join([f"{lab}. {int(val * 100)}%" for lab, val in zip(BIN_LABELS, BIN_VALUES)])


def build_scored_bin_messages(question: str, greedy_answer: str, entry_point: str = "") -> List[Dict[str, str]]:
    candidate_code = build_full_candidate_code(question, greedy_answer, entry_point)
    system_msg = (
        "You estimate confidence for Python HumanEval coding tasks. "
        "You cannot run the code. Choose the probability that the proposed solution passes the tests."
    )
    user_msg = (
        "Task prompt:\n"
        f"{question}\n\n"
        "Proposed solution:\n"
        "```python\n"
        f"{candidate_code}\n"
        "```\n\n"
        "Choose exactly one confidence option for correctness:\n"
        f"{BIN_TEXT}\n\n"
        "Answer with only the option letter."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def build_generate_messages(question: str, greedy_answer: str, entry_point: str = "") -> List[Dict[str, str]]:
    candidate_code = build_full_candidate_code(question, greedy_answer, entry_point)
    system_msg = (
        "You are estimating your own confidence for a Python HumanEval coding task. "
        "You are not allowed to run the code. Judge whether the proposed solution is likely to pass the hidden unit tests."
    )
    user_msg = (
        "Task prompt:\n"
        f"{question}\n\n"
        "Proposed solution:\n"
        "```python\n"
        f"{candidate_code}\n"
        "```\n\n"
        "What is the probability that the proposed solution is correct?\n"
        "Return only a JSON object in exactly this format:\n"
        "{\"confidence\": <a number between 0 and 1>}\n"
        "Do not include any explanation."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]






def compute_for_item(
    record: Dict[str, Any],
    idx: int,
    correct_map: Dict[str, int],
    conf_model: LocalConfidenceModel,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    qid = resolve_record_id(record, idx)
    question = str(record.get("question", ""))
    entry_point = get_entry_point(record)
    greedy_answer = get_greedy_prediction(record)
    correct = correct_map.get(qid, 0)

    if args.elicitation_mode == "generate":
        messages = build_generate_messages(question, greedy_answer, entry_point)
        gen = conf_model.generate_response(
            messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        raw_conf, conf_0_1, parse_mode = parse_confidence_from_text(gen["raw_text"])
        details: Dict[str, Any] = {
            "mode": "generate",
            "verbalized_response": gen["raw_text"],
            "parse_mode": parse_mode,
            "raw_numeric_confidence": raw_conf,
        }
        prompt_text = gen["prompt"]
    else:
        messages = build_scored_bin_messages(question, greedy_answer, entry_point)
        score = conf_model.score_labels(messages, BIN_LABELS)
        probs = score["label_probs"]
        conf_0_1 = float(sum(p * v for p, v in zip(probs, BIN_VALUES)))
        best_idx = max(range(len(probs)), key=lambda i: probs[i]) if probs else 0
        raw_conf = conf_0_1
        details = {
            "mode": "score_bins",
            "confidence_bins": [
                {
                    "label": lab,
                    "value": val,
                    "prob": probs[i],
                    "logit": score["label_logits"][i],
                    "token_ids": score["label_token_ids"].get(lab, []),
                }
                for i, (lab, val) in enumerate(zip(BIN_LABELS, BIN_VALUES))
            ],
            "top_label": BIN_LABELS[best_idx],
            "top_label_value": BIN_VALUES[best_idx],
            "parse_mode": "scored_bins_expected_value",
        }
        prompt_text = score["prompt"]

    uncertainty = None if conf_0_1 is None else float(1.0 - conf_0_1)
    parse_mode_out = details.get("parse_mode", "unknown")

    out: Dict[str, Any] = {
        "id": qid,
        "correct": int(correct),
        "verbalized_confidence_raw": raw_conf,
        "verbalized_confidence_0_1": conf_0_1,
        "verbalized_probability_uncertainty": uncertainty,
        "uncertainty": uncertainty,
        "parse_mode": parse_mode_out,
        "details": details,
        "meta": {
            "method": "Verbalized Probability",
            "dataset": "humaneval",
            "model_path": args.model_path,
            "model_family": conf_model.model_family,
            "correctness_source": "external_correct_file_only",
            "elicitation_mode": args.elicitation_mode,
            "confidence_definition": (
                "expected confidence bin probability from next-token logits"
                if args.elicitation_mode == "score-bins"
                else "model-generated verbalized probability parsed from text"
            ),
        },
    }

    if args.keep_question:
        out["question"] = question
    if args.keep_prediction:
        out["prediction"] = greedy_answer
        out["candidate_code"] = build_full_candidate_code(question, greedy_answer, entry_point)
    if args.keep_prompt:
        out["verbalized_prompt_messages"] = messages
        out["verbalized_prompt_text"] = prompt_text

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute robust Verbalized Probability uncertainty for HumanEval stage1 JSONL files."
    )
    parser.add_argument("--input-file", required=True, help="HumanEval stage1 model output JSONL")
    parser.add_argument("--correct-file", required=True, help="External correctness JSONL file")
    parser.add_argument("--output-file", required=True, help="Output JSONL file")
    parser.add_argument("--model-path", required=True, help="Local model path used to elicit confidence")
    parser.add_argument(
        "--model-family",
        default="auto",
        choices=["auto", "qwen", "llama", "generic"],
        help="Chat template family. auto infers from model path.",
    )
    parser.add_argument(
        "--elicitation-mode",
        default="score-bins",
        choices=["score-bins", "generate"],
        help="score-bins avoids parse failures by reading next-token probabilities over A..K bins.",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu; cuda uses device_map=auto")
    parser.add_argument("--dtype", default="auto", help="auto, bf16, fp16, fp32")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0, help="Only used in generate mode. 0 means greedy.")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--question-id-key", default="question_id")
    parser.add_argument("--correct-key", default="correct")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output instead of resuming")
    parser.add_argument("--keep-question", action="store_true")
    parser.add_argument("--keep-prediction", action="store_true")
    parser.add_argument("--keep-prompt", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--hf-endpoint", default=None, help="Optional HF_ENDPOINT, e.g. https://hf-mirror.com")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    safe_print(f"INPUT_FILE = {args.input_file}")
    safe_print(f"CORRECT_FILE = {args.correct_file}")
    safe_print(f"OUTPUT_FILE = {args.output_file}")
    safe_print("Correctness source: external correct file only")
    safe_print(f"Confidence model: {args.model_path}")
    safe_print(f"Model family: {args.model_family}")
    safe_print(f"Elicitation mode: {args.elicitation_mode}")

    correct_records = load_jsonl(args.correct_file)
    correct_map = build_correct_map(correct_records, args.question_id_key, args.correct_key)

    if args.overwrite and os.path.exists(args.output_file):
        os.remove(args.output_file)

    done_ids = set() if args.overwrite else load_done_ids(args.output_file)
    if done_ids:
        safe_print(f"Resume mode: found {len(done_ids)} existing ids in output")

    safe_print("Loading local confidence model...")
    conf_model = LocalConfidenceModel(
        model_path=args.model_path,
        model_family=args.model_family,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=True,
    )
    safe_print(f"Loaded model with family = {conf_model.model_family}")

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
                record = json.loads(line)
            except Exception as e:
                safe_print(f"skip invalid json line={line_idx}: {e}")
                continue

            qid = resolve_record_id(record, line_idx)
            if qid in done_ids:
                continue
            if qid not in correct_map:
                missing_correct += 1

            try:
                result = compute_for_item(record, line_idx, correct_map, conf_model, args)
                append_jsonl(args.output_file, result)
                processed += 1

                if args.verbose:
                    conf = result["verbalized_confidence_0_1"]
                    unc = result["verbalized_probability_uncertainty"]
                    conf_s = "None" if conf is None else f"{conf:.4f}"
                    unc_s = "None" if unc is None else f"{unc:.4f}"
                    safe_print(
                        f"done: {result['id']} | line={line_idx} | "
                        f"correct={result['correct']} | conf={conf_s} | unc={unc_s} | "
                        f"parse={result['parse_mode']}"
                    )
            except KeyboardInterrupt:
                raise
            except Exception as e:
                err = {
                    "id": qid,
                    "correct": correct_map.get(qid, 0),
                    "error": str(e),
                    "meta": {
                        "method": "Verbalized Probability",
                        "dataset": "humaneval",
                        "correctness_source": "external_correct_file_only",
                    },
                }
                if args.keep_question:
                    err["question"] = record.get("question", "")
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

    _parser = _ap.ArgumentParser(description="verbalized_probability / qwen / humaneval / answer_only (standalone)")
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
    if "verbalized_probability" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "verbalized_probability", "family": "qwen",
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

