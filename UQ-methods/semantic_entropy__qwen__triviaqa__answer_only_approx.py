#!/usr/bin/env python3
# Original paper/author repository: https://github.com/jlko/semantic_uncertainty
# Reference repository: https://github.com/IINemo/lm-polygraph

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
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
import re
import argparse
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification





INPUT_FILE = __import__("os").environ.get("UQ_INPUT_FILE", "")
OUTPUT_FILE = __import__("os").environ.get("UQ_OUTPUT_FILE", "")

NLI_MODEL_NAME = "microsoft/deberta-large-mnli"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NLI_LENGTH = 256
SEMANTIC_EQ_THRESHOLD = 0.5

OVERWRITE_OUTPUT = True
LIMIT = None   





def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"'`]", "", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def softmax_from_logprobs(logprobs: List[float]) -> List[float]:
    if not logprobs:
        return []
    x = torch.tensor(logprobs, dtype=torch.float32)
    probs = F.softmax(x, dim=0).tolist()
    return probs


def is_correct_trivia(pred: str, gt_obj: Any) -> bool:
    pred_norm = normalize_text(pred)

    gt_candidates = []
    if isinstance(gt_obj, dict):
        if "value" in gt_obj and gt_obj["value"] is not None:
            gt_candidates.append(str(gt_obj["value"]))
        aliases = gt_obj.get("aliases", [])
        if isinstance(aliases, list):
            gt_candidates.extend([str(x) for x in aliases if x is not None])
    elif gt_obj is not None:
        gt_candidates.append(str(gt_obj))

    gt_norms = {normalize_text(x) for x in gt_candidates if x is not None}
    return pred_norm in gt_norms if gt_norms else False





def get_entailment_label_id(model) -> int:
    label2id = getattr(model.config, "label2id", None)
    id2label = getattr(model.config, "id2label", None)

    if label2id:
        for k, v in label2id.items():
            if str(k).upper() == "ENTAILMENT":
                return int(v)

    if id2label:
        for k, v in id2label.items():
            if str(v).upper() == "ENTAILMENT":
                return int(k)

    
    return 2


class NLIPredictor:
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.entailment_id = get_entailment_label_id(self.model)

    @torch.no_grad()
    def entailment_prob(self, premise: str, hypothesis: str) -> float:
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=MAX_NLI_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits[0]
        probs = F.softmax(logits, dim=-1)
        return float(probs[self.entailment_id].item())

    @torch.no_grad()
    def bidirectional_score(self, a: str, b: str) -> Tuple[float, float, float]:
        p_ab = self.entailment_prob(a, b)
        p_ba = self.entailment_prob(b, a)
        score = min(p_ab, p_ba)
        return p_ab, p_ba, score





def cluster_samples_by_semantics(
    texts: List[str],
    nli: NLIPredictor,
    threshold: float,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    cluster_reps: List[str] = []
    cluster_ids: List[int] = []
    pairwise_debug: List[Dict[str, Any]] = []

    for idx, text in enumerate(texts):
        assigned = False
        text_norm = normalize_text(text)

        for cid, rep in enumerate(cluster_reps):
            rep_norm = normalize_text(rep)

            
            if text_norm == rep_norm:
                cluster_ids.append(cid)
                pairwise_debug.append(
                    {
                        "sample_idx": idx,
                        "sample_text": text,
                        "rep_cluster": cid,
                        "rep_text": rep,
                        "p_ab": 1.0,
                        "p_ba": 1.0,
                        "score": 1.0,
                        "matched": True,
                        "matched_by": "string_equal",
                    }
                )
                assigned = True
                break

            p_ab, p_ba, score = nli.bidirectional_score(text, rep)
            matched = score >= threshold

            pairwise_debug.append(
                {
                    "sample_idx": idx,
                    "sample_text": text,
                    "rep_cluster": cid,
                    "rep_text": rep,
                    "p_ab": p_ab,
                    "p_ba": p_ba,
                    "score": score,
                    "matched": matched,
                    "matched_by": "nli",
                }
            )

            if matched:
                cluster_ids.append(cid)
                assigned = True
                break

        if not assigned:
            new_cid = len(cluster_reps)
            cluster_reps.append(text)
            cluster_ids.append(new_cid)
            pairwise_debug.append(
                {
                    "sample_idx": idx,
                    "sample_text": text,
                    "rep_cluster": new_cid,
                    "rep_text": text,
                    "p_ab": 1.0,
                    "p_ba": 1.0,
                    "score": 1.0,
                    "matched": True,
                    "matched_by": "new_cluster",
                }
            )

    return cluster_ids, pairwise_debug





def compute_cluster_summary(
    cluster_ids: List[int],
    sample_probs: List[float],
    texts: List[str],
) -> List[Dict[str, Any]]:
    cluster_mass: Dict[int, float] = {}
    cluster_members: Dict[int, List[int]] = {}

    for i, cid in enumerate(cluster_ids):
        cluster_mass[cid] = cluster_mass.get(cid, 0.0) + sample_probs[i]
        cluster_members.setdefault(cid, []).append(i)

    summary = []
    for cid in sorted(cluster_mass.keys()):
        members = cluster_members[cid]
        summary.append(
            {
                "cluster_id": cid,
                "prob": float(cluster_mass[cid]),
                "size": len(members),
                "member_indices": members,
                "member_texts": [texts[i] for i in members],
                "representative": texts[members[0]] if members else "",
            }
        )
    return summary


def compute_semantic_entropy(cluster_summary: List[Dict[str, Any]]) -> Tuple[float, float]:
    probs = [max(0.0, float(c["prob"])) for c in cluster_summary]
    probs = [p for p in probs if p > 0]

    if not probs:
        return 0.0, 0.0

    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    k = len(probs)
    normalized = entropy / math.log(k + 1e-12) if k > 1 else 0.0
    return float(entropy), float(normalized)


def compute_semantic_density(cluster_summary: List[Dict[str, Any]]) -> Tuple[float, float]:
    probs = [max(0.0, float(c["prob"])) for c in cluster_summary]
    if not probs:
        return 1.0, 0.0

    density_confidence = sum(p * p for p in probs)
    density_uncertainty = 1.0 - density_confidence
    return float(density_confidence), float(density_uncertainty)





def process_one_record(
    record: Dict[str, Any],
    nli: NLIPredictor,
    threshold: float,
) -> Dict[str, Any]:
    qid = record.get("id", "unknown_id")
    question = record.get("question", "")
    gt = record.get("gt", None)

    greedy = record.get("greedy", {}) or {}
    greedy_text = str(greedy.get("text", "") or "")

    samples = record.get("samples", []) or []
    sample_texts = [str(s.get("text", "") or "") for s in samples]

    
    if not sample_texts:
        sample_texts = [greedy_text]

    seq_logprobs = [safe_float(s.get("seq_logprob", None), 0.0) for s in samples]
    if len(seq_logprobs) != len(sample_texts) or len(seq_logprobs) == 0:
        sample_probs = [1.0 / len(sample_texts)] * len(sample_texts)
    else:
        sample_probs = softmax_from_logprobs(seq_logprobs)

    cluster_ids, pairwise_debug = cluster_samples_by_semantics(
        texts=sample_texts,
        nli=nli,
        threshold=threshold,
    )

    cluster_summary = compute_cluster_summary(
        cluster_ids=cluster_ids,
        sample_probs=sample_probs,
        texts=sample_texts,
    )

    semantic_entropy, semantic_entropy_normalized = compute_semantic_entropy(cluster_summary)
    semantic_density_confidence, semantic_density_uncertainty = compute_semantic_density(cluster_summary)

    result = {
        "id": qid,
        "question": question,
        "gt": gt,
        "greedy_text": greedy_text,
        "correct": int(is_correct_trivia(greedy_text, gt)),

        "semantic_entropy": semantic_entropy,
        "semantic_entropy_normalized": semantic_entropy_normalized,

        "semantic_density_confidence": semantic_density_confidence,
        "semantic_density_uncertainty": semantic_density_uncertainty,

        "num_samples": len(sample_texts),
        "num_unique_strings": len(set(normalize_text(x) for x in sample_texts)),
        "num_clusters": len(cluster_summary),

        "cluster_summary": cluster_summary,
        "debug": {
            "sample_probs": sample_probs,
            "sample_texts": sample_texts,
            "cluster_ids": cluster_ids,
            "pairwise_details": pairwise_debug,
        },
    }
    return result





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=INPUT_FILE)
    parser.add_argument("--output_file", type=str, default=OUTPUT_FILE)
    parser.add_argument("--threshold", type=float, default=SEMANTIC_EQ_THRESHOLD)
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()

    input_file = args.input_file
    output_file = args.output_file
    threshold = args.threshold
    limit = args.limit
    overwrite = args.overwrite

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if os.path.exists(output_file) and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_file}. "
            f"Please set OVERWRITE_OUTPUT=True or use --overwrite."
        )

    print("Loading NLI model...", flush=True)
    print(f"NLI_MODEL_NAME = {NLI_MODEL_NAME}", flush=True)
    print(f"DEVICE = {DEVICE}", flush=True)
    print(f"INPUT_FILE = {input_file}", flush=True)
    print(f"OUTPUT_FILE = {output_file}", flush=True)
    print(f"THRESHOLD = {threshold}", flush=True)
    print(f"LIMIT = {limit}", flush=True)

    nli = NLIPredictor(model_name=NLI_MODEL_NAME, device=DEVICE)

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
                record = json.loads(line)
            except Exception as e:
                print(f"skip line {line_idx}: json parse error: {e}", flush=True)
                continue

            try:
                result = process_one_record(
                    record=record,
                    nli=nli,
                    threshold=threshold,
                )
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                processed += 1

                qid = result["id"]
                print(
                    f"done: {qid} | line={line_idx} | "
                    f"correct={result['correct']} | "
                    f"conf={result['semantic_density_confidence']:.4f} | "
                    f"unc={result['semantic_density_uncertainty']:.4f} | "
                    f"se={result['semantic_entropy']:.4f}",
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

    _parser = _ap.ArgumentParser(description="semantic_entropy / qwen / triviaqa / answer_only (standalone)")
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
    if "semantic_entropy" not in {"max_seq_prob", "mean_tok_ent", "max_tok_prob", "ppl"}:
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
    if "semantic_entropy" in {"bsdetector", "mars", "p_true", "verbalized_probability", "cot_uq_me", "cot_uq_mi", "cot_uq_sar", "cot_uq_sp", "topo_uq"} and not _args.model_path:
        _parser.error("this method requires --model-path")
    if _args.dry_run:
        print(_json.dumps({"script": str(_source), "method": "semantic_entropy", "family": "qwen",
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

