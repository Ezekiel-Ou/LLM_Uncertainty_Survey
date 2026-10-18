#!/usr/bin/env python3
# Implementation repository: https://github.com/IINemo/lm-polygraph

from __future__ import annotations
import argparse
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REVISION = '1b24f22d0cd426d3f89e7c98dc4b95115b6f3586'


METHODS = {
    'max_seq_prob': ('max_probability', 'MaximumSequenceProbability'),
    'max_tok_prob': ('max_probability', 'MaximumTokenProbability'),
    'mean_tok_ent': ('token_entropy', 'MeanTokenEntropy'),
    'ppl': ('perplexity', 'Perplexity'),
    'num_set': ('num_sem_sets', 'NumSemSets'),
    'eigv': ('eig_val_laplacian', 'EigValLaplacian'),
    'deg': ('deg_mat', 'DegMat'),
    'ecc': ('eccentricity', 'Eccentricity'),
    'semantic_entropy': ('semantic_entropy', 'SemanticEntropy'),
    'semantic_density': ('semantic_density', 'SemanticDensity'),
    'p_true': ('p_true', 'PTrue'),
    'verbalized_probability': ('verbalized_2s', 'Verbalized2S'),
    'token_sar': ('token_sar', 'TokenSAR'),
    'pmi': ('pointwise_mutual_information', 'PointwiseMutualInformation'),
    'mean_pmi': ('pointwise_mutual_information', 'MeanPointwiseMutualInformation'),
}

def normalize_method(value):
    import re
    key = re.sub('[^a-z0-9]', '', value.lower())
    aliases = {re.sub('[^a-z0-9]', '', k): k for k in METHODS}
    aliases.update({'perplexity':'ppl', 'maximumsequenceprobability':'max_seq_prob',
                    'maximumtokenprobability':'max_tok_prob', 'meantokenentropy':'mean_tok_ent'})
    if key not in aliases:
        raise argparse.ArgumentTypeError(f'No verified LM-Polygraph counterpart: {value}. Use --list; MARS and CoT-UQ are not aliases for TokenSAR.')
    return aliases[key]


def make_estimator(method, kwargs):
    module, name = METHODS[method]
    cls = getattr(importlib.import_module('lm_polygraph.estimators.' + module), name)
    if method == 'verbalized_probability':
        kwargs = {'confidence_prompt': 'Give your confidence that your previous answer is correct. Output exactly Confidence: followed by a number between 0 and 1.',
                  'confidence_regex': r'Confidence:\s*(0(?:\.\d+)?|1(?:\.0+)?)\b', **kwargs}
    return cls(**kwargs)


def numeric(value):
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [numeric(x) for x in value]
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Upstream returned a non-finite score; output not committed')
    return result


def evaluate_stats(estimator, stats):
    
    import numpy as np
    missing = set(estimator.stats_dependencies) - stats.keys()
    if missing:
        raise ValueError(f'Missing upstream statistics: {sorted(missing)}')
    converted = {}
    for key, value in stats.items():
        
        
        if 'matrix' in key or key == 'token_similarity':
            value = np.asarray(value)
        converted[key] = value
    
    
    if type(estimator).__name__ == 'PointwiseMutualInformation':
        if len(stats['greedy_log_likelihoods']) != 1:
            raise ValueError('stats must contain exactly one question')
        return numeric(estimator(converted))
    result = estimator(converted)
    if len(result) != 1:
        raise ValueError('stats must describe exactly one question (batch dimension = 1)')
    return numeric(result[0])


def load_model(args):
    from lm_polygraph.utils.generation_parameters import GenerationParameters
    params = GenerationParameters(max_new_tokens=args.max_new_tokens)
    if args.family == 'deepseek':
        from lm_polygraph.utils.model import BlackboxModel
        from openai import OpenAI
        key = os.environ.get('DEEPSEEK_API_KEY')
        if not key:
            raise ValueError('DEEPSEEK_API_KEY is required')
        model = BlackboxModel(openai_api_key=key, model_path=args.model_path,
                              generation_parameters=params, supports_logprobs=args.api_logprobs)
        model.openai_api = OpenAI(api_key=key, base_url=args.api_base_url)
        return model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lm_polygraph.utils.model import WhiteboxModel
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    base = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype='auto').to(args.device)
    base.eval()
    class RenderedPromptModel(WhiteboxModel):
        
        
        def tokenize(self, texts):
            rendered = []
            for text in texts:
                if isinstance(text, list):
                    options = {'enable_thinking':False} if args.family == 'qwen' else {}
                    text = self.tokenizer.apply_chat_template(text, tokenize=False,
                                add_generation_prompt=True, **options)
                rendered.append(text)
            return self.tokenizer(rendered, padding=True, return_tensors='pt', add_special_tokens=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    return RenderedPromptModel(base, tokenizer, model_path=args.model_path,
                               generation_parameters=params, instruct=False)


def provenance():
    dist = importlib.metadata.distribution('lm-polygraph')
    direct = dist.read_text('direct_url.json')
    return {'version':dist.version, 'installation':json.loads(direct) if direct else None,
            'audited_revision':REVISION, 'repository':'https://github.com/IINemo/lm-polygraph'}


def main(argv=None, fixed=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', type=normalize_method)
    parser.add_argument('--mode', choices=['generate','stats'], default='generate')
    parser.add_argument('--family', choices=['qwen','llama','deepseek'])
    parser.add_argument('--model-path')
    parser.add_argument('--dataset', choices=['gsm8k','triviaqa','humaneval'])
    parser.add_argument('--setting', choices=['answer_only','reasoning_augmented'], default='answer_only')
    parser.add_argument('--input-file')
    parser.add_argument('--output-file')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--estimator-kwargs', type=json.loads, default={})
    parser.add_argument('--human-eval', choices=['docker','skip'], default='docker')
    parser.add_argument('--api-base-url', default='https://api.deepseek.com')
    parser.add_argument('--api-logprobs', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.set_defaults(**(fixed or {}))
    args = parser.parse_args(argv)
    for key, value in (fixed or {}).items():
        if getattr(args,key) != value:
            parser.error(f'This file fixes --{key.replace("_","-")} to {value}')
    if args.list:
        print(json.dumps(METHODS, indent=2)); return
    for name in ['method','dataset','input_file','output_file']:
        if not getattr(args,name):parser.error(f'--{name.replace("_","-")} required')
    if not isinstance(args.estimator_kwargs,dict):parser.error('--estimator-kwargs must be a JSON object')
    if args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):parser.error('Token and question limits must be positive')
    if args.mode == 'generate' and not args.model_path:parser.error('--model-path required for generation')
    if args.mode == 'generate' and not args.family:
        args.family = next((f for f in ['qwen','llama','deepseek'] if f in args.model_path.lower()),None)
        if args.family is None:parser.error('Specify --family for an unrecognized model path')
    if args.mode == 'generate' and args.method == 'pmi':
        parser.error('Upstream high-level API truncates flattened PMI to one token; use --mode stats or method mean_pmi')
    if args.mode == 'stats' and args.method == 'verbalized_probability':
        parser.error('Verbalized2S requires a live model; use --mode generate')
    if args.family == 'deepseek' and args.mode == 'generate':
        if args.setting != 'answer_only':parser.error('DeepSeek reasoning_augmented is outside the verified experiment scope')
        if args.method not in {'num_set','eigv','deg','ecc'}:parser.error('Only verified blackbox graph routes are enabled for DeepSeek generation; use stats for other estimators')
    inp, out = Path(args.input_file).resolve(), Path(args.output_file).resolve()
    if not inp.is_file():parser.error('Input does not exist')
    if inp == out:parser.error('Output cannot replace input')
    if out.exists() and not args.overwrite:parser.error('Output exists; use --overwrite')
    if args.dry_run:
        print(json.dumps({'implementation':'lm_polygraph', 'method':args.method, 'class':METHODS[args.method],
                          'mode':args.mode, 'setting':args.setting, 'estimator_kwargs':args.estimator_kwargs,
                          'input':str(inp), 'output':str(out)},ensure_ascii=False)); return
    try:
        estimator = make_estimator(args.method,args.estimator_kwargs)
        source = provenance()
    except ImportError as exc:
        raise RuntimeError('Install UQ-methods/lm_polygraph_backend/requirements.txt in a separate environment') from exc
    from data_generation.schema import read_examples, normalize_example, build_messages, clean_answer, answer_span
    from data_generation.evaluation import Evaluator
    if args.mode == 'generate':
        from transformers import set_seed
        from lm_polygraph.utils import estimate_uncertainty
        set_seed(args.seed)
        evaluator = Evaluator(args.dataset,human_mode=args.human_eval)
        model = load_model(args)
    out.parent.mkdir(parents=True,exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.'+out.name+'.',suffix='.tmp',dir=out.parent)
    seen = set()
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            for index,row in enumerate(read_examples(inp)):
                if args.limit is not None and index >= args.limit:break
                if args.mode == 'stats':
                    qid = str(row.get('id',index))
                    score = evaluate_stats(estimator,row['stats'])
                    answer, correct = row.get('answer'), row.get('correct')
                    evaluation = {'status':'provided_upstream_statistics_label'}
                    question = row.get('question')
                else:
                    example = normalize_example(row,args.dataset,index)
                    qid, question = example['id'], example['question']
                    messages = build_messages(example,args.dataset,args.setting)
                    if args.family == 'deepseek':
                        prompt = '\n\n'.join(m['content'] for m in messages)
                    else:
                        kwargs = {'enable_thinking':False} if args.family == 'qwen' else {}
                        prompt = model.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**kwargs)
                    result = estimate_uncertainty(model,estimator,input_text=prompt)
                    score = numeric(result.uncertainty)
                    answer = result.generation_text
                    start, end, found = answer_span(answer,args.setting)
                    evaluation = (evaluator(example,clean_answer(answer[start:end],args.dataset))
                                  if found else {'correct':None,'status':'missing_final_answer'})
                    correct = evaluation['correct']
                if qid in seen:raise ValueError(f'Duplicate question ID: {qid}')
                seen.add(qid)
                if correct is not None and (isinstance(correct,str) or correct not in (0,1)):
                    raise ValueError(f'{qid}: correct must be 0, 1 or null')
                record = {'id':qid,'question':question,'answer':answer,'correct':correct,
                          'evaluation':evaluation, 'method':'lm_polygraph_' + args.method,'base_method':args.method,'implementation':'lm_polygraph',
                          'setting':args.setting,'dataset':args.dataset,'model_path':args.model_path,
                          'score':score,'score_type':'uncertainty','level':estimator.level,
                          'provenance':source,'estimator':str(estimator),
                          'run':{'mode':args.mode,'seed':args.seed,'max_new_tokens':args.max_new_tokens,
                                 'estimator_kwargs':args.estimator_kwargs, 'device':args.device,
                                 'sampling_configuration':'upstream registered calculator defaults (audited samples_n=10)'}}
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        if not seen:raise ValueError('No questions processed')
        os.replace(temporary,out)
    finally:
        Path(temporary).unlink(missing_ok=True)

if __name__ == '__main__':
    main()
