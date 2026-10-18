#!/usr/bin/env python3

from __future__ import annotations
import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

FOLDER=Path(__file__).resolve().parent
SETTINGS=('answer_only','reasoning_augmented')
DEFAULT_METHODS=('semantic_entropy','eigv','num_set','bsdetector')
COLUMNS=['dataset','setting','model','method','num_samples','accuracy','avg_uncertainty']


def summarize(setting,detail_dir=None,output_dir=None,methods=DEFAULT_METHODS,
              validate_sources=False,project_root=None,overwrite=False):
    if setting not in SETTINGS:raise ValueError(f'Unknown generation setting: {setting}')
    detail_dir=Path(detail_dir) if detail_dir else FOLDER/'detail'
    output_dir=Path(output_dir) if output_dir else FOLDER/'summary'
    root=Path(project_root) if project_root else FOLDER.parent
    source=detail_dir/f'uq_detail__{setting}.csv'
    destination=output_dir/f'scaling_law__{setting}.csv'
    if source.resolve()==destination.resolve():raise ValueError('Output must differ from input')
    if destination.exists() and not overwrite:raise ValueError(f'Output exists: {destination}; use --overwrite')
    groups=defaultdict(lambda:{'correct':[],'raw':[]});checked=set()
    with source.open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle)
        required={'dataset','setting','model','method','correct','raw_uncertainty_value','source_file'}
        if not required.issubset(reader.fieldnames or []):raise ValueError(f'Missing required detail columns: {source}')
        for line_number,row in enumerate(reader,2):
            if row['setting']!=setting:raise ValueError(f'Mixed generation settings at line {line_number}')
            if row['method'] not in methods:continue
            correct=float(row['correct']);raw=float(row['raw_uncertainty_value'])
            if correct not in (0.,1.) or not math.isfinite(raw):
                raise ValueError(f'Invalid correctness/raw score at line {line_number}; not filling missing values')
            if validate_sources and row['source_file'] not in checked:
                if not (root/row['source_file']).is_file():raise ValueError(f'Missing source file: {row["source_file"]}')
                checked.add(row['source_file'])
            key=(row['dataset'],row['setting'],row['model'],row['method'])
            groups[key]['correct'].append(correct);groups[key]['raw'].append(raw)
    if not groups:raise ValueError(f'No selected methods in {source}')
    output_dir.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.tmp')
    with temporary.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=COLUMNS);writer.writeheader()
        for (dataset,setting,model,method),values in sorted(groups.items()):
            writer.writerow(dict(dataset=dataset,setting=setting,model=model,method=method,
                                 num_samples=len(values['raw']),accuracy=mean(values['correct']),
                                 avg_uncertainty=mean(values['raw'])))
    os.replace(temporary,destination)
    print(f'Wrote {len(groups)} groups -> {destination}')
    return destination


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=['all',*SETTINGS],default='all')
    p.add_argument('--detail-dir',type=Path,default=FOLDER/'detail')
    p.add_argument('--output-dir',type=Path,default=FOLDER/'summary')
    p.add_argument('--methods',nargs='+',default=list(DEFAULT_METHODS))
    p.add_argument('--validate-sources',action='store_true')
    p.add_argument('--project-root',type=Path,default=FOLDER.parent)
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    try:
        for setting in SETTINGS:
            if args.split in ('all',setting):
                summarize(setting,args.detail_dir,args.output_dir,args.methods,args.validate_sources,args.project_root,args.overwrite)
    except (ValueError,OSError) as exc:p.exit(2,f'Error: {exc}\n')

if __name__=='__main__':main()
