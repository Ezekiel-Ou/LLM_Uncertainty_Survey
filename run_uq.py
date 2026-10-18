#!/usr/bin/env python3

import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--implementation', choices=['auto','local','lm-polygraph','approx'], default='auto')
    selected, remaining = selector.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining + (['--approx'] if selected.implementation == 'approx' else [])
    sys.path.insert(0,str(ROOT/'UQ-methods'))
    use_polygraph = selected.implementation == 'lm-polygraph'
    if selected.implementation == 'auto':
        
        route = argparse.ArgumentParser(add_help=False)
        route.add_argument('--method')
        route.add_argument('--list',action='store_true')
        requested, _ = route.parse_known_args(remaining)
        if requested.method and not requested.list:
            from uq_runtime.runner import normalize_method
            try:
                method = normalize_method(requested.method)
            except argparse.ArgumentTypeError:
                method = None
            registry = json.loads((ROOT/'registry.json').read_text())
            use_polygraph = any(v['method'] == method and v.get('implementation') == 'lm_polygraph'
                               for v in registry.values())
    if use_polygraph:
        main = importlib.import_module('lm_polygraph_backend.run').main
    else:
        from uq_runtime.runner import main
    main()
