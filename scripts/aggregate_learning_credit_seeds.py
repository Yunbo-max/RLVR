#!/usr/bin/env python3
"""Compare three independent training seeds for five-round EESD.

Requires each seed's full seven-arm matched report with identical frozen
RunBugRun task identities (dataset SHA). Report seed mean/range; no p-value
from a single seed is promoted to cross-seed evidence.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

PAIRS=('shapley_vs_direct','shapley_vs_sdpo','shapley_vs_frozen',
       'shapley_vs_random_subset','shapley_vs_matched_updates')


def aggregate(roots:list[Path])->dict:
    if len(roots)<2:raise ValueError('at least two independent training seeds required')
    seen=set();source_n=None;data_sha=None;measure={key:[] for key in PAIRS}
    per_seed=[]
    for root in roots:
        report=json.loads((root/'paired_summary.json').read_text())
        if report.get('schema')!='eesd-five-round-paired-report-v1':
            raise ValueError('invalid paired five-round report')
        identity=report['identity']; seed=identity['seed']
        if seed in seen:raise ValueError('duplicate training seed')
        seen.add(seed)
        if data_sha is None:
            data_sha=identity['dataset_manifest_sha256']
            source_n=report['source_count']
        if identity['dataset_manifest_sha256']!=data_sha or report['source_count']!=source_n:
            raise ValueError('cross-seed frozen final task population differs')
        primary=report['primary_comparisons']
        for key in PAIRS:measure[key].append(primary[key]['difference'])
        per_seed.append({'seed':seed,'effects':{key:primary[key]['difference'] for key in PAIRS}})
    results={}
    for key,values in measure.items():
        v=np.asarray(values)
        results[key]={'mean':float(v.mean()),'std_across_seeds':float(v.std(ddof=1)),
                      'min':float(v.min()),'max':float(v.max()),
                      'all_seeds_positive':bool(np.all(v>0)),
                      'per_seed':v.tolist()}
    return {'schema':'eesd-learning-credit-multi-seed-v1',
        'seeds':sorted(seen),'n_seeds':len(seen),
        'heldout_task_count_per_seed':source_n,'dataset_sha256':data_sha,
        'effects':results,'status':'multi-seed descriptive; preregister inference if needed'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('refuse to overwrite cross-seed summary')
    result=aggregate(a.roots)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))
