"""Statistical aggregation over actual paired problem identifiers."""
from pathlib import Path
import importlib.util
import json
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('summarize_joint_mix',ROOT/'scripts/summarize_joint_mix.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


def mk(tmp,arm,seed,seq,adapter='shared'):
    root=tmp/f'{arm}-{seed}'
    (root/'round_5').mkdir(parents=True)
    plan={'schema':'eesd-joint-mix-plan-v1','experiment':'train','arm':arm,
          'seed':seed,'dataset_manifest_sha256':'manifest','model_revision':'pinned',
          'train_distinct_problems':20,'final_distinct_problems':len(seq),
          'rounds':5,'initial_adapter_sha256':adapter}
    (root/'plan.json').write_text(json.dumps(plan))
    (root/'completion.json').write_text(json.dumps({'status':'completed'}))
    (root/'round_5'/'evaluation.json').write_text(json.dumps({'rows':[
        {'problem_id':str(i),'all_pass':bool(val)} for i,val in enumerate(seq)]}))
    return root


def test_paired_cluster_report(tmp_path):
    joint=mk(tmp_path,'joint_pair',1701,[1,1,0,1])
    mix=mk(tmp_path,'mix_pair',1701,[1,0,0,0])
    report=s.aggregate([joint,mix],draws=200)
    assert report['contrasts'][0]['delta_pp']==50.
    assert report['contrasts'][0]['individual_seed_results'][0]['fixes']==2
    assert report['contrasts'][0]['problem_ids']==4


def test_reject_mismatched_adapter_or_problem_ids(tmp_path):
    j=mk(tmp_path,'joint_pair',1701,[1,1,1],adapter='one')
    m=mk(tmp_path,'mix_pair',1701,[1,0,0],adapter='two')
    with pytest.raises(ValueError,match='different initial adapters'):s.aggregate([j,m],draws=100)
