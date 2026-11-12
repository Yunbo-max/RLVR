import importlib.util
import json
from pathlib import Path
import pytest

P=Path(__file__).resolve().parents[1]/'scripts'/'summarize_recursive_sdpo_matrix.py'
s=importlib.util.spec_from_file_location('summary_sdpo',P)
m=importlib.util.module_from_spec(s)
s.loader.exec_module(m)


def fake_tree(path):
    for a in m.ARMS:
        (path/a).mkdir(parents=True)
        (path/a/'plan.json').write_text(json.dumps({'model_id':'m','model_revision':'sha',
           'data_manifest_sha256':'h','train_sources':2,'eval_sources':3,'rollouts':2}))
        for r in range(4):
            (path/a/f'round_{r}').mkdir()
            flags=[0,1,0] if r==0 or a=='no_update' else ([1,1,0] if a=='shapley' else [0,1,0])
            rows=[{'source_id':f'p{i}','all_pass':bool(flags[i])} for i in range(3)]
            (path/a/f'round_{r}'/'evaluation.json').write_text(json.dumps({'rows':rows}))


def test_paired_summary_keeps_same_base(tmp_path):
    fake_tree(tmp_path)
    out=m.summarize(tmp_path,bootstrap_draws=100)
    assert out['arms']['shapley'][3]['delta_vs_sdpo']==pytest.approx(1/3)
    assert out['arms']['shapley'][3]['fixes_vs_round0']==1
    assert out['arms']['shapley'][3]['regressions_vs_round0']==0


def test_initial_baseline_mismatch_detected(tmp_path):
    fake_tree(tmp_path)
    file=tmp_path/'sdpo/round_0/evaluation.json'
    row=json.loads(file.read_text());row['rows'][0]['all_pass']=True
    file.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='initial model evaluation differs'):
        m.summarize(tmp_path)
