import importlib.util
import json
from pathlib import Path
import pytest

P=Path(__file__).resolve().parents[1]/'scripts'/'summarize_behavioral_shapley.py'
spec=importlib.util.spec_from_file_location('summary_behavioral',P)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def tree(path):
    for a in m.ARMS:
        arm=path/a;arm.mkdir()
        plan={'arm':a,'isolation':'bubblewrap_mandatory_no_soft_fallback',
              'model':'Qwen','revision':'rev','train_sources':2,'eval_sources':3,
              'rollouts':2,'feedback_tests':4,'calibration_budget_per_rollout':4,
              'rounds':3,'dataset_manifest_sha256':'sha','seed':1701,
              'temperature':.7,'max_new_tokens':256}
        (arm/'plan.json').write_text(json.dumps(plan))
        for r in range(4):
            folder=arm/f'round_{r}';folder.mkdir()
            vals=[0,1,0] if r==0 or a=='no_update' else [1,1,0] if a=='shapley_select' else [0,1,0]
            d={'schema':'eesd-behavioral-fresh-eval-v1','round':r,
               'rows':[{'source_id':f'id{i}','all_pass':bool(vals[i]),
                       'passes':int(vals[i]),'total_tests':1,
                       'code_sha256':f'base{i}' if r==0 else f'{a}-{r}-{i}'} for i in range(3)]}
            (folder/'evaluation.json').write_text(json.dumps(d))


def test_paired_compare_is_source_aware(tmp_path):
    tree(tmp_path)
    result=m.summarize(tmp_path,bootstraps=200)
    assert result['arms']['shapley_select'][3]['pass_at_1_delta_vs_sdpo']==pytest.approx(1/3)
    assert result['arms']['shapley_select'][3]['fixes_vs_initial']==1


def test_initial_checkpoint_mismatch_rejected(tmp_path):
    tree(tmp_path)
    f=tmp_path/'sdpo/round_0/evaluation.json';d=json.loads(f.read_text())
    d['rows'][0]['code_sha256']='different';f.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='identical initial checkpoint'):
        m.summarize(tmp_path)


def test_cross_arm_calibration_budget_mismatch_rejected(tmp_path):
    tree(tmp_path)
    f=tmp_path/'audit_gate'/'plan.json';d=json.loads(f.read_text())
    d['calibration_budget_per_rollout']=5;f.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='protocol mismatch'):
        m.summarize(tmp_path)
