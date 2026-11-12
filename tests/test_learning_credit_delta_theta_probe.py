import importlib.util
import json
from pathlib import Path
import pytest

p=Path(__file__).resolve().parents[1]/'scripts'/'probe_learning_credit_delta_theta.py'
spec=importlib.util.spec_from_file_location('probe_learning_credit_delta_theta',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_accepted_training_anchors_only_not_final_meta(tmp_path):
    root=tmp_path/'shapley'
    rd=root/'round_0';rd.mkdir(parents=True)
    rows=[{'source_id':'t1','step':True,'selected_mask':1,'candidate_sources':['print(1)','print(2)']},
          {'source_id':'t2','step':False,'selected_mask':0,'candidate_sources':['print(3)']}]
    (rd/'training.jsonl').write_text('\n'.join(json.dumps(x) for x in rows))
    sources=[{'source_id':'t1','problem_id':'train-p1'},
             {'source_id':'t2','problem_id':'train-p2'}]
    anchors=m._read_anchors(root,0,sources,5)
    assert [r['source_id'] for r,_ in anchors]==['t1']
    assert anchors[0][1]['selected_mask']==1
    with pytest.raises(ValueError):m._read_anchors(root,5,sources,5)


def test_tokenized_executed_code_stays_fixed_across_intervention_arms():
    class T:
        def __call__(self,text,add_special_tokens):
            return {'input_ids':[ord(c)%79 for c in text]}
    tokens=m._cached_candidates(T(),{'candidate_sources':['print(1)','print(2)']},100)
    assert len(tokens)==2 and tokens[0][1]=='print(1)'
    assert tokens==m._cached_candidates(T(),{'candidate_sources':['print(1)','print(2)']},100)
    with pytest.raises(ValueError):m._cached_candidates(T(),{'candidate_sources':['x']},100)


def test_intervention_arms_include_direct_random_and_full():
    assert m.ARMS==('shapley','direct_max','random_subset','full_feedback')
