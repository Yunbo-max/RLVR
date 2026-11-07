import importlib.util
import json
from pathlib import Path


BASE=Path(__file__).resolve().parents[1]/"scripts"


def _load(name):
    p=BASE/name
    spec=importlib.util.spec_from_file_location(name.replace(".","_"),p)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _raw(tid,split,before,after):
    return {
      "trajectory_id":tid,
      "source_component_id":tid,
      "split":split,
      "prompt":"repair this buggy program",
      "correction":"print(1)",
      "before_outcomes":before,
      "after_outcomes":after,
      "relevance":[1.,1.,1.,1.]
    }


def test_four_state_rebuttal_inputs_have_shared_schedule_and_different_weights(tmp_path):
    scorer=_load("prepare_submitted_dirichlet_rebuttal_weights.py")
    materializer=_load("materialize_submitted_dirichlet_training_inputs.py")
    rows=[
      _raw("a","train",[1]*4,[0]*4),
      _raw("b","train",[0]*4,[0]*4),
      _raw("c","train",[0,1,1,0],[0,0,1,0]),
      _raw("d","development",[1]*4,[0]*4)
    ]
    scored,summary=scorer.add_weights(rows)
    source=tmp_path/"weights.jsonl"
    source.write_text("".join(json.dumps(row)+"\n" for row in scored))
    out=tmp_path/"arms"
    receipt=materializer.materialize(source,out)
    assert receipt["scientific_result"].startswith("none")
    assert receipt["identity_order_sha256"]
    assert receipt["arms"]["equal_weight"]["train_positive"]==3
    matrix={}
    for arm in materializer.ARMS:
        data=[json.loads(x) for x in
              (out/arm/"scored-corrections.jsonl").read_text().split("\n") if x]
        assert [z["trajectory_id"] for z in data]==["a","b","c","d"]
        assert all(z["training_slot"]=="eesd_full" for z in data)
        assert len(data)==4
        assert all(z["training_weights"]["eesd_full"]==
                   z["rebuttal_weights"][arm] for z in data)
        matrix[arm]=data
    assert matrix["equal_weight"][0]["training_weights"]["eesd_full"]==1.0
    assert matrix["adaptive_eed"][1]["training_weights"]["eesd_full"]==0.0
    assert matrix["fixed_mass"][0]["training_weights"]["eesd_full"]>=matrix[
        "adaptive_eed"][0]["training_weights"]["eesd_full"]


def test_ensure_zero_kl_scheduling_is_comparable_between_arms():
    trainer=_load("run_eesd_weighted_sft.py")
    raw={"split":"train"}
    a=[dict(raw,_weight=0.),dict(raw,_weight=.5),
       {"split":"development","_weight":1.}]
    b=[dict(raw,_weight=.6),dict(raw,_weight=.7),
       {"split":"development","_weight":1.}]
    assert trainer.scheduled_training_indices(a,include_zero_weight_kl=True)==[0,1]
    assert trainer.scheduled_training_indices(b,include_zero_weight_kl=True)==[0,1]
    assert trainer.scheduled_training_indices(a,include_zero_weight_kl=False)==[1]
