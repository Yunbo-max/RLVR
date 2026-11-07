import importlib.util
import json
from pathlib import Path

import pytest

S=Path(__file__).resolve().parents[1]/"scripts"/"audit_eesd_future_outcome_pairs.py"
spec=importlib.util.spec_from_file_location("future_outcomes",S)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def _row(sid,future,*,seed=1701,visible=None):
    if visible is None:visible=["v1","v2","v3","v4"]
    return {
        "source_component_id":sid,"problem_id":"problem_"+sid,
        "seed":seed,"visible_test_ids":visible,
        "n_visible":len(visible),"future_test_id":future,
        "future_outcome":"PASS",
        "outcome_classes":["PASS","FAIL","EXCEPTION"],
        "fixed_probs":[.5,.4,.1],
        "adaptive_probs":[.7,.2,.1],
        "predictor_lock_sha256":"a"*64,
        "effective_mass":min(2.0,len(visible)),
    }


def _write(path,rows):
    path.write_text("".join(json.dumps(x)+"\n" for x in rows))
    return path


def test_raw_future_label_and_prediction_comparison(tmp_path):
    rows=[_row("a","future_a"),_row("b","future_b"),
          _row("a","future_a",seed=1702),_row("b","future_b",seed=1702)]
    file=_write(tmp_path/"raw.jsonl",rows)
    result=m.summarize(file,bootstrap=300,seed=11)
    assert result["seeds"]==[1701,1702]
    assert result["identical_future_test_populations_across_seeds"]
    r=result["results"]["1701"]["all"]
    assert r["adaptive_NLL"]<r["fixed_NLL"]
    assert r["argmax_agreement"]==1
    assert r["problem_cluster_gain_ci95"][0]>0
    assert r["observations"]==2


def test_future_id_cannot_be_part_of_visible_public_executions(tmp_path):
    raw=[_row("a","v3")]
    with pytest.raises(ValueError,match="leaks"):
        m.summarize(_write(tmp_path/"raw.jsonl",raw))


def test_prediction_probs_must_be_matched_valid_and_same_vocabulary(tmp_path):
    raw=[_row("a","future_a")]
    raw[0]["adaptive_probs"]=[.8,.2,.1]
    with pytest.raises(ValueError,match="sum to one"):
        m.summarize(_write(tmp_path/"raw.jsonl",raw))


def test_reused_future_in_a_different_visible_subset_rejected(tmp_path):
    raw=[_row("a","f1",visible=["v1","v2"]),
         _row("a","v1",visible=["v3","v4"])]
    with pytest.raises(ValueError,match="future test reused"):
        m.summarize(_write(tmp_path/"raw.jsonl",raw))


def test_predictions_across_seeds_with_different_populations_flagged(tmp_path):
    raw=[_row("a","f1",seed=1701),_row("a","f2",seed=1702)]
    result=m.summarize(_write(tmp_path/"raw.jsonl",raw),bootstrap=300)
    assert result["identical_future_test_populations_across_seeds"] is False
