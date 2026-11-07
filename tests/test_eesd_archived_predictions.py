"""Restored observation identities must be tied to saved labels/probabilities."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT=Path(__file__).resolve().parents[1]/"scripts/restore_eesd_archived_predictions.py"


def module():
    assert SCRIPT.is_file(), "archived prediction rows are not yet recoverable"
    spec=importlib.util.spec_from_file_location("restore_predictions",SCRIPT)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def source():
    return [{"source_component_id":"s0","problem_id":"real-p0","task_id":"candidate-0",
             "split":"primary","tests":[{"id":str(i),"input":str(i)} for i in range(6)],
             "outcomes":["PASS","WRONG","PASS","WRONG","PASS","WRONG"]}]


def test_restores_real_test_ids_and_problem_ids_in_order():
    m=module()
    rows=m.observation_rows(source(),seed=1701,classes=["PASS","WRONG"],
        labels=np.array([0,1]),clusters=np.array(["s0","s0"]),
        fixed=np.array([[.6,.4],[.4,.6]]),adaptive=np.array([[.7,.3],[.3,.7]]),
        masses=np.array([2.,3.]),lock="a"*64,visible=4)
    assert [r["future_test_id"] for r in rows]==["4","5"]
    assert all(r["problem_id"]=="real-p0" and r["visible_test_ids"]==["0","1","2","3"] for r in rows)
    assert [r["future_outcome"] for r in rows]==["PASS","WRONG"]


@pytest.mark.parametrize("fault",["label","cluster","overlap"])
def test_rejects_unbound_saved_observations(fault):
    m=module();records=source();labels=np.array([0,1]);clusters=np.array(["s0","s0"])
    if fault=="label":labels[0]=1
    if fault=="cluster":clusters[0]="s1"
    if fault=="overlap":records[0]["tests"][-1]["id"]="0"
    with pytest.raises(ValueError):
        m.observation_rows(records,seed=1701,classes=["PASS","WRONG"],labels=labels,
            clusters=clusters,fixed=np.array([[.6,.4],[.4,.6]]),
            adaptive=np.array([[.7,.3],[.3,.7]]),masses=np.array([2.,3.]),
            lock="a"*64,visible=4)
