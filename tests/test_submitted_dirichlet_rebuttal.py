import importlib.util
from pathlib import Path

import numpy as np
import pytest


S=Path(__file__).resolve().parents[1]/"scripts"/"prepare_submitted_dirichlet_rebuttal_weights.py"
spec=importlib.util.spec_from_file_location("submitted_four_state",S)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def row(tid,before,after,rel,source=None,split="train"):
    return {"trajectory_id":tid,"source_component_id":source or tid,
            "split":split,"before_outcomes":before,"after_outcomes":after,
            "relevance":rel,"prompt":"repair","correction":"print(1)"}


def test_reconstruct_paper_analytic_example_four_uniform_all_fix():
    got=m.weights_for_record([1]*4,[0]*4,[1]*4,alpha=.1,lam=.5)
    assert got["effective_mass"]==pytest.approx(4.)
    assert got["adaptive_eed"]==pytest.approx(0.832081,abs=2e-6)
    assert got["fixed_mass"]==pytest.approx(0.832081,abs=2e-6)


def test_reconstruct_concentrated_all_fix():
    got=m.weights_for_record([1]*4,[0]*4,[.97,.01,.01,.01],
                             alpha=.1,lam=.5)
    assert got["effective_mass"]==pytest.approx(1.062,abs=.002)
    assert got["adaptive_eed"]==pytest.approx(.541946,abs=2e-6)
    assert got["fixed_mass"]==pytest.approx(.832081,abs=2e-6)
    assert got["adaptive_eed"]<got["fixed_mass"]


def test_jtox_same_delta_different_final_quality():
    # One FIX, no REGRESS: rest UNRESOLVED vs rest PRESERVED.
    unresolved=m.weights_for_record([1,1,1,1],[0,1,1,1],[1]*4)
    preserved=m.weights_for_record([1,0,0,0],[0,0,0,0],[1]*4)
    assert unresolved["adaptive_eed"]==pytest.approx(preserved["adaptive_eed"])
    assert unresolved["absolute_final_public_pass_fraction"]==pytest.approx(.25)
    assert preserved["absolute_final_public_pass_fraction"]==1
    assert unresolved["eed_terminal_quality"]==pytest.approx(
        preserved["eed_terminal_quality"]*.25)


def test_pure_mass_is_never_larger_than_fixed_mass_same_proportions():
    import random
    rng=random.Random(4)
    for _ in range(120):
        before=[rng.randrange(5) for _ in range(8)]
        after=[rng.randrange(5) for _ in range(8)]
        rel=[rng.random()+.000001 for _ in range(8)]
        v=m.weights_for_record(before,after,rel)
        assert 0<=v["adaptive_eed"]<=v["fixed_mass"]+1e-12


def test_train_only_mean_and_shuffle_preserve_weight_distribution():
    rows=[
        row("one",[1]*4,[0]*4,[1]*4),
        row("two",[0]*4,[0]*4,[1]*4),
        row("three",[0,1,1,0],[0,0,1,0],[.97,.01,.01,.01]),
        row("dev",[1]*4,[0]*4,[1]*4,split="development"),
    ]
    scored,man=m.add_weights(rows,seed=17)
    assert man["schema"]=="eesd-submitted-four-state-rebuttal-weights-v1"
    train=[x for x in scored if x["split"]=="train"]
    w=[x["rebuttal_weights"]["adaptive_eed"] for x in train]
    shuffled=[x["rebuttal_weights"]["shuffled_eed"] for x in train]
    assert sorted(w)==sorted(shuffled)
    assert sum(w)/3==pytest.approx(man["mean_matched_weight_from_train_only"])
    assert all(x["rebuttal_weights"]["constant_mass_matched"]==pytest.approx(
        man["mean_matched_weight_from_train_only"]) for x in scored)
    scored2,_=m.add_weights(rows,seed=17)
    assert scored==scored2
    assert "8.0" in scored[0]["rebuttal_diagnostics"]["kappa_relative_sweep"]


def test_train_dev_source_leakage_is_rejected():
    a=row("a",[1]*4,[0]*4,[1]*4,source="p")
    b=row("b",[1]*4,[0]*4,[1]*4,source="p",split="development")
    with pytest.raises(ValueError,match="overlapping source"):
        m.add_weights([a,b])


def test_permuted_kappa_scores_respect_scale_and_mass():
    scores=[.97,.01,.01,.01]
    near=m.pow_relevance(scores,.5)
    far=m.pow_relevance(scores,2.)
    assert m.effective_mass(near)>m.effective_mass(scores)
    assert m.effective_mass(far)<m.effective_mass(scores)
