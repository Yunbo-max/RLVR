"""CPU-only source-pair provenance and token-granularity scientific invariants."""
import json

import numpy as np
import pytest

from eesd.token_aware import (
    original_program,
    edited_completion_mask,
    matched_random_mask,
    prefix_multipliers,
    token_multipliers,
)


class CharacterTokenizer:
    eos_token = "\u241F"  # unit separator symbol
    eos_token_id = 918273

    def __call__(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return {"input_ids": [
            self.eos_token_id if c == self.eos_token else ord(c)
            for c in text
        ]}


def test_source_recovered_from_exact_repair_prompt_json():
    program="def f(x):\n return x+1\n"
    row={"prompt":"Repair the following Python code.\n"+
         json.dumps({"program":program,"public_executions":[{"id":1}]})}
    assert original_program(row)==(program,"prompt.program")
    assert original_program({**row,"original":program})==(program,"original")
    with pytest.raises(ValueError,match="differs"):
        original_program({**row,"original":"return 3"})


def test_pure_synthesis_prompt_cannot_be_mislabeled_token_edit():
    with pytest.raises(ValueError,match="requires original code"):
        original_program({"prompt":"Write a program to add two integers."})


def test_single_edit_uses_correct_training_token_positions_not_whole_program():
    t=CharacterTokenizer()
    orig="abc"
    corrected="axc"
    ids=t(corrected+t.eos_token,add_special_tokens=False)["input_ids"]
    m=edited_completion_mask(t,original=orig,correction=corrected,target_ids=ids)
    assert m.tolist()==[False,True,False,False]
    expected=token_multipliers(m,boost=2.)
    assert np.allclose(expected,[2./3.,2.,2./3.,2./3.])
    assert expected.mean()==pytest.approx(1.,abs=1e-12)
    assert (np.ones(4)*expected).mean()==pytest.approx(1.)
    assert np.array_equal(prefix_multipliers(
        m,count=2,boost=2.,mode="edit",identity="x",seed=17
    ),[.5,1.5])


def test_all_or_no_edit_reduce_to_original_uniform_ce():
    assert np.array_equal(token_multipliers([True]*7,boost=2.),np.ones(7))
    assert np.array_equal(token_multipliers([False]*7,boost=2.),np.ones(7))
    assert np.array_equal(prefix_multipliers(
        [True,False],count=2,boost=2.,mode="uniform",
        identity="a",seed=1
    ),np.ones(2))


def test_random_token_control_matches_edited_count_and_is_reproducible():
    mask=[False]*20
    for i in [1,3,7,8]:mask[i]=True
    random1=matched_random_mask(mask,identity="a",seed=1701,eos_excluded=True)
    random2=matched_random_mask(mask,identity="a",seed=1701,eos_excluded=True)
    assert np.array_equal(random1,random2)
    assert random1.sum()==4 and not random1[-1]
    w=prefix_multipliers(mask,count=20,mode="random",boost=2.,
                         identity="a",seed=1701,eos_excluded=True)
    assert w.mean()==pytest.approx(1.)
    assert int(np.count_nonzero(w>w.mean()))==4


def test_truncated_prefix_is_re_normalized_and_bad_token_mismatch_fails():
    t=CharacterTokenizer()
    full=t("axc"+t.eos_token,add_special_tokens=False)["input_ids"]
    m=edited_completion_mask(t,original="abc",correction="axc",target_ids=full[:3])
    assert m.tolist()==[False,True,False]
    for count in (1,2,3):
        w=prefix_multipliers(m,count=count,mode="edit",boost=5.,
                             identity="x",seed=1)
        assert w.mean()==pytest.approx(1.,abs=1e-12)
    with pytest.raises(ValueError,match="exactly match"):
        edited_completion_mask(t,original="abc",correction="axc",target_ids=[77,88])


def test_zero_ce_weight_remains_zero_under_any_token_multiplier():
    mask=[False,True,False,False]
    weights=token_multipliers(mask,boost=2.)
    ce=np.array([1.,3.,2.,2.])
    assert 0.*(ce*weights).mean()==0.
    assert .6*(ce*weights).mean()!=pytest.approx(.6*ce.mean())


def test_bad_options_rejected():
    with pytest.raises(ValueError):
        token_multipliers([False,True],boost=float("nan"))
    with pytest.raises(ValueError):
        prefix_multipliers([False,True],count=4,boost=2.,
                           mode="edit",identity="a",seed=1)
