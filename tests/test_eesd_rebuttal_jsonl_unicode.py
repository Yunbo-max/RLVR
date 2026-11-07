"""Archived RunBugRun JSONL can contain Unicode NEL inside source/test text."""
import json

from eesd.distillation import TRAIN_RULES
from eesd.training_outcomes import read_training_rows


def test_jsonl_reader_does_not_split_unicode_nel_inside_quoted_prompt(tmp_path):
    def row(tid, split):
        return {
            "trajectory_id":tid,"source_component_id":tid,
            "split":split,
            "prompt":"repair a public test with a literal NEL: \u0085 end",
            "correction":"print('correct output')",
            "before_outcomes":[1,1,1,1],
            "after_outcomes":[0,0,0,0],
            "relevance":[1.0,1.0,1.0,1.0],
            "training_weights":{k:(0.6 if k=="eesd_full" else 0.)
                                for k in TRAIN_RULES},
        }
    records=[row("train1","train"),row("dev1","development")]
    p=tmp_path/"unicode.jsonl"
    p.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records))
    assert "\u0085" in p.read_text()
    rows,report=read_training_rows(p,"eesd_full")
    assert len(rows)==2
    assert report["train_rows"]==1
    assert report["development_rows"]==1
    assert rows[0]["prompt"]==records[0]["prompt"]
