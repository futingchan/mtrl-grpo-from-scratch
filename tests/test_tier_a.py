"""Tier A unit tests — no model downloads, no GPU."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # tier-a extra

from train.envs.sim_env import DocTodoEnv
from train.tier_a.common import (
    RunLogger,
    build_messages,
    completion_text,
    load_config,
    load_split,
    make_reward_bundle,
    make_reward_funcs,
)
from train.tier_a.grpo_minimal import completion_mask, grpo_advantages, grpo_loss_terms

REPO = Path(__file__).resolve().parent.parent


def test_build_messages_shape():
    m = build_messages("DOC TEXT")
    assert len(m) == 2
    assert m[0]["role"] == "system" and "JSON" in m[0]["content"]
    assert m[1] == {"role": "user", "content": "DOC TEXT"}


def test_completion_text():
    assert completion_text("hi") == "hi"
    assert completion_text([{"role": "assistant", "content": "yo"}]) == "yo"


def test_load_split_parses_columns():
    ds = load_split(REPO / "data" / "eval.parquet", n=3)
    assert len(ds) == 3
    row = ds[0]
    assert row["task_id"] == "eval-00000"
    assert isinstance(row["prompt"], list) and row["prompt"][0]["role"] == "system"
    assert isinstance(row["gold_todos"], list) and "task" in row["gold_todos"][0]
    assert isinstance(row["source_entities"], dict) and "names" in row["source_entities"]


def test_reward_funcs_perfect_and_garbage():
    ds = load_split(REPO / "data" / "eval.parquet", n=1)
    row = ds[0]
    funcs = make_reward_funcs(REPO / "configs" / "reward.yaml")
    total, schema = funcs
    perfect = json.dumps({"summary": "sum", "todos": row["gold_todos"]})
    out = total(
        completions=[perfect],
        doc=[row["doc"]],
        gold_todos=[row["gold_todos"]],
        source_entities=[row["source_entities"]],
    )
    assert out == [pytest.approx(5.0)]
    so = schema(
        completions=[perfect],
        doc=[row["doc"]],
        gold_todos=[row["gold_todos"]],
        source_entities=[row["source_entities"]],
    )
    assert so == [1.0]
    bad = total(
        completions=["garbage"],
        doc=[row["doc"]],
        gold_todos=[row["gold_todos"]],
        source_entities=[row["source_entities"]],
    )
    assert bad == [0.0]
    assert total.__name__ == "total_reward" and schema.__name__ == "schema_ok"


def test_runlogger_schema(tmp_path):
    lg = RunLogger(tmp_path)
    lg.log(0, 1.0, 0.5, 0.01, 100, 0.8)
    lg.log(1, 1.1, 0.4, 0.02, 95, 0.9)
    lg.set_sample_batch([10, 20], [1, 0], [0.1, -0.1], [0.5, 0.2])
    lg.close()
    run = json.loads((tmp_path / "run.json").read_text())
    assert len(run["steps"]) == 2
    assert set(run["steps"][0]) == {
        "step",
        "reward_mean",
        "reward_std",
        "kl",
        "completion_len",
        "schema_ok",
    }
    sb = run["sample_batch"]
    assert set(sb) == {"lengths", "correct", "logratio", "rewards"}
    assert all(len(v) == 8 for v in sb.values())
    assert sb["lengths"][:2] == [10, 20]
    lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2


def test_reward_bundle_scorer():
    ds = load_split(REPO / "data" / "eval.parquet", n=1)
    row = ds[0]
    bundle = make_reward_bundle(REPO / "configs" / "reward.yaml")
    assert len(bundle.funcs) == 2
    perfect = json.dumps({"summary": "sum", "todos": row["gold_todos"]})
    b = bundle.scorer(row["doc"], perfect, row["gold_todos"], row["source_entities"])
    assert b.total == pytest.approx(5.0) and b.schema_ok == 1.0


def test_load_config_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: {b: 1, c: x}\nsteps: 3\nflag: false\n")
    cfg = load_config(p, ["a.b=2", "steps=5", "flag=true", "a.new=hello"])
    assert cfg["a"]["b"] == 2 and cfg["steps"] == 5
    assert cfg["flag"] is True and cfg["a"]["new"] == "hello"


def test_doc_todo_env():
    ds = load_split(REPO / "data" / "eval.parquet", n=1)
    row = ds[0]
    env = DocTodoEnv()
    obs = env.reset(row)
    assert obs == row["doc"]
    _, reward, done, info = env.step(json.dumps({"summary": "s", "todos": row["gold_todos"]}))
    assert done and reward == pytest.approx(5.0)
    assert info["schema_ok"] == 1.0
    with pytest.raises(RuntimeError):
        env.step("again")


def test_grpo_advantages_groupwise():
    A = grpo_advantages([1.0, 3.0, 0.0, 0.0], G=2)
    assert A[0].item() == pytest.approx(-1.0) and A[1].item() == pytest.approx(1.0)
    assert A[2].item() == 0.0 and A[3].item() == 0.0  # constant group -> 0


def test_grpo_loss_terms_shapes_and_sign():
    B, T = 4, 6
    logp = torch.zeros(B, T)
    mask = torch.ones(B, T)
    adv = torch.tensor([1.0, -1.0, 2.0, 0.0])
    loss, sur, kl = grpo_loss_terms(logp, logp, logp, adv, mask, 0.2, 0.04)
    assert sur.item() == pytest.approx(adv.mean().item())
    assert kl.item() == pytest.approx(0.0)
    assert loss.item() == pytest.approx(-adv.mean().item())


def test_grpo_loss_terms_clipping():
    mask = torch.ones(1, 3)
    adv = torch.tensor([1.0])
    logp_old = torch.zeros(1, 3)
    logp_new = torch.full((1, 3), 0.5)  # ratio = e^0.5 ≈ 1.65 > 1+eps
    loss, sur, _kl = grpo_loss_terms(logp_new, logp_old, logp_old, adv, mask, 0.2, 0.0)
    assert sur.item() == pytest.approx(1.2)
    assert loss.item() == pytest.approx(-1.2)


def test_completion_mask_keeps_first_eos_only():
    ids = torch.tensor([[10, 11, 12, 99, 99, 99], [10, 11, 12, 13, 14, 15]])
    mask = completion_mask(ids, prompt_len=3, eos_id=99)
    assert mask.tolist() == [[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]

    ids = torch.tensor([[10, 11, 12, 13, 99, 99, 99, 99]])
    mask = completion_mask(ids, prompt_len=3, eos_id=99)
    assert mask.tolist() == [[1.0, 1.0, 0.0, 0.0, 0.0]]
