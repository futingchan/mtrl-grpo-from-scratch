import json
from pathlib import Path

import pytest

from train.reward.core import (
    RewardWeights,
    extract_json,
    load_weights,
    match_todos,
    parse_answer,
    score,
    token_f1,
)

DOC = "some document"
GOLD = [
    {
        "task": "Send the Q3 report to Dana",
        "owner": "Alice",
        "due": "2026-09-15",
        "priority": "high",
    },
    {"task": "Book the team offsite", "owner": None, "due": None, "priority": "low"},
]
ENTITIES = {"names": ["Alice", "Bob", "Dana"], "dates": ["2026-09-12", "2026-09-15"]}


def answer(summary="Short summary.", todos=None):
    return json.dumps({"summary": summary, "todos": todos if todos is not None else GOLD})


def test_perfect_answer_total():
    b = score(DOC, answer(), GOLD, ENTITIES)
    assert b.schema_ok == 1.0
    assert b.todo_recall == 1.0
    assert b.todo_precision == 1.0
    assert b.owner_accuracy == 1.0
    assert b.length_penalty == 1.0
    assert b.hallucination_rate == 0.0
    assert b.judge_score == 0.0
    assert b.format_reward == 1.0
    assert b.total == pytest.approx(5.0)


@pytest.mark.parametrize("bad", ["", "not json at all", "42", "[1,2]", "null"])
def test_bad_answers_all_zero(bad):
    b = score(DOC, bad, GOLD, ENTITIES)
    assert b.to_dict() == {k: 0.0 for k in b.to_dict()}


def test_json_fence():
    text = "```json\n" + answer() + "\n```"
    assert score(DOC, text, GOLD, ENTITIES).schema_ok == 1.0


def test_prose_then_json():
    text = "Here is the answer:\n" + answer() + "\nDone."
    assert score(DOC, text, GOLD, ENTITIES).schema_ok == 1.0


def test_extra_top_level_key_fails():
    obj = json.loads(answer())
    obj["extra"] = 1
    assert parse_answer(json.dumps(obj)) is None
    assert score(DOC, json.dumps(obj), GOLD, ENTITIES).total == 0.0


def test_missing_todos_key_fails():
    assert parse_answer(json.dumps({"summary": "s"})) is None


def test_todo_extra_key_fails():
    bad = [dict(GOLD[0], note="x")]
    assert parse_answer(answer(todos=bad)) is None


def test_bad_priority_fails():
    bad = [dict(GOLD[0], priority="medium")]
    assert parse_answer(answer(todos=bad)) is None


def test_bad_due_fails():
    bad = [dict(GOLD[0], due="tomorrow")]
    assert parse_answer(answer(todos=bad)) is None


def test_iso_due_ok():
    t = [dict(GOLD[0], due="2026-09-12")]
    assert parse_answer(answer(todos=t)) is not None


def test_owner_none_ok():
    t = [dict(GOLD[0], owner=None)]
    assert parse_answer(answer(todos=t)) is not None


def test_empty_pred_todos_nonempty_gold():
    b = score(DOC, answer(todos=[]), GOLD, ENTITIES)
    assert b.todo_recall == 0.0
    assert b.todo_precision == 0.0
    # total = schema + len only
    assert b.total == pytest.approx(1.0 + 0.5)


def test_empty_pred_and_gold():
    b = score(DOC, answer(todos=[]), [], ENTITIES)
    assert b.todo_recall == 1.0
    assert b.todo_precision == 1.0


def test_duplicate_predictions_precision():
    pred = [dict(GOLD[0]), dict(GOLD[0])]
    b = score(DOC, answer(todos=pred), [GOLD[0]], ENTITIES)
    assert b.todo_recall == 1.0
    assert b.todo_precision == 0.5


def test_hallucinated_owner():
    pred = [dict(GOLD[0], owner="Eve")]
    b = score(DOC, answer(todos=pred), GOLD, ENTITIES)
    # fields: owner (hallucinated) + due (in sources) → 1/2
    assert b.hallucination_rate == 0.5
    b2 = score(DOC, answer(todos=[dict(GOLD[0], owner="Eve", due=None)]), GOLD, ENTITIES)
    assert b2.hallucination_rate == 1.0
    assert b2.total < 5.0


def test_hallucinated_date():
    pred = [dict(GOLD[0], due="2026-01-01")]
    b = score(DOC, answer(todos=pred), GOLD, ENTITIES)
    assert b.hallucination_rate == 0.5


def test_length_penalty():
    long_summary = " ".join(["word"] * 200)
    b = score(DOC, answer(summary=long_summary), GOLD, ENTITIES)
    assert b.length_penalty == 0.0

    s61 = " ".join(["word"] * 61)
    b = score(DOC, answer(summary=s61), GOLD, ENTITIES)
    assert b.length_penalty == pytest.approx(1 - 1 / 60)

    b = score(DOC, answer(summary=""), GOLD, ENTITIES)
    assert b.length_penalty == 0.0


def test_fuzzy_match():
    pred = [{"task": "send the Q3 report to Dana", "owner": None, "due": None, "priority": "high"}]
    gold = [
        {"task": "Send Q3 report to Dana by Friday", "owner": None, "due": None, "priority": "med"}
    ]
    matches = match_todos(pred, gold, 0.5)
    assert len(matches) == 1


def test_token_f1():
    assert token_f1("a b c", "a b c") == 1.0
    assert token_f1("a b", "b a") == 1.0
    assert token_f1("", "x") == 0.0
    x = token_f1("send the report", "report to dana")
    assert x == token_f1("report to dana", "send the report")


def test_match_todos_greedy():
    pred = [{"task": "alpha"}, {"task": "alpha beta"}]
    gold = [{"task": "alpha beta"}, {"task": "alpha"}]
    m = match_todos(pred, gold, 0.1)
    assert (1, 0, pytest.approx(1.0)) in m and (0, 1, pytest.approx(1.0)) in m
    # one-to-one: same gold cannot be matched twice
    assert len({j for _, j, _ in m}) == len(m)


def test_load_weights(tmp_path):
    weights, config = load_weights(Path("configs/reward.yaml"))
    assert weights == RewardWeights()
    assert config["matcher"] == "f1"
    assert config["match_threshold"] == 0.5

    p = tmp_path / "w.yaml"
    p.write_text("weights: {schema: 2.0, recall: 1.0}\n")
    w2, _ = load_weights(p)
    assert w2.schema == 2.0 and w2.recall == 1.0 and w2.prec == 1.0


def test_custom_weights_scale():
    w = RewardWeights(schema=3.0, recall=1.0)
    b = score(DOC, answer(), GOLD, ENTITIES, w)
    assert b.format_reward == 3.0
    assert b.total == pytest.approx(3.0 + 1.0 + 1.0 + 0.5 + 0.5)


def test_extract_json_braces_in_strings():
    text = '{"summary": "use { and } here", "todos": []}'
    assert extract_json(text) == text
    assert json.loads(extract_json(text))["summary"] == "use { and } here"


def test_breakdown_to_dict_keys():
    b = score(DOC, answer(), GOLD, ENTITIES).to_dict()
    assert set(b) == {
        "schema_ok",
        "todo_recall",
        "todo_precision",
        "owner_accuracy",
        "length_penalty",
        "hallucination_rate",
        "judge_score",
        "format_reward",
        "content_reward",
        "total",
    }
