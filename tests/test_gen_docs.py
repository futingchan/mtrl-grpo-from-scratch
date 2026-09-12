import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from data.gen_docs import generate_doc
from train.reward import core

REPO = Path(__file__).resolve().parent.parent
SAMPLE = REPO / "data" / "harbor_tasks" / "sample"


def _rng(seed=0):
    import random

    return random.Random(seed)


def test_determinism_same_seed():
    a = generate_doc(_rng(7), "eval-00000")
    b = generate_doc(_rng(7), "eval-00000")
    assert a == b


def test_determinism_different_seed():
    a = generate_doc(_rng(0), "eval-00000")
    b = generate_doc(_rng(1), "eval-00000")
    assert a != b


def test_gold_todos_grounded():
    row = generate_doc(_rng(0), "t")
    joined = " ".join(core.normalize(row["doc"]))
    for todo in row["gold_todos"]:
        assert " ".join(core.normalize(todo["task"])) in joined
        if todo["owner"] is not None:
            assert todo["owner"] in row["source_entities"]["names"]
        if todo["due"] is not None:
            assert todo["due"] in row["source_entities"]["dates"]


def test_gold_answers_score_perfect():
    rng = _rng(0)
    for i in range(200):
        row = generate_doc(rng, f"eval-{i:05d}")
        gold_answer = json.dumps({"summary": row["gold_summary"], "todos": row["gold_todos"]})
        assert core.parse_answer(gold_answer) is not None
        b = core.score(row["doc"], gold_answer, row["gold_todos"], row["source_entities"])
        assert b.total == pytest.approx(5.0), f"{row['task_id']}: {b.to_dict()}"


def test_doc_word_counts():
    rng = _rng(0)
    for i in range(200):
        row = generate_doc(rng, f"eval-{i:05d}")
        words = len(row["doc"].split())
        assert 200 <= words <= 1100, f"{row['task_id']}: {words} words"


def _task_dirs():
    return sorted(p for p in (SAMPLE / "tasks").iterdir() if p.is_dir())


def test_sample_tasks_exist():
    assert len(_task_dirs()) == 5


def test_harbor_export_matches_eval_parquet():
    import pyarrow.parquet as pq

    row = pq.read_table(REPO / "data" / "eval.parquet").slice(0, 1).to_pylist()[0]
    assert row["task_id"] == "eval-00000"
    task_dir = SAMPLE / "tasks" / "eval-00000"
    assert (task_dir / "environment" / "doc.txt").read_text() == row["doc"]
    gold = json.loads((task_dir / "tests" / "gold.json").read_text())
    assert gold["gold_todos"] == json.loads(row["gold_todos"])
    assert gold["source_entities"] == json.loads(row["source_entities"])


def test_grader_parity_and_identity():
    assert len(_task_dirs()) == 5
    for task_dir in _task_dirs():
        env = task_dir / "environment"
        assert (env / "grader" / "core.py").read_bytes() == (
            REPO / "train" / "reward" / "core.py"
        ).read_bytes()

        gold = json.loads((task_dir / "tests" / "gold.json").read_text())
        answer = json.dumps({"summary": "ok", "todos": gold["gold_todos"]})
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(answer)
            answer_path = f.name

        out = subprocess.run(
            [
                sys.executable,
                "grader/grade.py",
                "--answer",
                answer_path,
                "--gold",
                "../tests/gold.json",
            ],
            cwd=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert out.returncode == 0, out.stderr
        expected = core.score(gold["doc"], answer, gold["gold_todos"], gold["source_entities"])
        assert float(out.stdout.strip()) == pytest.approx(expected.total)

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as bad:
            bad.write("{}")
        out = subprocess.run(
            [
                sys.executable,
                "grader/grade.py",
                "--answer",
                bad.name,
                "--gold",
                "../tests/gold.json",
            ],
            cwd=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert float(out.stdout.strip()) == 0.0


def test_missing_answer_scores_zero():
    task_dir = _task_dirs()[0]
    out = subprocess.run(
        [
            sys.executable,
            "grader/grade.py",
            "--answer",
            "/nonexistent/answer.txt",
            "--gold",
            "../tests/gold.json",
        ],
        cwd=task_dir / "environment",
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0
    assert float(out.stdout.strip()) == 0.0
