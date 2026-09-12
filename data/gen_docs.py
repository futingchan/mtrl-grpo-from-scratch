"""Synthetic document + gold-todo generator for the mtrl demo.

Stdlib + pyarrow only. Usage:

    python -m data.gen_docs --parquet data/ --n-train 2000 --n-eval 200 --seed 0
    python -m data.gen_docs --harbor data/harbor_tasks/sample --n 5 --seed 0 --split eval
"""

import argparse
import json
import random
import shutil
import stat
from datetime import date, timedelta
from pathlib import Path

POOL_PATH = Path(__file__).parent / "pool" / "pool.json"
POOL = json.loads(POOL_PATH.read_text())

CORE_PATH = Path(__file__).parent.parent / "train" / "reward" / "core.py"

BASE_DATE = date(2026, 9, 14)
DOC_TYPES = ["email", "meeting_notes", "prd", "chat_log"]

HIGH_CUES = ["urgent", "ASAP"]
LOW_CUES = ["when you get a chance", "low priority"]

GRADER_PY = '''"""Harbor task grader (stdlib only). Prints the scalar reward to stdout."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--breakdown", action="store_true")
    args = parser.parse_args()

    answer_path = Path(args.answer)
    if not answer_path.exists():
        print("0.0")
        return
    answer_text = answer_path.read_text()
    gold = json.loads(Path(args.gold).read_text())
    breakdown = score(
        gold["doc"],
        answer_text,
        gold["gold_todos"],
        gold["source_entities"],
    )
    if args.breakdown:
        print(json.dumps(breakdown.to_dict(), indent=2), file=sys.stderr)
    print(f"{breakdown.total:.6f}")


if __name__ == "__main__":
    main()
'''

TEST_SH = """#!/bin/bash
mkdir -p /logs/verifier
python /grader/grade.py --answer /workdir/answer.txt --gold /tests/gold.json > /logs/verifier/reward.txt
"""

INSTRUCTION_MD = """Read /workdir/doc.txt. Produce STRICT JSON {summary: <=60 words, todos: [{task, owner (name or null), due (YYYY-MM-DD or null), priority (high|med|low)}]}. Write it to /workdir/answer.txt (via the submit tool when available). No extra keys.
"""

DOCKERFILE = """FROM python:3.12-slim
WORKDIR /workdir
COPY doc.txt /workdir/doc.txt
COPY grader/ /grader/
"""


def _task_sentence(todo: dict) -> tuple[str | None, str]:
    """Render a todo as (speaker_hint, sentence). The hint is the owner, so
    chat-log lines read ``[HH:MM] <owner>: <task> ...`` instead of a random
    speaker; other renderers prefix the owner back onto the text."""
    task = todo["task"]
    owner = todo["owner"]
    if owner is not None:
        sentence = task
    else:
        sentence = f"Please {task[0].lower()}{task[1:]}"
    if todo["due"] is not None:
        sentence += f" by {todo['due']}"
    cue = todo.get("_cue")
    if cue:
        sentence += f" ({cue})"
    return owner, sentence + "."


def _fill(rng: random.Random, template: str, name: str | None = None) -> str:
    return template.format(
        project=rng.choice(POOL["projects"]),
        artifact=rng.choice(POOL["artifacts"]),
        name=name or rng.choice(POOL["names"]),
    )


def _with_hint(hint: str | None, text: str) -> str:
    return f"{hint}: {text}" if hint else text


def _render_doc(
    rng: random.Random,
    doc_type: str,
    sentences: list[tuple[str | None, str]],
    names: set[str],
    dates: set[str],
) -> str:
    """Wrap the shuffled (speaker_hint, sentence) list in a doc-type frame."""
    dates.add(BASE_DATE.isoformat())
    if doc_type == "email":
        sender = rng.choice(POOL["names"])
        recipient = rng.choice(POOL["names"])
        names.update([sender, recipient])
        header = (
            f"From: {sender}\nTo: {recipient}\n"
            f"Date: {BASE_DATE.isoformat()}\n"
            f"Subject: {rng.choice(POOL['projects'])} updates\n\n"
        )
        # group sentences into ~3 paragraphs
        rendered = [_with_hint(h, s) for h, s in sentences]
        chunks = [rendered[i::3] for i in range(3)]
        body = "\n\n".join(" ".join(c) for c in chunks if c)
        return header + body + "\n"
    if doc_type == "meeting_notes":
        attendees = rng.sample(POOL["names"], 4)
        names.update(attendees)
        header = (
            f"# {rng.choice(POOL['projects'])} sync notes\n"
            f"Date: {BASE_DATE.isoformat()}\n"
            f"Attendees: {', '.join(attendees)}\n\n"
        )
        return header + "\n".join(f"- {_with_hint(h, s)}" for h, s in sentences) + "\n"
    if doc_type == "prd":
        header = (
            f"# {rng.choice(POOL['projects'])} PRD\n"
            f"Last updated: {BASE_DATE.isoformat()}\n\n"
            "## Overview\n"
        )
        rendered = [_with_hint(h, s) for h, s in sentences]
        mid = len(rendered) // 2
        overview = " ".join(rendered[:mid])
        actions = "\n".join(f"- {s}" for s in rendered[mid:])
        return header + overview + "\n\n## Action items\n" + actions + "\n"
    # chat_log
    lines = []
    minutes = rng.randint(0, 59)
    hour = rng.randint(9, 17)
    for hint, s in sentences:
        speaker = hint or rng.choice(POOL["names"])
        names.add(speaker)
        lines.append(f"[{hour:02d}:{minutes:02d}] {speaker}: {s}")
        minutes += rng.randint(1, 4)
        if minutes >= 60:
            minutes -= 60
            hour = min(hour + 1, 23)
    return "\n".join(lines) + "\n"


def generate_doc(rng: random.Random, task_id: str) -> dict:
    doc_type = rng.choice(DOC_TYPES)
    names: set[str] = set()
    dates: set[str] = set()

    n_todos = rng.randint(2, 6)
    gold_todos = []
    sentences = []
    for _ in range(n_todos):
        task = _fill(rng, rng.choice(POOL["task_templates"]))
        owner = rng.choice(POOL["names"]) if rng.random() < 0.8 else None
        due = (
            (BASE_DATE + timedelta(days=rng.randint(0, 30))).isoformat()
            if rng.random() < 0.6
            else None
        )
        r = rng.random()
        if r < 0.3:
            priority, cue = "high", rng.choice(HIGH_CUES)
        elif r < 0.55:
            priority, cue = "low", rng.choice(LOW_CUES)
        else:
            priority, cue = "med", None
        todo = {
            "task": task,
            "owner": owner,
            "due": due,
            "priority": priority,
            "_cue": cue,
        }
        if owner:
            names.add(owner)
        if due:
            dates.add(due)
        sentences.append(_task_sentence(todo))
        del todo["_cue"]
        gold_todos.append(todo)

    n_filler = rng.randint(3, 8)
    for _ in range(n_filler):
        name = rng.choice(POOL["names"])
        names.add(name)
        sentences.append((None, _fill(rng, rng.choice(POOL["filler_sentences"]), name=name)))

    rng.shuffle(sentences)

    # pad to >= 200 words with more filler (before the single render pass,
    # so names added by the renderer can't leak into entities)
    while len(" ".join(s for _, s in sentences).split()) < 190:
        name = rng.choice(POOL["names"])
        names.add(name)
        sentences.append((None, _fill(rng, rng.choice(POOL["filler_sentences"]), name=name)))

    doc = _render_doc(rng, doc_type, sentences, names, dates)

    owners = [t["owner"] for t in gold_todos if t["owner"]] or ["unassigned"]
    gold_summary = rng.choice(POOL["summary_templates"]).format(
        doc_type=doc_type.replace("_", " "),
        project=rng.choice(POOL["projects"]),
        n=n_todos,
        owners=", ".join(owners[:3]),
    )

    return {
        "task_id": task_id,
        "doc": doc,
        "doc_type": doc_type,
        "gold_summary": gold_summary,
        "gold_todos": gold_todos,
        "source_entities": {"names": sorted(names), "dates": sorted(dates)},
    }


def write_parquet(rows: list[dict], out_dir: Path, split: str) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "task_id": [r["task_id"] for r in rows],
            "doc": [r["doc"] for r in rows],
            "doc_type": [r["doc_type"] for r in rows],
            "gold_summary": [r["gold_summary"] for r in rows],
            "gold_todos": [json.dumps(r["gold_todos"]) for r in rows],
            "source_entities": [json.dumps(r["source_entities"], sort_keys=True) for r in rows],
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split}.parquet"
    pq.write_table(table, path)
    return path


def write_harbor_task(row: dict, out_dir: Path) -> Path:
    task_dir = out_dir / "tasks" / row["task_id"]
    (task_dir / "environment" / "grader").mkdir(parents=True, exist_ok=True)
    (task_dir / "tests").mkdir(parents=True, exist_ok=True)

    (task_dir / "instruction.md").write_text(INSTRUCTION_MD)
    (task_dir / "task.toml").write_text(
        "[metadata]\n"
        f'task_id = "{row["task_id"]}"\n'
        f'doc_type = "{row["doc_type"]}"\n'
        f"difficulty = {len(row['gold_todos'])}\n"
        f"gold_summary = {json.dumps(row['gold_summary'])}\n\n"
        "[verifier]\n"
        "timeout_sec = 60\n"
    )
    (task_dir / "environment" / "Dockerfile").write_text(DOCKERFILE)
    (task_dir / "environment" / "doc.txt").write_text(row["doc"])
    shutil.copyfile(CORE_PATH, task_dir / "environment" / "grader" / "core.py")
    (task_dir / "environment" / "grader" / "grade.py").write_text(GRADER_PY)
    (task_dir / "tests" / "gold.json").write_text(
        json.dumps(
            {
                "gold_todos": row["gold_todos"],
                "source_entities": row["source_entities"],
                "doc": row["doc"],
            },
            indent=2,
        )
    )
    test_sh = task_dir / "tests" / "test.sh"
    test_sh.write_text(TEST_SH)
    test_sh.chmod(test_sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return task_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, help="output dir for train/eval parquet")
    parser.add_argument("--n-train", type=int, default=0)
    parser.add_argument("--n-eval", type=int, default=0)
    parser.add_argument("--harbor", type=Path, help="export Harbor task dirs here")
    parser.add_argument("--n", type=int, default=0, help="number of harbor tasks")
    parser.add_argument("--split", choices=["train", "eval"], default="eval")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.parquet:
        for split, n in (("train", args.n_train), ("eval", args.n_eval)):
            rng = random.Random(f"{args.seed}:{split}")
            rows = [generate_doc(rng, f"{split}-{i:05d}") for i in range(n)]
            path = write_parquet(rows, args.parquet, split)
            print(f"wrote {path} ({len(rows)} rows)")

    if args.harbor:
        # identical RNG stream to the parquet path so Harbor tasks and the
        # eval/train splits contain the same rows
        rng = random.Random(f"{args.seed}:{args.split}")
        for i in range(args.n):
            row = generate_doc(rng, f"{args.split}-{i:05d}")
            task_dir = write_harbor_task(row, args.harbor)
            print(f"wrote {task_dir}")


if __name__ == "__main__":
    main()
