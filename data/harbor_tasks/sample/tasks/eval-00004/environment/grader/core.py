"""Reward core for the mtrl demo.

Stdlib-only on purpose: this module is vendored into Harbor task images,
so it must import and run with no third-party dependencies. ``yaml`` is
imported lazily inside :func:`load_weights` only.
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

PRIORITIES = {"high", "med", "low"}
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

TODO_KEYS = {"task", "owner", "due", "priority"}
ANSWER_KEYS = {"summary", "todos"}


@dataclass(frozen=True)
class RewardWeights:
    schema: float = 1.0
    recall: float = 2.0
    prec: float = 1.0
    owner: float = 0.5
    len: float = 0.5
    hall: float = 1.0
    judge: float = 0.0


@dataclass
class RewardBreakdown:
    schema_ok: float
    todo_recall: float
    todo_precision: float
    owner_accuracy: float
    length_penalty: float
    hallucination_rate: float
    judge_score: float
    format_reward: float
    content_reward: float
    total: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def load_weights(path: str | Path) -> tuple[RewardWeights, dict]:
    """Load reward weights from a YAML config; returns (weights, full config dict)."""
    import yaml

    config = yaml.safe_load(Path(path).read_text())
    weights = RewardWeights(**(config.get("weights") or {}))
    return weights, config


def extract_json(text: str) -> str | None:
    """Extract the first balanced {...} substring.

    Strips ```json / ``` fences, then brace-counts from the first '{',
    ignoring braces inside string literals.
    """
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_answer(text: str) -> dict | None:
    """Parse a model answer into {"summary": str, "todos": [...]} or None."""
    raw = extract_json(text)
    if raw is None:
        return None
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or set(obj.keys()) != ANSWER_KEYS:
        return None
    summary, todos = obj["summary"], obj["todos"]
    if not isinstance(summary, str) or not isinstance(todos, list):
        return None
    for item in todos:
        if not isinstance(item, dict) or set(item.keys()) != TODO_KEYS:
            return None
        if not isinstance(item["task"], str) or not item["task"]:
            return None
        if not (item["owner"] is None or isinstance(item["owner"], str)):
            return None
        if not (item["due"] is None or ISO_DATE_RE.match(str(item["due"]))):
            return None
        if item["priority"] not in PRIORITIES:
            return None
    return obj


def normalize(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).split()


def token_f1(a: str, b: str) -> float:
    """Multiset token F1; 0.0 if either side is empty."""
    ta, tb = normalize(a), normalize(b)
    if not ta or not tb:
        return 0.0
    remaining: dict[str, int] = {}
    for t in tb:
        remaining[t] = remaining.get(t, 0) + 1
    common = 0
    for t in ta:
        if remaining.get(t, 0) > 0:
            remaining[t] -= 1
            common += 1
    if common == 0:
        return 0.0
    prec = common / len(ta)
    rec = common / len(tb)
    return 2 * prec * rec / (prec + rec)


def match_todos(
    pred: list[dict],
    gold: list[dict],
    threshold: float,
    scorer=token_f1,
) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching by descending score, pairs >= threshold."""
    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            s = scorer(p["task"], g["task"])
            if s >= threshold:
                pairs.append((s, i, j))
    pairs.sort(key=lambda x: -x[0])

    used_pred: set[int] = set()
    used_gold: set[int] = set()
    matches = []
    for s, i, j in pairs:
        if i in used_pred or j in used_gold:
            continue
        used_pred.add(i)
        used_gold.add(j)
        matches.append((i, j, s))
    return matches


def _norm_owner(owner) -> str | None:
    if owner is None:
        return None
    return " ".join(normalize(owner))


def _zero() -> RewardBreakdown:
    return RewardBreakdown(
        schema_ok=0.0,
        todo_recall=0.0,
        todo_precision=0.0,
        owner_accuracy=0.0,
        length_penalty=0.0,
        hallucination_rate=0.0,
        judge_score=0.0,
        format_reward=0.0,
        content_reward=0.0,
        total=0.0,
    )


def score(
    doc: str,
    answer_text: str,
    gold_todos: list[dict],
    source_entities: dict[str, list[str]],
    weights: RewardWeights = RewardWeights(),  # noqa: B008 - frozen dataclass, safe default
    *,
    threshold: float = 0.5,
    summary_max_words: int = 60,
    scorer=token_f1,
) -> RewardBreakdown:
    del doc  # reserved for future content signals (e.g. judge)

    parsed = parse_answer(answer_text)
    if parsed is None:
        return _zero()

    w = weights
    schema_ok = 1.0
    format_reward = w.schema * 1.0

    pred = parsed["todos"]
    matches = match_todos(pred, gold_todos, threshold, scorer)
    m, n_pred, n_gold = len(matches), len(pred), len(gold_todos)

    if n_gold == 0:
        todo_recall = 1.0 if n_pred == 0 else 0.0
    else:
        todo_recall = m / n_gold

    if n_pred == 0:
        todo_precision = 1.0 if n_gold == 0 else 0.0
    elif todo_recall == 0:
        todo_precision = 0.0
    else:
        todo_precision = m / n_pred

    if m == 0:
        owner_accuracy = 0.0
    else:
        correct = sum(
            1
            for i, j, _ in matches
            if _norm_owner(pred[i]["owner"]) == _norm_owner(gold_todos[j]["owner"])
        )
        owner_accuracy = correct / m

    words = len(parsed["summary"].split())
    if words == 0:
        length_penalty = 0.0
    elif words <= summary_max_words:
        length_penalty = 1.0
    else:
        length_penalty = max(0.0, 1.0 - (words - summary_max_words) / summary_max_words)

    gold_names = {_norm_owner(n) for n in source_entities.get("names", [])}
    gold_dates = set(source_entities.get("dates", []))
    hallucinated = 0
    n_fields = 0
    for item in pred:
        if item["owner"] is not None:
            n_fields += 1
            if _norm_owner(item["owner"]) not in gold_names:
                hallucinated += 1
        if item["due"] is not None:
            n_fields += 1
            if item["due"] not in gold_dates:
                hallucinated += 1
    hallucination_rate = hallucinated / n_fields if n_fields else 0.0

    judge_score = 0.0
    content_reward = (
        w.recall * todo_recall
        + w.prec * todo_precision
        + w.owner * owner_accuracy
        + w.len * length_penalty
        - w.hall * hallucination_rate
        + w.judge * judge_score
    )

    return RewardBreakdown(
        schema_ok=schema_ok,
        todo_recall=todo_recall,
        todo_precision=todo_precision,
        owner_accuracy=owner_accuracy,
        length_penalty=length_penalty,
        hallucination_rate=hallucination_rate,
        judge_score=judge_score,
        format_reward=format_reward,
        content_reward=content_reward,
        total=format_reward + content_reward,
    )


def reward_fn(
    doc: str,
    answer_text: str,
    gold_todos: list[dict],
    source_entities: dict[str, list[str]],
    config_path: str | Path | None = None,
) -> float:
    """Convenience scalar reward for trainer integration."""
    weights = RewardWeights()
    kwargs: dict = {}
    if config_path is not None:
        weights, config = load_weights(config_path)
        kwargs["threshold"] = config.get("match_threshold", 0.5)
        kwargs["summary_max_words"] = config.get("summary_max_words", 60)
    return score(doc, answer_text, gold_todos, source_entities, weights, **kwargs).total
