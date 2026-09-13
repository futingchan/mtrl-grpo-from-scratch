"""Shared plumbing for the Tier A trainers (A1 minimal, A2 TRL)."""

import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from train.reward.core import RewardBreakdown, load_weights, score

SYSTEM_PROMPT = (
    "You are an assistant that extracts action items from a document.\n"
    "Read the document, then output ONLY a JSON object of the form\n"
    '{"summary": "<=60 words", "todos": [{"task": str, "owner": name or null, '
    '"due": "YYYY-MM-DD" or null, "priority": "high"|"med"|"low"}]}\n'
    "No prose, no markdown fences, no extra keys."
)


def build_messages(doc: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": doc},
    ]


def load_split(path: str | Path, n: int | None = None):
    """Load a parquet split into a TRL-compatible datasets.Dataset."""
    import pyarrow.parquet as pq
    from datasets import Dataset

    table = pq.read_table(path)
    if n is not None:
        table = table.slice(0, n)
    rows = table.to_pylist()
    return Dataset.from_list(
        [
            {
                "prompt": build_messages(r["doc"]),
                "doc": r["doc"],
                "gold_todos": json.loads(r["gold_todos"]),
                "source_entities": json.loads(r["source_entities"]),
                "task_id": r["task_id"],
            }
            for r in rows
        ]
    )


def completion_text(c) -> str:
    """Accept a raw string or a chat-format [{'role':'assistant','content':...}]."""
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c:
        return c[-1].get("content", "")
    return str(c)


def make_scorer(
    reward_config_path: str | Path,
) -> Callable[[str, str, list[dict], dict[str, list[str]]], RewardBreakdown]:
    """Configured breakdown scorer closing over yaml weights/thresholds."""
    weights, config = load_weights(reward_config_path)
    kw = {
        "threshold": config.get("match_threshold", 0.5),
        "summary_max_words": config.get("summary_max_words", 60),
    }

    def scorer(
        doc: str,
        completion: str,
        gold_todos: list[dict],
        source_entities: dict[str, list[str]],
    ) -> RewardBreakdown:
        return score(doc, completion, gold_todos, source_entities, weights, **kw)

    return scorer


def make_reward_funcs(reward_config_path: str | Path) -> list:
    """TRL-signature reward fns backed by core.score."""
    scorer = make_scorer(reward_config_path)

    def total_reward(
        prompts=None,
        completions=None,
        doc=None,
        gold_todos=None,
        source_entities=None,
        **_kw,
    ) -> list[float]:
        return [
            scorer(d, completion_text(c), g, e).total
            for c, d, g, e in zip(completions, doc, gold_todos, source_entities)
        ]

    def schema_ok(
        prompts=None,
        completions=None,
        doc=None,
        gold_todos=None,
        source_entities=None,
        **_kw,
    ) -> list[float]:
        return [
            float(scorer(d, completion_text(c), g, e).schema_ok)
            for c, d, g, e in zip(completions, doc, gold_todos, source_entities)
        ]

    total_reward.__name__ = "total_reward"
    schema_ok.__name__ = "schema_ok"
    return [total_reward, schema_ok]


@dataclass
class RewardBundle:
    funcs: list
    scorer: Callable[[str, str, list[dict], dict[str, list[str]]], RewardBreakdown]


def make_reward_bundle(reward_config_path: str | Path) -> RewardBundle:
    return RewardBundle(
        funcs=make_reward_funcs(reward_config_path),
        scorer=make_scorer(reward_config_path),
    )


@dataclass
class LocalSpec:
    """The three trainer slots. Tier B replaces this with a HarborSpec-shaped spec."""

    train_dataset: Any
    eval_dataset: Any
    reward_funcs: list
    scorer: Callable[[str, str, list[dict], dict[str, list[str]]], RewardBreakdown]
    reward_weights: list = field(default_factory=lambda: [1.0, 0.0])

    @classmethod
    def from_config(cls, cfg: dict) -> "LocalSpec":
        bundle = make_reward_bundle(cfg["reward_config"])
        return cls(
            train_dataset=load_split(cfg["data"]["train"]),
            eval_dataset=load_split(cfg["data"]["eval"], cfg["data"].get("eval_n")),
            reward_funcs=bundle.funcs,
            scorer=bundle.scorer,
        )


class RunLogger:
    """metrics.jsonl + run.json (viz schema) + samples.jsonl under out_dir."""

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.steps: list[dict] = []
        self.sample_batch: dict | None = None
        self.eval: dict | None = None
        self._metrics = open(self.out_dir / "metrics.jsonl", "a")  # noqa: SIM115 - long-lived handle
        self._samples = open(self.out_dir / "samples.jsonl", "a")  # noqa: SIM115

    def log(
        self,
        step: int,
        reward_mean: float,
        reward_std: float,
        kl: float,
        completion_len: float,
        schema_ok: float,
    ) -> None:
        rec = {
            "step": int(step),
            "reward_mean": float(reward_mean),
            "reward_std": float(reward_std),
            "kl": float(kl),
            "completion_len": float(completion_len),
            "schema_ok": float(schema_ok),
        }
        self.steps.append(rec)
        self._metrics.write(json.dumps(rec) + "\n")
        self._metrics.flush()
        self._write_run_json()

    def set_sample_batch(
        self,
        lengths: list,
        correct: list,
        logratio: list,
        rewards: list,
    ) -> None:
        """Store one prompt's group for the viz. Callers pass a single group
        (G entries), not the flattened B*G batch."""
        self.sample_batch = {
            "lengths": [int(x) for x in lengths],
            "correct": [int(x) for x in correct],
            "logratio": [float(x) for x in logratio],
            "rewards": [float(x) for x in rewards],
        }
        self._write_run_json()

    def write_samples(self, step: int, samples: list[dict]) -> None:
        for s in samples:
            self._samples.write(json.dumps({"step": step, **s}) + "\n")
        self._samples.flush()

    def set_eval(self, before: dict, after: dict) -> None:
        self.eval = {"before": before, "after": after}
        self._write_run_json()

    def _write_run_json(self) -> None:
        payload = {
            "steps": self.steps,
            "sample_batch": self.sample_batch,
            "eval": self.eval,
        }
        (self.out_dir / "run.json").write_text(json.dumps(payload))

    def close(self) -> None:
        self._metrics.close()
        self._samples.close()


def evaluate(
    model,
    tokenizer,
    dataset,
    scorer,
    max_new_tokens: int,
    device: str,
    eval_batch_size: int = 8,
) -> dict:
    """Greedy-decode eval in chunks; all metrics derive from one breakdown."""
    was_training = model.training
    model.eval()
    tokenizer.padding_side = "left"
    totals, recalls, schemas = [], [], []
    for start in range(0, len(dataset), eval_batch_size):
        rows = [dataset[i] for i in range(start, min(start + eval_batch_size, len(dataset)))]
        texts = [
            tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
            for row in rows
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_width = enc["input_ids"].shape[1]
        completions = [out[i, prompt_width:] for i in range(out.shape[0])]
        texts_out = tokenizer.batch_decode(completions, skip_special_tokens=True)
        for row, ctext in zip(rows, texts_out):
            b = scorer(row["doc"], ctext, row["gold_todos"], row["source_entities"])
            totals.append(b.total)
            recalls.append(b.todo_recall)
            schemas.append(b.schema_ok)
    if was_training:
        model.train()
    n = len(totals)
    return {
        "reward_mean": sum(totals) / n,
        "todo_recall_mean": sum(recalls) / n,
        "schema_ok_rate": sum(schemas) / n,
        "n": n,
    }


def _coerce(v: str):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    if v.startswith(("[", "{")):
        return yaml.safe_load(v)
    return v


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for o in overrides or []:
        key, _, val = o.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(val)
    return cfg


def pick_device(flag: str) -> str:
    import torch

    if flag == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return flag


def batch_indices(rng: random.Random, n_rows: int, b: int) -> list[int]:
    return [rng.randrange(n_rows) for _ in range(b)]


def completion_mask(ids: torch.Tensor, prompt_len: int, eos_id: int) -> torch.Tensor:
    """Keep completion tokens through the first EOS, or all if no EOS exists."""
    tokens = ids[:, prompt_len:]
    mask = torch.ones(tokens.shape, dtype=torch.float32, device=ids.device)
    for row, row_tokens in enumerate(tokens):
        eos_positions = (row_tokens == eos_id).nonzero(as_tuple=False)
        if len(eos_positions):
            mask[row, eos_positions[0, 0] + 1 :] = 0
    return mask


def seq_logprobs(
    model,
    ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Return completion-token logprobs without materializing prompt logits.

    ``logits_to_keep=T+1`` retains the position immediately before the first
    completion token plus the T completion predictions. Taking ``[:-1]`` then
    aligns each row with ids[:, prompt_len:]. Per-row log-softmax avoids a
    second full fp32 vocabulary tensor for the whole batch.
    """
    T = ids.shape[1] - prompt_len
    out = model(input_ids=ids, attention_mask=attention_mask, logits_to_keep=T + 1)
    logits = out.logits[:, :-1, :]
    targets = ids[:, prompt_len:]
    values = []
    for row_logits, row_targets in zip(logits, targets):
        values.append(
            F.log_softmax(row_logits.float(), dim=-1).gather(1, row_targets.unsqueeze(1)).squeeze(1)
        )
    return torch.stack(values)


def sample_batch_from_model(
    model,
    tokenizer,
    row: dict,
    scorer,
    total_reward,
    gcfg: dict,
    device: str,
) -> dict:
    """Sample one prompt group and build the viz run.json sample_batch."""
    was_training = model.training
    model.eval()
    tokenizer.padding_side = "left"
    G = min(8, gcfg["num_generations"])
    text = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
    enc = tokenizer([text], return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        gen = model.generate(
            **enc,
            do_sample=True,
            temperature=gcfg["temperature"],
            num_return_sequences=G,
            max_new_tokens=gcfg["max_completion_length"],
            pad_token_id=tokenizer.pad_token_id,
        )
    comp_part = completion_mask(gen, prompt_len, tokenizer.eos_token_id)
    attn = torch.cat((enc["attention_mask"].repeat_interleave(G, dim=0), comp_part), dim=1)
    texts_out = tokenizer.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)
    correct = [
        int(scorer(row["doc"], c, row["gold_todos"], row["source_entities"]).schema_ok)
        for c in texts_out
    ]
    rewards = total_reward(
        completions=texts_out,
        doc=[row["doc"]] * G,
        gold_todos=[row["gold_todos"]] * G,
        source_entities=[row["source_entities"]] * G,
    )
    with torch.no_grad():
        logp_new = seq_logprobs(model, gen, attn, prompt_len)
        with model.disable_adapter():
            logp_ref = seq_logprobs(model, gen, attn, prompt_len)
    if was_training:
        model.train()
    comp_lens = comp_part.sum(1).clamp(min=1)
    seq_x = (((logp_ref - logp_new) * comp_part).sum(1) / comp_lens).tolist()
    return {
        "lengths": [int(x) for x in comp_part.sum(1).tolist()],
        "correct": correct,
        "logratio": [float(x) for x in seq_x],
        "rewards": [float(x) for x in rewards],
    }
