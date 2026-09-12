"""DocTodoEnv: trivial single-step environment over a generated doc row.

reset(task_row) -> instruction (the document text)
step(action_text) -> (obs=None, reward, done=True, info=breakdown dict)

This is the Tier A seam: Tier B swaps it for a real Harbor task env with the
same reset/step contract.
"""

from train.reward.core import score


class DocTodoEnv:
    def __init__(self, weights=None, **score_kwargs):
        self.weights = weights
        self.score_kwargs = score_kwargs
        self._row = None
        self._done = True

    def reset(self, task_row: dict) -> str:
        self._row = task_row
        self._done = False
        return task_row["doc"]

    def step(self, action_text: str):
        if self._done or self._row is None:
            raise RuntimeError("step() before reset() or after done")
        kw = dict(self.score_kwargs)
        if self.weights is not None:
            kw["weights"] = self.weights
        b = score(
            self._row["doc"],
            action_text,
            self._row["gold_todos"],
            self._row["source_entities"],
            **kw,
        )
        self._done = True
        return None, b.total, True, b.to_dict()
