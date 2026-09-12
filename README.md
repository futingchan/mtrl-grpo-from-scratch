# GRPO from scratch, on a 2017 GPU

I kept reading writeups of GRPO and RLVR ("RL with verifiable rewards") and
nodding along without owning the mechanics. The fix for that, for me, is always
the same: build the thing. So I trained a 0.5B parameter model with a
hand-written GRPO loop on the only GPU I own — a GTX 1080 Ti — and built a
small dashboard that animates what the loop is actually doing.

This repo is the whole story: the task, the reward, the ~230-line training
loop, the dashboard, and the numbers I got.

**Try the dashboard first** (no GPU, no install — it's a static page):

```bash
python3 -m http.server 4173 -d viz/dist    # → http://localhost:4173
```

Five panels: the RL loop animated end-to-end, a gridworld you can watch a
tabular policy learn, a PPO/GRPO/DPO side-by-side on the same batch of
completions, a reward-shaping slider, and the real training curve from the run
in this repo.

## The task

I wanted a task where the reward is *mostly checkable by code* — that's the
whole premise of RLVR, and it's what lets you skip training a reward model.
The task: given a document (email thread, meeting notes, chat log), emit
strict JSON:

```json
{
  "summary": "<= 60 words",
  "todos": [{"task": "...", "owner": "name|null", "due": "YYYY-MM-DD|null", "priority": "high|med|low"}]
}
```

The reward ([`train/reward/core.py`](train/reward/core.py)) is a weighted sum
of things a grader can verify deterministically:

- **schema** — does it parse, exactly `{summary, todos}`, right types, valid
  enum, ISO dates. Unparseable ⇒ total reward 0. Format first, content second.
- **todo recall** — fraction of gold todos found, matched greedily by
  multiset token-F1 ≥ 0.5. This is the dominant term (weight 2.0).
- **todo precision** — gated on recall > 0, so emitting *zero* todos can't
  score a free point on precision. First of several anti-hacking details;
  the reward is the only thing the policy can exploit, so it's written and
  tested like a compiler.
- **owner accuracy** — only over matched todos.
- **length** — linear decay past 60 words.
- **hallucination penalty** — any `owner`/`due` the model emits that doesn't
  appear in the source document's entity list counts against it.

The dataset is synthetic (`data/gen_docs.py`, 2k train / 200 eval, checked in
as parquet). Gold todos are known because the generator *injects* them —
you can't grade recall against documents whose todos you don't know.

## The loop

[`train/tier_a/grpo_minimal.py`](train/tier_a/grpo_minimal.py) is the point of
the repo — GRPO with no framework between me and the tensors:

```python
gen = model.generate(prompt, num_return_sequences=G, do_sample=True)   # G rollouts per prompt
rewards = reward_fn(doc, completions)                                  # [B*G], from the verifier
adv = (rewards - group_mean) / (group_std + eps)                       # per-prompt group baseline
ratio = exp(logp_new - logp_old)                                       # per-token
loss = -mean(min(ratio*adv, clip(ratio, 1±ε)*adv)) + β * KL(logp_new, logp_ref)
```

Things I had to actually understand to write this:

- **Why GRPO needs no critic.** PPO learns a value function to estimate "how
  much better than expected was this action." GRPO just uses the other
  samples of the *same prompt* as the baseline — advantage is standardized
  within the group of G rollouts. No second model, no GAE, no critic to
  debug. The cost: you need G>1 samples per prompt, and if all G get the
  same reward the advantage is ~0 and the step teaches nothing.
- **The reference model is free if you use LoRA.** `logp_ref` comes from the
  same weights with `model.disable_adapter()` — the frozen base model —
  instead of keeping a second copy in VRAM. On an 11 GB card that matters.
- **The KL estimator matters.** The code uses the k3 estimator,
  `exp(logp_ref − logp_new) − (logp_ref − logp_new) − 1`, which is unbiased
  and — unlike the naive log-ratio — can't go negative, so it doesn't
  inject noise with the wrong sign into the gradient.
- **Honest caveat:** this loop does one gradient step per batch, so
  `logp_old == logp_new` and the PPO clip is a no-op — ratio is 1
  everywhere. The clip machinery is real and tested, but it only *bites*
  once you do multiple epochs per batch or off-policy correction. I left it
  in because the code is more instructive with it, and because step-1
  correctness is what you verify against TRL anyway.

There's also [`train/tier_a/grpo_trl.py`](train/tier_a/grpo_trl.py), the same
config driven through TRL's `GRPOTrainer`, as a cross-check that my loop
produces the same learning curve.

## Why a hand-written loop: the 1080 Ti constraint

This part is half the reason the project exists. The card is Pascal,
compute capability 6.1:

- **vLLM is out** (needs ≥ 7.0/7.5 depending on version), which kills the
  fast-rollout path every modern RL framework assumes.
- **Current PyTorch wheels are out** — `torch 2.14.0+cu126` is the *last*
  wheel line still shipping `sm_61` kernels; cu128 dropped Pascal in torch
  2.8. `scripts/preflight.py` asserts this before training so you fail fast
  instead of 40 minutes in.
- **No bf16.** Pascal fp16 is supported but low-rate on GP102; amusingly the
  fp16 matmul microbenchmark still beats fp32 on this card (2.3 ms vs
  6.1 ms for a 2048³ matmul), but fp16 LoRA training with no bf16 fallback
  is a stability gamble I didn't need — at 0.5B params, fp32 + LoRA fits in
  ~9 GB. Boring choice, deliberately.

So the fashionable stack (TRL + vLLM rollouts + bf16) was unavailable, and
the unfashionable one turned out to be the better teacher anyway.

## Results

300 steps, `Qwen2.5-0.5B-Instruct`, LoRA r=16, B=2 prompts × G=4
completions, fp32, ~11 s/step — under an hour on the 1080 Ti:

| metric (32-doc eval) | before | after |
|---|---|---|
| schema_ok rate | 0.22 | 1.00 |
| todo recall (F1-match) | 0.11 | 0.44 |
| mean reward | 0.69 | 3.36 |

In-training `schema_ok` hits 1.0 around step 200 and stays there — the model
learns "emit valid JSON" fast, then spends the rest of the run grinding out
todo recall. That ordering is exactly what the format-vs-content split in
the reward is supposed to produce, and watching it happen on the curve is
more convincing than any blog post asserting it.

![training curves](docs/curves.png)

Two more things the curves show: completion length drifts down (~190 → ~130
tokens) as the schema converges — the model stops rambling once padding
stops paying — and KL climbs to ~0.02 nats and plateaus, i.e. the policy
moves but stays tethered to the base model. β=0.04 is doing its job.

Reproduce the eval numbers:

```bash
uv run python -m train.tier_a.grpo_minimal --config configs/tier_a_grpo.yaml   # ~1 h on a 1080 Ti
# or a 15-minute smell test: --set grpo.steps=50
uv run python scripts/plot_curves.py --runs runs/tier_a --copy-run-json viz/dist/run.json
```

## What I'd do next / honest limitations

- **The eval is small (n=32) and the data is synthetic.** Recall 0.28 is
  real improvement, not a solved task — there's headroom, and a few
  hand-written eval docs would make the number more meaningful.
- **Single update per batch** (see caveat above). Next step is minibatch
  reuse with real off-policy correction, which is where the clip earns its
  keep.
- **No judge term.** A second LM grading summary quality is the obvious
  extension, and also the obvious reward-hacking surface; I kept it off.
- **Tier B sketch:** the reward is stdlib-only and vendored into
  `data/harbor_tasks/` so the *same grader* can run inside Harbor sandboxed
  environments on a bigger GPU. Not wired to a trainer yet — that's the
  multi-turn version of this project.

## Repo map

```
train/tier_a/grpo_minimal.py   hand-written GRPO loop (the core)
train/tier_a/grpo_trl.py       same task through TRL's GRPOTrainer
train/reward/core.py           stdlib-only verifiable reward
train/envs/sim_env.py          single-step env + gridworld
data/gen_docs.py               synthetic doc/todo generator (+ Harbor export)
viz/                           dashboard source (React+TS); dist/ is committed
scripts/preflight.py           GPU/arch/pin assertions — run first
scripts/plot_curves.py         metrics → curves + run.json for panel 5
deploy/                        serve the dashboard on a free-tier cloud VM
docs/design.md                 design notes: constraints, tradeoffs, open questions
```

## References that were actually useful

- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — where GRPO comes from
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — RLVR working at scale
- [PPO](https://arxiv.org/abs/1707.06347) · [DPO](https://arxiv.org/abs/2305.18290)
- Jimmy Shi, [*a vision researcher's guide to PPO & GRPO*](https://yugeten.github.io/posts/2025/01/ppogrpo) — the clearest derivation I found
- Cameron Wolfe's [GRPO](https://cameronrwolfe.substack.com/p/grpo) and [GRPO tricks](https://cameronrwolfe.substack.com/p/grpo-tricks)
- [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/en/grpo_trainer) — for the cross-check path
- [CUDA compute capability and why it matters](https://dev.to/maxvyaznikov/cuda-compute-capability-what-it-is-and-why-it-matters-for-ml-engineers-1mhg) — background for §hardware

---

Setup details, CPU smoke test, and preflight: see below.

## Setup

```bash
uv python install 3.12
uv sync --extra tier-a
uv run python scripts/preflight.py           # strict: expects the 1080 Ti
uv run python scripts/preflight.py --cpu-ok  # laptop-safe variant
uv run pytest -q                             # reward + data + loop unit tests
```

Data is checked in (`data/train.parquet`, `data/eval.parquet`). Regenerate:

```bash
uv run python -m data.gen_docs --parquet data/ --n-train 2000 --n-eval 200 --seed 0
uv run python -m data.gen_docs --harbor data/harbor_tasks/sample --n 5 --seed 0 --split eval
```

CPU smoke test (no GPU, tiny model, 2 steps):

```bash
uv run python -m train.tier_a.grpo_minimal --config configs/tier_a_grpo.yaml \
  --set model=HuggingFaceTB/SmolLM2-360M-Instruct \
  --set grpo.steps=2 --set grpo.num_generations=2 \
  --set grpo.max_completion_length=32 --set grpo.batch_prompts=1 \
  --set data.eval_n=2 --device cpu
```

Each run writes `metrics.jsonl`, `run.json` (dashboard panel-5 schema),
`samples.jsonl`, `eval_before.json`, `eval_after.json`, and LoRA checkpoints
under `runs/<name>/`.

License: Apache-2.0.
