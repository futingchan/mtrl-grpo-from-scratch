# Design notes

Working notes for this project: the constraints that shaped it, the
decisions I made, and the ones I'm still unsure about. Written before and
while building, lightly edited after. Where a prediction turned out wrong
I've left the prediction in place and noted what actually happened.

## The constraint that shaped everything

The GPU is a GTX 1080 Ti: Pascal, compute capability 6.1, 11 GB.

| Property | Consequence |
|---|---|
| CC 6.1 | vLLM needs ≥ 7.0, FlashAttention needs ≥ 8.0, Unsloth needs ≥ 7.0. The standard fast-rollout stack is simply unavailable. |
| 11 GB VRAM | ~1B params max for RL with LoRA; ~3B for inference only. |
| fp16 but no bf16, no tensor cores | fp16 matmul is technically faster on a microbenchmark (2.3 ms vs 6.1 ms fp32 for 2048³) but fp16 AdamW on LoRA with no bf16 escape hatch is a stability risk for zero benefit at this size. Trained fp32. |
| `torch 2.14.0+cu126` | Last wheel line with `sm_61` kernels; cu128 dropped Maxwell/Pascal/Volta in torch 2.8. Pinned. `scripts/preflight.py` asserts arch list before anything runs. |

The consequence I didn't fully appreciate at the start: with no vLLM,
*generation is the bottleneck*, and on a 0.5B fp32 model it runs about
11 s/step for B=2 × G=4 × 256 tokens. That is slow enough to matter for
iteration speed and fast enough to finish a 300-step run in under an hour,
which felt like a reasonable trade for not renting anything.

## Why this task

Constraints I wanted the task to satisfy:

1. **Verifiable reward.** Graded by code, not a learned reward model. This
   is the RLVR premise and the whole reason GRPO is practical at hobby
   scale.
2. **Rich enough to be interesting.** Not "output 42". I wanted multiple
   reward terms with different learning speeds so the curve tells a story.
3. **A product-shaped task**, not a puzzle. Summarize + extract todos is a
   thing people actually want from a model.

Doc → `{summary, todos[]}` hits all three. Schema validity, todo
recall/precision against known gold, owner attribution, length budget, and
hallucination checks against the source entities are all deterministic.

## Reward design decisions

The reward is the attack surface. It is the only thing the policy
optimizes, so every term got an anti-gaming test:

- **Precision is gated on recall.** Without this, emitting zero todos
  trivially maxes precision. `test_reward_core.py` has a case that would
  catch this regression.
- **Hallucination uses the source entity list**, not the gold todos. The
  model shouldn't be able to invent plausible-looking owners that happen
  to not be in gold.
- **Format is a separate additive term** from content, so "learn JSON
  first" doesn't require content competence and vice versa.
- **token-F1 over embedding cosine** for todo matching. Cosine with a small
  sentence encoder is more forgiving of paraphrase but adds a ~90 MB model
  dependency to the grader, and the grader is meant to be vendored into
  minimal sandbox images. F1 at 0.5 threshold is a defensible compromise;
  cosine sits behind a config flag if paraphrase turns out to matter.

The stdlib-only constraint on `train/reward/core.py` is deliberate: the same
file is vendored into Harbor task images (`data/harbor_tasks/`). One grader,
two runtimes. If Tier A and Tier B ever disagree on a score, that is a bug,
not a difference of opinion.

## The GRPO loop, and its one dishonest line

`grpo_minimal.py` is as small as I could make it without hiding a concept:
group-standardized advantage, per-token clipped surrogate, k3 KL to a
reference model obtained by disabling the LoRA adapter on the same weights.

The one dishonest line: `logp_old = logp_new.detach()`. With a single
gradient step per sampled batch, the importance ratio is identically 1 and
the PPO clip never fires. This is fine in practice, since even standard
GRPO recipes often do ~1 epoch per batch, but it does mean the clip is
aspirational. Making it *do* something requires minibatch reuse (multiple
epochs per batch), which is the next experiment. It doubles gradient compute
per rollout, which is exactly the thing that rollout-rich, compute-cheap
GPUs amortize and mine doesn't.

`grpo_trl.py` exists to answer "is my loop right?". Same data, same reward,
TRL's `GRPOTrainer` with `use_vllm=False`. The curves landing within noise
of each other is the validation.

## How I sequenced it

1. Reward core + 24 unit tests, including adversarial cases (empty JSON,
   extra keys, hallucinated owner, 200-word summary). Reward first because
   everything else is downstream of it being right.
2. Data generator, with gold known by construction.
3. Dashboard panels 1–4 (concept animations, toy numbers).
4. Tier A loop, run until the eval metrics actually move.
5. Panel 5 wired to the real `run.json`.
6. TRL cross-check.
7. (future) Tier B: Harbor sandboxed tasks on a CC ≥ 8.0 rental GPU.

## Still open

- **Is token-F1 too strict?** If trained outputs paraphrase gold tasks well
  but mismatch tokens, recall underreports real progress. Cheap check: run
  the eval completions through both scorers and diff.
- **Does the KL coefficient β=0.04 matter at this scale?** KL stayed under
  ~0.05 nats for the whole run; the policy never got close to reward-hacking
  gibberish. Possibly the KL term is doing nothing here and could be
  removed, or possibly it is precisely *why* nothing went wrong. An ablation
  would tell, and it is cheap: one 300-step run with β=0.
- **Multi-turn.** The env contract (`train/envs/sim_env.py`) is already
  shaped like an OpenEnv single-step env. The interesting version gives the
  model `read_document`/`grep_document`/`submit` tools inside Harbor
  sandboxes, which is where "multi-turn RL" actually earns the name.
