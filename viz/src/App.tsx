import { useMemo, useState } from 'react'

import { makeBatch, type ToyBatch } from './lib/batch'
import Gridworld from './panels/Gridworld'
import RewardShaping from './panels/RewardShaping'
import TheLoop from './panels/TheLoop'
import TrainingRun from './panels/TrainingRun'
import UpdateComparison from './panels/UpdateComparison'

export default function App() {
  const [seed, setSeed] = useState(7)
  const [lambda, setLambda] = useState(0.5)
  const [realBatch, setRealBatch] = useState<ToyBatch | null>(null)

  const batch = useMemo(() => realBatch ?? makeBatch(seed), [seed, realBatch])

  return (
    <>
      <h1>RL for LLMs — the loop, visually</h1>
      <div className="intro">
        <p>
          The goal is to demonstrate the full RL fine-tuning loop end to end, with
          minimal framework dependency, implementing the basic pieces from scratch where
          it improves understanding. The flow, at a high level: prompts go in, the
          policy samples a group of completions, a deterministic verifier scores each
          one, advantages are computed relative to the group, and the policy is updated.
        </p>
        <p>
          This page is that loop made visible: five animated panels, ordered from toy to
          real, with the actual numbers from a 300-step run of{' '}
          <code>Qwen2.5-0.5B-Instruct</code> underneath.
        </p>
        <h3>The mechanism</h3>
        <p>
          A policy is a probability distribution over the model's next tokens, and
          training it with RL means shifting mass toward outputs that score well. The
          catch is that neither sampling nor the reward program is differentiable, so
          the estimator is the policy-gradient trick: draw samples, score each, and move
          the logits by advantage times ∇log π. Everything in the loop exists to compute
          that one quantity well.
        </p>
        <p>
          The reward is a program, not a learned model. Each completion must be strict
          JSON (a summary under 60 words, a <code>todos[]</code> array of{' '}
          <code>{'{task, owner, due, priority}'}</code>), checked against gold todos the
          data generator injected, with a hallucination penalty for owners or dates
          absent from the source. A deterministic grader cannot be sweet-talked, so the
          signal stays honest for the whole run; that is the "verifiable" in RLVR.
        </p>
        <p>
          The baseline is what turns a raw score into a signal. A reward of 3.4 means
          nothing alone; what matters is whether it beats what the policy would typically
          have done. GRPO estimates "typical" for free: sample a group of G completions
          from one prompt and set <code>A = (r − mean)/(std)</code>. Above-average
          siblings go positive, below-average go negative, and dividing by the group std
          keeps the gradient scale constant whether rewards are 0.1 or 10. No critic
          network, no human rater: the group is the baseline. The degenerate case is a
          group where every rollout scores identically (zero advantage, zero signal),
          which is why the task needs variance.
        </p>
        <p>
          The update itself is a clipped surrogate. Each completion's contribution is
          scaled by its new-vs-old probability ratio and clipped to{' '}
          <code>[1−ε, 1+ε]</code>: a trust region, so one lucky rollout cannot dominate
          the step. On top sits a <code>β·KL</code> penalty against a frozen reference
          copy of the policy (the k3 estimator <code>exp(x)−x−1</code>, non-negative by
          construction), which tethers the trained model to the base model and keeps
          reward hacking bounded. That is the whole of GRPO: group sampling for the
          baseline, standardization for scale, clipping and KL for stability.
        </p>
        <p>Putting it together, one iteration end to end:</p>
        <div className="bigloop">
          {[
            ['1. prompt: a document', 'panels 1 · 5'],
            ['2. policy samples a group of G', 'panels 1 · 3'],
            ['3. program scores each rᵢ', 'panels 1 · 4'],
            ['4. advantage = rᵢ − group mean', 'panel 3'],
            ['5. update θ: upweight winners', 'panels 1 · 2'],
          ].map(([label, tag], i, arr) => (
            <span key={label} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className="loop-item">
                <span className="stage">{label}</span>
                <span className="loop-tag">{tag}</span>
              </span>
              <span className="arrow">{i < arr.length - 1 ? '→' : '↺'}</span>
            </span>
          ))}
          <span className="loop-repeat">repeat ×300; that is the whole training run</span>
        </div>
        <h3>The five panels</h3>
        <p>
          Ordered from toy to real. Panel 1 animates one iteration of the loop on a
          policy made of eight numbers. Panel 2 strips the LLM out entirely: REINFORCE on
          a gridworld, the ancestral form of the same update. Panel 3 puts three update
          rules side by side on one fixed batch. Panel 4 is about the reward itself, and
          how small changes to it re-sort what gets reinforced. Panel 5 is the real
          300-step run, and its checkbox feeds an actual prompt group back into panels
          3–4. Everything is live; press the buttons.
        </p>
      </div>
      <label>
        seed{' '}
        <input
          type="number"
          value={seed}
          onChange={(e) => setSeed(+e.target.value || 0)}
          style={{ width: 80 }}
        />
      </label>
      <TheLoop batch={batch} lambda={lambda} />
      <Gridworld />
      <UpdateComparison batch={batch} lambda={lambda} />
      <RewardShaping batch={batch} lambda={lambda} setLambda={setLambda} />
      <TrainingRun onBatch={setRealBatch} />
    </>
  )
}
