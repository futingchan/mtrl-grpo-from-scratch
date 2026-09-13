import { useState } from 'react'

import { Bars, HowTo, Panel } from '../components'
import type { ToyBatch } from '../lib/batch'
import {
  clippedSurrogate,
  dpoLoss,
  grpoAdvantages,
  k3Kl,
  mean,
  ppoAdvantages,
  shapedReward,
} from '../lib/rl'

export default function UpdateComparison({
  batch,
  lambda,
}: {
  batch: ToyBatch
  lambda: number
}) {
  const [eps, setEps] = useState(0.2)
  const [beta, setBeta] = useState(0.04)

  // real run.json batches carry the task's own rewards; toy batches are shaped
  const real = batch.rewards !== undefined
  const r = batch.rewards ?? batch.tokens.map((t, i) => shapedReward(batch.correct[i], t, lambda))
  const clippedFlags = batch.ratio.map((rho) => rho < 1 - eps || rho > 1 + eps)

  // PPO column
  const V = mean(r)
  const advPpo = ppoAdvantages(r, V)
  const ppo = clippedSurrogate(batch.ratio, advPpo, eps)

  // GRPO column
  const advGrpo = grpoAdvantages(r)
  const sur = clippedSurrogate(batch.ratio, advGrpo, eps)
  const kl = k3Kl(batch.logRatio)
  const grpoTotal = sur.loss + beta * mean(kl)

  // DPO column: sort by reward desc, pair best↔worst
  const order = r.map((v, i) => [v, i] as const).sort((a, b) => b[0] - a[0])
  const pairs = []
  for (let k = 0; k < order.length / 2; k++) {
    pairs.push({
      chosenLogRatio: batch.logRatio[order[k][1]],
      rejectedLogRatio: batch.logRatio[order[order.length - 1 - k][1]],
    })
  }
  const dpo = dpoLoss(pairs, beta)

  return (
    <Panel
      title="3. One batch, three updates: PPO vs GRPO vs DPO"
      idea="One scored batch, three update rules. The difference between PPO, GRPO and DPO is almost entirely one question: what do you subtract from the reward?"
      blurb={[
        `One fixed group of ${batch.tokens.length} completions, three algorithms turning the same rewards into a loss. Left, PPO answers the question with a critic: a value network trained alongside the policy, giving A = r − V (here V=mean(r) stands in, so the advantages come out small). Middle, GRPO's answer is free: subtract the group's own mean and divide by its std. The standardization does real work: it keeps the gradient scale constant whether raw rewards are 0.1 or 10, and it stretches even small intra-group differences into usable signal. Right, DPO refuses the question: no rollouts and no scalar reward, just best↔worst pairs and a log-sigmoid loss that sits near log 2 whenever pairs barely differ. Each bar is one sample's contribution to the loss, negated: green means reinforce.`,
        `The two sliders are the guardrails every on-policy method needs. ε clips the importance ratio ρ = π_new/π_old, a trust region so a single lucky rollout cannot dominate the step (dashed outlines mark clipped samples). β scales a KL penalty that tethers the policy to a frozen reference copy of itself; the k3 estimator exp(x)−x−1 is non-negative by construction, so drift is always paid for.`,
        `Our trainer is the middle column: no critic means no second network in memory and nothing for the policy to exploit. The honest caveat: with one gradient step per batch, π_old = π_new and ρ = 1, so the clip never fires; the machinery is real and unit-tested, but it only bites once you reuse batches. Tick "use real batch" in panel 5 to rerun this comparison on an actual post-training group.`,
      ]}
      formula={
        'PPO/GRPO:  L = −(1/G)Σ min(ρᵢAᵢ, clip(ρᵢ,1−ε,1+ε)Aᵢ) + β·(1/G)Σ[exp(xᵢ)−xᵢ−1]\n' +
        '           ρᵢ = π_θ(oᵢ)/π_old(oᵢ),  xᵢ = log π_ref(oᵢ) − log π_θ(oᵢ)\n' +
        'advantages: PPO Aᵢ = rᵢ − V   |   GRPO Aᵢ = (rᵢ − mean(r))/(std(r)+1e-6)\n' +
        'DPO:  L = −Σ log σ(β·(logπ_θ/π_ref(chosen) − logπ_θ/π_ref(rejected)))'
      }
    >
      <div className="controls">
        <label>
          ε clip = {eps.toFixed(2)}{' '}
          <input
            type="range"
            min={0.05}
            max={0.5}
            step={0.01}
            value={eps}
            onChange={(e) => setEps(+e.target.value)}
          />
        </label>
        <label>
          β = {beta.toFixed(2)}{' '}
          <input
            type="range"
            min={0}
            max={0.5}
            step={0.01}
            value={beta}
            onChange={(e) => setBeta(+e.target.value)}
          />
        </label>
        <span className="legend">dashed outline = ratio clipped (outside [1−ε, 1+ε])</span>
      </div>
      {real && (
        <p className="note">
          using the real rewards exported by the run (schema + todo recall/precision +
          length), and ρ = 1 because rollouts are on-policy (π_old = π_θ at generation time)
        </p>
      )}
      <div className="cols">
        <div className="col">
          <h3>PPO (toy critic)</h3>
          <p className="note">V = mean(r) = {V.toFixed(3)} (stand-in critic)</p>
          <Bars
            values={ppo.perSample}
            height={130}
            clipped={clippedFlags}
            colors={(v) => (v < 0 ? '#3d9e50' : '#b3261e')}
            labels={advPpo.map((a) => `A=${a.toFixed(2)}`)}
          />
          <p style={{ marginTop: 22 }}>
            <b>L_ppo = {ppo.loss.toFixed(4)}</b>
          </p>
          <p className="note">
            with a critic, GAE at γ=λ=1 single-step = r − V
          </p>
        </div>
        <div className="col">
          <h3>GRPO</h3>
          <p className="note">A = (r − mean(r))/(std(r)+ε) — group baseline, no critic</p>
          <Bars
            values={sur.perSample}
            height={130}
            clipped={clippedFlags}
            colors={(v) => (v < 0 ? '#3d9e50' : '#b3261e')}
            labels={advGrpo.map((a) => `A=${a.toFixed(2)}`)}
          />
          <p style={{ marginTop: 22 }}>
            surrogate = {sur.loss.toFixed(4)} · β·KL ={' '}
            {(beta * mean(kl)).toFixed(4)} · <b>L = {grpoTotal.toFixed(4)}</b>
          </p>
          <p className="note">bars = −surrogate per sample; green = descent direction</p>
        </div>
        <div className="col">
          <h3>DPO</h3>
          <p className="note">{pairs.length} pairs: best↔worst, 2nd↔2nd-worst, … (sorted by reward)</p>
          <Bars
            values={dpo.perPair}
            height={130}
            labels={dpo.implicitRewards.map(
              (ir) => `r̂ ${ir.chosen.toFixed(2)}/${ir.rejected.toFixed(2)}`,
            )}
          />
          <p style={{ marginTop: 22 }}>
            <b>L_dpo = {dpo.loss.toFixed(4)}</b>
          </p>
          <p className="note">
            no rollouts, no advantages; needs preference pairs. bar label: implicit
            rewards β·logRatio chosen/rejected
          </p>
        </div>
      </div>
      <HowTo>
        <li>
          A bar is how much that sample moves the loss: green (negative) pushes
          probability up, red (positive) pushes it down. Labels show the advantage each
          method computed.
        </li>
        <li>
          Slide <b>ε</b> toward 0.05: dashed outlines appear on samples whose ratio ρ left
          the trust region; clipping caps their influence so one lucky rollout can't wreck
          the step.
        </li>
        <li>
          Slide <b>β</b> up and the GRPO column's β·KL term grows. x = log π_ref − log π_θ
          is per-sequence; <code>exp(x)−x−1</code> is always ≥ 0, so drift from the
          reference is always penalized.
        </li>
        <li>
          PPO vs GRPO differ only in the baseline (V vs group mean+std). GRPO's
          standardization still spreads advantages when all rewards are similar.
        </li>
        <li>
          DPO needs no environment at all: pairs come from sorting rewards best↔worst. Its
          loss drops only when the chosen log-ratio beats the rejected one.
        </li>
        <li>
          Tick "use real batch" in panel 5 to rerun this comparison on an actual
          prompt group from the trained run (ρ=1, on-policy).
        </li>
      </HowTo>
    </Panel>
  )
}
