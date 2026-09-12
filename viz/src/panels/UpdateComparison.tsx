import { useState } from 'react'

import { Bars, Panel } from '../components'
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

  const r = batch.tokens.map((t, i) => shapedReward(batch.correct[i], t, lambda))
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
      title="3. One batch, three updates — PPO vs GRPO vs DPO"
      blurb="Same 8 completions, same rewards. PPO subtracts a learned critic (here V=mean(r) as a stand-in). GRPO replaces the critic with the group mean and standardizes, plus a k3-KL regularizer toward a reference policy. DPO skips rollouts entirely: it needs preference pairs and compares log-ratios directly."
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
          <p className="note">4 pairs: best↔worst, 2nd↔7th, … (sorted by reward)</p>
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
    </Panel>
  )
}
