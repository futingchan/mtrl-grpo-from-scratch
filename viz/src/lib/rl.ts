/** Toy RL math shared by the panels. Population std (ddof=0) throughout. */

export function mean(xs: number[]): number {
  if (xs.length === 0) return 0
  return xs.reduce((a, b) => a + b, 0) / xs.length
}

export function std(xs: number[]): number {
  const m = mean(xs)
  return Math.sqrt(mean(xs.map((x) => (x - m) ** 2)))
}

export function softmax(logits: number[]): number[] {
  const m = Math.max(...logits)
  const e = logits.map((x) => Math.exp(x - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((x) => x / s)
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v))
}

/** r_i = correct_i − λ·tokens_i/1000 */
export function shapedReward(correct: number, tokens: number, lambda: number): number {
  return correct - (lambda * tokens) / 1e3
}

/** GRPO group baseline: A_i = (r_i − mean(r)) / (std(r) + eps). */
export function grpoAdvantages(r: number[], eps = 1e-6): number[] {
  const m = mean(r)
  const s = std(r)
  return r.map((x) => (x - m) / (s + eps))
}

/** PPO with a value stand-in: A_i = r_i − V (single-step GAE at γ=λ=1). */
export function ppoAdvantages(r: number[], v: number): number[] {
  return r.map((x) => x - v)
}

/**
 * Clipped surrogate loss, returned negated per sample so bars point in the
 * descent direction: L_i = −min(ρA, clip(ρ,1−ε,1+ε)A).
 */
export function clippedSurrogate(
  ratio: number[],
  adv: number[],
  eps: number,
): { perSample: number[]; loss: number } {
  const perSample = ratio.map((rho, i) => {
    const a = adv[i]
    return -Math.min(rho * a, clamp(rho, 1 - eps, 1 + eps) * a)
  })
  return { perSample, loss: mean(perSample) }
}

/** k3 KL estimator per sequence: KL̂ = exp(x) − x − 1, x = log π_ref − log π_θ. */
export function k3Kl(logRatio: number[]): number[] {
  return logRatio.map((x) => Math.exp(x) - x - 1)
}

export interface DpoPair {
  chosenLogRatio: number
  rejectedLogRatio: number
}

/** DPO: L = −mean log σ(β·(logRatio_chosen − logRatio_rejected)). */
export function dpoLoss(
  pairs: DpoPair[],
  beta: number,
): {
  perPair: number[]
  loss: number
  implicitRewards: { chosen: number; rejected: number }[]
} {
  const sig = (x: number) => 1 / (1 + Math.exp(-x))
  const perPair = pairs.map(
    (p) => -Math.log(sig(beta * (p.chosenLogRatio - p.rejectedLogRatio))),
  )
  const implicitRewards = pairs.map((p) => ({
    chosen: beta * p.chosenLogRatio,
    rejected: beta * p.rejectedLogRatio,
  }))
  return { perPair, loss: mean(perPair), implicitRewards }
}
