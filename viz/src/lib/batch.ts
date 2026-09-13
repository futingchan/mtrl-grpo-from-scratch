import { mulberry32 } from './rng'

/**
 * One GRPO group: G completions of a single prompt, with the per-sequence
 * quantities the update rules consume. `correct` is a 0/1 verifiable flag
 * (schema_ok in the real task). `ratio` is π_θ/π_old; `logRatio` is
 * log π_ref − log π_θ (the k3-KL input). `rewards` is only set when the
 * batch comes from a real run.json; toy batches derive reward from
 * correct/tokens via shapedReward.
 */
export interface ToyBatch {
  tokens: number[]
  correct: (0 | 1)[]
  logRatio: number[]
  ratio: number[]
  rewards?: number[]
}

/** Box–Muller normal driven by the seeded rng. */
function randn(rng: () => number, mean = 0, std = 1): number {
  const u = Math.max(rng(), 1e-12)
  const v = rng()
  return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

/**
 * Synthetic batch: lengths in [40, 900], correctness increasingly likely for
 * longer completions (p = .35 → .75) so the length penalty actually trades
 * off against being right.
 */
export function makeBatch(seed: number, n = 8): ToyBatch {
  const rng = mulberry32(seed)
  const tokens: number[] = []
  const correct: (0 | 1)[] = []
  const logRatio: number[] = []
  const ratio: number[] = []
  for (let i = 0; i < n; i++) {
    const t = 40 + Math.floor(rng() * 861)
    tokens.push(t)
    const p = 0.35 + 0.4 * ((t - 40) / 860)
    correct.push(rng() < p ? 1 : 0)
    logRatio.push(randn(rng, 0, 0.3))
    ratio.push(Math.exp(randn(rng, 0, 0.25)))
  }
  return { tokens, correct, logRatio, ratio }
}
