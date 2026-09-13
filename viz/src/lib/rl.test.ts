import { describe, expect, it } from 'vitest'

import { makeBatch } from './batch'
import {
  GOAL,
  GRID,
  initTheta,
  policy,
  reinforceEpisode,
} from './gridworld'
import { mulberry32 } from './rng'
import {
  clippedSurrogate,
  dpoLoss,
  grpoAdvantages,
  k3Kl,
  mean,
  ppoAdvantages,
  shapedReward,
  softmax,
  std,
} from './rl'

describe('rng', () => {
  it('is deterministic and uniform in [0,1)', () => {
    const a = mulberry32(7)
    const b = mulberry32(7)
    for (let i = 0; i < 100; i++) {
      const v = a()
      expect(v).toBe(b())
      expect(v).toBeGreaterThanOrEqual(0)
      expect(v).toBeLessThan(1)
    }
  })
})

describe('batch', () => {
  it('is deterministic per seed', () => {
    expect(makeBatch(3)).toEqual(makeBatch(3))
    expect(makeBatch(3)).not.toEqual(makeBatch(4))
  })

  it('produces in-range fields for G=8', () => {
    const b = makeBatch(7)
    expect(b.tokens).toHaveLength(8)
    for (const t of b.tokens) {
      expect(t).toBeGreaterThanOrEqual(40)
      expect(t).toBeLessThanOrEqual(900)
    }
    expect(b.correct.every((c) => c === 0 || c === 1)).toBe(true)
    expect(b.ratio.every((r) => r > 0)).toBe(true)
  })
})

describe('rl basics', () => {
  it('mean/std use population std', () => {
    expect(mean([])).toBe(0)
    expect(mean([1, 2, 3])).toBe(2)
    expect(std([2, 2, 2])).toBe(0)
    expect(std([0, 2])).toBeCloseTo(1)
  })

  it('softmax sums to 1 and is shift-stable', () => {
    const p = softmax([1000, 1001, 999])
    expect(p.reduce((a, b) => a + b)).toBeCloseTo(1)
    expect(p[1]).toBeGreaterThan(p[0])
    expect(p[0]).toBeGreaterThan(p[2])
  })

  it('shapedReward trades correctness for length', () => {
    expect(shapedReward(1, 0, 0.5)).toBe(1)
    expect(shapedReward(1, 1000, 0.5)).toBeCloseTo(0.5)
    expect(shapedReward(0, 2000, 1)).toBeCloseTo(-2)
  })
})

describe('advantages', () => {
  it('grpoAdvantages standardizes to mean 0 / std 1', () => {
    const a = grpoAdvantages([0.2, 0.5, 1.0, 0.1])
    expect(mean(a)).toBeCloseTo(0)
    expect(std(a)).toBeCloseTo(1)
  })

  it('grpoAdvantages on a constant group gives zeros, not NaN', () => {
    const a = grpoAdvantages([1, 1, 1])
    expect(a.every((x) => x === 0)).toBe(true)
  })

  it('ppoAdvantages subtract the critic', () => {
    expect(ppoAdvantages([1, 2, 3], 1.5)).toEqual([-0.5, 0.5, 1.5])
  })
})

describe('clippedSurrogate', () => {
  it('is unclipped inside [1−ε, 1+ε]', () => {
    const { perSample, loss } = clippedSurrogate([1.1], [2.0], 0.2)
    expect(perSample[0]).toBeCloseTo(-2.2)
    expect(loss).toBeCloseTo(-2.2)
  })

  it('clips the pessimistic branch only', () => {
    // ρ high + positive A: clipped branch binds at (1+ε)A
    const up = clippedSurrogate([1.5], [1.0], 0.2)
    expect(up.perSample[0]).toBeCloseTo(-1.2)
    // ρ high + negative A: the raw ρA term is worse, clip does not save it
    const down = clippedSurrogate([1.5], [-1.0], 0.2)
    expect(down.perSample[0]).toBeCloseTo(1.5)
  })
})

describe('k3Kl', () => {
  it('is 0 at x=0 and positive elsewhere', () => {
    const kl = k3Kl([0, 0.5, -0.5])
    expect(kl[0]).toBe(0)
    expect(kl[1]).toBeGreaterThan(0)
    expect(kl[2]).toBeGreaterThan(0)
  })
})

describe('dpoLoss', () => {
  it('prefers chosen over rejected', () => {
    const { perPair, loss, implicitRewards } = dpoLoss(
      [{ chosenLogRatio: 0.5, rejectedLogRatio: -0.5 }],
      0.1,
    )
    expect(perPair[0]).toBeLessThan(0.7) // < log 2
    expect(loss).toBe(perPair[0])
    expect(implicitRewards[0]).toEqual({ chosen: 0.05, rejected: -0.05 })
  })

  it('loss grows when the pair is flipped', () => {
    const good = dpoLoss([{ chosenLogRatio: 0.5, rejectedLogRatio: -0.5 }], 0.1)
    const bad = dpoLoss([{ chosenLogRatio: -0.5, rejectedLogRatio: 0.5 }], 0.1)
    expect(bad.loss).toBeGreaterThan(good.loss)
  })
})

describe('gridworld', () => {
  it('policy is uniform at init and a valid distribution', () => {
    const th = initTheta()
    expect(th).toHaveLength(GRID * GRID)
    const pi = policy(th, 0)
    expect(pi).toEqual([0.25, 0.25, 0.25, 0.25])
  })

  it('episodes terminate and stay inside the grid', () => {
    const rng = mulberry32(42)
    const th = initTheta()
    const ep = reinforceEpisode(rng, th, 0)
    expect(ep.actions.length).toBeLessThanOrEqual(30)
    expect(ep.states.length).toBe(ep.actions.length + 1)
    for (const s of ep.states) {
      expect(s).toBeGreaterThanOrEqual(0)
      expect(s).toBeLessThan(GRID * GRID)
    }
  })

  it('repeated episodes learn to reach the goal more often', () => {
    const rng = mulberry32(42)
    const th = initTheta()
    let base = 0
    let reached = 0
    for (let i = 0; i < 400; i++) {
      const ep = reinforceEpisode(rng, th, base)
      base += (ep.ret - base) / (i + 1)
      if (ep.states[ep.states.length - 1] === GOAL) reached++
    }
    expect(reached).toBeGreaterThan(200)
  })
})
