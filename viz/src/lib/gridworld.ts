import { softmax } from './rl'

/** 4×4 grid, start at cell 0, goal at cell 15. Actions: up/right/down/left. */
export const GRID = 4
export const GOAL = 15
export const T_MAX = 30
export const STEP_R = -0.04
export const GOAL_R = 1
export const GAMMA = 0.95
export const ALPHA = 0.3

const DR = [-1, 0, 1, 0]
const DC = [0, 1, 0, -1]

export type Theta = number[][] // [16 states][4 actions] logits

export function initTheta(): Theta {
  return Array.from({ length: 16 }, () => [0, 0, 0, 0])
}

/** Bump into a wall and you stay put (and still pay the step cost). */
function stepFn(s: number, a: number): { s2: number; r: number; done: boolean } {
  const row = Math.floor(s / GRID)
  const col = s % GRID
  const nr = Math.min(GRID - 1, Math.max(0, row + DR[a]))
  const nc = Math.min(GRID - 1, Math.max(0, col + DC[a]))
  const s2 = nr * GRID + nc
  return s2 === GOAL
    ? { s2, r: GOAL_R, done: true }
    : { s2, r: STEP_R, done: false }
}

export function policy(theta: Theta, s: number): number[] {
  return softmax(theta[s])
}

export interface Episode {
  states: number[]
  actions: number[]
  rewards: number[]
  ret: number // undiscounted sum, for display
}

function rollout(rng: () => number, theta: Theta): Episode {
  let s = 0
  const states = [s]
  const actions: number[] = []
  const rewards: number[] = []
  let ret = 0
  for (let t = 0; t < T_MAX; t++) {
    const pi = policy(theta, s)
    const u = rng()
    let a = 3
    let cum = 0
    for (let k = 0; k < 4; k++) {
      cum += pi[k]
      if (u < cum) {
        a = k
        break
      }
    }
    const { s2, r, done } = stepFn(s, a)
    actions.push(a)
    rewards.push(r)
    ret += r
    s = s2
    states.push(s)
    if (done) break
  }
  return { states, actions, rewards, ret }
}

/**
 * One REINFORCE episode: roll out, back out discounted returns G_t, then
 * θ[s][a] += α·(G_t − baseline)·(1[a=a_t] − π(a|s)). Mutates theta in place.
 */
export function reinforceEpisode(
  rng: () => number,
  theta: Theta,
  baseline: number,
): Episode {
  const ep = rollout(rng, theta)
  const T = ep.actions.length
  const G = new Array<number>(T).fill(0)
  let g = 0
  for (let t = T - 1; t >= 0; t--) {
    g = ep.rewards[t] + GAMMA * g
    G[t] = g
  }
  for (let t = 0; t < T; t++) {
    const s = ep.states[t]
    const a = ep.actions[t]
    const pi = policy(theta, s)
    for (let y = 0; y < 4; y++) {
      theta[s][y] += ALPHA * (G[t] - baseline) * ((y === a ? 1 : 0) - pi[y])
    }
  }
  return ep
}
