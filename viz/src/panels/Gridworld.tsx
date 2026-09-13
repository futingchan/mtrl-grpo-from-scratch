import { useEffect, useReducer, useRef, useState } from 'react'

import { LineChart, Panel, HowTo } from '../components'
import {
  ALPHA,
  GOAL,
  GRID,
  initTheta,
  policy,
  reinforceEpisode,
  type Episode,
  type Theta,
} from '../lib/gridworld'
import { mulberry32 } from '../lib/rng'

const CELL = 72

interface Sim {
  theta: Theta
  episodes: number
  returns: number[]
  lastEp: Episode | null
  base: number // running mean return
  rng: () => number
}

function freshSim(): Sim {
  return {
    theta: initTheta(),
    episodes: 0,
    returns: [],
    lastEp: null,
    base: 0,
    rng: mulberry32(42),
  }
}

function Arrow({ a, prob }: { a: number; prob: number }) {
  // triangles pointing up/right/down/left inside the cell
  const shapes = [
    '36,4 44,18 28,18', // up
    '68,36 54,28 54,44', // right
    '36,68 44,54 28,54', // down
    '4,36 18,28 18,44', // left
  ]
  return (
    <polygon
      className="cell-tri"
      points={shapes[a]}
      fill={a === 1 ? '#2c6fbb' : '#3a7ec2'}
      opacity={0.15 + 0.85 * prob}
    />
  )
}

export default function Gridworld() {
  // sim state lives in a ref: interval callbacks always see the latest
  const simRef = useRef<Sim>(freshSim())
  const [, force] = useReducer((x: number) => x + 1, 0)
  const [playing, setPlaying] = useState(false)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const run = (n: number) => {
    const s = simRef.current
    for (let k = 0; k < n; k++) {
      const ep = reinforceEpisode(s.rng, s.theta, s.base)
      s.base += (ep.ret - s.base) / (s.episodes + 1)
      s.episodes += 1
      s.returns.push(ep.ret)
      s.lastEp = ep
    }
    force()
  }

  const ticksLeft = useRef(Infinity)

  const startPlaying = (n: number) => {
    if (timerRef.current) return
    ticksLeft.current = n
    timerRef.current = setInterval(() => {
      run(1)
      if (--ticksLeft.current <= 0) stopPlaying()
    }, 500)
    setPlaying(true)
  }

  const stopPlaying = () => {
    if (timerRef.current) clearInterval(timerRef.current)
    timerRef.current = null
    setPlaying(false)
  }

  const togglePlay = () => (playing ? stopPlaying() : startPlaying(Infinity))

  const reset = () => {
    stopPlaying()
    simRef.current = freshSim()
    force()
  }

  useEffect(
    () => () => {
      if (timerRef.current) clearInterval(timerRef.current)
    },
    [],
  )

  const sim = simRef.current
  const cells = []
  for (let s = 0; s < 16; s++) {
    const row = Math.floor(s / GRID)
    const col = s % GRID
    const pi = policy(sim.theta, s)
    cells.push(
      <g key={s} transform={`translate(${col * CELL},${row * CELL})`}>
        <rect
          width={CELL}
          height={CELL}
          fill={s === GOAL ? '#cdebc9' : '#fff'}
          stroke="#ccc"
        />
        {s !== GOAL && pi.map((prob, a) => <Arrow key={a} a={a} prob={prob} />)}
        {s === GOAL && (
          <text x={CELL / 2} y={CELL / 2 + 6} textAnchor="middle" fontSize={18}>
            ★
          </text>
        )}
      </g>,
    )
  }

  const path =
    sim.lastEp &&
    sim.lastEp.states
      .map(
        (s, i) =>
          `${i === 0 ? 'M' : 'L'}${(s % GRID) * CELL + CELL / 2},${
            Math.floor(s / GRID) * CELL + CELL / 2
          }`,
      )
      .join(' ')

  return (
    <Panel
      title="2. REINFORCE on gridworld"
      idea="The same loop with the LLM removed: a tabular policy learning to reach a goal from reward alone. GRPO's group mean is REINFORCE's baseline, computed across siblings instead of across time."
      blurb={[
        "A 4×4 grid. The agent starts top-left, the goal ★ is bottom-right, and walking off an edge costs a step and leaves you in place. The policy is a table of logits θ[s][a] (one per direction per cell) drawn as four triangles whose opacity is the action probability. This is the policy-gradient theorem at its most legible: Williams' REINFORCE, 1992.",
        "Each press runs one episode: roll out to the goal or 30 steps, compute the discounted return G_t, then nudge the logits of every (state, action) on the trajectory toward whatever beat the running mean b. The baseline is a variance trick, not a bias: subtracting it does not change the expected gradient, it just stops the update from being dominated by noise.",
        "Mapping to the loop: REINFORCE estimates what is typical by averaging returns over time; GRPO estimates it by averaging over the G siblings sampled from one prompt at one step. Same estimator, different axis. Watch the returns hover near −1 while the policy wanders, then jump toward +0.76 once it stumbles onto the goal path (6 steps × −0.04 + 1): a baseline learning to tell luck from skill.",
      ]}
      formula={
        'G_t = Σ γ^k r_{t+k},  γ=0.95, r_step=−0.04, r_goal=+1, T≤30\n' +
        'b = running mean return\n' +
        `θ[s][a] ← θ[s][a] + α·(G_t − b)·(1[a=a_t] − π(a|s)),  α=${ALPHA}`
      }
    >
      <div className="controls">
        <button onClick={() => run(1)}>Run 1 episode</button>
        <button onClick={() => run(10)}>Run 10</button>
        <button onClick={() => startPlaying(10)}>Play ×10</button>
        <button onClick={togglePlay}>{playing ? 'Pause' : 'Play'}</button>
        <button onClick={reset}>Reset</button>
        <span>
          episodes <b>{sim.episodes}</b> · last return{' '}
          <b>{sim.lastEp ? sim.lastEp.ret.toFixed(2) : '—'}</b> · baseline b ={' '}
          {sim.base.toFixed(3)}
        </span>
      </div>
      <div style={{ display: 'flex', gap: 24, alignItems: 'flex-start' }}>
        <svg width={GRID * CELL} height={GRID * CELL}>
          {cells}
          {path && (
            <path
              d={path}
              fill="none"
              stroke="#e08000"
              strokeWidth={2.5}
              strokeDasharray="5 3"
              opacity={0.8}
            />
          )}
        </svg>
        <LineChart series={sim.returns} yLabel="return" width={420} height={GRID * CELL} />
      </div>
      <p className="legend">
        legend: triangle opacity = π(a|s), orange dashes = last trajectory, ★ = goal
      </p>
      <HowTo>
        <li>
          Press <b>Run 1 episode</b> a few times. Early trajectories (orange dashes)
          wander: every cell starts uniform at 25% per direction, and returns sit near −1.
        </li>
        <li>
          The update is REINFORCE with a baseline: actions whose discounted return{' '}
          <code>G_t</code> beats the running mean <code>b</code> get reinforced. That
          baseline is the ancestor of GRPO's group-mean subtraction.
        </li>
        <li>
          <b>Run 10</b> or <b>Play</b>, give it ~100 episodes. The right/down triangles on
          the start→goal path brighten as the policy sharpens; the return curve climbs
          toward ≈ +0.76 (6 steps × −0.04 + 1).
        </li>
        <li>
          Same loop as panel 1, same update. Only difference: logits are per state and
          reward arrives at the end of the episode instead of per sample.
        </li>
      </HowTo>
    </Panel>
  )
}
