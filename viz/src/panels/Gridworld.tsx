import { useMemo, useRef, useState } from 'react'

import { LineChart, Panel } from '../components'
import {
  ALPHA,
  GOAL,
  GRID,
  initTheta,
  policy,
  reinforceEpisode,
  type Episode,
} from '../lib/gridworld'
import { mulberry32 } from '../lib/rng'

const CELL = 72

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
  const [theta, setTheta] = useState<number[][]>(initTheta)
  const [episodes, setEpisodes] = useState(0)
  const [returns, setReturns] = useState<number[]>([])
  const [lastEp, setLastEp] = useState<Episode | null>(null)
  const [playing, setPlaying] = useState(false)
  const baseRef = useRef(0) // running mean return
  const rngRef = useRef(mulberry32(42))
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const runOne = (th: number[][], eps: number, base: number) => {
    const ep = reinforceEpisode(rngRef.current, th, base)
    return { ep, newBase: base + (ep.ret - base) / (eps + 1) }
  }

  const run = (n: number) => {
    setTheta((th) => {
      const copy = th.map((r) => [...r])
      setReturns((rets) => {
        const newRets = [...rets]
        let base = baseRef.current
        let eps = episodes
        let last: Episode | null = null
        for (let k = 0; k < n; k++) {
          const { ep, newBase } = runOne(copy, eps, base)
          base = newBase
          eps += 1
          newRets.push(ep.ret)
          last = ep
        }
        baseRef.current = base
        setEpisodes(eps)
        setLastEp(last)
        return newRets
      })
      return copy
    })
  }

  const togglePlay = () => {
    if (playing) {
      if (timerRef.current) clearInterval(timerRef.current)
      setPlaying(false)
    } else {
      timerRef.current = setInterval(() => run(1), 500)
      setPlaying(true)
    }
  }

  const reset = () => {
    if (timerRef.current) clearInterval(timerRef.current)
    setTheta(initTheta())
    setEpisodes(0)
    setReturns([])
    setLastEp(null)
    setPlaying(false)
    baseRef.current = 0
    rngRef.current = mulberry32(42)
  }

  const grid = useMemo(() => {
    const cells = []
    for (let s = 0; s < 16; s++) {
      const row = Math.floor(s / GRID)
      const col = s % GRID
      const pi = policy(theta, s)
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
    return cells
  }, [theta])

  const path =
    lastEp &&
    lastEp.states
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
      blurb="Before LLMs: a 4×4 grid, tabular softmax policy π(a|s) = softmax(θ_s). Each episode we roll out once, compute discounted returns, and update logits by REINFORCE with a running-mean baseline. Triangle opacity = π(a|s); watch the right/down arrows light up on the path to the goal."
      formula={
        'G_t = Σ γ^k r_{t+k},  γ=0.95, r_step=−0.04, r_goal=+1, T≤30\n' +
        'b = running mean return\n' +
        `θ[s][a] ← θ[s][a] + α·(G_t − b)·(1[a=a_t] − π(a|s)),  α=${ALPHA}`
      }
    >
      <div className="controls">
        <button onClick={() => run(1)}>Run 1 episode</button>
        <button onClick={() => run(10)}>Run 10</button>
        <button onClick={togglePlay}>{playing ? 'Pause' : 'Play'}</button>
        <button onClick={reset}>Reset</button>
        <span>
          episodes <b>{episodes}</b> · last return{' '}
          <b>{lastEp ? lastEp.ret.toFixed(2) : '—'}</b> · baseline b ={' '}
          {baseRef.current.toFixed(3)}
        </span>
      </div>
      <div style={{ display: 'flex', gap: 24, alignItems: 'flex-start' }}>
        <svg width={GRID * CELL} height={GRID * CELL}>
          {grid}
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
        <LineChart series={returns} yLabel="return" width={420} height={GRID * CELL} />
      </div>
      <p className="legend">
        legend: triangle opacity = π(a|s), orange dashes = last trajectory, ★ = goal
      </p>
    </Panel>
  )
}
