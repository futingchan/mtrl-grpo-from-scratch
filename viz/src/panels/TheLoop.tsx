import { useEffect, useRef, useState } from 'react'

import { Bars, Panel } from '../components'
import type { ToyBatch } from '../lib/batch'
import { mulberry32 } from '../lib/rng'
import { grpoAdvantages, shapedReward, softmax } from '../lib/rl'

const STAGES = [
  'prompt',
  'sample G=8',
  'reward r [G]',
  'advantage A [G]',
  '∇: θᵢ += η·Aᵢ·(1−pᵢ)',
  'new p = softmax(θ) [G]',
]
const ETA = 0.5
const LAMBDA = 0.5

export default function TheLoop({ batch }: { batch: ToyBatch }) {
  const [theta, setTheta] = useState<number[]>(() => Array(8).fill(0))
  const [stage, setStage] = useState(0)
  const [iter, setIter] = useState(0)
  const [playing, setPlaying] = useState(false)
  const rngRef = useRef(mulberry32(1))

  const p = softmax(theta)
  const rewards = batch.tokens.map((t, i) => shapedReward(batch.correct[i], t, LAMBDA))
  const adv = grpoAdvantages(rewards)

  const advance = () => {
    if (stage === STAGES.length - 1) {
      // apply gradient step, then restart the cycle
      setTheta((th) => th.map((t, i) => t + ETA * adv[i] * (1 - p[i])))
      setIter((n) => n + 1)
      setStage(0)
      rngRef.current() // keep stream moving for determinism
    } else {
      setStage((s) => s + 1)
    }
  }

  useEffect(() => {
    if (!playing) return
    const id = setInterval(advance, 900)
    return () => clearInterval(id)
  })

  const reset = () => {
    setTheta(Array(8).fill(0))
    setStage(0)
    setIter(0)
    setPlaying(false)
  }

  const stageShowsAdv = stage >= 3
  const under = stageShowsAdv ? adv : stage >= 2 ? rewards : undefined

  return (
    <Panel
      title="1. The RL loop"
      blurb="A toy 1-d policy: logits θ ∈ R⁸ over the 8 completions of one prompt. Each iteration we reward the samples (correctness − 0.5·tokens/1000), standardize rewards into advantages, and nudge logits toward the high-advantage completions. Watch probability mass concentrate."
      formula={
        'r_i = correct_i − 0.5·tokens_i/1000         r ∈ R^8\n' +
        'A_i = (r_i − mean(r)) / (std(r)+1e-6)      A ∈ R^8\n' +
        'θ_i ← θ_i + η·A_i·(1−p_i),  η=0.5          θ ∈ R^8\n' +
        'p = softmax(θ)                              p ∈ R^8, Σp=1'
      }
    >
      <div className="pipeline">
        {STAGES.map((s, i) => (
          <span key={s} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span className={'stage' + (i === stage ? ' active' : '')}>{s}</span>
            {i < STAGES.length - 1 && <span className="arrow">→</span>}
          </span>
        ))}
      </div>
      <div className="controls">
        <button onClick={() => setPlaying(!playing)}>{playing ? 'Pause' : 'Play'}</button>
        <button onClick={advance}>Step</button>
        <button onClick={reset}>Reset</button>
        <span>
          iteration <b>{iter}</b>
        </span>
      </div>
      <p className="legend">policy p = softmax(θ):</p>
      <Bars
        values={p}
        labels={batch.tokens.map((t, i) => `${t}tok${batch.correct[i] ? '✓' : '✗'}`)}
        height={170}
      />
      {under && (
        <>
          <p className="legend" style={{ marginTop: 26 }}>
            {stageShowsAdv ? 'advantages A' : 'rewards r'} (green &gt;0 / red &lt;0):
          </p>
          <Bars
            values={under}
            height={110}
            colors={(v) => (v >= 0 ? '#3d9e50' : '#b3261e')}
          />
        </>
      )}
    </Panel>
  )
}
