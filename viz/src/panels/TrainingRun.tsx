import { useEffect, useState } from 'react'

import { LineChart, Panel } from '../components'
import type { ToyBatch } from '../lib/batch'

interface RunStep {
  step: number
  reward_mean: number
  reward_std: number
  kl: number
  completion_len: number
  schema_ok: number
}
interface RunJson {
  steps: RunStep[]
  sample_batch?: { lengths: number[]; correct: (0 | 1)[]; logratio: number[] }
}

export default function TrainingRun({
  onBatch,
}: {
  onBatch: (b: ToyBatch | null) => void
}) {
  const [run, setRun] = useState<RunJson | null>(null)
  const [failed, setFailed] = useState(false)
  const [useReal, setUseReal] = useState(false)

  useEffect(() => {
    fetch('./run.json')
      .then((res) => {
        if (!res.ok) throw new Error('no run')
        return res.json()
      })
      .then((j: RunJson) => setRun(j))
      .catch(() => setFailed(true))
  }, [])

  const toggle = (v: boolean) => {
    setUseReal(v)
    if (v && run?.sample_batch) {
      const sb = run.sample_batch
      onBatch({
        tokens: sb.lengths,
        correct: sb.correct,
        logRatio: sb.logratio,
        ratio: sb.lengths.map(() => 1),
      })
    } else {
      onBatch(null)
    }
  }

  if (failed) {
    return (
      <Panel title="5. Training run" blurb="" formula="">
        <p>No run.json yet — produced by train/tier_a (M4)</p>
      </Panel>
    )
  }
  if (!run) return null

  const steps = run.steps
  const get = (k: keyof RunStep) => steps.map((s) => s[k] as number)

  return (
    <Panel
      title="5. Training run"
      blurb="Metrics from a real GRPO run (run.json, loaded relatively so the page works from file://)."
      formula={'reward_mean ± reward_std, kl, completion_len, schema_ok  vs  step'}
    >
      <div className="cols" style={{ flexWrap: 'wrap' }}>
        <LineChart
          series={get('reward_mean')}
          band={{
            lo: steps.map((s) => s.reward_mean - s.reward_std),
            hi: steps.map((s) => s.reward_mean + s.reward_std),
          }}
          yLabel="reward_mean±std"
        />
        <LineChart series={get('kl')} yLabel="kl" />
        <LineChart series={get('completion_len')} yLabel="completion_len" />
        <LineChart series={get('schema_ok')} yLabel="schema_ok" />
      </div>
      {run.sample_batch && (
        <label className="controls">
          <input
            type="checkbox"
            checked={useReal}
            onChange={(e) => toggle(e.target.checked)}
          />{' '}
          use real batch in panels 3–4
        </label>
      )}
    </Panel>
  )
}
