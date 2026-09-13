import { useEffect, useState } from 'react'

import { HowTo, LineChart, Panel } from '../components'
import type { ToyBatch } from '../lib/batch'

interface RunStep {
  step: number
  reward_mean: number
  reward_std: number
  kl: number
  completion_len: number
  schema_ok: number
}
interface EvalMetrics {
  reward_mean: number
  todo_recall_mean: number
  schema_ok_rate: number
  n: number
}
interface RunJson {
  steps: RunStep[]
  sample_batch?: {
    lengths: number[]
    correct: (0 | 1)[]
    logratio: number[]
    rewards?: number[]
  }
  eval?: { before: EvalMetrics; after: EvalMetrics }
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
        rewards: sb.rewards,
      })
    } else {
      onBatch(null)
    }
  }

  if (failed) {
    return (
      <Panel title="5. Training run">
        <p>No run.json yet; produced by train/tier_a (M4)</p>
      </Panel>
    )
  }
  if (!run) return null

  const steps = run.steps
  const get = (k: keyof RunStep) => steps.map((s) => s[k] as number)

  return (
    <Panel
      title="5. Training run"
      idea="The toys end here. These are the real curves from a 300-step run of the hand-written GRPO loop, and they carry the mechanism's signature: format first, then content, while KL stays tethered."
      blurb={[
        "300 steps of train/tier_a/grpo_minimal.py: Qwen2.5-0.5B-Instruct + LoRA r=16, B=2 prompts × G=4 completions per step, ~11s per step. Four curves per step: mean group reward with its ±std band, k3-KL against the frozen reference, mean completion length, and the schema_ok rate.",
        "Read them in order and the mechanism shows through. reward_mean climbs while the band narrows: groups converge as the policy sharpens. schema_ok saturates by ~step 50, the model taking the cheapest reward first, exactly what panel 4 predicts. completion_len drifts ~190→~130 as rambling stops paying. And kl climbs to ~0.02 nats then plateaus: the policy moves, but stays tethered. Unbounded KL growth is what reward hacking looks like from the outside.",
        "The eval table is the honest check: greedy decoding on 32 held-out documents, before vs after. The checkbox feeds one real post-training prompt group (true rewards, lengths, log-ratios) back into panels 3 and 4, closing the loop between the toy math and the real tensors. Everything above this panel is what produced everything in it.",
      ]}
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
          xs={get('step')}
        />
        <LineChart series={get('kl')} yLabel="kl" xs={get('step')} />
        <LineChart
          series={get('completion_len')}
          yLabel="completion_len"
          xs={get('step')}
        />
        <LineChart series={get('schema_ok')} yLabel="schema_ok" xs={get('step')} />
      </div>
      {run.eval && (
        <table className="rt" style={{ maxWidth: 560 }}>
          <thead>
            <tr>
              <th>eval (greedy, n={run.eval.after.n})</th>
              <th>before</th>
              <th>after</th>
            </tr>
          </thead>
          <tbody>
            {(
              [
                ['reward_mean', 'reward_mean'],
                ['todo_recall_mean', 'todo recall'],
                ['schema_ok_rate', 'schema ok'],
              ] as const
            ).map(([k, label]) => (
              <tr key={k}>
                <td>{label}</td>
                <td>{run.eval!.before[k].toFixed(3)}</td>
                <td>{run.eval!.after[k].toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
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
      <HowTo>
        <li>
          <b>reward_mean ± std</b>: mean group reward per step; the band is the spread
          inside groups. Climbs as the policy learns, narrows as completions converge.
        </li>
        <li>
          <b>schema_ok → 1.0</b> within ~50 steps: the model learns JSON format first (the
          format-reward trick working as intended), then todo recall keeps improving.
        </li>
        <li>
          <b>kl</b> stays small: the β·KL term holds the policy near the frozen reference;
          drift is how policies learn to game the reward.
        </li>
        <li>
          <b>completion_len</b> drops as the length penalty trims rambling. The panel-4
          tradeoff, live.
        </li>
        <li>
          The <b>eval table</b>: greedy decoding on a held-out eval set, before vs after
          {run.eval &&
            `. Reward ${run.eval.before.reward_mean.toFixed(2)}→${run.eval.after.reward_mean.toFixed(2)}, todo recall ${run.eval.before.todo_recall_mean.toFixed(2)}→${run.eval.after.todo_recall_mean.toFixed(2)}, schema ${run.eval.before.schema_ok_rate.toFixed(2)}→${run.eval.after.schema_ok_rate.toFixed(2)}`}
          .
        </li>
        <li>
          The checkbox feeds one real post-training group (a prompt's G completions, true
          rewards, log-ratios) into panels 3–4. Hover the bars to compare.
        </li>
      </HowTo>
    </Panel>
  )
}
