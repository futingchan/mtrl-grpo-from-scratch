import { Panel } from '../components'
import type { ToyBatch } from '../lib/batch'
import { grpoAdvantages, shapedReward } from '../lib/rl'

export default function RewardShaping({
  batch,
  lambda,
  setLambda,
}: {
  batch: ToyBatch
  lambda: number
  setLambda: (v: number) => void
}) {
  const rows = batch.tokens
    .map((t, i) => ({
      i,
      tokens: t,
      correct: batch.correct[i],
      reward: shapedReward(batch.correct[i], t, lambda),
    }))
    .sort((a, b) => b.reward - a.reward)
  const adv = grpoAdvantages(rows.map((r) => r.reward))

  return (
    <Panel
      title="4. Reward shaping — cost of thinking longer"
      blurb="The reward mixes a correctness signal with a token penalty, mirroring thought-length penalties in reasoning models. Slide λ and watch the ranking — and therefore the GRPO advantages — flip between 'be right at any length' and 'be short'."
      formula={
        'reward_i = correct_i − λ·tokens_i/1000\n' +
        'A_i = (reward_i − mean)/(std + 1e-6)   →   which completions get reinforced'
      }
    >
      <div className="controls">
        <label>
          λ = {lambda.toFixed(2)}{' '}
          <input
            type="range"
            min={0}
            max={2}
            step={0.05}
            value={lambda}
            onChange={(e) => setLambda(+e.target.value)}
          />
        </label>
        <span className="legend">shared with panel 3; sorted by reward (winner highlighted)</span>
      </div>
      <table className="rt">
        <thead>
          <tr>
            <th>completion</th>
            <th>tokens</th>
            <th>correct</th>
            <th>reward</th>
            <th>GRPO advantage</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, k) => (
            <tr key={row.i} className={k === 0 ? 'winner' : ''} style={{ order: k }}>
              <td>#{row.i}</td>
              <td>{row.tokens}</td>
              <td>{row.correct ? '✓' : '✗'}</td>
              <td>{row.reward.toFixed(3)}</td>
              <td>{adv[k].toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="note">
        reward = correctness − λ·tokens/1000 — a small λ keeps long correct answers on top; past
        λ≈1 short wrong answers outrank long right ones.
      </p>
    </Panel>
  )
}
