import { HowTo, Panel } from '../components'
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
      realReward: batch.rewards?.[i],
    }))
    .sort((a, b) => b.reward - a.reward)
  const adv = grpoAdvantages(rows.map((r) => r.reward))

  return (
    <Panel
      title="4. Reward shaping: cost of thinking longer"
      idea="Reward design is policy design. The update reinforces ranking, not score: whoever wins the group is what the model becomes."
      blurb={[
        `The reward is the entire specification of what the model becomes, which means the only question that matters is who wins the group. This table re-sorts one prompt's ${batch.tokens.length} completions by a shaped reward, correctness minus λ·tokens/1000, and shows the GRPO advantage each ranking produces. The highlighted row is what the next update reinforces most.`,
        `Slide λ and you are watching the mechanics of reward hacking. At λ=0 correctness is all that counts, so a long rambling-but-right answer beats a terse right one and the policy learns to ramble. Past λ≈1 the ranking inverts: being briefly wrong outranks being thoroughly right. There is always a λ where the incentive flips; the craft is writing a spec where that point sits beyond the behavior you actually want.`,
        `The real task reward is a weighted sum (schema validity, todo recall and precision, a length budget, a hallucination penalty): messier than this toy, same principle. Load the real batch in panel 5 and the "run reward" column shows what the trainer actually optimized; with every completion already schema-valid, length alone decides the ranking. The format-first-then-content ordering in panel 5 is this mechanism at scale: valid JSON is simply the cheapest reward on the table.`,
      ]}
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
        <span className="legend">shared with panels 1 and 3; sorted by reward (winner highlighted)</span>
      </div>
      <table className="rt">
        <thead>
          <tr>
            <th>completion</th>
            <th>tokens</th>
            <th>correct</th>
            <th>reward</th>
            {batch.rewards && <th>run reward</th>}
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
              {row.realReward !== undefined && <td>{row.realReward.toFixed(3)}</td>}
              <td>{adv[k].toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="note">
        reward = correctness − λ·tokens/1000: a small λ keeps long correct answers on top;
        past λ≈1 short wrong answers outrank long right ones.
        {batch.rewards &&
          ' With a real batch loaded, “correct” is schema_ok and “run reward” is the actual task reward the trainer optimized.'}
      </p>
      <HowTo>
        <li>
          Rows are the {batch.tokens.length} completions of one prompt, re-sorted by
          shaped reward. The highlighted winner is what the update reinforces most.
        </li>
        <li>
          Drag <b>λ</b> from 0 to 2 and watch the ranking flip. Below ≈1 the long correct
          completions stay on top; past it short wrong answers win and their advantages go
          negative.
        </li>
        <li>
          The thought-length tradeoff from reasoning-model training: correctness alone
          rewards rambling; a heavy token penalty teaches the model to quit early, or to
          prefer being briefly wrong.
        </li>
        <li>
          With a real batch (panel 5 checkbox), every completion was already
          schema-valid, so the shaped ranking is decided by length alone and the shortest
          answer wins. The "run reward" column shows the actual reward the trainer used
          (schema + todo recall/precision + length) next to this deliberately simpler toy
          shaping.
        </li>
      </HowTo>
    </Panel>
  )
}
