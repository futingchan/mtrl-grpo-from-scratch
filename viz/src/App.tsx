import { useMemo, useState } from 'react'

import { makeBatch, type ToyBatch } from './lib/batch'
import Gridworld from './panels/Gridworld'
import RewardShaping from './panels/RewardShaping'
import TheLoop from './panels/TheLoop'
import TrainingRun from './panels/TrainingRun'
import UpdateComparison from './panels/UpdateComparison'

export default function App() {
  const [seed, setSeed] = useState(7)
  const [lambda, setLambda] = useState(0.5)
  const [realBatch, setRealBatch] = useState<ToyBatch | null>(null)

  const batch = useMemo(() => realBatch ?? makeBatch(seed), [seed, realBatch])

  return (
    <>
      <h1>RL for LLMs — the loop, visually</h1>
      <label>
        seed{' '}
        <input
          type="number"
          value={seed}
          onChange={(e) => setSeed(+e.target.value || 0)}
          style={{ width: 80 }}
        />
      </label>
      <TheLoop batch={batch} />
      <Gridworld />
      <UpdateComparison batch={batch} lambda={lambda} />
      <RewardShaping batch={batch} lambda={lambda} setLambda={setLambda} />
      <TrainingRun onBatch={setRealBatch} />
    </>
  )
}
