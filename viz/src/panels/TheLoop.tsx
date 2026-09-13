import { useEffect, useRef, useState } from 'react'

import { Bars, HowTo, Panel } from '../components'
import type { ToyBatch } from '../lib/batch'
import { mulberry32 } from '../lib/rng'
import { grpoAdvantages, mean, shapedReward, softmax, std } from '../lib/rl'

const STAGES = [
  'prompt',
  'sample G',
  'reward r [G]',
  'advantage A [G]',
  '∇: update θ on drawn arms',
  'new p = softmax(θ)',
]
const stageInfo = [
  'One document (the prompt) is held fixed. The bars are its possible completions: each is just a token count plus a correct flag, which is all the reward function sees.',
  'The policy draws a fresh group of G completions from p. The same completion can be drawn twice; which arms appear is the randomness of rollout.',
  'Each drawn completion is scored: r = correct − λ·tokens/1000. r is fixed per completion; what changes across iterations is which arms get drawn.',
  'Rewards are standardized inside the drawn group: A = (r − mean)/(std). This is the GRPO baseline: no critic needed.',
  'The update: each drawn arm moves by the sum of its advantages in the group. Undrawn arms get no update: only what you sample can be reinforced.',
  'Re-normalize: p = softmax(θ). Drawn winners took mass; every other bar shrank by normalization. The next iteration draws a new group.',
]
const ETA = 1

/** Draw one index from a categorical distribution. */
function drawFrom(p: number[], u: number): number {
  let acc = 0
  for (let i = 0; i < p.length; i++) {
    acc += p[i]
    if (u < acc) return i
  }
  return p.length - 1
}

export default function TheLoop({
  batch,
  lambda,
}: {
  batch: ToyBatch
  lambda: number
}) {
  const G = batch.tokens.length
  const [theta, setTheta] = useState<number[]>(() => Array(G).fill(0))
  const [stage, setStage] = useState(0)
  const [iter, setIter] = useState(0)
  const [draw, setDraw] = useState<number[]>([])
  const [playing, setPlaying] = useState(false)
  const rngRef = useRef(mulberry32(1))

  // a new batch is a new prompt group: start the toy policy over
  useEffect(() => {
    setTheta(Array(batch.tokens.length).fill(0))
    setStage(0)
    setIter(0)
    setDraw([])
    rngRef.current = mulberry32(1)
  }, [batch])

  const p = softmax(theta)
  // reward is a fixed property of each completion; the group is resampled per iteration
  const armR = batch.tokens.map((t, i) => shapedReward(batch.correct[i], t, lambda))
  const drawn = draw.length > 0
  const drawnR = draw.map((i) => armR[i])
  const m = mean(drawnR)
  const s = std(drawnR)
  const count = Array(G).fill(0)
  draw.forEach((i) => count[i]++)
  // per-arm chart values: an arm contributes only if it was drawn this iteration
  const rShow = drawn ? armR.map((r, i) => (count[i] ? r : 0)) : armR
  const aShow = drawn
    ? armR.map((_, i) => (count[i] ? (armR[i] - m) / (s + 1e-6) : 0))
    : grpoAdvantages(armR)
  // net logit update per arm: (η/G)·Σ advantages over the times it was drawn
  const dTheta = armR.map(
    (_, i) => (ETA / G) * count[i] * ((armR[i] - m) / (s + 1e-6)),
  )

  const advance = () => {
    if (stage === 0) {
      // sample a fresh group of G from the current policy
      const rng = rngRef.current
      setDraw(Array.from({ length: G }, () => drawFrom(p, rng())))
      setStage(1)
    } else if (stage === STAGES.length - 2) {
      // apply the group update, then show the renormalized policy
      setTheta((th) => th.map((t, i) => t + dTheta[i]))
      setIter((n) => n + 1)
      setStage(STAGES.length - 1)
    } else if (stage === STAGES.length - 1) {
      setStage(0)
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
    setTheta(Array(G).fill(0))
    setStage(0)
    setIter(0)
    setDraw([])
    setPlaying(false)
    rngRef.current = mulberry32(1)
  }

  // all charts always visible; the current stage's chart is lit
  const lit = (stages: number[]) =>
    stage === 0 || stages.includes(stage) ? 1 : 0.3

  const rLabels = drawn
    ? count.map((c, i) => (c ? `×${c}` : `${i}`))
    : undefined

  return (
    <Panel
      title="1. The RL loop"
      idea={`The whole mechanism in miniature: sample a fresh group of ${G}, score each draw, shift probability toward the arms that beat their siblings. Everything else in RL fine-tuning is plumbing around this.`}
      blurb={[
        `A policy is just a probability distribution over choices, and this one is reduced to the smallest object that still counts: ${G} logits θ, one per possible completion of a single fixed prompt, normalized by softmax into the bars in the top chart. Each bar is a whole completion stripped to the two numbers the reward sees: a token count and a correct flag, so "50tok✓" is a 50-token right answer (r ≈ 0.97) and "115tok✗" is a 115-token wrong one (r ≈ −0.06). A real model's policy is the same object at a different scale: a distribution over ~150k vocabulary tokens per position, emitted by hundreds of millions of parameters.`,
        `Step through one iteration. Each pass, the policy draws a fresh group of ${G} from p (a bar marked ×n was sampled n times this round), a program scores each draw (correct minus a per-token cost; the λ is shared with panel 4), the drawn rewards are standardized, and each drawn arm's logit moves by the sum of its advantages. Undrawn arms get no update: only what you sample can be reinforced, which is why the group needs spread.`,
        `Watch what convergence looks like: the best arm gets drawn more often, gains more mass, and within ~15 iterations the group is almost all one completion. That is exploitation collapse, and it is real: the drawn group turns homogeneous, advantages shrink to ~0, and the updates die out. Eight arms on one softmax collapse fast; the real run resists longer because it juggles B prompts per step, an astronomically larger action space, and the β·KL tether of panel 3.`,
        `The mapping to the real trainer is one-to-one. "Sample G" is a real generate() call on Qwen2.5-0.5B, θ is LoRA weights instead of a table, and the advantage formula on screen is the formula in the loss. What changes between this panel and panel 5 is scale, not mechanism.`,
      ]}
      formula={
        `r_i = correct_i − ${lambda.toFixed(2)}·tokens_i/1000        fixed per completion\n` +
        `group: i_1..i_G ~ Categorical(p)                fresh draws each iteration\n` +
        `A_k = (r_{i_k} − mean)/(std + 1e-6)             over the drawn group\n` +
        `θ_i ← θ_i + (η/G)·Σ_{k: i_k=i} A_k,  η=${ETA}       undrawn arms: no update\n` +
        `p = softmax(θ)                                  p ∈ R^${G}, Σp=1`
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
      <p className="stage-info">{stageInfo[stage]}</p>
      <div className="controls">
        <button onClick={() => setPlaying(!playing)}>{playing ? 'Pause' : 'Play'}</button>
        <button onClick={advance}>Step</button>
        <button onClick={reset}>Reset</button>
        <span>
          iteration <b>{iter}</b>
        </span>
      </div>
      <p className="legend">policy p = softmax(θ):</p>
      <div style={{ opacity: lit([1, 5]), transition: 'opacity .4s' }}>
        <Bars
          values={p}
          labels={batch.tokens.map((t, i) => `${t}tok${batch.correct[i] ? '✓' : '✗'}`)}
          height={150}
        />
      </div>
      <p className="legend" style={{ marginTop: 24 }}>
        rewards r{drawn ? ' of the drawn group (×n = sampled n times)' : ' per completion'}
        :
      </p>
      <div style={{ opacity: lit([2]), transition: 'opacity .4s' }}>
        <Bars
          values={rShow}
          labels={rLabels}
          height={95}
          colors={(v) => (v >= 0 ? '#3d9e50' : '#b3261e')}
        />
      </div>
      <p className="legend" style={{ marginTop: 24 }}>
        advantages A over the drawn group:
      </p>
      <div style={{ opacity: lit([3]), transition: 'opacity .4s' }}>
        <Bars
          values={aShow}
          labels={rLabels}
          height={95}
          colors={(v) => (v >= 0 ? '#3d9e50' : '#b3261e')}
        />
      </div>
      <p className="legend" style={{ marginTop: 24 }}>
        update applied Δθᵢ = (η/G)·ΣAᵢ over draws:
      </p>
      <div style={{ opacity: lit([4]), transition: 'opacity .4s' }}>
        <Bars
          values={dTheta}
          labels={rLabels}
          height={95}
          colors={(v) => (v >= 0 ? '#3d9e50' : '#b3261e')}
        />
      </div>
      <HowTo>
        <li>
          Press <b>Step</b> to walk one stage at a time; <b>Play</b> runs whole
          iterations. The yellow caption says what the highlighted stage does, and the
          lit chart is the one that stage touches.
        </li>
        <li>
          The top chart is the policy: <code>p = softmax(θ)</code> over the {G} possible
          completions. Labels show each one's token count and correct flag (✓/✗).
        </li>
        <li>
          "sample G" draws a fresh group every iteration: <code>×n</code> labels mean an
          arm was sampled n times. The r and A charts only show drawn arms, so they
          change every round; the reward of a given arm never changes.
        </li>
        <li>
          The bottom chart is the logit update actually applied. Watch it shrink as the
          policy converges: once the draws are all the same good arms, their advantages
          approach zero and learning stalls.
        </li>
        <li>
          In the real trainer θ is LoRA weights on a 0.5B model and "sample G" is a real
          generate() call. The advantage formula is identical.
        </li>
      </HowTo>
    </Panel>
  )
}
