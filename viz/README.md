# viz — D1 visual RL demo

Static React+TS page explaining the RL training loop for LLMs. No backend, no
storage; everything deterministic via `mulberry32` (see `src/lib/rng.ts`).

```bash
npm install
npm run test      # vitest unit tests for src/lib/rl.ts
npm run build     # outputs dist/ (base './', openable from file://)
npm run preview   # or: python3 -m http.server 4173 -d dist
```

## run.json

Panel 5 (`TrainingRun`) fetches `./run.json` relative to `index.html`. If the
fetch fails it renders a placeholder — to preview, copy
`public/run.example.json` to `dist/run.json` (vite copies `public/*` into
dist; a real run writes `run.json` next to `index.html`).

Schema:

```jsonc
{
  "steps": [
    {
      "step": 0,              // int
      "reward_mean": 0.0,     // float
      "reward_std": 0.0,      // float
      "kl": 0.0,              // float
      "completion_len": 0.0,  // float
      "schema_ok": 0.0        // float in [0,1]
    }
  ],
  "sample_batch": {           // optional; drives panels 3–4 via checkbox
    "lengths": [int × 8],
    "correct": [0|1 × 8],
    "logratio": [float × 8],  // log π_ref − log π_θ
    "rewards": [float × 8]
  }
}
```
