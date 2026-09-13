# viz — D1 visual RL demo

Static React+TS page explaining the RL training loop for LLMs. No backend, no
storage; everything deterministic via `mulberry32` (see `src/lib/rng.ts`).

```bash
npm install
npm run test      # vitest unit tests for src/lib/rl.ts
npm run build     # outputs dist/ (base './', all paths relative)
npm run preview   # or: python3 -m http.server 4173 -d dist
```

## Hosting

`dist/` is fully static: `fetch('./run.json')` resolves relative to
`index.html`, so any static host works. The repo deploys it to GitHub Pages
via `.github/workflows/pages.yml` (rebuilds from source on every push to
`main` that touches `viz/`). One-time setup: on
github.com/futingchan/mtrl-grpo-from-scratch, Settings → Pages → "Build and
deployment" → Source: **GitHub Actions**. Then Actions → "Deploy viz demo to
GitHub Pages" → Run workflow for the first deploy. Live at
https://futingchan.github.io/mtrl-grpo-from-scratch/.

## run.json

Panel 5 (`TrainingRun`) fetches `./run.json` relative to `index.html`. If the
fetch fails it renders a placeholder. `public/run.json` is a copy of a real
Tier A run (`runs/tier_a/run.json`), so dev and build both show real curves;
`public/run.example.json` is the synthetic fallback. Refresh it after a new
run with `cp runs/<name>/run.json viz/public/run.json && npm run build`
(or `scripts/plot_curves.py --copy-run-json` to patch a built dist directly).

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
    "lengths": [int × G],     // G = completions for ONE prompt
    "correct": [0|1 × G],     // schema_ok per completion
    "logratio": [float × G],  // log π_ref − log π_θ
    "rewards": [float × G]    // real task reward per completion
  },
  "eval": {                   // optional; greedy eval before/after training
    "before": { "reward_mean": 0.0, "todo_recall_mean": 0.0, "schema_ok_rate": 0.0, "n": 0 },
    "after":  { "reward_mean": 0.0, "todo_recall_mean": 0.0, "schema_ok_rate": 0.0, "n": 0 }
  }
}
```
