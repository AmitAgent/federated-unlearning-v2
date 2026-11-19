# Contributing

## Workflow

- Read `README.md` and `docs/CODE_MAP.md` first; they say where each part of the
  project lives.
- Run the tests before and after a change: `python -m pytest tests -q`.
- Add or update a test with every behavior change, and keep new methods opt-in so
  stored results do not change.
- Be honest in docs: do not claim a result that has not been measured and recorded.

## Repo layout

```
frontend/     React (Vite) dashboard — UI only, no ML logic
backend/      Node/Express API — routes requests to a ComputeBackend
core/         Python ML logic (datasets, federated, unlearning, evaluation)
experiments/  Generated experiment artifacts (gitignored contents)
docs/         Architecture doc, specification, results, guides, code map
```

## Coding conventions

- **Frontend**: functional React components, one component per file,
  colocate page components under `src/pages/`. No backend credentials
  or ML logic in frontend code, ever (spec Ch. 71.17).
- **Backend**: routes stay thin; anything backend-specific goes behind
  the `ComputeBackend` interface (`backend/src/compute/`) so the
  dashboard never depends on which backend is active.
- **Core**: standard Python, one concern per module, following the
  `core/federated/`, `core/unlearning/`, `core/evaluation/`,
  `core/datasets/` split from spec Ch. 49.
- Keep changes small and focused: one concern per change.

## Environment variables

- Never commit `.env` files. Copy `.env.example` to `.env` locally.
- No secrets (SSH keys, GPU credentials, API keys) ever go in frontend
  code — server-side/env only (spec Ch. 71.17).
