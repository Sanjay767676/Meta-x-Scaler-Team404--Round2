# Workspace

## Overview

pnpm workspace monorepo using TypeScript. Each package manages its own dependencies.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `pnpm --filter @workspace/api-server run dev` — run API server locally

See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.

## FORGE-v4  (Python — Adversarial RL Environment)

Located at `FORGE-v4/`. A standalone Python project; run independently of the pnpm workspace.

### Quick start
```bash
cd FORGE-v4
python3 app.py                         # improving_coder vs tiered Breaker
python3 app.py --coder weak_coder_v1   # bubble sort strategy
python3 app.py --coder weak_coder_v2   # selection sort w/ abs() bug
python3 app.py --steps 5              # override step count
```

### Key files
| File | Purpose |
|------|---------|
| `app.py` | CLI entry point |
| `env.py` | `FORGEEnv` — reset/step/get_state |
| `agents.py` | Coder strategies + `BreakerAgent` (tiered) |
| `tasks.py` | Task and hidden test generation |
| `sandbox.py` | Subprocess code execution with timeout |
| `rewards.py` | `coder_reward()` / `breaker_reward()` |
| `memory.py` | `CoachMemory` — JSON-backed lessons |
| `logger.py` | Writes `logs/rewards.json`, `logs/episodes.csv`, `logs/summary.json` |
| `trainer.py` | Training loop + TRL/Unsloth hook placeholders |
| `config.py` | All constants |

### Coder strategies
- `weak_coder_v1` — bubble sort (O(n²), slow on large arrays)
- `weak_coder_v2` — selection sort with abs() bug (fails on negatives)
- `improving_coder` — bubble sort → selection sort → `sorted()` by episode

### Breaker tiers
- Tier 1: empty / single element / tiny arrays
- Tier 2: duplicates, negatives, sorted/reverse-sorted
- Tier 3: large arrays, heavy duplicates, stress cases
- Tier 4: boundary integers (±100), extreme stress

Tier unlocks at 60% break rate; Tier 3 needs episode ≥ 4, Tier 4 needs episode ≥ 7.
