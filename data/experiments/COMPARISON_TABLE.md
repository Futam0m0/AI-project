# Experiment results — 50 battles × 5 strategies

## Strategy matrix

| Strategy          | Win rate % | Avg rounds | Avg HP left | Avg damage dealt | Avg healing done | Clearance rate % |
|-------------------|-----------:|-----------:|------------:|-----------------:|-----------------:|-----------------:|
| astar             |      98.0% |      13.5 |       55.9 |            29.8 |             3.9 |            98.0% |
| baseline          |      98.0% |      12.9 |       58.1 |            29.9 |             1.7 |            98.0% |
| bfs               |      98.0% |      13.1 |       56.7 |            29.7 |             3.6 |            98.0% |
| ga-opt            |     100.0% |      10.4 |       69.3 |            30.0 |             0.1 |           100.0% |
| sa-opt            |      96.0% |      15.6 |       57.4 |            29.8 |             2.9 |            96.0% |

## Before / after — SA and GA optimisation vs handwritten baseline

| Metric              | Baseline | SA-optimised | Δ (SA)     | GA-optimised | Δ (GA)     |
|---------------------|---------:|-------------:|-----------:|-------------:|-----------:|
| Win rate %          |    98.0% |        96.0% |      −2.0% |       100.0% |      +2.0% |
| Avg party HP left   |      58.1  |        57.4  |      −1.2% |        69.3  |     +19.3% |
| Avg damage dealt    |      29.9  |        29.8  |      −0.1% |        30.0  |      +0.5% |
| Avg healing done    |       1.7  |         2.9  |     +71.8% |         0.1  |     −95.3% |
| Clearance rate %    |    98.0% |        96.0% |      −2.0% |       100.0% |      +2.0% |

## Search comparison: uninformed (BFS) vs optimal (A*)

| Metric              | Baseline | BFS          | Δ(BFS)   | A*            | Δ(A*)   |
|---------------------|---------:|-------------:|---------:|--------------:|--------:|
| Win rate %          |    98.0% |        98.0% |    +0.0% |        98.0% |   +0.0% |
| Avg rounds          |    12.9  |        13.1  |    +1.7% |        13.5  |   +5.0% |
| Avg party HP left   |     58.1  |        56.7  |    −2.5% |        55.9  |   −3.8% |
| Clearance rate %    |    98.0% |        98.0% |    +0.0% |        98.0% |   +0.0% |

## Implementation notes (§3 honesty requirement)

- **Search algorithms (BFS / A*)** are only used for dungeon navigation when no enemies are within Chebyshev ≤ 2 tiles. In-melee actions fall back to the role-based baseline so the party does actually engage — search is *not* used to warp characters through enemy-occupied tiles.
- **SA / GA optimisation** ran over the 10-weight `BehaviourWeights` chromosome described in `algorithms/baseline_decision.py`. Objective = maximise party HP remaining across 2 short battles (same score as the dropdown caches use). SA: 30 steps / cooling 0.92. GA: population 10 / 5 generations — the cached versions used in the dashboard are the exact chromosomes reported by these runs.
- **Minimax/Alpha-Beta/Expectimax** (dropdown only, not part of the 5-column experiment matrix) evaluate 1v1 sub-trees at melee. Tree depth is bounded: MM/AB max_depth=5, Expectimax max_depth=4, leaves evaluated by `heuristic_eval = (our_hp − their_hp)/max_hp` ∈ [−1, +1]. Expectimax attack action is stochastic → 0.2 miss, 0.7 hit, 0.1 crit — probabilities sum exactly 1.0.
- **Seed reproducibility**: 50 seeds = [100000, 100049]. `GameState.new_default(seed=s)` produces the same dungeon/party/enemies run-to-run via zlib-CRC32–derived per-actor PRNG seeding described in Phase 2.

