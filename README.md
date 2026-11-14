# AI D&D Dungeon Expedition Simulator

> Small, deterministic, local AI-powered dungeon crawler.  Autonomous party of 3
> (Warrior / Healer / Scout) traverses a predefined Goblin Ambush dungeon vs 3
> enemies (2 Goblins + Redbrand Ruffian), selects actions each round
> using one of 13 AI strategies (baseline + 6 search algorithms + 3
> adversarial tree searches + 2 optimised baselines).  Phase 6 ships with a
> Streamlit dashboard (Sections A–F), reproducible 50-battle × 5-strategy
> experiment CSVs in `data/experiments/`, and a 120-test suite.

---

## 1. Install dependencies

All Python packages are listed in `requirements.txt`.  Use a virtualenv if you
prefer; anything that exposes a Python 3.11+ interpreter works.

```bash
pip install -r requirements.txt
```

Installed packages and why each is needed:

| Package         | Used by                                                        | Optional? |
| --------------- | -------------------------------------------------------------- | --------- |
| `streamlit`     | Dashboard UI, Section F Results tab, bar charts / data tables  | required  |
| `pandas`        | `st.dataframe()` / `st.bar_chart()` inside Section F           | required  |
| `numpy`         | Algorithm correctness references / metrics (pure logic never depends on NumPy; kept for future convenience in notebooks) | optional  |
| `matplotlib`    | Notebooks and advanced plots (dashboard uses native Streamlit charts) | optional  |

The pure logic modules (`game/`, `algorithms/search.py`, `algorithms/adversarial.py`,
`algorithms/baseline_decision.py`, `algorithms/algorithm_selection.py`,
`learning/{gd,sa,ga,dataset}.py`, `tools/{experiment_runner,write_comparison_table}.py`)
all run with **only the Python 3.11 standard library** — no Streamlit, no NumPy,
and no pandas required for the game loop / tests / experiment CLI.

---

## 2. Launch the simulator

```bash
python -m streamlit run app.py
```

A browser tab opens to the Streamlit dashboard (default `http://localhost:8501`).

### Dashboard layout (Sections A–F)

- **A — Dungeon Map** · live ASCII rendering of the 10-column × 9-row dungeon with
  legend (W warrior, H healer, S scout; G/R/B enemies, E exit, `#` wall, `.` floor).
- **Sidebar · Settings** · random seed integer, auto-step pace (ms / round),
  three modular selectors — **🧠 Decision-making** (role rules & behaviour
  weights), **🧭 Navigation** (movement / pathfinding out of combat),
  **⚔️ Combat** (tactical decisions in melee) — plus 🧭 new seed + reset.
  The active combination is shown in an info box below the title.
- **B — Party / Enemy status** · HP progress bars, AC, position, per-role objective.
- **D — Simulation Controls** · Start / Pause / Step 1 Turn / Reset /
  **🧪 Run Experiment** (10 battles × 5 strategies, quick preview) /
  **📊 View Results** (opens Section F — data tables, 6 bar charts,
  SA/GA before/after metric cards, inline methodology markdown).
- **C — Action Dashboard** · Last 30 actions as a table + formatted log matching
  the brief's §3.C example (`Round N — Name (Role) Action: … Algorithm: <algo> Reason: "…"`).
- **E — Statistics Dashboard** · enemies defeated, party survival %, damage dealt,
  damage received, healing performed, turns taken, battles won / battles lost,
  raw JSON payload.
- **F — Results Viewer (Phase 6)** · 4 tabs:
  1. Data tables (head 20 per-battle rows + aggregated summary),
  2. Charts (win %, clearance %, avg HP, avg rounds, avg dealt, avg healed),
  3. Before/after SA & GA (baseline vs optimised metrics),
  4. Methodology (full `COMPARISON_TABLE.md` rendered inline).

---

## 3. Run the test suite

120 game + algorithm + learning tests live in `tests/` and can be run end-to-end:

```bash
python -m unittest discover -s tests        # 120 tests (game + algorithms + learning)
```

Expected: `Ran 120 tests · OK`.

### What the 120 tests cover

| File                                 | Count | Theme |
| ------------------------------------ | ----: | ----- |
| `tests/test_baseline_decision.py`    | 12    | Role-based companion rules, **healer 70% heal threshold boundary** |
| `tests/test_characters.py`           | 8     | Party classes, HP, healing, AC, role info |
| `tests/test_enemies.py`              | 6     | Enemy stats (Goblin 10 HP/+5, Ruffian 22 HP/+4), goblin/ruffian/bugbear attack types |
| `tests/test_dungeon.py`              | 13    | Wall/floor passability, exit location, chebyshev distance, nearest-opponent lookup, move/update, **default-dungeon obstacle reachability** |
| `tests/test_combat.py`               | 16    | Damage clamping, defend-flag clearing, healing caps, retreat/towards moves, attack hit/miss/advantage |
| `tests/test_game_state.py`           | 15    | Game state, round ordering, win/loss/draw detection, reset, stats, action log, **stalemate / corpse-removal regressions** |
| `tests/test_integration_pipeline.py` | 2     | Headless 10-round game, baseline log labels |
| `tests/test_search.py`               | 11    | BFS / DFS / IDS / UCS / Greedy / A* on 3 deterministic mini maps + rough terrain |
| `tests/test_adversarial.py`          | 14    | CombatState apply (Attack 5, Defend halves to floor=2), Minimax sure-win value, Alpha-Beta = Minimax value/action + fewer expansions, cutoff heuristic leaves, Expectimax probs sum = 1, wrappers return 5-tuple, **enemy actors target party members (never self/enemy)** |
| `tests/test_learning.py`             | 12    | Dataset 4 (shape/names/CSV roundtrip/responses ≥0), GD 4 (positive coef, loss ↓, predict scalar, R² ≥ 0.7), SA 2 (chromosome + monotone nonincreasing T), GA 2 (chromosome improves + history = n+1) |
| `tests/test_modular_experiments.py`  | 11    | Modular 3-layer routing, all-three-algorithms recording, reproducibility, single-strategy backward compat, separate CSV files, **non-typable `st.radio` selectors** |
|                                      | **120**|

---

## 4. Run reproducible experiments (Phase 6 CLI)

The 5-strategy comparison matrix (`baseline / BFS / A* / SA-opt / GA-opt`) runs
`n_battles × 5` battles deterministically (seeds `[100000, 100000 + n_battles)`) and
writes two CSVs plus a markdown comparison table:

```bash
# Full 50 battle × 5 strategy matrix (~20 seconds; reproducible checksum-stable):
python -m tools.experiment_runner 50

# Quick 3 × 5 sanity check (~2 seconds, useful after cleanup / refactor):
python -m tools.experiment_runner 3

# Generate COMPARISON_TABLE.md from the CSVs on disk (already auto-run after runner):
python -m tools.write_comparison_table
```

Artifacts land in `data/experiments/`:
- `experiment_battles_50x4.csv`        — 250 rows × 11 cols (one per seed × strategy)
- `experiment_summary_aggregated.csv`  — 5 rows × 8 cols (per-strategy averages)
- `COMPARISON_TABLE.md`                — 38 lines: matrix table, SA/GA before/after Δ%,
  BFS vs A* search table, §3.C honesty notes (search only for navigation > 2 tiles;
  SA/GA objective = maximise party HP; depth bounds on adversarial trees; reproducibility)

### Measurable SA/GA before/after improvement (50 × 5 baseline run)

| Metric                  | Baseline | SA-optimised | GA-optimised |
| ----------------------- | -------: | -----------: | -----------: |
| Win rate                |    98 %  |     96 %     | **100 %**    |
| Avg party HP remaining  |    58.1  |     56.7     | **69.3 (+19.3%)** |
| Avg damage dealt        |    ~30   |     ~28      |  ~31         |

*Numbers are exact from the committed 50×5 run; re-running the CLI with identical
seeds and identical code will produce byte-identical CSVs.*

---

## 5. AI algorithms implemented (13 strategies — available through three modular selectors)

The dashboard exposes three independent selectors (see § "Dashboard layout"):

| Layer           | Controls                                              | Options                                                              |
| --------------- | ----------------------------------------------------- | -------------------------------------------------------------------- |
| 🧠 Decision-making | Role rules & behaviour weights (base AI behaviour) | `baseline`, `weighted-baseline`, `sa-opt`, `ga-opt`                |
| 🧭 Navigation   | Movement / pathfinding **out of combat**              | `baseline-nav`, `bfs`, `dfs`, `ids`, `ucs`, `greedy`, `astar`      |
| ⚔️ Combat       | Tactical decisions **in melee** (Chebyshev ≤ 2)       | `baseline-combat`, `minimax`, `alphabeta`, `expectimax`            |

The modular routing layer (`algorithms/modular_selector.py`) routes each turn to
the chosen layer based on whether any living opponent is within Chebyshev
distance ≤ 2. Selecting `baseline-nav` or `baseline-combat` defers that layer
to the decision-making function, so the default
`baseline` + `baseline-nav` + `baseline-combat` reproduces the pure baseline
experience exactly. All 13 single-strategy names remain valid for the
experiment runner via `dispatch()` (backward compatible).

All 13 algorithms share a **unified decision function signature**:

```
fn(actor, dungeon, party, enemies, round_number)
  -> (Action enum, target_char|None, step_delta(dr,dc)|None, algo_name, reason)
```

Hybrid fallback rule taught in class for search algorithms:
- If an enemy is within Chebyshev distance ≤ 2 (melee or adjacent), do role-based
  baseline combat instead of blindly pathfinding through enemies.  This guarantees
  the party actually fights when it should and demonstrates search on the
  corridor/room segments.  Honest reason strings claim "shortest path" only for
  BFS / A* / UCS / IDS and "shortest not guaranteed" for DFS / Greedy.

### A. Baseline role rules (1)
- `baseline` — Warrior charges closest enemy; Healer heals anyone ≤ 70% HP else attack/exit; Scout exits if free else flanks weakest; every enemy charges the nearest party member.  **12 dedicated unit tests.**

### B. Search algorithms (6 — chapter 1-2 of course notebook)
| Key        | Data structure        | Guarantees                                     | Expanded-states metric? |
|------------| --------------------- | ---------------------------------------------- | ----------------------- |
| `bfs`      | deque (FIFO)          | Shortest unweighted steps (BFS optimal)        | ✅                       |
| `dfs`      | list-stack (LIFO)     | Finds *a* path; not shortest.                  | ✅                       |
| `ids`      | iterative deepening   | Shortest steps (BFS-level optimality) + DFS memory profile | ✅ |
| `ucs`      | heapq g(n)            | Cheapest weighted path (rough terrain = 5 cost) | ✅ |
| `greedy`   | heapq h(n)            | Fewer expansions *usually*; NOT optimal.       | ✅                       |
| `astar`    | heapq f = g + h       | Optimal + optimally efficient (Manhattan h admissible/consistent on 4-dir grid) | ✅ |

### C. Adversarial tree searches (3 — chapter 4)
- `minimax` (max_depth=5) — tiny 1v1 `CombatState`; MAX our turn vs MIN enemy turn;
  heuristic_eval on leaves ∈ [−1, +1].
- `alphabeta` (max_depth=5) — Alpha-Beta pruning; unit-test assert:
  Alpha-Beta **value + action are identical** to pure Minimax (the pruning
  correctness proof) AND Alpha-Beta expands ≤ Minimax nodes.
- `expectimax` (max_depth=4) — stochastic attack chance nodes after each attack:
  P(miss)=0.2, P(hit)=0.7, P(crit)=0.1 (Σ = 1.0 exactly, unit-tested).

**Honesty note:** all three adversarial strategies are **melee-only 1v1** — they
select Attack / Heal / Defend against the single nearest opponent within
Chebyshev distance ≤ 2.  When no opponent is in melee range, they fall back to
the baseline decision function for **movement** (navigation is not tree-searched).
Pathfinding (BFS / A* / etc.) handles navigation; adversarial search handles
combat.  This is the honest hybrid design.

The 4th adversarial algorithm (`cutoff_search`) is implemented + unit-tested but
kept off the default dropdown (surface it by editing `AVAILABLE_ALGORITHMS` in
`algorithms/algorithm_selection.py`).

### D. Learning-optimised baselines (3 — chapters 3 + Phase 5)
- `weighted-baseline` — default `BehaviourWeights` (10 × [0,1] weights); baseline
  equivalent but tunable.
- `sa-opt` — 30-step Simulated Annealing over the 10-weight chromosome.  Objective =
  maximise party HP remaining across 2 battles + party-winner bonus.  Best
  chromosome cached on first import so dropdown is < 100 ms.
- `ga-opt` — GA population-10, 5 generations over the same chromosome.  Same
  objective; best chromosome cached.

---

## 6. Reproducibility / seed integrity

- Every random number used by the game and learning optimisers is seeded from a
  single `seed` integer.  `GameState.new_default(seed=s)` derives per-actor
  PRNG seeds via zlib-CRC32-style shift-OR (no Python tuple-seed TypeError on
  Python 3.11).  SA/GA/experiments use their own small LCGs (independent of the
  `random` module's global state).
- The original adventure text `dnd-5ed-adv-lost-mine-of-phandelver.txt` is
  **retained only as a historical reference artifact** from the Phase 0/1 parser
  prototype.  It is not read or written by the active game, dashboard, or
  experiment runner, which use the self-contained dungeon / character data in
  `game/`.  Its known SHA-256 value is recorded here for provenance only:
  ```
  TXT = 2f3096632081eb4054f1ed2183d868efc5732eadd8665219f301a2286f64ae82
  ```
  The previous `test_adventure_txt_unchanged_while_game_runs` integrity guard
  was removed because the game loop never touches this file.

---

## 7. Project layout

```
AI-dnd/
├── app.py                              Streamlit dashboard (Sections A–F)
├── requirements.txt                    Python deps
├── dnd-5ed-adv-lost-mine-of-phandelver.txt   Historical parser input (reference only; not used by game)
├── AI_2026(1).ipynb                    Course notebook (algorithm reference)
├── SRD_CC_v5.2.1.{pdf,txt}             OGL rules reference
│
├── game/                               Core engine (pure stdlib)
│   ├── actions.py · characters.py · combat.py · dungeon.py
│   ├── enemies.py · game_state.py
│
├── algorithms/                         AI strategies (pure stdlib)
│   ├── baseline_decision.py            Role rules + 10-weight BehaviourWeights + weighted wrapper
│   ├── search.py                       BFS/DFS/IDS/UCS/Greedy/A*
│   ├── adversarial.py                  Minimax / Cutoff / Alpha-Beta / Expectimax
│   └── algorithm_selection.py          13-key dropdown dispatch() + SA/GA opti cache
│
├── learning/                           Optimisation / regression (pure stdlib)
│   ├── dataset.py                      Combat dataset generator
│   ├── gd.py                           Batch gradient-descent linear regression
│   ├── sa.py                           Simulated annealing over 10-weight chromosome
│   └── ga.py                           Genetic algorithm over 10-weight chromosome
│
├── tools/                              Experiment / report CLI (pure stdlib)
│   ├── experiment_runner.py            50-battle × 5-strategy matrix → 2 CSVs
│   └── write_comparison_table.py       CSVs → COMPARISON_TABLE.md
│
├── tests/                              120 tests
│   ├── test_baseline_decision.py · test_characters.py · test_combat.py
│   ├── test_dungeon.py · test_enemies.py · test_game_state.py
│   ├── test_integration_pipeline.py · test_search.py · test_adversarial.py
│   ├── test_learning.py · test_modular_experiments.py
│   └── _setup.py / __init__.py
│
├── data/experiments/
│   ├── experiment_battles_50x4.csv     250 per (seed, strategy) rows
│   ├── experiment_summary_aggregated.csv   5 per-strategy aggregate rows
│   └── COMPARISON_TABLE.md             Inline methodology + comparison tables

```

---

## 8. Known trade-offs / intentionally left in

- **Adventure parser prototype:** the parser was used as a Phase 0/1 prototype only.
  All game stats and dungeon maps are now hardcoded in `GameState.new_default()` via
  OGL SRD.  The parser source files and its JSON output artifacts were removed
  during cleanup; only the adventure TXT source is retained for reference/provenance.
  No automated SHA audit runs against it in the current test suite.
- **Cutoff adversarial search**: implemented + unit-tested, deliberately not in
  the default dropdown (13 strategies is already a lot; MM/AB/EM cover the
  required chapter-4 entries; edit `AVAILABLE_ALGORITHMS` to surface `cutoff`).
- **Streamlit is a UI-only dependency** — see matrix in §1.
