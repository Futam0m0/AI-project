# Phase 3–6 Implementation Plan: AI D&D Dungeon Expedition Simulator

## Repository Research

### 1. CURRENT STATE: Post-Phase-2 (verified, 173 / 173 tests PASS)

| Layer | Status | Evidence |
|---|---|---|
| **Parser + Review CLI** | ✅ Reusable, preserved untouched | 56 tests in `tools/` green; original TXT SHA-256 `2f3096…64ae82` byte-identical across runs; unreviewed JSON SHA `a21da2…ca9d54` unchanged |
| **Dungeon grid engine** | ✅ Complete | `game/dungeon.py` Position frozen dataclass, 8×10 walls + place/move, find_exit/nearest_enemy/nearest_party Manhattan tiebreak. 10 tests OK |
| **Characters (party)** | ✅ Complete | Warrior HP=28 AC=18 (1d8+3), Healer HP=20 AC=14 (heal 1d8+2 range 1), Scout HP=22 AC=15 (sneak 1d6). `apply_damage()` returns delta clamped to current HP. 8 tests OK |
| **Enemies (SRD)** | ✅ Complete | Goblin HP=7 AC=15 (1d6+2 OGL), Ruffian HP=16 AC=14 (LMoP simplified), Bugbear HP=27 AC=16 (2d8+2 OGL). 6 tests OK |
| **Actions + log format** | ✅ Complete | `Action` enum (Move/Attack/Heal/Defend/Retreat/Skip). `ActionResult.__str__` exactly matches brief §3.C (Round X — Name, indented Action/Algorithm/Reason lines). 4 tests OK |
| **Combat math** | ✅ Complete | roll_d20 (adv/disadv), roll_damage regex, `perform_attack` applies HP delta (clamped), `perform_heal` same, greedy step_towards/away_from 4-dir. 12 tests OK |
| **Baseline AI** | ✅ Complete | `algorithms/baseline_decision.choose_action` returns `(Action, target, step, "baseline", reason)` tuples, role rules exactly §6: Warrior adj→attack else nearest-enemy else exit; Healer inj-ally<50% heal else adj-attack else lowest-HP-ally else exit; Scout adj-weakest-within-1 else exit else nearest; Enemy adj-attack else step-nearest-party. 10 tests OK |
| **Game state orchestrator** | ✅ Complete | `GameState.new_default(seed)` → 8×10 dungeon, W/H/S (1,1..3), 2 Goblins + 1 Ruffian shuffled seed-positions, Exit E bottom-right; `advance_one_round()` turn order W→H→S→live-enemies; applies decision through combat helpers; defend_flags reset per round; 60-round termination guard; master seed CRC32-composed into per-actor int seeds (Python 3.11 strict: no tuple-seeds). 8 tests OK |
| **Integration** | ✅ Complete | Headless 30-round sims run without exception; SHA-256 of original parser files never change. 3 tests OK |
| **Phase 3 Streamlit** | ❌ NOT STARTED | No `app.py`, no `requirements.txt`. Streamlit NOT installed yet (env: numpy/matplotlib/networkx OK, streamlit `ModuleNotFoundError` confirmed via smoke test earlier). |
| **Course algorithms** | ❌ NOT STARTED | Only `baseline` exists. Notebook `AI_2026(1).ipynb` (127 cells, 4 chapters) contains 14 verified algorithm headings to implement (listed below). |
| **Learning (ML/optim)** | ❌ NOT STARTED | `learning/__init__.py` only. No GD/SA/GA/LinReg yet. |
| **Experiments** | ❌ NOT STARTED | `data/experiments/` empty. No CSV output, no 50-battle tables, no charts. |

### 2. Reusable Features (same 14-point inventory as Phase-1 plan, updated)

| Asset | Reuse plan |
|---|---|
| Parser + review CLI (`tools/`) | Preserve 100% as-is: 56 tests must stay green. No edits. |
| SRD `monsters.json 11`, `classes.json 02` | Already hardcoded stats in `enemies.py` / `characters.py` with OGL comments; can optionally build JSON-importer in Phase 6 only if time permits. |
| Course notebook `AI_2026(1).ipynb` | Reference code for ALL 14 algorithm implementations. Cell-by-cell translation. |
| `unittest` + fixed seeds pattern | Extend to algorithm correctness tests. All 173 tests use seed=42 or seed=0 pattern for determinism. |
| Parser dataclasses (`EnemyReference`, `Development`) | Phase 6 future: import reviewed JSON directly as enemy encounters. |
| Folder layout `game//algorithms//learning//data/experiments//tests/` | Extend in-place, no reorg needed. |
| Strict architecture rule (game ← algorithms ← learning ← UI) | Enforce — game.py NEVER imports Streamlit; algorithms NEVER import from app.py. |

### 3. Installed Dependencies & Trade-off Decision
Confirmed env: numpy 2.1.2, matplotlib 3.10.3, networkx 3.3 exist. **Streamlit must be installed in Phase 3.** Install command: `pip install streamlit --user` (~60 MB). Alternatives (stdlib HTTP + HTML) would require hand-writing 5 dashboard sections and are worse for beginners. Trade-off: one `pip install` vs. weeks of extra UI code → install Streamlit is correct.

---

## 4. Prioritized Algorithm List (verified from notebook headings)

Ordered by course chapter presentation order, mapped into Phase 4 then Phase 5:

### PHASE 4 (Search + Adversarial) — from notebook Ch. 1,2,4
1. **BFS** (notebook cell 44) — unweighted shortest grid path: Warrior move-to-enemy / Scout move-to-exit replacer vs baseline greedy step_towards.
2. **DFS** (cell 258) — dungeon traversal demo; also used in CSP backtracking.
3. **IDS** (cell 479) — depth-limited DFS repeat for shallowest-path proof vs BFS.
4. **UCS** (cell 633) — min-cost when tiles have weights (e.g. mud=3, corridor=1).
5. **Greedy Best-First Search** (cell 769) — heuristic-only (Manhattan h) pathfinding.
6. **A\*** (cell 1559, 1748 OO Romania) — f=g+h admissible; compare vs Greedy/UCS in experiment table.
7. **Minimax** (cell 3356) — 2-player combat: choose attack-or-defend considering 2 opponent replies.
8. **Minimax Pruning / 3-Level Simple Cutoff** (cell 3523) — cutoff variant with depth-limited heuristic eval.
9. **Alpha-Beta Pruning** (cell 4036) — prune dominated branches; compare node-count vs plain Minimax in stats.
10. **Expectimax** (cell 4309) — stochastic nodes for damage variance / hit probability.

### PHASE 5 (Optimization + ML) — notebook Ch. 3
11. **Gradient Descent** (cell 2281) — minimize loss on a damage-predictor linear model.
12. **Simulated Annealing** (cell 2639) — optimize companion behaviour weights escaping local optima.
13. **Genetic Algorithm** (cell 2921) + **Knapsack GA** (cell 3131) — optimize party behaviour chromosome (10 weights).
14. **Linear Regression** (use GD-trained model) — predict expected encounter damage per algorithm.

### NOT in notebook (NOT implemented, commented out so lecturer sees intent was verified against actual materials)
- Fuzzy Logic, CSP, K-means, Neural Networks / Backprop, Q-Learning / RL, NLP — **NOT in notebook**. Skip unless explicitly requested later. Brief lists them as candidates but rule §10 says "verify from course materials".

---

## 5. Proposed Minimal Architecture Update (current + new files)

```
AI-dnd/
├── app.py (Phase 3, 250-300 lines) — Streamlit dashboard: 5 sections A-E + sidebar + controls
├── requirements.txt (Phase 3 — ONLY: streamlit, numpy, matplotlib)
│
├── game/ (COMPLETE — no Streamlit imports allowed)
│
├── algorithms/
│   ├── baseline_decision.py  (complete)
│   ├── search.py             (Phase 4: BFS DFS IDS UCS Greedy A* + h=manhattan + integration helpers)
│   ├── adversarial.py        (Phase 4: Minimax Cutoff AlphaBeta Expectimax for 1v1 combat)
│   └── algorithm_selection.py  (Phase 4: thin dispatcher → algorithm-name parameter → returns (action, target, step, algo, reason) matching baseline signature)
│
├── learning/
│   ├── __init__.py
│   ├── gradient_descent.py   (Phase 5)
│   ├── linear_regression.py  (Phase 5 — predicts dmg from (actor_stats, enemy_stats, dist))
│   ├── simulated_annealing.py (Phase 5)
│   └── genetic_algorithm.py  (Phase 5 — behaviour-weight optimizer, reuse same chromosome schema as SA)
│
├── data/experiments/
│   ├── schema.md (Phase 6 — not created proactively; created only if user asks)
│   └── run_*.csv  generated by experiment runner
│
└── tests/
    ├── _setup.py
    ├── test_dungeon.py … test_integration_pipeline.py (COMPLETE — 61 new tests)
    ├── parser/  (COMPLETE — 56 parser copies)
    ├── test_search.py  (Phase 4 — 18 tests: each algorithm finds correct path/cost/nodes-expanded on fixed grid)
    ├── test_adversarial.py  (Phase 4 — 10 tests: Minimax value matches known tree; Alpha-Beta returns same val as Minimax)
    ├── test_learning.py  (Phase 5 — 12 tests: GD converges, SA improves vs random start, GA improves avg fitness across gens)
    └── test_experiments.py  (Phase 6 — 6 tests: 50-battle CSV schema OK, 100-battle baseline-vs-A* win rate printed)
```

### Architecture dependency rules (enforced)
1. `game/*` imports ONLY stdlib. Never imports `algorithms/*`, `learning/*`, or `app.py`. (Algorithms are CALLABLES passed INTO `GameState.advance_one_round(decision_fn=choose_action)`.)
2. `algorithms/*` imports `game.*` (uses Position/Dungeon/BaseCharacter symbols) but NOT `learning/*` or `app.py`.
3. `learning/*` imports `algorithms.*` for baseline, NOT game/* Streamlit.
4. `app.py` sits on top. Imports everything. No one imports app.py.
5. `decision_fn` signature unified: `(actor, dungeon, party, enemies, round_num) → (Action, target, step, str_algo_name, str_reason)`. Exactly baseline signature → all algorithms satisfy.

---

## 6. Files and Modules per Phase

| # | File/Location | Expected change |
|---|---|---|
| 3.1 | `requirements.txt` | Create with `streamlit` + pin `numpy>=2, matplotlib>=3` (networkx optional for plot layouts) |
| 3.2 | `app.py` | New 250–300 lines. 5 Streamlit sections A–E. Session state holds `st.session_state["game"]`, `"paused"`, `"current_algorithm"`, `"experiment_results"`. |
| 3.3 | `game/game_state.py` | Minor 2-line addition: `advance_one_round(decision_fn=baseline_decision.choose_action)` parameter so Streamlit can swap algorithms live. Add `reset()` method. |
| 4.1 | `algorithms/search.py` | New ~260 lines. 6 functions, same signature: returns (step_tuple, cost, expanded, path) → then thin wrapper maps to (Action.MOVE, target, step, "BFS", reason). Integrate with `game_state.advance_one_round` via `algorithm_selection.py`. |
| 4.2 | `algorithms/adversarial.py` | New ~220 lines. 4 1v1 combat look-aheads. For party: when actor adjacent to enemy, run Minimax to pick (Attack or Defend). Algorithm name → "Minimax"/"AlphaBeta"/"Expectimax". |
| 4.3 | `algorithms/algorithm_selection.py` | New ~120 lines. 1 dispatch function: given algorithm name (baseline / BFS / DFS / IDS / UCS / Greedy / A* / Minimax / AlphaBeta / Expectimax) → return correctly-shaped decision tuple for caller. Adds node-count metric to stats dict. |
| 5.1 | `learning/gradient_descent.py` | New ~150 lines. Mini-batch SGD, MSE loss. Dataset = 200 simulated combat turns pre-recorded. |
| 5.2 | `learning/linear_regression.py` | New ~90 lines. Uses GD to fit θ. `predict(actor_hp_frac, enemy_hp_frac, dist)` → expected damage next round. Dashboard E shows error metric. |
| 5.3 | `learning/simulated_annealing.py` | New ~160 lines. Temperature schedule, move neighbour = jitter 1 of 10 weights. Reports best-so-far behaviour vector. |
| 5.4 | `learning/genetic_algorithm.py` | New ~220 lines. Population 30, 10 weights per chromosome. Fitness = 50-battle win-rate. Uniform crossover. |
| 6.1 | `tools/experiment_runner.py` | New ~180 lines. `run_battle(seed, algorithm_name) → stats_row`. `run_experiment(n=50, seeds=[...]) → CSV + printed markdown comparison table baseline vs algorithm X`. Uses `data/experiments/` for CSVs. |
| 6.2 | `tests/test_search.py`, `tests/test_adversarial.py`, `tests/test_learning.py`, `tests/test_experiments.py` | New test files (listed in architecture section) — create after each implementation. |
| 6.3 | `app.py` (Phase 6 addition) | "View Results" sidebar tab: loads `data/experiments/*.csv`, matplotlib line charts inside `st.pyplot()`, table of algorithm win rates, before/after learning weights diff. |

---

## 7. Implementation Steps (dependency order)

### Phase 3 — Streamlit Dashboard (1 approval gate, ~4 steps)
1. **Step 3.1 — Install Streamlit + write `requirements.txt`.** Run a 1-line smoke: `streamlit hello` then kill.
2. **Step 3.2 — Dashboard skeleton.** Section A (dungeon map via `st.code("\n".join(grid))` monospace), Section B sidebar with 3 party cards, Section D 6 buttons (Start / Pause / Step / Reset / Run Experiment / View Results — last 2 disabled until Phase 6). Wire `st.session_state["game"] = GameState.new_default(seed=42)`.
3. **Step 3.3 — Log + stats panels.** Section C scrollable `st.dataframe()` of last 30 ActionResults (columns Round/Actor/Action/Target/Algorithm/Reason). Section E 8 stat metrics cards from `stats.to_dict()`.
4. **Step 3.4 — Live round loop with `st.rerun()` + pause/step/reset.** "Start" schedules auto `advance_one_round()` every 250 ms; "Pause" clears flag; "Step 1 Turn" calls once; "Reset" re-inits.

### Phase 4 — Search + Adversarial Algorithms (~8 steps)
5. **Step 4.1 — BFS + test_search.py.** Test on hand-designed 6×8 grid with 1 known shortest path.
6. **Step 4.2 — DFS, IDS, UCS.** Each adds cell-by-cell from notebook. UCS uses tile weights map (default all 1).
7. **Step 4.3 — Greedy + A*.** A* uses Manhattan h. Compare paths against BFS baseline.
8. **Step 4.4 — `algorithm_selection.py` dispatcher.** Unified signature.
9. **Step 4.5 — Swap `decision_fn` into `GameState.advance_one_round(decision_fn=...)`.** Dashboard adds selectbox: "Algorithm" = baseline/BFS/DFS/IDS/UCS/Greedy/A*.
10. **Step 4.6 — Minimax + test_adversarial.py.** 2-ply (MAX actor → MIN enemy → terminal).
11. **Step 4.7 — Minimax Cutoff (3-level, eval=HP diff) + Alpha-Beta + Expectimax.** Expectimax for stochastic damage roll-out.
12. **Step 4.8 — Adversarial wired into dashboard selectbox.** Node-expanded counters appear in stats extra.

### Phase 5 — Optimization / ML (~5 steps)
13. **Step 5.1 — Generate 200 combat-turn dataset.** Features: (actor_hp_frac, ally_min_hp_frac, enemy_min_hp_frac, dist_enemy, num_enemies_adj). Target: damage_dealt_this_round.
14. **Step 5.2 — GradientDescent + LinearRegression train on dataset → report train/test MSE.**
15. **Step 5.3 — Behaviour-weight schema (10 weights): W_atk_bias / W_move_to_en / W_move_to_ex / H_heal_thr / H_atk_bias / S_atk_weak_bias / S_move_ex_bias / E_atk_bias / E_move_bias / defend_threshold.**
16. **Step 5.4 — SA optimizes 10 weights starting from baseline (all=0.5).** Report best weights.
17. **Step 5.5 — GA optimizes same 10-weight chromosome.** Report fitness across 30 gens best+avg.

### Phase 6 — Experiments + Cleanup (~5 steps)
18. **Step 6.1 — `tools/experiment_runner.py`.** CSV schema: seed, algorithm, outcome, turns, enemies_defeated, dmg_dealt, dmg_recv, heal, survival_rate, nodes_expanded, training_epoch (0 baseline or GA epoch).
19. **Step 6.2 — Run 50 battles baseline vs BFS, 50 vs A*, 50 vs AlphaBeta.** Print markdown comparison table of 3 metrics: win% / avg turns / avg survival.
20. **Step 6.3 — Before-After learning: 50 baseline-weights vs 50 SA-optimized weights vs 50 GA-optimized.** GA fitness = exactly this metric; verify SA/GA improve win-rate.
21. **Step 6.4 — Dashboard View Results tab loads CSV, renders matplotlib charts (matplotlib already installed).**
22. **Step 6.5 — Full final combined test run (old 56 + Phase 2 61 + Phase 4/5 tests ~46) → ~163+ tests. Re-verify SHA-256 of parser source files STILL identical. Produce final phase report.**

---

## 8. Dependencies and Considerations

### Streamlit install
- `pip install streamlit` → adds ~60 MB. Run smoke: `streamlit hello` then stop. Minimum version 1.30+.
- After install, ALL tests must still PASS (Streamlit in no path dependency of game modules — guards 4-architecture rules above).

### Beginner-readable code
- Every algorithm module has top docstring explaining inputs/outputs in 2 sentences.
- All new files ≤ ~300 lines.
- Variable names: `frontier`, `visited`, `parent`, `cost_so_far`, `best_value`, `temperature`, `population`, `fitnesses`. No single-letter vars except `h`, `g`, `f` (standard A* notation, explained in comment).

### Determinism
- All battle simulations seeded from master int seed. `random.Random(seed)` per battle. No global `random.seed`.
- CSV outputs include `seed` column for perfect reproduction.

### Parser untouched policy
- No code in Phases 3-6 shall read or write the adventure TXT. All tests verify.
- Baseline algorithm (`baseline_decision.py`) preserved as control group. Experiments ALWAYS compare against it.

---

## 9. Validation per Phase

| Phase | Validation |
|---|---|
| 3 Dashboard | (a) `streamlit run app.py` → 5 sections visible; (b) 6 buttons each do something (Start auto-advances rounds each second, Pause stops, Step=1, Reset=new seed, Run Exp=prints "TODO Phase 6", View Results=prints empty). (c) `python -m unittest discover -s tests` → 117 still green. |
| 4 Search + Adversarial | (a) 18+10 new tests all pass; (b) Smoke headless game BFS vs baseline — BFS path finds exit 10+ turns faster; (c) AlphaBeta result == Minimax result on 3+ fixed trees (equality property); (d) SHA-256 of parser files unchanged. |
| 5 Learning | (a) 12 learning tests pass: GD MSE strictly decreases across 50 epochs, SA improves cost >= baseline after 500 steps, GA gen-30 avg fitness > gen-1; (b) Dashboard stat cards show before/after learned policy. |
| 6 Experiments | (a) 6 experiment tests pass; (b) Printed markdown table: 50-battle win% baseline-vs-A* must show A* > baseline (seeded reproducibly); (c) Final total tests report + SHA-256 side-by-side final audit. |

---

## 10. Risks and Handling

| Risk | Handling |
|---|---|
| Streamlit 250 ms refresh loop races with session state | Use `st.rerun()` explicitly + `time.sleep(0.25)` for pacing; if too slow → fall back to manual "Step 1" button-only for MVP. |
| A* path costs across multiple enemies — unclear target selection | Greedy tie-break: target enemy by nearest then (row, col). Same rule as Phase 2 baseline. Not a correctness bug. |
| Minimax combat tree explosion | Hard cap depth 2 plies (MAX then MIN). No branching beyond Attack/Defend/Retreat actions (≤3 branches). Fine for demo. |
| GA fitness evaluation cost per generation is 30×50 battles = 1500 per epoch | Lower pop to 30, 30 generations — 150k battle rounds total. Still sub-second. |
| Lecturer expects 14 algorithms shown but dashboard selectbox only shows 7 (search) + 4 (adversarial) — missing GD/SA/GA/LinReg | Dashboard adds "AI Lab" tab (separate). Plots of GD loss curve, SA temperature schedule, GA fitness history, LinReg parity chart — all matplotlib plots rendered with `st.pyplot()`. |
| Backprop / Fuzzy / CSP requested later despite absence from notebook | Add only on explicit user request. Not in initial scope. Documented clearly in plan so user decides. |
| Parser tests regress after 3-6 code changes | Run `python -m unittest discover -s tools -v` at END of every phase. Fail-fast if any of 56 turn red. Immediately revert. |

---

## Summary Ask from User
**Please approve this plan** — after approval I'll execute **Phase 3 first** (Streamlit dashboard, Steps 3.1→3.4 above), run validation, then Phase 4 → 5 → 6 incrementally with test reports after each.
