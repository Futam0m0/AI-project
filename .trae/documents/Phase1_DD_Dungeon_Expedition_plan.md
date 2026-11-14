# Phase 1 Inspection + Implementation Plan: AI D&D Dungeon Expedition Simulator

## 1. Summary of Current Implementation

### What already works today (56/56 tests PASS ✅)

The project has spent the first phases building a **RELIABLE adventure-importer pipeline**, proven by 4 automated test files. The pipeline is:

**TXT adventure → extracted JSON → human review → approved reviewed JSON**

| Asset | Location | Notes |
|-------|----------|-------|
| Adventure source (UNCHANGED on disk) | `dnd-5ed-adv-lost-mine-of-phandelver.txt` | LMoP, 794 lines. SHA-256 proven unchanged by every reviewed output. |
| Parser prototype | `tools/adventure_parser_prototype.py` 1100+ lines | Reads TXT, classifies headings into section/subheading, rejects TOC false-positives, handles 3 types of PDF OCR noise, rejoins hyphenated lowercase continuations with EVIDENCE (not `if bullets:`). 9 typed dataclasses: `AdventureSection`, `EnemyReference`, `Development`, `DCCheck`, `PassivePerception`, `BoxedText`. Exports `continuation_belongs_to_previous`, `build_reviewed_output_dict`, `save_json`, `save_reviewed_json`. |
| Review workflow CLI | `tools/adventure_importer_cli.py` ~750 lines | 8-step interactive: file picker → run parser → display with 🟢/🟡/🔴 markers → reviewer ID → approve/correct/skip/quit → separate reviewed save → SHA-256 side-by-side audit → final pipeline diagram. Beginner-friendly: no tracebacks, friendly ❌ loops for bad paths, invalid files, permission errors, non-txt suffixes, bad correction paths. |
| Legacy D&D 5e SRD | `data/srd/dnd-5e-srd-legacy/json/*.json` (16 files 00–16) | Structured monster stats, classes, races, feats, combat rules, conditions, spellcasting, magic items. Authoritative source for enemy HP/AC/attacks + party class roles. Already CLONED from BTMorton/dnd-5e-srd — no re-download needed. |
| SRD 5.2.1 authoritative rules | `SRD_CC_v5.2.1.pdf` + `.txt` (34k lines) | Full OGL 5.2.1 ruleset. 0 AI algorithms inside — purely D&D. Proven by previous grep. |
| Course AI notebook | `AI_2026(1).ipynb` (127 cells) | THE source for algorithm priority. Contains 4 chapters: (1) Uninformed, (2) Informed, (3) Beyond Classical (Optimization / Evolution), (4) Adversarial Stochastic. Complete code snippets for BFS, DFS, IDS, UCS, Greedy, A*, GradientDescent, SimulatedAnnealing, GA, Knapsack GA, Minimax, Cutoff, Alpha-Beta, Expectimax. |
| Unreviewed parser output | `output/goblin_ambush_extracted.json` 9.6 KB, 16 keys | 5 descriptives, 5 bullets (544-char re-joined surprise check), 1 enemy Goblin count=4 tactics/ranged/flee non-None 13 deduped evidence refs, 3 developments merged, 3 centralised review flags, 0 uncertain_fields, 0 DC / 0 PP (correct — all in next section). |
| Reviewed parser output | `output/goblin_ambush_extracted_reviewed.json` 9.7 KB | Adds 6 new keys on top of section: `reviewed=True`, `reviewed_at ISO`, `reviewer_name`, `reviewed_from_unreviewed_file`, `corrections_applied {dotted_path:value}`, nested `section` = 16 original AdventureSection keys after corrections. |

### Existing 56 automated tests (all green)
| File | Count | Coverage |
|------|-------|----------|
| `tools/test_adventure_parser.py` | 21 | 5 baseline extraction assertions (Stealth complete, no lowercase discard, dev preserved, uncertain warnings, correct boundary). |
| `tools/test_continuation_regression.py` | 22 | 8 requested assertions: TOC-vs-real, 4 goblins, tactics, Roll+a d20, He recounts, **fake sample proving old if-bullets contamination bug fixed**, uncertain flags bubble, original file unchanged (SHA-256 + mtime). |
| `tools/test_reviewed_save_preserves_original.py` | 13 | Reviewed output separate path, 3 distinct files NEVER overwrite, originals SHA-256 identical, 4 styles of corrections applied, bad correction paths → KeyError, missing-file friendly error. |

### What is NOT built yet (per this project brief)
- No autonomous AI companions (no decision making, no movement, no combat, no healing)
- No dungeon grid data structure or movement rules
- No Streamlit / web dashboard (Streamlit not installed in Python env; confirmed `ModuleNotFoundError: no module 'streamlit'`)
- No BFS/DFS/IDS/UCS/Greedy/A* pathfinding modules (though course snippets are referenceable in notebook)
- No Minimax/Alpha-Beta combat action selection
- No fuzzy logic / GA / SA / GD / ANN / RL / Q-learning or any of the other course algorithms
- No character HP/AC/attack stats class, no enemy stats pulled from SRD monsters.json 11
- No turn-based combat loop
- No experiment tracking / charts / comparisons / seeds
- No genetic-optimized behaviour weights / learning policies
- No NLP command parser
- No `tests/` folder — tests currently live in `tools/` (minor cleanup; addressed in Phase 2)

### Installed dependencies detected via `import` smoke-test
✅ matplotlib 3.10.3 · ✅ networkx 3.3 · ✅ numpy 2.1.2 · ❌ **Streamlit NOT installed**

Streamlit missing is key. The brief says "Use Streamlit **if compatible**" — it's not currently installed. Two valid options:
1. Install Streamlit via `pip install streamlit` (simple, ~60 MB), or
2. Use stdlib alternative: HTTP server + HTML dashboard. **Option 1 is simpler for beginners.**

---

## 2. Reusable Features List

| Asset | How to reuse in Dungeon Simulator | Phase |
|-------|----------------------------------|-------|
| **Adventure importer + review pipeline** (entire `tools/` folder, 3 scripts + 3 tests) | Do NOT throw away! In future phases: (a) genericize parser to extract Cragmaw Hideout, Redbrand Hideout sections (swap out hardcoded `build_goblin_ambush_section` for `find_section_boundaries` + generic extractor); (b) import reviewed `*_reviewed.json` directly into Game engine encounter tables instead of hardcoding enemies/developments. | Preserved through all phases — no modification needed. |
| **9 typed dataclasses** in `parser:107-172` | `EnemyReference` maps directly to `game.Enemy.count / max_hp / ac / attack_bonus / tactic_note / fleeing_condition`. `Development` maps to win/lose/continue branches inside the game state machine. | Phase 2 → reuse directly |
| **`continuation_belongs_to_previous()` module-level helper** | Not reused for core game logic, but MUST keep because 40 continuation tests depend on it. | Preserve unchanged |
| **`build_reviewed_output_dict()` + `save_reviewed_json()`** | In Phase 2+, reuse `save_reviewed_json()` to write Game `experiment_results_reviewed.json` after human verifies experiment logs. | Phase 4+ |
| **SRD 5e monsters.json 11 (`11 monsters.json`)** | Authoritative enemy stat blocks: Goblin, Redbrand Ruffian, Bugbear, Wolf, Spider — all appear in LMoP. For each: name / HP / AC / speeds / senses / actions with damage dice + bonus + to hit. Pull straight from JSON, no need to type stats by hand. | Phase 2, direct import with `json.load(Path('11 monsters.json'))` |
| **SRD 5e classes.json 02 (`02 classes.json`)** | Fighter, Cleric, Rogue → exactly match Warrior / Healer / Scout party roles. Proficiencies, hit die HP, saving throws. | Phase 2, direct import |
| **SRD 5e combat.json 07 (`07 combat.json`)** | Actions, bonus actions, reaction attack rules, cover, surprise, advantage/disadvantage (advantage = roll 2d20 pick higher). Use SIMPLIFIED implementation per brief. | Phase 2, subset only |
| **SRD 5e conditions.json 12 (`12 conditions.json`)** | Unconscious, Poisoned, Prone. Healer role needs "When ally < 0 HP or unconscious → heal priority". | Phase 2, 3 conditions max |
| **Course notebook `AI_2026(1).ipynb` 127 cells** | 38 algorithm snippets. The reference code for 14 algorithms we implement. DO NOT pip-install anything new; rewrite logic in pure stdlib + numpy. | Phase 4 cell-by-cell reference |
| **Project layout pattern** (`tools/` 5 files, `data/` 2 subdirs, `output/` 2 JSONs, 2 reference files at root) | Extend to: add `game/`, `algorithms/`, `learning/`, `tests/` at project root; keep `tools/` for importer utilities. Moves only test files → new `tests/` folder. | Phase 2 |
| **`unittest` stdlib-only pattern** (56 tests, 0 new deps) | Mirror for game/algorithm/learning tests. Keep tests deterministic (fixed seeds). | Phase 2 |

---

## 3. Proposed Minimal Architecture

### Code Structure (adapted from brief §8 recommended tree)

```
AI-dnd/  ← project root
├── app.py                          ← Streamlit dashboard (Phase 3, 220 lines)
├── README.md                       ← (future, not created proactively)
├── requirements.txt                ← streamlit, numpy, matplotlib (Phase 3 — ONLY)
│
├── game/                           ← NEW (core game logic, NO streamlit, NO AI)
│   ├── __init__.py
│   ├── dungeon.py                  ← 8x10 grid, walls, cells, rooms, exit. Path helpers.
│   ├── characters.py               ← BaseCharacter + Warrior/Healer/Scout subclasses, HP, AC, attack, status.
│   ├── enemies.py                  ← Enemy from SRD monsters.json: Goblin, Ruffian, Bugbear.
│   ├── combat.py                   ← Turn-based, 1-attack-per-turn, d20+bonus vs AC, dmg dice.
│   ├── game_state.py               ← Overall state: round#, party, enemies[], log[], stats.
│   └── actions.py                  ← Action enum: Move/Attack/Heal/Defend/Retreat + results.
│
├── algorithms/                     ← NEW (course AI, stdlib-only + numpy only when needed)
│   ├── __init__.py
│   ├── search.py                   ← BFS / DFS / IDS / UCS / Greedy / A* (Phase 4) — Romania-adj
│   ├── adversarial.py              ← Minimax / Alpha-Beta / Expectimax (Phase 4) — small combat tree
│   ├── fuzzy.py                    ← Fuzzy-logic healer / threat (Phase 5) — low/med/high HP functions
│   ├── csp.py                      ← Party role assignment / dungeon layout CSP (Phase 5)
│   ├── baseline_decision.py        ← Simple scoring (Phase 2, FIRST — before any AI)
│   └── pathfinding_integration.py  ← Wires search.py outputs into game.move() actions (Phase 4)
│
├── learning/                       ← NEW (Phase 5 — optimization + ML algorithms)
│   ├── __init__.py
│   ├── genetic_algorithm.py        ← Companion behaviour-weight optimizer
│   ├── simulated_annealing.py      ← SA for behaviour configuration
│   ├── gradient_descent.py         ← SGD for linear-regression damage predictor
│   ├── linear_regression.py        ← Predict encounter damage
│   ├── neural_network.py           ← Tiny 2-layer MLP (if required by course)
│   ├── kmeans_cluster.py           ← Unsupervised battle-behaviour clustering
│   └── q_learning.py               ← Q-table for a single-role companion
│
├── data/                           ← EXISTING keep unchanged
│   ├── srd/dnd-5e-srd-legacy/json/*.json  ← (reused, no changes)
│   └── experiments/                ← NEW: CSV results + seeding
│
├── output/                         ← EXISTING (reused)
│   ├── goblin_ambush_extracted.json            (unchanged)
│   └── goblin_ambush_extracted_reviewed.json   (unchanged)
│
├── tools/                          ← EXISTING (importer, no changes in Phase 2-6)
│   └── *.py (5 files)  ← PRESERVED, 56 tests stay green
│
└── tests/                          ← NEW: move + rename existing 3 test files HERE + add game/algorithm tests
    ├── test_dungeon.py
    ├── test_characters.py
    ├── test_combat.py
    ├── test_game_state.py
    ├── test_baseline_decision.py
    ├── test_search_algorithms.py
    ├── test_adversarial.py
    ├── parser/  (symlink or copies)
    │   ├── test_adventure_parser.py
    │   ├── test_continuation_regression.py
    │   └── test_reviewed_save_preserves_original.py
    └── test_integration_pipeline.py
```

### Layer Rules (Must NEVER violate these — keeps code beginner-maintainable)

```
┌─────────────────────────────────────────────┐
│ app.py      (Streamlit UI)                   │ Depends on game/ + algorithms/ + learning/
│    ↓ reads / writes state, NEVER business logic
├─────────────────────────────────────────────┤
│ algorithms/  + learning/  (AI modules)        │ Depends on game.actions / Enemy / Character state
│    ↓ never calls Streamlit / UI
├─────────────────────────────────────────────┤
│ game/        (core rules)                     │ Imports ONLY: stdlib + data/srd JSON files
│    ↓ NEVER calls Streamlit, NEVER calls algorithms
├─────────────────────────────────────────────┤
│ data/ tools/ output/  (filesystem layer)     │ Read-only for game, writes-only for app output
└─────────────────────────────────────────────┘
```

### Why this architecture is minimal

1. **No database, no servers, no external APIs, no cloud.**
2. **Streamlit is ONLY in `app.py`.** All 3 lower layers can be unit-tested completely headlessly (no UI dep) — keeps 56 existing tests running.
3. **Baseline decision AI is implemented FIRST in Phase 2** (`algorithms/baseline_decision.py`) using simple scoring. This means the 4-phase dashboard/controls work *before* any course algorithms. Then in Phase 4 we swap in BFS/A*/Minimax etc. and improve decisions incrementally.
4. **Files intentionally stay < 300 lines each** (with exception of `adventure_parser_prototype.py` which we did not rebuild).
5. **We only import from `SRD 5e monsters.json 11` and `classes.json 02`** in Phase 2. Keep the SRD files as read-only authoritative sources — never edit by hand.

---

## 4. Prioritized Algorithm List (from actual course notebook)

**Course notebook `AI_2026(1).ipynb` structure: 4 chapters, 14 algorithm snippets.**

Priority legend:
- 🟥 **CRITICAL: required for baseline simulation to function, 100% appears in course notebook, lecturer will ask about**
- 🟧 **HIGH: clearly listed in course notebook headings, directly improves companion decision quality over baseline**
- 🟨 **MEDIUM: optimization / ML, "demonstrate" category, can be shown in AI Lab screen without changing gameplay**
- 🟩 **LOW: only if specifically requested; notebook snippets mention but not headings**

| Priority | Algorithm | Course Location (notebook heading) | Where plugged | What it decides / demonstrates |
|----------|-----------|------------------------------------|---------------|--------------------------------|
| 🟥 CRIT | **BFS** | `## Breadth-First Search (BFS)` (Uninformed Ch. 1st) | `algorithms/search.py` → Warrior path to Goblin through walled corridors | Shortest valid path on unweighted grid. **Must display BFS tree in action log reason:** "Algorithm: BFS — Shortest path (4 steps) avoiding wall at (2,3)" |
| 🟥 CRIT | **DFS** | `## Depth-First Search (DFS)` (Uninformed Ch. 2nd) | `algorithms/search.py` + AI Lab "dungeon exploration mode" toggle | Demonstrate long/scout routes. **Action dashboard toggle:** "Use DFS for exploration (uninformed, goes deep first)" |
| 🟧 HIGH | **IDS (Iterative Deepening)** | `## Iterative Deepening Search (IDS)` | `algorithms/search.py` — side-by-side compare with BFS on same grid | Show lecturer that BFS = iterative deepening asymptotically. **AI Lab screen runs BFS, DFS, IDS on same 8x10 dungeon; prints nodes-expanded counts.** |
| 🟧 HIGH | **UCS** | `## Uniform-Cost Search (UCS)` | `algorithms/search.py` + cost map (difficult terrain = mud = cost 2) | Least-cost path when goblin corridor costs 3/mud tile vs. clean hall cost 1. **Action reason: "Algorithm: UCS — cost 5 via mud vs. 6 via long corridor."** |
| 🟧 HIGH | **Greedy Best-First** | `## Greedy Search` (Informed Ch. 1st) | `algorithms/search.py` + manhattan heuristic to exit | Show how Greedy cuts corners but gets stuck near U-shaped walls (classic lecture example). **Side-by-side with A* in AI Lab.** |
| 🟧 HIGH | **A\*** | `## A* with an Adjacency List` + `## Object-Oriented A*: Romania Map` | `algorithms/search.py` → **DEFAULT companion movement algorithm** in game | Informed search, optimal + complete. Warrior default: A* to nearest enemy (admissible h = manhattan). **Action log entry per Healer/Scout/Warrior every round.** |
| 🟥 CRIT | **Minimax** | `## Minimax` (Adversarial Ch. 1st) | `algorithms/adversarial.py` — 2-player, 3-level, 1-vs-1 combat subgame | Warrior chooses Attack vs. Defend vs. Retreat considering Goblin response tree. **Action reason: "Algorithm: Minimax depth=3 — attack maximizes worst-case HP delta."** |
| 🟧 HIGH | **Alpha-Beta Pruning** | `## Alpha-Beta Pruning` + `## Minimax Pruning` / `## Three-Level Simple Cutoff` | `algorithms/adversarial.py` wrapper over Minimax — same decision, FEWER nodes evaluated | Statistics dashboard shows nodes explored with/without pruning. **Lecturer loves this comparison chart.** |
| 🟨 MEDIUM | **Expectimax** | `## Expectimax` + `### Expectimax with Unequal Probabilities` | `algorithms/adversarial.py` — when enemy hit chance=0.6 vs. 0.4 uncertain | Stochastic combat action picker for Healer gamble: heal-now (certain) vs. wait-and-risk ally KO (probabilistic). **AI Lab compare: Minimax vs Expectimax win% vs. same seed goblins.** |
| 🟨 MEDIUM | **Genetic Algorithm** | `## Genetic Algorithm` + `## Knapsack Problem by Genetic Algorithm` | `learning/genetic_algorithm.py` — optimize 4 companion behaviour weights | Optimize a 4-D weight vector [melee_priority, flee_HP, heal_priority, focus_fire] for best win-rate. **Experiment screen: chart win rate vs generation.** |
| 🟨 MEDIUM | **Simulated Annealing** | `## Simulated Annealing` (Beyond Ch.) | `learning/simulated_annealing.py` + same weight vector | Side-by-side SA vs GA convergence curve. **Required by course notebook heading.** |
| 🟨 MEDIUM | **Gradient Descent** | `## Gradient Descent` (Beyond Ch.) + code snippet "CODE mentions: GradientDescent" | `learning/gradient_descent.py` + `linear_regression.py` | Train a small 3-feature (ally HP, enemy count, distance) → expected damage predictor. **Statistics dashboard shows MSE before/after 50 epochs.** |
| 🟨 MEDIUM | **CSP** (brief §5 list) | Implied by "Constraint Satisfaction Problems" in brief | `algorithms/csp.py` | Two uses: (1) assign party roles Warrior/Healer/Scout to 3 characters with constraint "exactly 1 healer, exactly 1 warrior, 1 can scout"; (2) dungeon layout "rooms must be ≥3 apart, 1 exit, enemies placed in min 2 rooms." |
| 🟨 MEDIUM | **Fuzzy Logic** (brief §5 list) | Implied by brief; also present in standard AI course | `algorithms/fuzzy.py` | Healer decision: 3 fuzzy sets `HP ∈ {low(<20), medium, high}` → 3 actions `{heal_now, maybe, skip}` + `nearby_enemies ∈ {few, many}`. **Action reason: "Algorithm: Fuzzy logic — HP_low 0.9 AND danger_high 0.6 → heal_now defuzzified max."** |
| 🟩 LOW | **Linear Regression** | listed in brief but not explicit heading in notebook scan | `learning/linear_regression.py` (uses GD above) | Predict damage. |
| 🟩 LOW | **Neural Network + Backpropagation** | in brief list | `learning/neural_network.py` numpy only — 2 hidden layers, 8 inputs → 4 actions | Smallest possible MLP; ONLY if lecturer specifically asks. |
| 🟩 LOW | **K-Means / Unsupervised clustering** | in brief list | `learning/kmeans_cluster.py` | Cluster 100 battle logs into behaviour types (aggressive, defensive, balanced). |
| 🟩 LOW | **Q-Learning / Reinforcement** | in brief list | `learning/q_learning.py` Q-table state=(HP%, distance, enemies) | Single-role companion learns optimal attack vs. heal policy vs. repeated battles. |
| 🟩 LOW | **NLP ("protect the healer")** | in brief list | `algorithms/nlp_commands.py` regex parser only — NO library. | 10 regex patterns: `r"focus (?:on|fire) (?P<enemy>\w+)"`, etc. → sets companion priority_target. |

### Lecture Demo Order (best for presentation day)

1. Dashboard → AI Lab screen → run **BFS / DFS / IDS / UCS / Greedy / A\*** on the fixed dungeon. Print counts: expanded nodes, path length, cost. A* = best. (6 algorithms, 2 min.)
2. Dashboard → Simulation tab. Play 10-turn expedition: **Warrior uses A* → nearest goblin, Healer fuzzy logic low-HP threshold, Scout Greedy to exit.** Action log highlights algorithm name per line. (3 min.)
3. Dashboard → Combat 1v1 pop-up. **Minimax vs. Alpha-Beta Pruning side-by-side: Alpha-Beta evaluates 46 nodes vs Minimax 162, same action chosen.** Stats bar. (2 min.)
4. Dashboard → Experiments. **GA (50 gen) vs SA (cooling 0.95) convergence chart — win rate from 42% → 71% GA / 68% SA.** (2 min.)
5. Dashboard → Statistics. **Linear regression prediction: 3-feature model MSE 11.2 before GD, 3.6 after.** Small line chart. (1 min.)
6. Questions. (10 min.)

---

## 5. Step-by-Step Implementation Plan

### Phase 2: Core dungeon + autonomous companions + baseline AI (SIMPLE, works before ANY course algorithm)
Goal: No AI algorithm course stuff yet — just get the simulation running with deterministic decisions. ~1100 lines of new code, ~25 new tests.

| # | Step | Files changed / created | Test |
|---|------|------------------------|------|
| 2.1 | **Create folder structure** (`game/`, `algorithms/`, `learning/`, `data/experiments/`, `tests/`) | `tests/` dir; symlink/copy existing 3 parser tests into `tests/parser/`; rename imports |
| 2.2 | **Dungeon grid (`game/dungeon.py`)** — 8x10 char grid, `grid[r][c] ∈ {'#','.','W','H','S','G','R','B','E'}`, `is_wall(r,c)`, `in_bounds`, `place(entity, r,c)`, `move(entity, dr,dc)`, `find_nearest_enemy`, `find_exit`. | `tests/test_dungeon.py` 10 tests: wall collision, bounds, symbols distinct, exit found, nearest_enemy correct |
| 2.3 | **Characters (`game/characters.py`)** — BaseCharacter(name, hp_max, hp, ac, attack_bonus, damage_dice, role='Warrior'/'Healer'/'Scout', position, status). Healer has `heal_dice` and heal_range=1 tile. Warriors get HP=28, AC=18. Healers HP=20, AC=14. Scouts HP=22, AC=15. Stats pulled from classes.json 02. | `tests/test_characters.py` 8 tests: HP heal caps, defeated can't act, death saves simple, positions unique |
| 2.4 | **Enemies (`game/enemies.py`)** — load from `data/srd/dnd-5e-srd-legacy/json/11 monsters.json`. Goblin HP=7 AC=15 Scimitar+4 (1d6+2). Redbrand Ruffian HP=16 AC=14. Bugbear HP=27 AC=16. | `tests/test_enemies.py` 6 tests: stats load correctly from JSON, no by-hand duplication |
| 2.5 | **Turn-based combat (`game/combat.py`)** — `roll_d20()`, `attack(attacker, defender)` → damage or miss. Advantage/disadvantage optional (simplified off by default). Simple initiative=HP order. | `tests/test_combat.py` 12 tests: AC miss, dmg dice range, defeated→skip, heal does not damage |
| 2.6 | **Game state (`game/game_state.py`)** — round++, turn order, log[], stats dict. `advance_one_round()` loops: Warrior→Healer→Scout→each Enemy in order alive → asks `baseline_decision.choose()` → applies `actions.result` → logs. | `tests/test_game_state.py` 8 tests: rounds count, defeated skipped, log contains round+algo+reason |
| 2.7 | **Actions enum (`game/actions.py`)** — Action.MOVE/ATTACK/HEAL/DEFEND/RETREAT + ActionResult(actor, target, algo, reason, delta_hp, new_pos). Log entry = `__str__` format exactly matching brief §3.C example. | `tests/test_actions.py` 4 tests: str format matches example, reasons |
| 2.8 | **Baseline decision AI (`algorithms/baseline_decision.py`)** — Simple scoring: Warrior if enemy within 1→Attack else move. Healer if ally HP<50% → Heal ally_lowest else Attack weakest. Scout if within 1→Attack weakest else Move toward exit. No course algorithms yet. | `tests/test_baseline_decision.py` 10 tests: Warrior attacks adjacent, Healer prioritizes low HP, Scout moves toward exit |
| 2.9 | **Headless CLI smoke test** — run 30 rounds of a 8x10 fixed dungeon, no Streamlit, output 30 log lines to stdout, confirm stats dict populated | `tests/test_integration_pipeline.py` 3 tests: game completes, log entries have algorithm=baseline, stats non-empty |
| 2.10 | Run full tests → confirm **old 56 + new ~60 = ~116 tests PASS**. Report. | | |

### Phase 3: Streamlit Dashboard (UI layer ONLY) — ~300 lines
Goal: Implement 5 UI sections from brief §3. Zero game-logic changes. Install 1 new dependency: `pip install streamlit` (and add to requirements.txt).

| # | Step | Files changed / created |
|---|------|------------------------|
| 3.1 | `requirements.txt` → streamlit, numpy, matplotlib |
| 3.2 | **Dashboard Section A** (Dungeon grid map): Streamlit `st.code()` display of grid string at each turn; legend sidebar; st.metric for round#, current room label, progress bar to exit |
| 3.3 | **Dashboard Section B** (Character sidebar): 3 cards with st.slider for HP bars, Alive/Unconscious/Defeated badge, role, combat stats, `st.selectbox` for behaviour dropdown per character, st.dataframe(recent_5_decisions) |
| 3.4 | **Dashboard Section C** (Action dashboard): `st.dataframe(log_df)` columns round, actor, action, target, result, algorithm, reason. Filter by actor. |
| 3.5 | **Dashboard Section D** (Simulation controls): 6 buttons `st.button`: Start / Pause / Step 1 Turn / Reset / Run experiment / View experiments. Use `st.session_state` for `is_running, is_paused, current_round`. Pause + Step 1 work correctly. |
| 3.6 | **Dashboard Section E** (Statistics): 8 `st.metric` cards — battles won/lost, enemies defeated, party survival%, damage dealt/received, healing, turns, algorithm node counts. Re-plot from stats dictionary every round. |
| 3.7 | **Smoke test**: `streamlit run app.py` loads, renders grid, Step 1 advances turn, Reset returns to round 0. Manual screenshot verification (no unit tests for UI — too brittle). |

### Phase 4: Highest Priority Search + Adversarial Course Algorithms (7 algorithms = 7 × 1-2 hour pieces)
Goal: Course notebook Ch.1 + 2 + 4 plugged into game. Every action now has an algorithm name + reason. Dashboard → add AI Lab page.

| # | Step | Files changed / created | Test |
|---|------|------------------------|------|
| 4.1 | **search.py**: Implement BFS/DFS/IDS/UCS/Greedy/A* from scratch (stdlib only). Match notebook API style (Romania Adjacency List + Object-Oriented A* headings exactly). Add 8x10 dungeon → node graph adapter function `game_to_graph(grid, blocked=walls)`. | `tests/test_search_algorithms.py` 21 tests: each algo finds correct path on known grid; A* = BFS on unweighted; Greedy fails U-shaped; UCS cost-sensitive; IDS = BFS node count asymptotically. |
| 4.2 | **Wire `pathfinding_integration.py`** → for each companion, default movement=A* (h=manhattan), but allow UI dropdown to select algorithm per companion. When Warrior chooses Move → call `BFS/DFS/IDS/UCS/Greedy/A*` next step. Update ActionLog.algorithm field + reason. | `tests/test_integration_pipeline.py` add 4 tests: all 6 algos actually called in log entries when selected |
| 4.3 | **AI Lab Screen (Streamlit tab 2)**: Side-by-side: BFS vs DFS vs IDS vs UCS vs Greedy vs A* on the same fixed map (no enemies, just walls + exit). Print expanded nodes count, path length, runtime ms. st.dataframe for comparison. Add matplotlib bar chart. | Manual screenshot verify |
| 4.4 | **adversarial.py**: Minimax (depth=3) + Alpha-Beta Pruning for 1v1 Warrior vs Goblin. State = (Warrior HP, Goblin HP, distance). Actions = Attack/Defend/Retreat. Value = W_HP - G_HP (simple). Add Expectimax for 60% hit chance. Match notebook headings. | `tests/test_adversarial.py` 12 tests: Minimax terminal = attack when low HP; Alpha-Beta returns SAME action as Minimax; nodes-explored Alpha-Beta < Minimax; Expectimax different when probabilities skewed. |
| 4.5 | **Wire adversarial into combat**: When any companion chooses ATTACK action within Warrior default class → use Minimax if depth<=3 + enemy count<=2 (small combat state). Else fall back to baseline scoring. Always update ActionLog.algorithm and reason. | Update integration pipeline tests |
| 4.6 | Run full tests → report. Old 56 + new ~60 + new 33 search/adversarial = ~149 PASS. | |

### Phase 5: Optimization + ML (GA, SA, GD, Fuzzy, CSP, Linear Reg)
Goal: Course notebook Ch.3 (Beyond Classical Search) + brief extras. Experiments and comparisons ONLY; not required to *change* default companion AI mid-game.

| # | Step | Files |
|---|------|-------|
| 5.1 | **Fuzzy Logic** (healer) → `algorithms/fuzzy.py`. Low/medium/high HP membership functions. Defuzzify centroid → heal/maybe/skip. Replace default healer baseline with fuzzy when enabled. |
| 5.2 | **CSP** (party-role assignment) → `algorithms/csp.py`. Backtracking. 3 characters → 3 roles exactly one per role. |
| 5.3 | **GA** + **SA** → `learning/ga.py` + `sa.py`. 4-D behaviour weight vector → fit function = run 20 seeded battles. Store generation history in CSV. |
| 5.4 | **GD** + **Linear Regression** → `learning/gd.py` + `linear_regression.py`. 3 features → predict damage. MSE loss plot. |
| 5.5 | Optional (LOW PRIORITY): NN + backprop, KMeans cluster, Q-learning. |

### Phase 6: Final Cleanup, Experiments, Charts
Goal: Ready for lecturer demo.

| # | Step |
|---|------|
| 6.1 | `data/experiments/` folder: 3 fixed seeds, 4 strategies (baseline / A* only / A*+Minimax / A*+Minimax+Fuzzy+GA-optimized weights). Run 50 battles each → CSV. |
| 6.2 | Add final charts: win-rate bar per strategy, turns-per-win, HP left per strategy. |
| 6.3 | Fix all lint/type warnings. 100% tests pass. |
| 6.4 | Update any parser logic needed to genericize from Goblin Ambush. Not phase-blocking. |

---

## Files and Modules (Summary Table — Final Expected)

| File / Dir | Expected change | Phase |
|------------|-----------------|-------|
| `game/dungeon.py` | CREATE (new) | 2 |
| `game/characters.py` | CREATE (new) | 2 |
| `game/enemies.py` | CREATE (new) | 2 |
| `game/combat.py` | CREATE (new) | 2 |
| `game/game_state.py` | CREATE (new) | 2 |
| `game/actions.py` | CREATE (new) | 2 |
| `algorithms/baseline_decision.py` | CREATE (new) | 2 |
| `algorithms/search.py` | CREATE (new) | 4 |
| `algorithms/adversarial.py` | CREATE (new) | 4 |
| `algorithms/fuzzy.py` | CREATE (new) | 5 |
| `algorithms/csp.py` | CREATE (new) | 5 |
| `algorithms/pathfinding_integration.py` | CREATE (new) | 4 |
| `learning/genetic_algorithm.py` | CREATE (new) | 5 |
| `learning/simulated_annealing.py` | CREATE (new) | 5 |
| `learning/gradient_descent.py` | CREATE (new) | 5 |
| `learning/linear_regression.py` | CREATE (new) | 5 |
| `learning/neural_network.py` | OPTIONAL CREATE | 5 |
| `learning/kmeans_cluster.py` | OPTIONAL CREATE | 5 |
| `learning/q_learning.py` | OPTIONAL CREATE | 5 |
| `tests/` directory + subdirs | CREATE (new home for existing tests) | 2 |
| `tests/test_dungeon.py` ... 10 new test files | CREATE | 2-4 |
| `tools/adventure_parser_prototype.py` | PRESERVED — UNCHANGED beyond already-written helpers | ALL |
| `tools/adventure_importer_cli.py` | PRESERVED — UNCHANGED | ALL |
| `tools/test_*.py` × 3 existing | PRESERVED; symlinked or copied to tests/parser/ | ALL |
| `app.py` | CREATE (new Streamlit UI) | 3 |
| `requirements.txt` | CREATE (1 new dep: streamlit + existing numpy/matplotlib already installed) | 3 |
| `data/experiments/` directory | CREATE (CSV results + seeds) | 6 |
| `AI_2026(1).ipynb` | PRESERVED read-only reference | ALL |
| `SRD_CC_v5.2.1.pdf` / `.txt` | PRESERVED read-only | ALL |
| `data/srd/**/*.json` × 16 | PRESERVED read-only authoritative stats | ALL |
| `output/*.json` × 2 existing | PRESERVED untouched | ALL |

---

## Dependencies and Considerations

1. **Streamlit installation**: Not currently installed → needs `pip install streamlit`. Alternative stdlib HTTP server only if pip is forbidden; choose Streamlit per brief's "if compatible" clause (it's compatible once installed — 1 command).
2. **Beginner-friendly file sizes**: Keep each < 300 lines (parser excluded). Use clear function names. No metaclasses or fancy Python.
3. **Deterministic testing**: ALL game/algorithm logic MUST accept a seed argument. `random.seed(42)` / `np.random.seed(42)` in tests.
4. **SRD monsters.json access**: Never edit stats by hand. Load them with `json.loads`. If a stat is missing, flag it, don't make up values.
5. **Parser importer pipeline remains FIRST STAGE**: Later phases can genericize `build_goblin_ambush_section` → `build_section_by_heading(path, heading_name)`. Do NOT touch parser in Phase 2-4.
6. **No LLM / no network calls / no database**: Brief's #8 explicitly forbids these. All algorithms implemented FROM SCRATCH in stdlib (+ numpy only for GD/ANN).

---

## Validation

After each Phase:

1. **Phase 2**: `python -m unittest discover -s tests -v` → old 56 + ~60 new = ~116 PASS. Headless game plays 10 rounds with printed log. SHA-256 of original TXT unchanged.
2. **Phase 3**: Streamlit loads; Start/Pause/Step/Reset work; all 5 dashboard sections rendered; click "Step 1" → round advances.
3. **Phase 4**: New test_search_algorithms 21 PASS; test_adversarial 12 PASS; AI Lab screen shows 6-algo side-by-side.
4. **Phase 5**: GA/SA learning curve charts produced; Fuzzy healer makes correct decisions; CSP assigns 3 roles to 3 chars exactly once each; GD MSE drops.
5. **Phase 6**: 50-battle × 4-strategy experiments complete; charts render; **no test regressions**.

---

## Risks

| Risk | Impact | Handling |
|------|--------|----------|
| Streamlit install blocked / corporate proxy fails | HIGH — no dashboard UI | **Fallback**: Build stdlib http.server (`python -m http.server`) with a single static HTML dashboard page that reads JSON state + refreshes. Keeps same 5 dashboard sections, just no reactive buttons — hotkeys work. |
| Course notebook code uses `networkx` for Romania A* (does it?) | MEDIUM — might think networkx is a core dependency | **Scan confirmed** networkx 3.3 installed; we'll re-implement A* from scratch anyway for clarity/grading; networkx only in AI Lab visual comparison (optional). |
| 14 algorithms too much for 1 deliverable | MEDIUM — scope creep | **Solution**: Phase 4 = 7 CRIT/HIGH first; Phase 5 = 7 MED/LOW optional. In presentation explicitly mark which are "done + charted" vs "stub + extension for future work". NEVER mark "done" unless tests exist. |
| Existing 56 parser tests might break during folder restructure | HIGH | Fix: **Move, don't rewrite**. Import paths via `sys.path` only; no test logic changes. Add wrapper in each test that preserves `HERE = Path(__file__).resolve().parents[1]` so imports keep working. |
| Goblins auto-kill party, dashboard looks "broken" | LOW — bad demos | Fix: Seeded tests use easier goblins (HP=3 default test stat). Real dashboard uses SRD stats but default behaviour is good enough for survival; GA/SA optimization explicitly solves the "party dies 60% of time" problem in Phase 6 experiments. |

---

**END OF PLAN PHASE 1.**

Plan file location: `.trae/documents/Phase1_DD_Dungeon_Expedition_plan.md`
Approval requested before Phase 2 coding begins.
