"""
Streamlit dashboard for the AI D&D Dungeon Expedition Simulator.

Sections (per brief §3):
  A — Dungeon grid + room / round / progress metrics (top).
  B — Character sidebar (3 party cards with HP/AC/status/objective).
  C — Action dashboard (last 30 ActionResult entries rendered as scrollable table + plain text).
  D — Simulation controls: 6 buttons (Start / Pause / Step 1 Turn / Reset /
      Run Experiment / View Results). Last 2 are placeholders for Phase 6.
  E — Statistics dashboard: 8 metric cards.

Run with:  python -m streamlit run app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from algorithms.modular_selector import (
    COMBAT_LABELS,
    COMBAT_OPTIONS,
    DECISION_LABELS,
    DECISION_OPTIONS,
    NAVIGATION_LABELS,
    NAVIGATION_OPTIONS,
    build_decision_fn,
    combination_label,
)
from game.game_state import GameState
from tools.experiment_runner import (
    load_experiment_results,
    run_strategy_matrix,
    aggregate_rows,
    run_modular_matrix,
    aggregate_modular_rows,
    load_modular_results,
)
from tools.write_comparison_table import build_comparison_markdown

# ---------------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI D&D Dungeon Expedition Simulator",
    page_icon="🏰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session-state init — one GameState + control flags
# ---------------------------------------------------------------------------
DEFAULT_SEED = 42

if "seed" not in st.session_state:
    st.session_state.seed = DEFAULT_SEED
if "game" not in st.session_state:
    st.session_state.game = GameState.new_default(seed=st.session_state.seed)
if "running" not in st.session_state:
    st.session_state.running = False
if "paused" not in st.session_state:
    st.session_state.paused = False
# Three modular algorithm selectors (defaults reproduce the baseline experience).
if "decision_algo" not in st.session_state:
    st.session_state.decision_algo = "baseline"
if "navigation_algo" not in st.session_state:
    st.session_state.navigation_algo = "baseline-nav"
if "combat_algo" not in st.session_state:
    st.session_state.combat_algo = "baseline-combat"
if "pace_ms" not in st.session_state:
    st.session_state.pace_ms = 300
if "experiment_results" not in st.session_state:
    st.session_state.experiment_results = []
if "modular_results" not in st.session_state:
    st.session_state.modular_results = []
if "view_results" not in st.session_state:
    st.session_state.view_results = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _current_decision_fn():
    """Build the combined decision function from the three selectors."""
    return build_decision_fn(
        st.session_state.decision_algo,
        st.session_state.navigation_algo,
        st.session_state.combat_algo,
    )


def _reset_all(seed: int | None = None) -> None:
    """Reset the simulator state (but keep previous experiment logs)."""
    if seed is not None:
        st.session_state.seed = seed
    st.session_state.game = GameState.new_default(seed=st.session_state.seed)
    st.session_state.running = False
    st.session_state.paused = False


def _log_df(game: GameState, max_rows: int = 30):
    rows = []
    for r in game.action_log[-max_rows:]:
        rows.append(
            {
                "Round": r.round_number,
                "Actor": f"{r.actor.symbol} · {r.actor.name}",
                "Action": r.action.value,
                "Target": r.target.name if r.target else "—",
                "Algorithm": r.algorithm,
                "Reason": r.reason,
                "ΔHP target": r.delta_hp_target,
            }
        )
    df = pd.DataFrame.from_records(rows)
    if len(df) == 0:
        df = pd.DataFrame(
            columns=["Round", "Actor", "Action", "Target", "Algorithm", "Reason", "ΔHP target"]
        )
    return df


def _win_loss_banner(game: GameState) -> tuple[str, str]:
    if game.winner == "party":
        return ("🎉 Party wins!", "success")
    if game.winner == "enemies":
        return ("💀 Enemies win.", "error")
    if game.winner == "draw":
        return ("🤝 Draw — stalemate after max rounds.", "warning")
    return ("", "")


LEGEND = {
    "#": "Wall",
    ".": "Floor",
    "W": "Warrior",
    "H": "Healer",
    "S": "Scout",
    "G": "Goblin",
    "R": "Redbrand Ruffian",
    "B": "Bugbear",
    "E": "Exit",
}

# ===========================================================================
# Sidebar (Section B — character cards + settings)
# ===========================================================================
with st.sidebar:
    st.title("🏰 Expedition Control")
    st.caption("Phase 3 · Baseline AI dashboard — 5 sections A–E wired")

    st.markdown("## ⚙️ Settings")

    st.markdown("#### 🧠 Decision-making")
    decision_key = st.radio(
        "Role rules & behaviour weights",
        DECISION_OPTIONS,
        index=DECISION_OPTIONS.index(st.session_state.decision_algo),
        format_func=lambda k: DECISION_LABELS.get(k, k),
        help=(
            "Controls the party's role rules and behaviour weights. "
            "Used directly when navigation or combat is set to 'Baseline'."
        ),
        horizontal=False,
    )
    st.session_state.decision_algo = decision_key

    st.markdown("#### 🧭 Navigation")
    nav_key = st.radio(
        "Movement / pathfinding (out of combat)",
        NAVIGATION_OPTIONS,
        index=NAVIGATION_OPTIONS.index(st.session_state.navigation_algo),
        format_func=lambda k: NAVIGATION_LABELS.get(k, k),
        help=(
            "Controls movement when no enemy is within melee range "
            "(Chebyshev ≤ 2). 'Baseline navigation' defers to the "
            "decision-making selector."
        ),
        horizontal=False,
    )
    st.session_state.navigation_algo = nav_key

    st.markdown("#### ⚔️ Combat")
    combat_key = st.radio(
        "Tactical decisions (in combat)",
        COMBAT_OPTIONS,
        index=COMBAT_OPTIONS.index(st.session_state.combat_algo),
        format_func=lambda k: COMBAT_LABELS.get(k, k),
        help=(
            "Controls decisions when an enemy is within melee range "
            "(Chebyshev ≤ 2). 'Baseline combat' defers to the "
            "decision-making selector."
        ),
        horizontal=False,
    )
    st.session_state.combat_algo = combat_key

    st.caption(
        "ℹ️ Each layer operates at a different level. Navigation handles "
        "movement between fights; Combat handles melee decisions; "
        "Decision-making provides the base role rules."
    )
    seed_val = st.number_input(
        "Random seed (integer)",
        min_value=0,
        max_value=9_999_999,
        value=int(st.session_state.seed),
        step=1,
    )
    pace = st.slider(
        "Auto-step pace (ms per round)",
        min_value=50, max_value=3_000, value=int(st.session_state.pace_ms), step=25,
    )
    st.session_state.pace_ms = pace

    if st.button("🧭 New seed + Reset", use_container_width=True):
        import random
        _reset_all(random.randint(0, 9_999_999))
        st.rerun()

    st.markdown("---")
    st.markdown("## 🧝 Party — Section B")

    g = st.session_state.game
    ROLE_LABEL = {"W": "Warrior (Tank)", "H": "Healer (Support)", "S": "Scout (Damage)"}

    for char in g.party:
        with st.container(border=True):
            col1, col2 = st.columns([1, 5])
            col1.markdown(f"### `{char.symbol}`")
            col2.markdown(f"**{char.name}** — {ROLE_LABEL.get(char.symbol, char.symbol)}")
            pct = 0 if char.hp_max <= 0 else max(0, min(100, int(100 * char.hp / char.hp_max)))
            col2.progress(pct, text=f"HP: {char.hp} / {char.hp_max} · AC {char.ac}")
            status_colour = {"alive": "🟢", "unconscious": "🟠", "defeated": "🔴"}.get(
                char.status, "⚪"
            )
            col2.caption(f"{status_colour} Status: {char.status}")
            if char.position is not None:
                col2.caption(f"📍 Position: ({char.position.row}, {char.position.col})")
            st.caption(f"🎯 Objective: {char.objective}")

    st.markdown("---")
    st.markdown("### 📖 Legend")
    for sym, name in LEGEND.items():
        st.markdown(f"- **`{sym}`** → {name}")

# ===========================================================================
# Main dashboard
# ===========================================================================
st.title("⚔️ AI D&D Dungeon Expedition Simulator")
st.caption(
    "Autonomous party explores dungeon · modular AI selectors · "
    "see §3 brief for 5 sections A-E layout."
)

# Show the currently selected 3-layer combination clearly.
_cur_label = combination_label(
    st.session_state.decision_algo,
    st.session_state.navigation_algo,
    st.session_state.combat_algo,
)
st.info(f"**Active configuration** — {_cur_label}")

# ===========================================================================
# Section A — Dungeon grid header (round, room, progress) + map
# ===========================================================================
st.markdown("## A — Dungeon Map")
game = st.session_state.game

banner, banner_kind = _win_loss_banner(game)
if banner:
    if banner_kind == "success":
        st.success(banner)
    else:
        st.error(banner)

meta1, meta2, meta3, meta4 = st.columns(4)
meta1.metric("Round", f"{game.round_number}", delta=None)
room_label = "Goblin Ambush — Hall 1"
meta2.metric("Room", room_label)
exit_pos = game.dungeon.find_exit()
if exit_pos is not None:
    # Distance to exit — smallest Manhattan from any alive party member.
    best = min(
        (p.position.manhattan(exit_pos) for p in game.party_alive() if p.position is not None),
        default="—",
    )
    meta3.metric("Distance to Exit", f"{best}")
else:
    meta3.metric("Distance to Exit", "—")

alive = len(game.party_alive()); total = max(1, len(game.party))
progress_pct = int(100 * alive / total)
meta4.metric("Party intact", f"{alive}/{total}", delta=f"{progress_pct}% alive")

st.code(game.dungeon.to_string(), language=None)

# ===========================================================================
# Section D — Simulation controls (after the map so user can press Step easily)
# ===========================================================================
st.markdown("## D — Simulation Controls")
b_start, b_pause, b_step, b_reset, b_exp, b_mod, b_view = st.columns(7)
with b_start:
    if st.button("▶️ Start", use_container_width=True, disabled=bool(game.winner)):
        st.session_state.running = True
        st.session_state.paused = False
with b_pause:
    if st.button("⏸️ Pause", use_container_width=True):
        st.session_state.paused = True
        st.session_state.running = False
with b_step:
    step_pressed = st.button("⏭️ Step 1 Turn", use_container_width=True, disabled=bool(game.winner))
with b_reset:
    reset_pressed = st.button("🔄 Reset", use_container_width=True)
with b_exp:
    exp_pressed = st.button("🧪 Run Experiment", use_container_width=True, disabled=False)
    st.caption("10 battles × 5 strategies (quick)")
with b_mod:
    mod_exp_pressed = st.button("🧬 Run Modular Exp.", use_container_width=True, disabled=False)
    st.caption("Runs the current 3-layer selection")
with b_view:
    view_pressed = st.button("📊 View Results", use_container_width=True, disabled=False)
    st.caption("Phase 6 — CSV / charts")

if step_pressed:
    if not game.winner:
        game.advance_one_round(decision_fn=_current_decision_fn())
        st.rerun()
if reset_pressed:
    _reset_all(int(seed_val))
    st.rerun()
if exp_pressed:
    with st.spinner("Running 10 battles × 5 strategies (~10 s)..."):
        rows = run_strategy_matrix(n_battles=10, verbose=False)
        summary = aggregate_rows(rows)
        st.session_state.experiment_results = rows
        st.session_state.experiment_summary = summary
    st.success(f"Experiment complete: {len(rows)} rows, {len(summary)} strategies.")
if mod_exp_pressed:
    cfg = (st.session_state.decision_algo, st.session_state.navigation_algo, st.session_state.combat_algo)
    with st.spinner(f"Running 10 battles with {cfg[0]} / {cfg[1]} / {cfg[2]}..."):
        mod_rows = run_modular_matrix([cfg], n_battles=10, verbose=False)
        mod_summary = aggregate_modular_rows(mod_rows)
        st.session_state.modular_results = mod_rows
        st.session_state.modular_summary = mod_summary
    st.success(f"Modular experiment complete: {len(mod_rows)} rows, {len(mod_summary)} configuration(s).")
if view_pressed:
    st.session_state.view_results = not bool(st.session_state.get("view_results", False))
    st.rerun() if view_pressed else None

# Live auto-loop: if running and not paused and not won/lost → sleep, step once, rerun.
if st.session_state.running and not st.session_state.paused and not game.winner:
    time.sleep(st.session_state.pace_ms / 1000.0)
    game.advance_one_round(decision_fn=_current_decision_fn())
    st.rerun()

# ===========================================================================
# Section C — Action dashboard (chronological log)
# ===========================================================================
st.markdown("## C — Action Dashboard (§3.C log format)")
log_c1, log_c2 = st.columns([3, 2])
with log_c1:
    st.markdown("### 🗂 Last 30 actions (table)")
    df = _log_df(game)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Round": st.column_config.NumberColumn("Round", width="small"),
            "ΔHP target": st.column_config.NumberColumn("ΔHP target", width="small"),
        },
        height=320,
    )
with log_c2:
    st.markdown("### 📜 Formatted log (matches brief §3.C example)")
    display_log = game.action_log[-15:]
    if not display_log:
        st.caption("(Press Step 1 Turn or Start to generate entries.)")
    text = "\n\n".join(str(r) for r in display_log)
    st.code(text, language=None, line_numbers=False)

# ===========================================================================
# Section E — Statistics dashboard (8 metric cards)
# ===========================================================================
st.markdown("## E — Statistics Dashboard")
stats = game.stats.to_dict()
e1, e2, e3, e4 = st.columns(4)
e5, e6, e7, e8 = st.columns(4)
e1.metric("Battles Won", f"{stats['battles_won']}")
e2.metric("Battles Lost", f"{stats['battles_lost']}")
e3.metric("Enemies Defeated", f"{stats['enemies_defeated']}")
e4.metric("Party Survival %", f"{100*stats['party_survival_rate']:.0f} %")
e5.metric("Damage Dealt", f"{stats['damage_dealt']}")
e6.metric("Damage Received", f"{stats['damage_received']}")
e7.metric("Healing Performed", f"{stats['healing_performed']}")
e8.metric("Turns Taken", f"{stats['turns_taken']}")

with st.expander("📋 Raw stats payload (JSON)"):
    st.json(stats)

# ===========================================================================
# Small UI diagnostic: show how to run (for Phase 3 validation report)
# ===========================================================================
with st.expander("▶️ How to run this dashboard"):
    st.code(
        "cd AI-dnd\n"
        "pip install -r requirements.txt      # (done in Phase 3.1)\n"
        "python -m streamlit run app.py",
        language="bash",
    )
    st.markdown(
        "- Press **Step 1 Turn** to step one round at a time.\n"
        "- Press **Start** to auto-advance every %.0f ms (tune pace in sidebar).\n"
        "- Press **Pause** then **Reset** to restart with the same seed.\n"
        "- Press **Run Experiment** for a 10-battle single-strategy preview.\n"
        "- Press **Run Modular Exp.** to run the current 3-layer selection across 10 seeds.\n"
        "- Press **View Results** to open the analysis (single-strategy and modular results).\n"
        % st.session_state.pace_ms
    )

# ===========================================================================
# Section F — Results Viewer (View Results tab)
# ===========================================================================
if st.session_state.get("view_results", False):
    st.markdown("---")
    st.markdown("## F — Results Viewer (Phase 6: View Results)")
    # 1) Load either from session or from CSVs on disk.
    rows = st.session_state.get("experiment_results") or []
    summary = st.session_state.get("experiment_summary") or []
    if not rows or not summary:
        disk_rows, disk_summary = load_experiment_results(PROJECT_ROOT / "data" / "experiments")
        rows = rows or disk_rows
        summary = summary or disk_summary
    if not rows or not summary:
        st.info("No experiment results yet. Press **Run Experiment** above, or run the CLI "
                "`python -m tools.experiment_runner 50` to generate the 50×5 matrix in "
                "`data/experiments/` and then re-open this tab.")
    else:
        tab_df, tab_chart, tab_before_after, tab_markdown = st.tabs(
            ["📋 Data tables", "📈 Charts", "🎯 Before / After (SA & GA)", "📝 Methodology"]
        )
        with tab_df:
            st.markdown("#### Per-battle rows (head 20 shown)")
            df_rows = pd.DataFrame(rows)
            st.dataframe(df_rows.head(20), use_container_width=True, hide_index=True)
            st.markdown("#### Aggregated summary (per strategy)")
            df_sum = pd.DataFrame(summary)
            st.dataframe(df_sum, use_container_width=True, hide_index=True)
        with tab_chart:
            df_sum = pd.DataFrame(summary)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### Win rate % per strategy")
                st.bar_chart(data=df_sum, x="strategy", y="win_rate_pct", use_container_width=True)
            with c2:
                st.markdown("##### Clearance rate % per strategy")
                st.bar_chart(data=df_sum, x="strategy", y="pct_party_complete", use_container_width=True)
            c3, c4 = st.columns(2)
            with c3:
                st.markdown("##### Avg party HP left at end")
                st.bar_chart(data=df_sum, x="strategy", y="avg_party_hp_left", use_container_width=True)
            with c4:
                st.markdown("##### Avg rounds per battle")
                st.bar_chart(data=df_sum, x="strategy", y="avg_rounds", use_container_width=True)
            c5, c6 = st.columns(2)
            with c5:
                st.markdown("##### Avg party damage dealt")
                if "avg_party_damage_dealt" in df_sum.columns:
                    st.bar_chart(data=df_sum, x="strategy", y="avg_party_damage_dealt", use_container_width=True)
                else:
                    st.caption("(Not in this CSV)")
            with c6:
                st.markdown("##### Avg party healing done")
                if "avg_party_healing_done" in df_sum.columns:
                    st.bar_chart(data=df_sum, x="strategy", y="avg_party_healing_done", use_container_width=True)
                else:
                    st.caption("(Not in this CSV)")
        with tab_before_after:
            df_rows_pd = pd.DataFrame(rows)
            baselines = [r for r in rows if r.get("strategy") == "baseline"]
            optims = {k: [r for r in rows if r.get("strategy") == k] for k in ("sa-opt", "ga-opt")}
            def _avg(rows_list, col):
                if not rows_list:
                    return 0.0
                return sum(r[col] for r in rows_list) / float(len(rows_list))
            bl_wins = _avg(baselines, "party_hp_left")
            sa_wins = _avg(optims.get("sa-opt") or [], "party_hp_left")
            ga_wins = _avg(optims.get("ga-opt") or [], "party_hp_left")
            m1, m2, m3 = st.columns(3)
            m1.metric("Baseline HP left", f"{bl_wins:.1f}")
            m2.metric("SA-optimized HP left", f"{sa_wins:.1f}",
                      delta=f"{(sa_wins - bl_wins):+0.1f} ({100 * (sa_wins - bl_wins) / max(1e-9, bl_wins):+.1f}%)")
            m3.metric("GA-optimized HP left", f"{ga_wins:.1f}",
                      delta=f"{(ga_wins - bl_wins):+0.1f} ({100 * (ga_wins - bl_wins) / max(1e-9, bl_wins):+.1f}%)")
            bl_winrate_pct = (100.0 * sum(1 for r in baselines if r.get("winner") == "party")
                              / float(max(1, len(baselines))))
            sa_winrate = (100.0 * sum(1 for r in (optims.get("sa-opt") or []) if r.get("winner") == "party")
                          / float(max(1, len(optims.get("sa-opt") or []))))
            ga_winrate = (100.0 * sum(1 for r in (optims.get("ga-opt") or []) if r.get("winner") == "party")
                          / float(max(1, len(optims.get("ga-opt") or []))))
            st.markdown("#### Win rate comparison")
            wr_df = pd.DataFrame({
                "Strategy": ["baseline", "sa-opt", "ga-opt"],
                "Win rate %": [bl_winrate_pct, sa_winrate, ga_winrate],
            })
            st.bar_chart(data=wr_df, x="Strategy", y="Win rate %", use_container_width=True)
            st.markdown("#### Distribution of rounds per battle (baseline vs SA/GA)")
            small = df_rows_pd[df_rows_pd["strategy"].isin({"baseline", "sa-opt", "ga-opt"})]
            if not small.empty:
                st.dataframe(small[["strategy","seed","winner","rounds","party_hp_left"]],
                             use_container_width=True, hide_index=True)
        with tab_markdown:
            try:
                md = build_comparison_markdown(rows, summary)
                st.markdown(md)
            except Exception as ex:
                st.warning(f"Could not render methodology markdown: {ex}")

    # ----- Modular (3-layer) experiment results -----
    mod_rows = st.session_state.get("modular_results") or []
    mod_summary = st.session_state.get("modular_summary") or []
    if not mod_rows or not mod_summary:
        disk_mod_rows, disk_mod_summary = load_modular_results(PROJECT_ROOT / "data" / "experiments")
        mod_rows = mod_rows or disk_mod_rows
        mod_summary = mod_summary or disk_mod_summary
    if mod_rows and mod_summary:
        st.markdown("---")
        st.markdown("### 🧬 Modular (3-layer) experiment results")
        st.caption(
            "These results record the **decision**, **navigation**, and **combat** "
            "selections explicitly. The `strategy` column is a combined key "
            "(`decision|navigation|combat`) used for grouping only."
        )
        m_tab_df, m_tab_chart = st.tabs(["📋 Modular data tables", "📈 Modular charts"])
        with m_tab_df:
            st.markdown("#### Per-battle rows (head 20)")
            st.dataframe(pd.DataFrame(mod_rows).head(20), use_container_width=True, hide_index=True)
            st.markdown("#### Aggregated summary (per 3-layer configuration)")
            st.dataframe(pd.DataFrame(mod_summary), use_container_width=True, hide_index=True)
        with m_tab_chart:
            df_mod = pd.DataFrame(mod_summary)
            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown("##### Win rate % per configuration")
                st.bar_chart(data=df_mod, x="strategy", y="win_rate_pct", use_container_width=True)
            with mc2:
                st.markdown("##### Avg rounds per battle")
                st.bar_chart(data=df_mod, x="strategy", y="avg_rounds", use_container_width=True)

    if st.button("Close Results Viewer", use_container_width=True):
        st.session_state.view_results = False
        st.rerun()
