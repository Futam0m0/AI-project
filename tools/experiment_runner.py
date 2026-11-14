"""Phase 6.1 — experiment runner: 50 battles × 4/5 strategy matrix CSV exporter.

Produces two CSVs inside ``data/experiments/`` (folders are auto-created):

1. ``experiment_battles_50x4.csv`` — one row per (seed, strategy) pair.
   Columns: seed, strategy, winner, rounds, party_hp_left, enemy_hp_left,
            total_dealt, total_healed, party_battles_won, party_battles_lost,
            action_log_lines.

2. ``experiment_summary_aggregated.csv`` — one row per strategy aggregated
   over the 50 seeds.  Columns = strategy, battle_count, win_rate_pct,
   avg_rounds, avg_party_hp, avg_dealt, avg_healed, pct_party_complete.

Running
-------
    python -m tools.experiment_runner          # runs 50×4 (the default plan)
    python -m tools.experiment_runner 10       # runs a smaller 10×4 (quick dev)

This module *does not* import Streamlit — it's pure CLI/CSV and safe to call
from tests or scheduled scripts.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import Iterable

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from algorithms.algorithm_selection import dispatch
from algorithms.modular_selector import build_decision_fn
from game.dungeon import PARTY_SYMBOLS
from game.actions import Action as _Action
from game.game_state import GameState


STRATEGY_MATRIX: list[str] = [
    "baseline",
    "bfs",
    "astar",
    "sa-opt",
    "ga-opt",
]

DEFAULT_N_BATTLES = 50
MAX_ROUNDS_PER_BATTLE = 60
CSV_PER_BATTLE_NAME = "experiment_battles_50x4.csv"
CSV_AGGREGATED_NAME = "experiment_summary_aggregated.csv"

# Modular-experiment CSVs (separate files so historical single-strategy
# results are never overwritten or confused with 3-layer combinations).
CSV_MODULAR_BATTLE_NAME = "experiment_modular_battles.csv"
CSV_MODULAR_AGGREGATED_NAME = "experiment_modular_summary.csv"


# ---------------------------------------------------------------------------
# Shared metric extraction (used by both single-strategy and modular modes)
# ---------------------------------------------------------------------------
def _battle_metrics(gs: GameState, rounds: int) -> dict:
    """Extract per-battle numeric metrics from a finished GameState."""
    party_hp_left = sum(max(0, p.hp) for p in gs.party)
    enemy_hp_left = sum(max(0, e.hp) for e in gs.enemies)
    total_dealt = 0
    total_healed = 0
    for a in gs.action_log:
        is_party = getattr(getattr(a, "actor", None), "symbol", None) in PARTY_SYMBOLS
        if not is_party:
            continue
        act = getattr(a, "action", None)
        dht = int(getattr(a, "delta_hp_target", 0) or 0)
        if act == _Action.ATTACK and dht < 0:
            total_dealt += -dht
        elif act == _Action.HEAL and dht > 0:
            total_healed += dht
    winner = gs.winner if gs.winner else "draw"
    return {
        "winner": winner,
        "rounds": int(rounds),
        "party_hp_left": float(party_hp_left),
        "enemy_hp_left": float(enemy_hp_left),
        "party_damage_dealt": float(total_dealt),
        "party_healing_done": float(total_healed),
        "stats_battles_won": int(getattr(gs.stats, "battles_won", 0)),
        "stats_battles_lost": int(getattr(gs.stats, "battles_lost", 0)),
        "log_lines": int(len(gs.action_log)),
    }


def _run_battle_with_df(dfn, seed: int, *, max_rounds: int = MAX_ROUNDS_PER_BATTLE) -> dict:
    """Run one battle with an arbitrary decision function + return metrics."""
    gs = GameState.new_default(seed=seed)
    rounds = 0
    for r in range(max_rounds):
        rounds = r
        if gs.winner is not None:
            break
        gs.advance_one_round(decision_fn=dfn)
    rounds = max(rounds, gs.round_number)
    return _battle_metrics(gs, rounds)


# ---------------------------------------------------------------------------
# Core battle evaluator — single-strategy (backward-compatible) mode
# ---------------------------------------------------------------------------
def run_single_battle(strategy: str, seed: int, *, max_rounds: int = MAX_ROUNDS_PER_BATTLE) -> dict:
    """Run a single battle and return a results-row dict (ready for CSV).

    The 5-tuple decision_fn is resolved via the same dispatch() used by the
    Streamlit dashboard, so results here are 1:1 reproducible in the UI.
    """
    metrics = _run_battle_with_df(dispatch(strategy), seed, max_rounds=max_rounds)
    return {"seed": int(seed), "strategy": strategy, **metrics}


def run_strategy_matrix(
    strategies: Iterable[str] = STRATEGY_MATRIX,
    *,
    n_battles: int = DEFAULT_N_BATTLES,
    seed_start: int = 100_000,
    verbose: bool = True,
) -> list[dict]:
    """Run ``n_battles × len(strategies)`` battles.  Returns list of row dicts."""
    strategies = list(strategies)
    rows: list[dict] = []
    t0 = time.perf_counter()
    for si, strat in enumerate(strategies):
        for bi in range(n_battles):
            seed = seed_start + bi
            row = run_single_battle(strat, seed)
            rows.append(row)
            if verbose and bi % 10 == 0:
                elapsed = time.perf_counter() - t0
                done = si * n_battles + bi + 1
                total = len(strategies) * n_battles
                print(f"[exp] {done:>4}/{total} strategy={strat:10s} seed={seed} "
                      f"winner={row['winner']:8s} rounds={row['rounds']:2d} "
                      f"t={elapsed:5.1f}s")
    return rows


def aggregate_rows(rows: list[dict]) -> list[dict]:
    """Aggregate per-(seed,strategy) rows → per-strategy summary rows."""
    by_strat: dict[str, list[dict]] = {}
    for r in rows:
        by_strat.setdefault(r["strategy"], []).append(r)
    summary = []
    for strat, rrs in sorted(by_strat.items()):
        n = len(rrs)
        wins = sum(1 for r in rrs if r["winner"] == "party")
        exit_clear = sum(1 for r in rrs if r["enemy_hp_left"] == 0 or r["winner"] == "party")
        summary.append({
            "strategy": strat,
            "battle_count": int(n),
            "win_rate_pct": 100.0 * wins / float(max(1, n)),
            "avg_rounds": sum(r["rounds"] for r in rrs) / float(max(1, n)),
            "avg_party_hp_left": sum(r["party_hp_left"] for r in rrs) / float(max(1, n)),
            "avg_party_damage_dealt": sum(r["party_damage_dealt"] for r in rrs) / float(max(1, n)),
            "avg_party_healing_done": sum(r["party_healing_done"] for r in rrs) / float(max(1, n)),
            "pct_party_complete": 100.0 * exit_clear / float(max(1, n)),
        })
    return summary


# ---------------------------------------------------------------------------
# Modular (3-layer) experiment support
# ---------------------------------------------------------------------------

def modular_strategy_key(decision: str, navigation: str, combat: str) -> str:
    """Stable combined key for a 3-layer configuration (used for grouping)."""
    return f"{decision}|{navigation}|{combat}"


def run_single_modular_battle(
    decision: str,
    navigation: str,
    combat: str,
    seed: int,
    *,
    max_rounds: int = MAX_ROUNDS_PER_BATTLE,
) -> dict:
    """Run one battle with a 3-layer modular configuration.

    Uses :func:`algorithms.modular_selector.build_decision_fn` so the routing
    layer (not a specialised algorithm) decides which layer acts on each turn.
    The returned row records all three algorithm names explicitly so it is
    clear which layers were selected.
    """
    dfn = build_decision_fn(decision, navigation, combat)
    metrics = _run_battle_with_df(dfn, seed, max_rounds=max_rounds)
    return {
        "seed": int(seed),
        "strategy": modular_strategy_key(decision, navigation, combat),
        "decision": decision,
        "navigation": navigation,
        "combat": combat,
        **metrics,
    }


def run_modular_matrix(
    configs: Iterable[tuple[str, str, str]],
    *,
    n_battles: int = DEFAULT_N_BATTLES,
    seed_start: int = 100_000,
    verbose: bool = True,
) -> list[dict]:
    """Run ``n_battles × len(configs)`` modular battles.

    Each config is a ``(decision, navigation, combat)`` tuple.  Uses the same
    seeds for every config so configurations are directly comparable.
    """
    configs = list(configs)
    rows: list[dict] = []
    t0 = time.perf_counter()
    for ci, (dec, nav, com) in enumerate(configs):
        key = modular_strategy_key(dec, nav, com)
        for bi in range(n_battles):
            seed = seed_start + bi
            row = run_single_modular_battle(dec, nav, com, seed)
            rows.append(row)
            if verbose and bi % 10 == 0:
                elapsed = time.perf_counter() - t0
                done = ci * n_battles + bi + 1
                total = len(configs) * n_battles
                print(f"[exp] {done:>4}/{total} cfg={key:35s} seed={seed} "
                      f"winner={row['winner']:8s} rounds={row['rounds']:2d} "
                      f"t={elapsed:5.1f}s")
    return rows


def aggregate_modular_rows(rows: list[dict]) -> list[dict]:
    """Aggregate modular per-battle rows → per-configuration summary rows.

    Each summary row keeps ``decision``, ``navigation`` and ``combat`` columns
    in addition to the aggregate metrics.
    """
    by_key: dict[str, list[dict]] = {}
    for r in rows:
        by_key.setdefault(r["strategy"], []).append(r)
    summary = []
    for key, rrs in sorted(by_key.items()):
        n = len(rrs)
        wins = sum(1 for r in rrs if r["winner"] == "party")
        exit_clear = sum(1 for r in rrs if r["enemy_hp_left"] == 0 or r["winner"] == "party")
        first = rrs[0]
        summary.append({
            "strategy": key,
            "decision": first["decision"],
            "navigation": first["navigation"],
            "combat": first["combat"],
            "battle_count": int(n),
            "win_rate_pct": 100.0 * wins / float(max(1, n)),
            "avg_rounds": sum(r["rounds"] for r in rrs) / float(max(1, n)),
            "avg_party_hp_left": sum(r["party_hp_left"] for r in rrs) / float(max(1, n)),
            "avg_party_damage_dealt": sum(r["party_damage_dealt"] for r in rrs) / float(max(1, n)),
            "avg_party_healing_done": sum(r["party_healing_done"] for r in rrs) / float(max(1, n)),
            "pct_party_complete": 100.0 * exit_clear / float(max(1, n)),
        })
    return summary


def save_modular_results(
    rows: list[dict],
    summary: list[dict],
    out_dir: str | os.PathLike,
) -> tuple[Path, Path]:
    """Save modular experiment rows + summary to separate CSV files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    battle_csv = out_dir / CSV_MODULAR_BATTLE_NAME
    agg_csv = out_dir / CSV_MODULAR_AGGREGATED_NAME
    fieldnames_battle = list(rows[0].keys()) if rows else [
        "seed", "strategy", "decision", "navigation", "combat",
        "winner", "rounds", "party_hp_left", "enemy_hp_left",
        "party_damage_dealt", "party_healing_done",
        "stats_battles_won", "stats_battles_lost", "log_lines",
    ]
    with battle_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames_battle)
        w.writeheader()
        w.writerows(rows)
    fieldnames_agg = list(summary[0].keys()) if summary else [
        "strategy", "decision", "navigation", "combat", "battle_count",
        "win_rate_pct", "avg_rounds", "avg_party_hp_left",
        "avg_party_damage_dealt", "avg_party_healing_done", "pct_party_complete",
    ]
    with agg_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames_agg)
        w.writeheader()
        w.writerows(summary)
    return battle_csv, agg_csv


def load_modular_results(out_dir: str | os.PathLike) -> tuple[list[dict], list[dict]]:
    """Load modular CSVs from disk → (rows, summary).  Empty lists if missing."""
    out_dir = Path(out_dir)
    battle_csv = out_dir / CSV_MODULAR_BATTLE_NAME
    agg_csv = out_dir / CSV_MODULAR_AGGREGATED_NAME
    rows = _load_rows(battle_csv) if battle_csv.exists() else []
    summary = _load_rows(agg_csv) if agg_csv.exists() else []
    return rows, summary


def save_results(rows: list[dict], summary: list[dict], out_dir: str | os.PathLike) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    battle_csv = out_dir / CSV_PER_BATTLE_NAME
    agg_csv = out_dir / CSV_AGGREGATED_NAME
    fieldnames_battle = list(rows[0].keys()) if rows else [
        "seed","strategy","winner","rounds","party_hp_left","enemy_hp_left",
        "party_damage_dealt","party_healing_done","stats_battles_won","stats_battles_lost","log_lines",
    ]
    with battle_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames_battle)
        w.writeheader()
        w.writerows(rows)
    fieldnames_agg = list(summary[0].keys()) if summary else [
        "strategy","battle_count","win_rate_pct","avg_rounds","avg_party_hp_left",
        "avg_party_damage_dealt","avg_party_healing_done","pct_party_complete"
    ]
    with agg_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames_agg)
        w.writeheader()
        w.writerows(summary)
    return battle_csv, agg_csv


def _load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    # Numberise numeric columns.
    numeric_cols = {"seed","rounds","party_hp_left","enemy_hp_left","party_damage_dealt",
                    "party_healing_done","stats_battles_won","stats_battles_lost","log_lines"}
    numeric_agg = {"battle_count","win_rate_pct","avg_rounds","avg_party_hp_left",
                   "avg_party_damage_dealt","avg_party_healing_done","pct_party_complete"}
    for r in rows:
        for c in (numeric_cols | numeric_agg):
            if c in r:
                try:
                    r[c] = int(r[c]) if c in {"seed","rounds","log_lines","battle_count",
                                              "stats_battles_won","stats_battles_lost"} else float(r[c])
                except (TypeError, ValueError):
                    pass
    return rows


def load_experiment_results(out_dir: str | os.PathLike) -> tuple[list[dict], list[dict]]:
    """Load both CSVs from disk → (rows, summary). Returns empty lists if missing."""
    out_dir = Path(out_dir)
    battle_csv = out_dir / CSV_PER_BATTLE_NAME
    agg_csv = out_dir / CSV_AGGREGATED_NAME
    rows = _load_rows(battle_csv) if battle_csv.exists() else []
    summary = _load_rows(agg_csv) if agg_csv.exists() else []
    return rows, summary


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    n_battles = int(argv[0]) if argv else DEFAULT_N_BATTLES
    data_dir = _ROOT / "data" / "experiments"
    print(f"[Phase 6 experiment runner] running {n_battles} battles × {len(STRATEGY_MATRIX)} "
          f"strategies -> {data_dir}")
    rows = run_strategy_matrix(n_battles=n_battles)
    summary = aggregate_rows(rows)
    bfile, afile = save_results(rows, summary, data_dir)
    print(f"[Phase 6] Saved {len(rows)} rows -> {bfile}")
    print(f"[Phase 6] Saved {len(summary)} strategies summary -> {afile}")
    print("\n=== AGGREGATE SUMMARY ===")
    print(f"{'strategy':13s} {'win%':>7s} {'rnds':>6s} {'hpL':>7s} {'dealt':>6s} {'heal':>6s} {'clear%':>7s}")
    for s in summary:
        print(f"{s['strategy']:13s} {s['win_rate_pct']:6.1f}% {s['avg_rounds']:6.1f} "
              f"{s['avg_party_hp_left']:7.1f} {s['avg_party_damage_dealt']:6.1f} "
              f"{s['avg_party_healing_done']:6.1f} {s['pct_party_complete']:6.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
