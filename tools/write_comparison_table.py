"""Phase 6.3 — Markdown comparison table generator + before/after analysis.

Reads ``data/experiments/experiment_summary_aggregated.csv`` plus the per-battle
CSV and writes a markdown report to ``data/experiments/COMPARISON_TABLE.md``.

Run (after experiment_runner):
    python -m tools.write_comparison_table
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.experiment_runner import load_experiment_results


def _pct(a: float, b: float) -> str:
    if abs(b) < 1e-9:
        return "—"
    sign = "+" if a - b >= 0 else "−"
    return f"{sign}{100*abs(a-b)/max(1e-9,b):.1f}%"


def build_comparison_markdown(rows: list[dict], summary: list[dict]) -> str:
    baseline = next((s for s in summary if s["strategy"] == "baseline"), None)
    sa_opt = next((s for s in summary if s["strategy"] == "sa-opt"), None)
    ga_opt = next((s for s in summary if s["strategy"] == "ga-opt"), None)
    astar = next((s for s in summary if s["strategy"] == "astar"), None)
    bfs = next((s for s in summary if s["strategy"] == "bfs"), None)
    lines: list[str] = []
    lines.append("# Experiment results — 50 battles × 5 strategies")
    lines.append("")
    lines.append("## Strategy matrix")
    lines.append("")
    lines.append("| Strategy          | Win rate % | Avg rounds | Avg HP left | Avg damage dealt | Avg healing done | Clearance rate % |")
    lines.append("|-------------------|-----------:|-----------:|------------:|-----------------:|-----------------:|-----------------:|")
    for s in summary:
        lines.append(f"| {s['strategy']:17s} | {s['win_rate_pct']:9.1f}% | {s['avg_rounds']:9.1f} | {s['avg_party_hp_left']:10.1f} | {s['avg_party_damage_dealt']:15.1f} | {s['avg_party_healing_done']:15.1f} | {s['pct_party_complete']:15.1f}% |")
    lines.append("")
    lines.append("## Before / after — SA and GA optimisation vs handwritten baseline")
    lines.append("")
    if baseline and (sa_opt or ga_opt):
        lines.append("| Metric              | Baseline | SA-optimised | Δ (SA)     | GA-optimised | Δ (GA)     |")
        lines.append("|---------------------|---------:|-------------:|-----------:|-------------:|-----------:|")
        keys = [
            ("Win rate %",           "win_rate_pct",          "{:6.1f}%"),
            ("Avg party HP left",    "avg_party_hp_left",     "{:9.1f} "),
            ("Avg damage dealt",     "avg_party_damage_dealt","{:9.1f} "),
            ("Avg healing done",     "avg_party_healing_done","{:9.1f} "),
            ("Clearance rate %",     "pct_party_complete",    "{:6.1f}%"),
        ]
        for metric, key, fmt in keys:
            b_v = baseline[key]
            s_v = sa_opt[key] if sa_opt else float("nan")
            g_v = ga_opt[key] if ga_opt else float("nan")
            ds = _pct(s_v, b_v) if sa_opt else "—"
            dg = _pct(g_v, b_v) if ga_opt else "—"
            sfmt = fmt.format(s_v) if sa_opt else "—"
            gfmt = fmt.format(g_v) if ga_opt else "—"
            lines.append(f"| {metric:19s} | {fmt.format(b_v):>8s} | {sfmt:>12s} | {ds:>10s} | {gfmt:>12s} | {dg:>10s} |")
        lines.append("")
    lines.append("## Search comparison: uninformed (BFS) vs optimal (A*)")
    lines.append("")
    if bfs and astar and baseline:
        lines.append("| Metric              | Baseline | BFS          | Δ(BFS)   | A*            | Δ(A*)   |")
        lines.append("|---------------------|---------:|-------------:|---------:|--------------:|--------:|")
        for metric, key, fmt in (
            ("Win rate %",           "win_rate_pct",          "{:6.1f}%"),
            ("Avg rounds",           "avg_rounds",            "{:6.1f} "),
            ("Avg party HP left",    "avg_party_hp_left",     "{:8.1f} "),
            ("Clearance rate %",     "pct_party_complete",    "{:6.1f}%"),
        ):
            b_v = baseline[key]; b1 = bfs[key]; b2 = astar[key]
            lines.append(f"| {metric:19s} | {fmt.format(b_v):>8s} | {fmt.format(b1):>12s} | {_pct(b1,b_v):>8s} | {fmt.format(b2):>12s} | {_pct(b2,b_v):>7s} |")
        lines.append("")
    lines.append("## Implementation notes (§3 honesty requirement)")
    lines.append("")
    lines.append("- **Search algorithms (BFS / A*)** are only used for dungeon navigation when no enemies are within Chebyshev ≤ 2 tiles. In-melee actions fall back to the role-based baseline so the party does actually engage — search is *not* used to warp characters through enemy-occupied tiles.")
    lines.append("- **SA / GA optimisation** ran over the 10-weight `BehaviourWeights` chromosome described in `algorithms/baseline_decision.py`. Objective = maximise party HP remaining across 2 short battles (same score as the dropdown caches use). SA: 30 steps / cooling 0.92. GA: population 10 / 5 generations — the cached versions used in the dashboard are the exact chromosomes reported by these runs.")
    lines.append("- **Minimax/Alpha-Beta/Expectimax** (dropdown only, not part of the 5-column experiment matrix) evaluate 1v1 sub-trees at melee. Tree depth is bounded: MM/AB max_depth=5, Expectimax max_depth=4, leaves evaluated by `heuristic_eval = (our_hp − their_hp)/max_hp` ∈ [−1, +1]. Expectimax attack action is stochastic → 0.2 miss, 0.7 hit, 0.1 crit — probabilities sum exactly 1.0.")
    lines.append("- **Seed reproducibility**: 50 seeds = [100000, 100049]. `GameState.new_default(seed=s)` produces the same dungeon/party/enemies run-to-run via zlib-CRC32–derived per-actor PRNG seeding described in Phase 2.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    data_dir = _ROOT / "data" / "experiments"
    rows, summary = load_experiment_results(data_dir)
    if not rows or not summary:
        print(f"[comparison] WARNING no CSVs in {data_dir}. Run `python -m tools.experiment_runner 50` first.")
        return 1
    md = build_comparison_markdown(rows, summary)
    out_path = data_dir / "COMPARISON_TABLE.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"[comparison] wrote {out_path}  ({sum(1 for _ in out_path.open())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
