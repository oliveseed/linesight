# =======================================================================================================================
# Offline plotting for scripts/infer_survival.py output
# =======================================================================================================================
"""
Reads the rollouts.json produced by scripts/infer_survival.py and generates two plots:

1. Survival rate: x = zone index, y = number of rollouts that reached at least that zone.
2. Race time scatter: x = zone index where the rollout was truncated, y = race time in seconds
   (simulator clock, not wall clock).

Usage:
    python scripts/plot_inference_survival.py --data save/inference_survival/<run_name>/<timestamp>/rollouts.json
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config_files import config_copy
from trackmania_rl.bng_trajectory import get_vcp_centers_from_file

BASE_DIR = Path(__file__).resolve().parents[1]

REASON_STYLE = {
    "finish": {"color": "seagreen", "marker": "o", "label": "finish"},
    "crash": {"color": "crimson", "marker": "x", "label": "crash (damage)"},
    "timeout": {"color": "darkorange", "marker": "s", "label": "timeout / no progress"},
    "error": {"color": "gray", "marker": "^", "label": "interface error"},
}


def find_latest_rollouts_json() -> Path:
    candidates = list(BASE_DIR.glob("save/inference_survival/**/rollouts.json"))
    if not candidates:
        raise FileNotFoundError(
            "No rollouts.json found under save/inference_survival/. Run scripts/infer_survival.py first, "
            "or pass --data explicitly."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_rollouts(data_path: Path) -> dict:
    with open(data_path, encoding="utf-8") as f:
        payload = json.load(f)
    rollouts = payload["rollouts"]
    if not rollouts:
        raise ValueError(f"{data_path} contains no rollouts.")
    return payload


def track_zone_bounds() -> tuple[int, int]:
    """Return (first_zone_idx, finish_idx) for the configured map."""
    map_path = BASE_DIR / "maps" / "nord1.npy"
    zone_centers = get_vcp_centers_from_file(trajectory_path=map_path, config=config_copy)
    finish_idx = len(zone_centers) - 1 - config_copy.n_zone_centers_extrapolate_after_end_of_map
    first_zone_idx = config_copy.n_zone_centers_extrapolate_before_start_of_map
    return first_zone_idx, finish_idx


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to rollouts.json (default: newest under save/inference_survival/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the plots (default: same directory as --data)",
    )
    args = parser.parse_args()

    data_path = args.data if args.data is not None else find_latest_rollouts_json()
    output_dir = args.output_dir if args.output_dir is not None else data_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_rollouts(data_path)
    rollouts = payload["rollouts"]
    n = len(rollouts)

    try:
        first_zone_idx, finish_idx = track_zone_bounds()
    except Exception as exc:  # noqa: BLE001 — fall back to data bounds if the map cannot be loaded
        print(f"Warning: could not load map for zone bounds ({exc}); using data bounds instead.")
        first_zone_idx = min(r["end_zone_idx"] for r in rollouts)
        finish_idx = max(r["end_zone_idx"] for r in rollouts)

    furthest = np.array([r["furthest_zone_idx"] for r in rollouts], dtype=np.int64)
    end_zone = np.array([r["end_zone_idx"] for r in rollouts], dtype=np.int64)
    race_time_s = np.array([r["race_time_ms"] for r in rollouts], dtype=np.float64) / 1000.0
    reasons = [r["truncation_reason"] for r in rollouts]

    # ========================================================
    # Plot 1: survival rate — number of rollouts that reached each zone
    # ========================================================
    zones = np.arange(first_zone_idx, finish_idx + 1)
    # A rollout is considered to have "survived" zone k if it ever reached zone k.
    survival_counts = np.array([np.count_nonzero(furthest >= k) for k in zones])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(zones, survival_counts, drawstyle="steps-post", color="steelblue", linewidth=1.5)
    ax.fill_between(zones, survival_counts, step="post", alpha=0.2, color="steelblue")
    ax.set_xlim(first_zone_idx, finish_idx)
    ax.set_ylim(0, n)
    ax.set_xlabel("Zone index")
    ax.set_ylabel(f"Number of rollouts survived (out of {n})")
    ax.set_title(f"Survival rate — {payload.get('run_name', '?')} ({n} rollouts)")
    ax.grid(True, alpha=0.3)
    survival_path = output_dir / "survival_rate.png"
    fig.tight_layout()
    fig.savefig(survival_path, dpi=150)
    plt.close(fig)

    # ========================================================
    # Plot 2: race time vs truncation zone (scatter)
    # ========================================================
    fig, ax = plt.subplots(figsize=(12, 7))
    for reason, style in REASON_STYLE.items():
        mask = np.array([r == reason for r in reasons])
        if not np.any(mask):
            continue
        ax.scatter(
            end_zone[mask],
            race_time_s[mask],
            color=style["color"],
            marker=style["marker"],
            label=f"{style['label']} (n={int(mask.sum())})",
            alpha=0.8,
            s=50,
        )
    ax.set_xlim(first_zone_idx, finish_idx)
    ax.set_xlabel("Zone index at truncation")
    ax.set_ylabel("Race time (s, simulator clock)")
    ax.set_title(f"Race time vs truncation zone — {payload.get('run_name', '?')} ({n} rollouts)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    scatter_path = output_dir / "race_time_scatter.png"
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)

    # ========================================================
    # Console summary
    # ========================================================
    reason_counts = {r: reasons.count(r) for r in sorted(set(reasons))}
    finished_times = race_time_s[np.array([r == "finish" for r in reasons])]
    print(f"Data:        {data_path}")
    print(f"Rollouts:    {n}")
    print(f"Reasons:     {reason_counts}")
    print(f"Zone bounds: first={first_zone_idx}  finish={finish_idx}")
    if len(finished_times):
        print(
            f"Finish time: mean={finished_times.mean():.1f}s  median={np.median(finished_times):.1f}s  "
            f"min={finished_times.min():.1f}s  max={finished_times.max():.1f}s"
        )
    print(f"Wrote: {survival_path}")
    print(f"Wrote: {scatter_path}")


if __name__ == "__main__":
    main()
