"""How does the neuron go from silence to firing, and what happens when the drive is too strong?

Simulates the frequency-current (f-I) relationship, determines single-spike rheobase
versus repetitive firing onset, and characterizes the collapse into depolarisation block.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hhlab.analysis import (
    detect_block_onset,
    fi_curve,
    find_repetitive_onset,
    find_rheobase,
    run_step,
)


def main():
    """Run f-I curve analysis, save figures and summary data, and display results."""
    parser = argparse.ArgumentParser(description="Analyse Hodgkin-Huxley f-I curve and block.")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory")
    parser.add_argument("--show", action="store_true", help="Display interactive figure")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Stimulus current grid across sub-threshold, repetitive firing, and block regimes
    currents = np.concatenate([
        np.arange(0.0, 12.5, 0.5),
        np.arange(15.0, 50.0, 5.0),
        np.arange(50.0, 90.0, 2.5),
        np.array([100.0, 120.0, 150.0]),
    ])

    print("Computing f-I curve across current levels (500 ms steps, 100 ms transient ignored)...")
    res_chunks = []
    chunk_size = 10
    for idx in range(0, len(currents), chunk_size):
        sub_curr = currents[idx : idx + chunk_size]
        print(
            f"  Simulating currents {idx + 1} to {min(idx + chunk_size, len(currents))}"
            f" / {len(currents)} (current {sub_curr[0]:.1f} - {sub_curr[-1]:.1f} µA/cm²)..."
        )
        res_chunks.append(fi_curve(sub_curr, t_pre=50.0, duration=500.0, transient=100.0))

    res = {
        "current": np.concatenate([c["current"] for c in res_chunks]),
        "rate_hz": np.concatenate([c["rate_hz"] for c in res_chunks]),
        "n_spikes": np.concatenate([c["n_spikes"] for c in res_chunks]),
        "v_max": np.concatenate([c["v_max"] for c in res_chunks]),
    }

    # 2. Compute excitability thresholds and depolarisation block onset
    print("Finding rheobase via bisection...")
    rheobase = find_rheobase()
    print(f"  Rheobase: {rheobase:.4f} µA/cm²")

    print("Finding onset of repetitive firing via bisection (1000 ms step)...")
    onset = find_repetitive_onset()
    print(f"  Repetitive onset: {onset:.4f} µA/cm²")

    block_onset = detect_block_onset(res["current"], res["rate_hz"])
    print(f"  Depolarisation block onset: {block_onset:.2f} µA/cm²")

    # 3. Generate 2 x 2 diagnostic figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # (a) Firing rate against current with transition thresholds
    ax_a = axes[0, 0]
    ax_a.plot(res["current"], res["rate_hz"], "b.-", lw=1.5, ms=5, label="Firing rate")
    ax_a.axvline(rheobase, color="g", linestyle="--", lw=1.3, label=f"Rheobase ({rheobase:.2f})")
    ax_a.axvline(onset, color="m", linestyle="--", lw=1.3, label=f"Onset ({onset:.2f})")
    if np.isfinite(block_onset):
        ax_a.axvline(
            block_onset, color="r", linestyle="--", lw=1.3, label=f"Block ({block_onset:.1f})"
        )
    ax_a.set_xlabel("Injected Current (µA/cm²)")
    ax_a.set_ylabel("Firing Rate (Hz)")
    ax_a.set_title("(a) Firing Rate vs Injected Current (f-I Curve)")
    ax_a.legend(loc="upper right", fontsize=9)

    # (b) Maximum membrane voltage against current
    ax_b = axes[0, 1]
    ax_b.plot(res["current"], res["v_max"], "r.-", lw=1.5, ms=5)
    ax_b.set_xlabel("Injected Current (µA/cm²)")
    ax_b.set_ylabel("Maximum Voltage V_max (mV)")
    ax_b.set_title("(b) Maximum Membrane Potential vs Current")

    # (c) Traces just below and above repetitive firing onset
    ax_c = axes[1, 0]
    i_below = onset - 0.2
    i_above = onset + 0.2
    tr_below = run_step(i_below, t_pre=50.0, duration=1000.0, t_post=0.0)
    tr_above = run_step(i_above, t_pre=50.0, duration=1000.0, t_post=0.0)

    mask_zoom = (tr_below.t >= 50.0) & (tr_below.t <= 350.0)
    t_zoom = tr_below.t[mask_zoom] - 50.0
    ax_c.plot(t_zoom, tr_below.V[mask_zoom], "k--", lw=1.2, label=f"Below onset ({i_below:.2f})")
    ax_c.plot(t_zoom, tr_above.V[mask_zoom], "m-", lw=1.2, label=f"Above onset ({i_above:.2f})")
    ax_c.set_xlabel("Time Since Step Onset (ms)")
    ax_c.set_ylabel("Membrane Potential V (mV)")
    ax_c.set_title("(c) Bifurcation: Traces Below and Above Repetitive Onset")
    ax_c.legend(loc="upper right", fontsize=9)

    # (d) Illustrative traces at 10, 50, and 70 uA/cm^2
    ax_d = axes[1, 1]
    tr_10 = run_step(10.0, t_pre=10.0, duration=300.0, t_post=10.0)
    tr_50 = run_step(50.0, t_pre=10.0, duration=300.0, t_post=10.0)
    tr_70 = run_step(70.0, t_pre=10.0, duration=300.0, t_post=10.0)

    ax_d.plot(tr_10.t, tr_10.V, "b-", lw=1.2, label="10 µA/cm² (regular)")
    ax_d.plot(tr_50.t, tr_50.V, "g-", lw=1.2, label="50 µA/cm² (fast)")
    ax_d.plot(tr_70.t, tr_70.V, "r-", lw=1.2, label="70 µA/cm² (spikes die out: block)")
    ax_d.set_xlabel("Time (ms)")
    ax_d.set_ylabel("Membrane Potential V (mV)")
    ax_d.set_title("(d) Firing Regimes Across Drive Strengths (300 ms steps)")
    ax_d.legend(loc="upper right", fontsize=9)

    fig_path = out_dir / "exp2_fi_curve.png"
    plt.savefig(fig_path, dpi=150)
    if args.show:
        plt.show()
    plt.close(fig)

    # 4. Save JSON summary with tabulated rates and arrays
    idx_10 = int(np.argmin(np.abs(res["current"] - 10.0)))
    idx_50 = int(np.argmin(np.abs(res["current"] - 50.0)))
    rate_at_10 = float(res["rate_hz"][idx_10])
    rate_at_50 = float(res["rate_hz"][idx_50])

    peak_rate_idx = int(np.argmax(res["rate_hz"]))
    max_rate = float(res["rate_hz"][peak_rate_idx])
    curr_at_max = float(res["current"][peak_rate_idx])

    summary_data = {
        "rheobase_uA_cm2": float(rheobase),
        "repetitive_onset_uA_cm2": float(onset),
        "onset_over_rheobase": float(onset / rheobase),
        "block_onset_uA_cm2": float(block_onset) if np.isfinite(block_onset) else None,
        "rate_hz_at_10": rate_at_10,
        "rate_hz_at_50": rate_at_50,
        "max_rate_hz": max_rate,
        "lowest_nonzero_rate_hz": (
            float(res["rate_hz"][res["rate_hz"] > 0].min()) if np.any(res["rate_hz"] > 0) else None
        ),
        "current_at_max_rate_uA_cm2": curr_at_max,
        "current": [float(x) for x in res["current"]],
        "rate_hz": [float(x) for x in res["rate_hz"]],
        "n_spikes": [int(x) for x in res["n_spikes"]],
        "v_max": [float(x) for x in res["v_max"]],
    }

    json_path = out_dir / "exp2_fi_curve_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 5. Print results table and conclusion sentence
    ratio = onset / rheobase
    print("=" * 60)
    print(" Experiment 2: Frequency-Current (f-I) Curve Summary")
    print("=" * 60)
    print(f" {'rheobase_uA_cm2':<35} : {rheobase:10.4f}")
    print(f" {'repetitive_onset_uA_cm2':<35} : {onset:10.4f}")
    print(f" {'onset_over_rheobase':<35} : {ratio:10.4f}")
    print(f" {'block_onset_uA_cm2':<35} : {block_onset:10.4f}")
    print(f" {'rate_hz_at_10':<35} : {rate_at_10:10.2f}")
    print(f" {'rate_hz_at_50':<35} : {rate_at_50:10.2f}")
    print(f" {'max_rate_hz':<35} : {max_rate:10.2f}")
    print(f" {'current_at_max_rate_uA_cm2':<35} : {curr_at_max:10.2f}")
    print("=" * 60)

    # Class 2 excitability means firing starts abruptly at a non-zero rate, not at ~0 Hz.
    # Report both signs: the rheobase/onset gap and the lowest non-zero rate on the grid.
    nonzero = res["rate_hz"][res["rate_hz"] > 0.0]
    lowest = float(nonzero.min()) if nonzero.size else float("nan")
    print(
        f"Rheobase {rheobase:.2f} and repetitive onset {onset:.2f} µA/cm² differ by a factor of "
        f"{ratio:.2f}; the lowest non-zero rate on the grid is {lowest:.1f} Hz."
    )
    if ratio > 2.0 and lowest > 20.0:
        print("Both signs are consistent with class 2 excitability (abrupt onset of firing).")
    else:
        print("These numbers alone do not show a clear class 2 pattern.")


if __name__ == "__main__":
    main()