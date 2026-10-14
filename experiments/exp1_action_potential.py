"""What does one action potential look like, and what makes it?

Simulates a single Hodgkin-Huxley action potential in response to a 40 ms depolarising step,
dissects the underlying ionic currents and channel conductances, and benchmarks solver accuracy.
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
    afterhyperpolarisation,
    detect_spikes,
    run_step,
    spike_peak,
    spike_width,
)
from hhlab.neuron import rk4_simulate
from hhlab.stimuli import step


def main():
    """Run action potential characterisation, save outputs, and print summary."""
    parser = argparse.ArgumentParser(
        description="Characterise a single Hodgkin-Huxley action potential."
    )
    parser.add_argument("--outdir", type=str, default="results", help="Output directory")
    parser.add_argument("--show", action="store_true", help="Display interactive figure")
    parser.add_argument("--amplitude", type=float, default=10.0, help="Step amplitude (uA/cm^2)")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run LSODA adaptive solution and fixed-step RK4 cross-check
    t_pre = 5.0
    duration = 40.0
    t_post = 5.0
    t_max = t_pre + duration + t_post

    tr_lsoda = run_step(args.amplitude, t_pre=t_pre, duration=duration, t_post=t_post)
    stim_fn = step(args.amplitude, t_pre, t_pre + duration)
    tr_rk4 = rk4_simulate(stim_fn, t_max=t_max, dt=0.005)

    # 2. Extract electrophysiological metrics
    crossings_lsoda = detect_spikes(tr_lsoda.t, tr_lsoda.V)
    crossings_rk4 = detect_spikes(tr_rk4.t, tr_rk4.V)

    peak_v = spike_peak(tr_lsoda.t, tr_lsoda.V, 0)
    first_cross = float(crossings_lsoda[0]) if len(crossings_lsoda) > 0 else float("nan")
    width = spike_width(tr_lsoda.t, tr_lsoda.V, 0)
    ahp_val = afterhyperpolarisation(tr_lsoda.t, tr_lsoda.V, 0)

    conds = tr_lsoda.conductances()
    peak_gna = float(np.max(conds["gNa"]))
    peak_gk = float(np.max(conds["gK"]))

    # Spikes within the 40 ms stimulation window
    spikes_in_step = np.sum((crossings_lsoda >= t_pre) & (crossings_lsoda < t_pre + duration))
    n_spikes = int(spikes_in_step)
    rest_v = float(tr_lsoda.V[0])

    if len(crossings_lsoda) > 0 and len(crossings_rk4) > 0:
        diff_crossing = float(abs(crossings_lsoda[0] - crossings_rk4[0]))
    else:
        diff_crossing = float("nan")

    # 3. Generate 2 x 3 diagnostic figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    # (a) Membrane potential and stimulus current
    ax_a = axes[0, 0]
    ax_a.plot(tr_lsoda.t, tr_lsoda.V, "b-", lw=1.8, label="V (mV)")
    ax_a.set_xlabel("Time (ms)")
    ax_a.set_ylabel("Membrane Potential V (mV)", color="b")
    ax_a.tick_params(axis="y", labelcolor="b")
    ax_a.set_title("(a) Membrane Potential & Stimulus")
    ax_a_twin = ax_a.twinx()
    ax_a_twin.plot(tr_lsoda.t, tr_lsoda.I_stim, "k--", lw=1.0, alpha=0.7, label="I_stim")
    ax_a_twin.set_ylabel("Stimulus Current (µA/cm²)", color="k")
    ax_a_twin.tick_params(axis="y", labelcolor="k")

    # (b) Gating variable kinetics
    ax_b = axes[0, 1]
    ax_b.plot(tr_lsoda.t, tr_lsoda.m, "g-", lw=1.5, label="m (Na activation)")
    ax_b.plot(tr_lsoda.t, tr_lsoda.h, "r-", lw=1.5, label="h (Na inactivation)")
    ax_b.plot(tr_lsoda.t, tr_lsoda.n, "b-", lw=1.5, label="n (K activation)")
    ax_b.set_xlabel("Time (ms)")
    ax_b.set_ylabel("Gating Probability [0, 1]")
    ax_b.set_title("(b) Gating Variables")
    ax_b.legend(loc="center right", fontsize=8)

    # (c) Active conductances
    ax_c = axes[0, 2]
    ax_c.plot(tr_lsoda.t, conds["gNa"], "r-", lw=1.6, label="gNa")
    ax_c.plot(tr_lsoda.t, conds["gK"], "b-", lw=1.6, label="gK")
    ax_c.set_xlabel("Time (ms)")
    ax_c.set_ylabel("Conductance (mS/cm²)")
    ax_c.set_title("(c) Sodium & Potassium Conductances")
    ax_c.legend(loc="upper right", fontsize=9)

    # (d) Ionic currents
    currs = tr_lsoda.currents()
    ax_d = axes[1, 0]
    ax_d.plot(tr_lsoda.t, currs["I_Na"], "r-", lw=1.4, label="I_Na")
    ax_d.plot(tr_lsoda.t, currs["I_K"], "b-", lw=1.4, label="I_K")
    ax_d.plot(tr_lsoda.t, currs["I_L"], "c--", lw=1.2, label="I_L")
    ax_d.plot(tr_lsoda.t, tr_lsoda.I_stim, "k:", lw=1.2, label="I_stim")
    ax_d.set_xlabel("Time (ms)")
    ax_d.set_ylabel("Current Density (µA/cm²)")
    ax_d.set_title("(d) Transmembrane Currents")
    ax_d.legend(loc="upper right", fontsize=8)

    # (e) Phase plane: V against n
    ax_e = axes[1, 1]
    ax_e.plot(tr_lsoda.V, tr_lsoda.n, "purple", lw=1.5)
    ax_e.set_xlabel("Membrane Potential V (mV)")
    ax_e.set_ylabel("Potassium Gate n")
    ax_e.set_title("(e) Phase Plane (V vs n)")

    # (f) RK4 vs LSODA solver voltage difference
    ax_f = axes[1, 2]
    # Interpolate RK4 trajectory onto the LSODA evaluation time grid
    v_rk4_interp = np.interp(tr_lsoda.t, tr_rk4.t, tr_rk4.V)
    v_diff = v_rk4_interp - tr_lsoda.V
    ax_f.plot(tr_lsoda.t, v_diff, "k-", lw=1.2)
    ax_f.set_xlabel("Time (ms)")
    ax_f.set_ylabel("V_RK4 - V_LSODA (mV)")
    ax_f.set_title("(f) Solver Difference (RK4 vs LSODA)")

    fig_path = out_dir / "exp1_action_potential.png"
    plt.savefig(fig_path, dpi=150)
    if args.show:
        plt.show()
    plt.close(fig)

    # 4. Save JSON summary
    summary_data = {
        "amplitude_uA_cm2": float(args.amplitude),
        "rest_mV": float(rest_v),
        "peak_mV": float(peak_v),
        "first_crossing_ms": float(first_cross),
        "half_width_ms": float(width),
        "ahp_mV": float(ahp_val),
        "peak_gNa_mS_cm2": float(peak_gna),
        "peak_gK_mS_cm2": float(peak_gk),
        "n_spikes": int(n_spikes),
        "rk4_vs_lsoda_first_crossing_diff_ms": float(diff_crossing),
    }

    json_path = out_dir / "exp1_action_potential_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 5. Print formatted aligned table
    print("=" * 60)
    print(" Experiment 1: Single Action Potential Characterisation")
    print("=" * 60)
    for key, value in summary_data.items():
        if isinstance(value, float):
            print(f" {key:<38} : {value:10.4f}")
        else:
            print(f" {key:<38} : {value:10d}")
    print("=" * 60)


if __name__ == "__main__":
    main()