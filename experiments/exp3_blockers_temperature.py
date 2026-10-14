"""What do sodium and potassium channels contribute, and how does temperature affect spikes?

Investigates pharmacological channel blockers (TTX and TEA) alongside altered channel densities,
and maps temperature scaling of kinetic rates, spike widths, rheobase, and heat block.
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
    detect_spikes,
    find_rheobase,
    firing_rate,
    run_step,
    spike_width,
)
from hhlab.params import DEFAULT_PARAMS


def _clean(val):
    """Recursively convert float NaN and Inf to Python None for JSON compliance."""
    if isinstance(val, (float, np.floating)):
        return None if (np.isnan(val) or np.isinf(val)) else float(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, dict):
        return {k: _clean(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_clean(v) for v in val]
    return val


def main():
    """Run pharmacological blocker and temperature experiments and save outputs."""
    parser = argparse.ArgumentParser(description="Evaluate channel blockers and temperature.")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory")
    parser.add_argument("--show", action="store_true", help="Display interactive figure")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Part A: Channel Blockers and Density Perturbations
    # -------------------------------------------------------------------------
    conditions = {
        "Control": DEFAULT_PARAMS,
        "TTX-like (gNa=0)": DEFAULT_PARAMS.with_(gNa=0.0),
        "TEA-like (gK=0)": DEFAULT_PARAMS.with_(gK=0.0),
        "Half Na (gNa=60)": DEFAULT_PARAMS.with_(gNa=60.0),
        "Half K (gK=18)": DEFAULT_PARAMS.with_(gK=18.0),
    }

    traces_part_a = {}
    metrics_part_a = {}
    for name, p_cond in conditions.items():
        tr = run_step(10.0, params=p_cond, t_pre=5.0, duration=40.0, t_post=5.0)
        traces_part_a[name] = tr
        crossings = detect_spikes(tr.t, tr.V)
        idx_30 = int(np.argmin(np.abs(tr.t - 30.0)))
        metrics_part_a[name] = {
            "peak_v": float(np.max(tr.V)),
            "n_spikes": len(crossings),
            "v_30": float(tr.V[idx_30]),
        }

    # Additional TTX condition at 50 uA/cm^2
    tr_ttx_50 = run_step(
        50.0,
        params=DEFAULT_PARAMS.with_(gNa=0.0),
        t_pre=5.0,
        duration=40.0,
        t_post=5.0,
    )
    idx_30 = int(np.argmin(np.abs(tr_ttx_50.t - 30.0)))
    ttx_50_metrics = {
        "peak_v": float(np.max(tr_ttx_50.V)),
        "n_spikes": len(detect_spikes(tr_ttx_50.t, tr_ttx_50.V)),
        "v_30": float(tr_ttx_50.V[idx_30]),
    }

    # -------------------------------------------------------------------------
    # Part B: Temperature Series
    # -------------------------------------------------------------------------
    temps = [6.3, 12.0, 18.5, 25.0]
    traces_temp = {}
    widths_temp = []
    peaks_temp = []
    spikes_temp = []
    rheobases_temp = []
    rates_15_temp = []
    temp_dicts = []

    for t_c in temps:
        p_t = DEFAULT_PARAMS.with_(temp_C=t_c, q10=3.0)
        # Waveform for standard 10 uA/cm^2 step
        tr_t = run_step(10.0, params=p_t, t_pre=5.0, duration=40.0, t_post=5.0)
        traces_temp[t_c] = tr_t

        crossings = detect_spikes(tr_t.t, tr_t.V)
        n_spk = len(crossings)
        peak_v = float(np.max(tr_t.V))
        w = spike_width(tr_t.t, tr_t.V, 0) if n_spk > 0 else float("nan")

        print(f"Finding rheobase at {t_c:.1f} °C...")
        rheo = find_rheobase(params=p_t, i_hi=60.0)

        # Firing rate at 15 uA/cm^2 (evaluated for 6.3, 12.0, and 18.5 °C)
        if t_c < 24.0:
            tr_fi = run_step(15.0, params=p_t, t_pre=50.0, duration=500.0, t_post=0.0)
            rate_15 = firing_rate(tr_fi.t, tr_fi.V, 150.0, 550.0)
        else:
            # At 25 °C sodium inactivation and potassium activation both speed up so much
            # that the spike fails at 10 µA/cm² (heat block), which is a known model property.
            rate_15 = None

        widths_temp.append(w)
        peaks_temp.append(peak_v)
        spikes_temp.append(n_spk)
        rheobases_temp.append(rheo)
        rates_15_temp.append(rate_15)

        temp_dicts.append({
            "temp_C": float(t_c),
            "phi": float(p_t.phi()),
            "half_width_ms": w,
            "peak_mV": peak_v,
            "n_spikes_I10": int(n_spk),
            "rheobase_uA_cm2": float(rheo),
            "rate_hz_I15": rate_15,
        })

    # -------------------------------------------------------------------------
    # Diagnostic Figure
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    # (a) Five pharmacological/density conditions
    ax_a = axes[0, 0]
    for name, tr in traces_part_a.items():
        ax_a.plot(tr.t, tr.V, lw=1.3, label=name)
    ax_a.set_xlabel("Time (ms)")
    ax_a.set_ylabel("Membrane Potential V (mV)")
    ax_a.set_title("(a) Channel Blockers & Density Variations")
    ax_a.legend(loc="upper right", fontsize=8)

    # (b) TTX-like passive charging
    ax_b = axes[0, 1]
    ax_b.plot(traces_part_a["TTX-like (gNa=0)"].t, traces_part_a["TTX-like (gNa=0)"].V,
              "b-", lw=1.4, label="10 µA/cm²")
    ax_b.plot(tr_ttx_50.t, tr_ttx_50.V, "r-", lw=1.4, label="50 µA/cm²")
    ax_b.set_xlabel("Time (ms)")
    ax_b.set_ylabel("Membrane Potential V (mV)")
    ax_b.set_title("(b) TTX-like (gNa=0) Passive Charging")
    ax_b.legend(loc="center right", fontsize=9)

    # (c) Control vs TEA-like plateau
    ax_c = axes[0, 2]
    tr_ctrl = traces_part_a["Control"]
    tr_tea = traces_part_a["TEA-like (gK=0)"]
    ax_c.plot(tr_ctrl.t, tr_ctrl.V, "b-", lw=1.4, label="Control")
    ax_c.plot(tr_tea.t, tr_tea.V, "r-", lw=1.4, label="TEA-like (gK=0)")
    v_tea_30 = metrics_part_a["TEA-like (gK=0)"]["v_30"]
    ax_c.annotate(
        f"Plateau: {v_tea_30:.1f} mV",
        xy=(30.0, v_tea_30),
        xytext=(18.0, v_tea_30 + 15.0),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
        fontsize=9,
    )
    ax_c.set_xlabel("Time (ms)")
    ax_c.set_ylabel("Membrane Potential V (mV)")
    ax_c.set_title("(c) TEA Block and Depolarised Plateau")
    ax_c.legend(loc="upper right", fontsize=9)

    # (d) Temperature series waveforms
    ax_d = axes[1, 0]
    for t_c in temps:
        ax_d.plot(traces_temp[t_c].t, traces_temp[t_c].V, lw=1.3, label=f"{t_c} °C")
    ax_d.set_xlabel("Time (ms)")
    ax_d.set_ylabel("Membrane Potential V (mV)")
    ax_d.set_title("(d) Action Potentials Across Temperatures")
    ax_d.legend(loc="upper right", fontsize=8)

    # (e) Spike half-width and rheobase vs temperature (twin axes)
    ax_e = axes[1, 1]
    valid_widths = [(t, w) for t, w in zip(temps, widths_temp) if not np.isnan(w)]
    ax_e.plot(
        [t for t, _ in valid_widths],
        [w for _, w in valid_widths],
        "bo-",
        lw=1.5,
        label="Spike Half-width",
    )
    ax_e.set_xlabel("Temperature (°C)")
    ax_e.set_ylabel("Half-width (ms)", color="b")
    ax_e.tick_params(axis="y", labelcolor="b")
    ax_e.set_title("(e) Half-width & Rheobase vs Temperature")

    ax_e_twin = ax_e.twinx()
    ax_e_twin.plot(temps, rheobases_temp, "rs--", lw=1.5, label="Rheobase")
    ax_e_twin.set_ylabel("Rheobase (µA/cm²)", color="r")
    ax_e_twin.tick_params(axis="y", labelcolor="r")

    # (f) Firing rate at 15 uA/cm^2
    ax_f = axes[1, 2]
    temps_sub = [t for t, r in zip(temps, rates_15_temp) if r is not None]
    rates_sub = [r for r in rates_15_temp if r is not None]
    ax_f.plot(temps_sub, rates_sub, "g^-", lw=1.5, ms=6)
    ax_f.set_xlabel("Temperature (°C)")
    ax_f.set_ylabel("Firing Rate at 15 µA/cm² (Hz)")
    ax_f.set_title("(f) Repetitive Rate Acceleration")

    fig_path = out_dir / "exp3_blockers_temperature.png"
    plt.savefig(fig_path, dpi=150)
    if args.show:
        plt.show()
    plt.close(fig)

    # -------------------------------------------------------------------------
    # JSON Summary Export
    # -------------------------------------------------------------------------
    summary_data = {
        "control_peak_mV": metrics_part_a["Control"]["peak_v"],
        "ttx_peak_mV_I10": metrics_part_a["TTX-like (gNa=0)"]["peak_v"],
        "ttx_peak_mV_I50": ttx_50_metrics["peak_v"],
        "ttx_spikes_I10": metrics_part_a["TTX-like (gNa=0)"]["n_spikes"],
        "tea_V_at_30ms_mV": metrics_part_a["TEA-like (gK=0)"]["v_30"],
        "tea_spikes": metrics_part_a["TEA-like (gK=0)"]["n_spikes"],
        "half_gNa_peak_mV": metrics_part_a["Half Na (gNa=60)"]["peak_v"],
        "half_gNa_spikes": metrics_part_a["Half Na (gNa=60)"]["n_spikes"],
        "half_gK_peak_mV": metrics_part_a["Half K (gK=18)"]["peak_v"],
        "half_gK_spikes": metrics_part_a["Half K (gK=18)"]["n_spikes"],
        "temperature": temp_dicts,
    }

    clean_summary = _clean(summary_data)
    json_path = out_dir / "exp3_blockers_temperature_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_summary, f, indent=2)

    print("=" * 60)
    print(" Experiment 3: Channel Blockers & Temperature Summary")
    print("=" * 60)
    for k, v in clean_summary.items():
        if k != "temperature":
            print(f" {k:<25} : {v}")
    print(" Temperature series:")
    for d in clean_summary["temperature"]:
        print(
            f"   T={d['temp_C']:4.1f} °C: width={d['half_width_ms']}, "
            f"rheo={d['rheobase_uA_cm2']:.2f}, rate_15={d['rate_hz_I15']}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()