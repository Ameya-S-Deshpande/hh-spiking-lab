"""How does inhibition change firing rate, synchrony, and rhythm in a spiking network?

Simulates a 50-neuron sparse excitatory-inhibitory network of Hodgkin-Huxley neurons,
evaluates network-level synchrony and power spectra, and sweeps inhibitory synaptic feedback.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hhlab.network import (
    NetworkConfig,
    peak_frequency,
    population_rates,
    population_spectrum,
    simulate_network,
    synchrony_chi,
)

# This is a toy network with identical Hodgkin-Huxley cells, random connectivity,
# and instantaneous synapses, so rhythm frequencies and synchrony patterns are qualitative.


def _clean(val):
    """Recursively convert float NaN and Inf to Python None for JSON export."""
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
    """Run E-I network simulation, execute feedback weight sweep, and save figures."""
    parser = argparse.ArgumentParser(description="Evaluate E-I network dynamics and synchrony.")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory")
    parser.add_argument("--show", action="store_true", help="Display interactive figure")
    parser.add_argument("--t-max", type=float, default=500.0, help="Simulation duration (ms)")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Main network run with default parameters (40 E + 10 I, seed 1)
    print(f"Simulating default 50-neuron E-I network for {args.t_max:.1f} ms...")
    t0 = time.perf_counter()
    config = NetworkConfig()
    result = simulate_network(config, t_max=args.t_max)
    runtime_s = time.perf_counter() - t0
    print(f"  Simulation completed in {runtime_s:.2f} s")

    exc_rate, inh_rate = population_rates(result, t_start=100.0)
    chi = synchrony_chi(result, t_start=100.0)
    freqs, power = population_spectrum(result, population="exc", t_start=100.0)
    peak_f = peak_frequency(freqs, power)
    n_spikes = len(result.spike_times)

    print(f"  Excitatory rate : {exc_rate:.2f} Hz")
    print(f"  Inhibitory rate : {inh_rate:.2f} Hz")
    print(f"  Synchrony chi   : {chi:.4f}")
    print(f"  Peak frequency  : {peak_f:.2f} Hz")

    # 2. Sweep inhibitory-to-excitatory weight w_ie
    weights = [0.0, 0.1, 0.2, 0.35, 0.5]
    sweep_results = []
    print("\nRunning w_ie parameter sweep...")
    for w in weights:
        t_w0 = time.perf_counter()
        cfg_w = dataclasses.replace(config, w_ie=w)
        res_w = simulate_network(cfg_w, t_max=args.t_max)
        dur_w = time.perf_counter() - t_w0

        er_w, ir_w = population_rates(res_w, t_start=100.0)
        chi_w = synchrony_chi(res_w, t_start=100.0)
        freqs_w, power_w = population_spectrum(res_w, population="exc", t_start=100.0)
        peak_w = peak_frequency(freqs_w, power_w)

        print(
            f"  w_ie = {w:4.2f} -> E: {er_w:5.1f} Hz, I: {ir_w:5.1f} Hz, "
            f"chi: {chi_w:.3f}, peak: {peak_w:.1f} Hz ({dur_w:.2f} s)"
        )
        sweep_results.append({
            "w_ie": float(w),
            "exc_rate_hz": float(er_w),
            "inh_rate_hz": float(ir_w),
            "chi": float(chi_w),
            "peak_frequency_hz": float(peak_w) if np.isfinite(peak_w) else float("nan"),
        })

    # 3. Diagnostic Figure (2 x 3 panels)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    # (a) Spike raster plot
    ax_a = axes[0, 0]
    exc_spk = result.spike_ids < config.n_exc
    inh_spk = result.spike_ids >= config.n_exc
    ax_a.scatter(
        result.spike_times[exc_spk],
        result.spike_ids[exc_spk],
        s=3,
        c="navy",
        alpha=0.7,
        label=f"Excitatory (0-{config.n_exc - 1})",
    )
    ax_a.scatter(
        result.spike_times[inh_spk],
        result.spike_ids[inh_spk],
        s=3,
        c="crimson",
        alpha=0.7,
        label=f"Inhibitory ({config.n_exc}-{config.n_exc + config.n_inh - 1})",
    )
    ax_a.set_xlabel("Time (ms)")
    ax_a.set_ylabel("Neuron Index")
    ax_a.set_title("(a) Spike Raster Plot")
    ax_a.legend(loc="upper right", fontsize=8)

    # (b) Population rate histograms (2 ms bins)
    ax_b = axes[0, 1]
    bin_w = 2.0
    bins = np.arange(0.0, result.t_max + bin_w, bin_w)
    counts_exc, _ = np.histogram(result.spike_times[exc_spk], bins=bins)
    counts_inh, _ = np.histogram(result.spike_times[inh_spk], bins=bins)
    rate_exc_t = counts_exc / (config.n_exc * (bin_w / 1000.0))
    rate_inh_t = counts_inh / (config.n_inh * (bin_w / 1000.0))
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    ax_b.plot(bin_centers, rate_exc_t, "navy", lw=1.2, label="Excitatory")
    ax_b.plot(bin_centers, rate_inh_t, "crimson", lw=1.2, label="Inhibitory")
    ax_b.set_xlabel("Time (ms)")
    ax_b.set_ylabel("Population Rate (Hz)")
    ax_b.set_title("(b) Population Firing Rates (2 ms bins)")
    ax_b.legend(loc="upper right", fontsize=8)

    # (c) Sample voltage traces over 100 ms window
    ax_c = axes[0, 2]
    win_mask = (result.t_rec >= 200.0) & (result.t_rec <= 300.0)
    t_win = result.t_rec[win_mask]
    v_win = result.v_rec[win_mask]
    for i in range(min(3, config.n_exc)):
        ax_c.plot(t_win, v_win[:, i], lw=1.0, label=f"E neuron {i}")
    ax_c.plot(t_win, v_win[:, config.n_exc], "crimson", linestyle="--", lw=1.2, label="I neuron")
    ax_c.set_xlabel("Time (ms)")
    ax_c.set_ylabel("Membrane Potential V (mV)")
    ax_c.set_title("(c) Membrane Potentials (200-300 ms)")
    ax_c.legend(loc="upper right", fontsize=8)

    # (d) Network mean membrane potential
    ax_d = axes[1, 0]
    v_mean = np.mean(result.v_rec, axis=1)
    ax_d.plot(result.t_rec, v_mean, "k-", lw=1.1)
    ax_d.set_xlabel("Time (ms)")
    ax_d.set_ylabel("Population Mean <V> (mV)")
    ax_d.set_title("(d) Mean Membrane Potential (Synchrony Signal)")

    # (e) Excitatory population power spectrum
    ax_e = axes[1, 1]
    mask_f = freqs <= 150.0
    ax_e.plot(freqs[mask_f], power[mask_f], "navy", lw=1.3, label="E-spectrum")
    if np.isfinite(peak_f) and peak_f <= 150.0:
        pk_idx = int(np.argmin(np.abs(freqs - peak_f)))
        ax_e.plot(peak_f, power[pk_idx], "ro", ms=6, label=f"Peak: {peak_f:.1f} Hz")
    ax_e.set_xlabel("Frequency (Hz)")
    ax_e.set_ylabel("Power")
    ax_e.set_title("(e) Excitatory Power Spectrum")
    ax_e.legend(loc="upper right", fontsize=8)

    # (f) Feedback weight sweep vs rate and synchrony
    ax_f = axes[1, 2]
    sw_w = [d["w_ie"] for d in sweep_results]
    sw_er = [d["exc_rate_hz"] for d in sweep_results]
    sw_chi = [d["chi"] for d in sweep_results]

    ax_f.plot(sw_w, sw_er, "bo-", lw=1.4, label="Excitatory Rate")
    ax_f.set_xlabel("Inhibitory Weight w_ie (mS/cm²)")
    ax_f.set_ylabel("Excitatory Rate (Hz)", color="b")
    ax_f.tick_params(axis="y", labelcolor="b")
    ax_f.set_title("(f) Inhibition Sweep: Rate and Synchrony")

    ax_f_twin = ax_f.twinx()
    ax_f_twin.plot(sw_w, sw_chi, "ms--", lw=1.4, label="Synchrony χ")
    ax_f_twin.set_ylabel("Synchrony Parameter χ", color="m")
    ax_f_twin.tick_params(axis="y", labelcolor="m")

    fig_path = out_dir / "exp4_ei_network.png"
    plt.savefig(fig_path, dpi=150)
    if args.show:
        plt.show()
    plt.close(fig)

    # 4. Save JSON summary
    summary_data = {
        "t_max_ms": float(args.t_max),
        "exc_rate_hz": float(exc_rate),
        "inh_rate_hz": float(inh_rate),
        "chi": float(chi),
        "peak_frequency_hz": float(peak_f) if np.isfinite(peak_f) else None,
        "n_spikes": int(n_spikes),
        "runtime_s": float(runtime_s),
        "sweep": sweep_results,
    }

    clean_summary = _clean(summary_data)
    json_path = out_dir / "exp4_ei_network_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_summary, f, indent=2)

    # 5. Print sweep table and monotonicity statement
    print("\n" + "=" * 65)
    print(" Experiment 4: Excitatory-Inhibitory Network Sweep Summary")
    print("=" * 65)
    header = f" {'w_ie':<8} {'Exc Rate (Hz)':<15} {'Inh Rate (Hz)':<15}"
    print(header + f" {'chi':<10} {'Peak Freq (Hz)':<15}")
    print("-" * 65)
    for row in clean_summary["sweep"]:
        peak = row["peak_frequency_hz"]
        pk_str = f"{peak:10.2f}" if peak is not None else "      None"
        print(
            f" {row['w_ie']:<8.2f} {row['exc_rate_hz']:<15.2f} "
            f"{row['inh_rate_hz']:<15.2f} {row['chi']:<10.4f} {pk_str}"
        )
    print("=" * 65)

    rates = [r["exc_rate_hz"] for r in clean_summary["sweep"]]
    is_decreasing = all(rates[i] > rates[i + 1] for i in range(len(rates) - 1))
    if is_decreasing:
        print("The excitatory rate decreased monotonically with increasing w_ie.")
    else:
        print("The excitatory rate did not decrease strictly monotonically with increasing w_ie.")


if __name__ == "__main__":
    main()