"""Interactive Streamlit dashboard for Hodgkin-Huxley single-cell and network dynamics."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from hhlab.analysis import (
    afterhyperpolarisation,
    detect_spikes,
    spike_peak,
    spike_width,
)
from hhlab.network import (
    NetworkConfig,
    peak_frequency,
    population_rates,
    population_spectrum,
    simulate_network,
    synchrony_chi,
)
from hhlab.neuron import simulate
from hhlab.params import DEFAULT_PARAMS
from hhlab.stimuli import step

st.set_page_config(page_title="Hodgkin–Huxley Spiking Lab", layout="wide")
st.title("Hodgkin–Huxley Spiking Lab")
st.caption(
    "Educational model of a squid giant-axon neuron. Built with AI assistance "
    "(Claude and Gemini); the author reviewed and ran all code."
)


def _fmt(val: float | None, unit: str = "", decimals: int = 2) -> str:
    """Format numerical metrics with em-dash fallback for missing/NaN values."""
    if val is None or np.isnan(val):
        return "—"
    return f"{val:.{decimals}f} {unit}".strip()


@st.cache_data
def simulate_single_neuron(
    amplitude: float,
    start_time: float,
    duration: float,
    temp_c: float,
    gna_pct: float,
    gk_pct: float,
) -> dict[str, np.ndarray]:
    """Run and cache single-compartment simulation from scalar parameters."""
    t_max = start_time + duration + 10.0
    params = DEFAULT_PARAMS.with_(
        temp_C=temp_c,
        gNa=120.0 * (gna_pct / 100.0),
        gK=36.0 * (gk_pct / 100.0),
    )
    stim = step(amplitude, start_time, start_time + duration)
    tr = simulate(stim, t_max=t_max, dt=0.01, params=params)
    conds = tr.conductances()

    return {
        "t": tr.t,
        "V": tr.V,
        "m": tr.m,
        "h": tr.h,
        "n": tr.n,
        "I_stim": tr.I_stim,
        "gNa": conds["gNa"],
        "gK": conds["gK"],
    }


@st.cache_data
def simulate_network_cached(
    n_exc: int,
    n_inh: int,
    w_ie: float,
    drive_mean: float,
    t_max: float,
    seed: int,
) -> dict:
    """Run and cache network simulation from scalar parameters."""
    cfg = NetworkConfig(
        n_exc=n_exc,
        n_inh=n_inh,
        w_ie=w_ie,
        drive_exc_mean=drive_mean,
        seed=seed,
    )
    res = simulate_network(cfg, t_max=t_max)
    t_start = min(100.0, t_max / 2.0)

    exc_rate, inh_rate = population_rates(res, t_start=t_start)
    chi = synchrony_chi(res, t_start=t_start)
    freqs, power = population_spectrum(res, population="exc", t_start=t_start)
    peak_f = peak_frequency(freqs, power)

    return {
        "spike_times": res.spike_times,
        "spike_ids": res.spike_ids,
        "exc_rate": exc_rate,
        "inh_rate": inh_rate,
        "chi": chi,
        "freqs": freqs,
        "power": power,
        "peak_f": peak_f,
        "n_exc": n_exc,
        "n_inh": n_inh,
        "t_max": t_max,
    }


tab1, tab2 = st.tabs(["Single Neuron", "Network"])

# -----------------------------------------------------------------------------
# Tab 1: Single Neuron Simulation
# -----------------------------------------------------------------------------
with tab1:
    col1, col2, col3 = st.columns(3)
    amplitude = col1.slider(
        "Stimulus amplitude (µA/cm²)", min_value=0.0, max_value=100.0, value=10.0, step=0.5
    )
    start_time = col2.slider("Start time (ms)", min_value=0.0, max_value=50.0, value=5.0, step=1.0)
    duration = col3.slider("Duration (ms)", min_value=1.0, max_value=500.0, value=40.0, step=1.0)

    col4, col5, col6 = st.columns(3)
    temp_c = col4.slider(
        "Temperature (°C)", min_value=0.0, max_value=35.0, value=6.3, step=0.1
    )
    gna_scale = col5.slider(
        "Sodium conductance scale (%)", min_value=0, max_value=150, value=100, step=5
    )
    gk_scale = col6.slider(
        "Potassium conductance scale (%)", min_value=0, max_value=150, value=100, step=5
    )

    data = simulate_single_neuron(
        amplitude, start_time, duration, temp_c, float(gna_scale), float(gk_scale)
    )

    t_arr = data["t"]
    v_arr = data["V"]
    spikes = detect_spikes(t_arr, v_arr)
    n_spikes = len(spikes)
    peak_v = spike_peak(t_arr, v_arr, 0)
    width = spike_width(t_arr, v_arr, 0)
    ahp = afterhyperpolarisation(t_arr, v_arr, 0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Number of Spikes", n_spikes)
    m2.metric("Peak Voltage", _fmt(peak_v, "mV"))
    m3.metric("First-Spike Half-Width", _fmt(width, "ms"))
    m4.metric("Afterhyperpolarisation", _fmt(ahp, "mV"))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)

    # Panel 1: Membrane potential & stimulus
    ax1.plot(t_arr, v_arr, "b-", lw=1.5, label="V")
    ax1.set_ylabel("V (mV)", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.set_title("Membrane Potential & Stimulus Current")

    ax1_twin = ax1.twinx()
    ax1_twin.plot(t_arr, data["I_stim"], "k--", lw=1.0, alpha=0.6, label="I_stim")
    ax1_twin.set_ylabel("I_stim (µA/cm²)", color="k")
    ax1_twin.tick_params(axis="y", labelcolor="k")

    # Panel 2: Gating kinetics
    ax2.plot(t_arr, data["m"], "g-", lw=1.3, label="m (Na act)")
    ax2.plot(t_arr, data["h"], "r-", lw=1.3, label="h (Na inact)")
    ax2.plot(t_arr, data["n"], "b-", lw=1.3, label="n (K act)")
    ax2.set_ylabel("Gating State [0, 1]")
    ax2.set_title("Gating Variable Kinetics")
    ax2.legend(loc="center right", fontsize=8)

    # Panel 3: Conductances
    ax3.plot(t_arr, data["gNa"], "r-", lw=1.4, label="gNa")
    ax3.plot(t_arr, data["gK"], "b-", lw=1.4, label="gK")
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("Conductance (mS/cm²)")
    ax3.set_title("Membrane Conductances")
    ax3.legend(loc="upper right", fontsize=8)

    st.pyplot(fig)
    plt.close(fig)

    with st.expander("What am I looking at?"):
        st.write(
            "The upstroke is sodium activation. "
            "The downstroke is sodium inactivation plus potassium activation. "
            "The undershoot is potassium channels closing slowly."
        )

# -----------------------------------------------------------------------------
# Tab 2: Network Simulation
# -----------------------------------------------------------------------------
with tab2:
    ncol1, ncol2, ncol3 = st.columns(3)
    n_exc = ncol1.slider("Excitatory neurons", min_value=10, max_value=60, value=40, step=1)
    n_inh = ncol2.slider("Inhibitory neurons", min_value=0, max_value=20, value=10, step=1)
    w_ie = ncol3.slider(
        "I→E weight (mS/cm²)", min_value=0.0, max_value=1.0, value=0.2, step=0.05
    )

    ncol4, ncol5, ncol6 = st.columns(3)
    drive_mean = ncol4.slider(
        "Excitatory drive mean (µA/cm²)", min_value=4.0, max_value=12.0, value=8.0, step=0.5
    )
    sim_time = ncol5.slider(
        "Simulated time (ms)", min_value=100.0, max_value=500.0, value=300.0, step=50.0
    )
    seed_val = ncol6.number_input("Random seed", min_value=0, max_value=999999, value=1, step=1)

    if st.button("Run network"):
        with st.spinner("Simulating network dynamics..."):
            st.session_state["net_res"] = simulate_network_cached(
                n_exc, n_inh, float(w_ie), float(drive_mean), float(sim_time), int(seed_val)
            )

    if "net_res" in st.session_state:
        net = st.session_state["net_res"]

        nm1, nm2, nm3, nm4 = st.columns(4)
        nm1.metric("Excitatory Rate", _fmt(net["exc_rate"], "Hz", 1))
        nm2.metric("Inhibitory Rate", _fmt(net["inh_rate"], "Hz", 1))
        nm3.metric("Synchrony χ", _fmt(net["chi"], "", 3))
        nm4.metric("Spectrum Peak", _fmt(net["peak_f"], "Hz", 1))

        fig_net, (ax_r, ax_s) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

        # Raster plot
        if len(net["spike_times"]) > 0:
            exc_mask = net["spike_ids"] < net["n_exc"]
            inh_mask = net["spike_ids"] >= net["n_exc"]
            if np.any(exc_mask):
                ax_r.scatter(
                    net["spike_times"][exc_mask],
                    net["spike_ids"][exc_mask],
                    s=4,
                    c="navy",
                    alpha=0.7,
                    label="Excitatory",
                )
            if np.any(inh_mask):
                ax_r.scatter(
                    net["spike_times"][inh_mask],
                    net["spike_ids"][inh_mask],
                    s=4,
                    c="crimson",
                    alpha=0.7,
                    label="Inhibitory",
                )
            ax_r.legend(loc="upper right", fontsize=8)

        ax_r.set_xlim(0.0, net["t_max"])
        ax_r.set_ylim(-1, max(1, net["n_exc"] + net["n_inh"]))
        ax_r.set_xlabel("Time (ms)")
        ax_r.set_ylabel("Neuron Index")
        ax_r.set_title("Network Spike Raster")

        # Population Spectrum
        if len(net["freqs"]) > 0:
            mask_f = net["freqs"] <= 150.0
            ax_s.plot(net["freqs"][mask_f], net["power"][mask_f], "navy", lw=1.3)
            if net["peak_f"] is not None and np.isfinite(net["peak_f"]):
                pk_idx = int(np.argmin(np.abs(net["freqs"] - net["peak_f"])))
                ax_s.plot(
                    net["peak_f"],
                    net["power"][pk_idx],
                    "ro",
                    label=f"Peak: {net['peak_f']:.1f} Hz",
                )
                ax_s.legend(loc="upper right", fontsize=8)

        ax_s.set_xlabel("Frequency (Hz)")
        ax_s.set_ylabel("Power")
        ax_s.set_title("Excitatory Spike Count Spectrum")

        st.pyplot(fig_net)
        plt.close(fig_net)

        st.caption("Toy network, identical cells, random connectivity.")