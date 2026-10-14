"""Vectorised simulation and analysis of excitatory-inhibitory Hodgkin-Huxley networks.

Advances all neurons simultaneously using NumPy array operations and fixed-step RK4.
A time step of dt = 0.02 ms provides sufficient numerical convergence without the
computational overhead of adaptive solvers, running a 500 ms 50-neuron simulation in ~6 s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hhlab.channels import alpha_beta, ionic_currents, steady_state
from hhlab.params import DEFAULT_PARAMS, HHParams


@dataclass(frozen=True)
class NetworkConfig:
    """Configuration parameters for a sparsely connected E-I Hodgkin-Huxley network.

    Parameters
    ----------
    n_exc : int
        Number of excitatory neurons (default 40).
    n_inh : int
        Number of inhibitory neurons (default 10).
    p_connect : float
        Connection probability between directed neuron pairs (default 0.2).
    w_ee : float
        E -> E synaptic weight in mS/cm^2 (default 0.02).
    w_ei : float
        E -> I synaptic weight in mS/cm^2 (default 0.10).
    w_ie : float
        I -> E synaptic weight in mS/cm^2 (default 0.20).
    w_ii : float
        I -> I synaptic weight in mS/cm^2 (default 0.10).
    tau_exc : float
        Excitatory synaptic decay time constant in ms (default 2.0).
    tau_inh : float
        Inhibitory synaptic decay time constant in ms (default 8.0).
    e_exc : float
        Excitatory synaptic reversal potential in mV (default 0.0).
    e_inh : float
        Inhibitory synaptic reversal potential in mV (default -75.0).
    drive_exc_mean : float
        Mean constant input current for excitatory population in uA/cm^2 (default 8.0).
    drive_exc_std : float
        Standard deviation of constant excitatory input current in uA/cm^2 (default 0.5).
    drive_inh_mean : float
        Mean constant input current for inhibitory population in uA/cm^2 (default 0.0).
    drive_inh_std : float
        Standard deviation of constant inhibitory input current in uA/cm^2 (default 0.0).
    noise_std : float
        White noise current standard deviation in uA/cm^2 (default 0.0).
    seed : int
        Random seed for connectivity and initial conditions (default 1).
    dt : float
        Integration time step in ms (default 0.02).
    params : HHParams
        Single-cell Hodgkin-Huxley parameters (default DEFAULT_PARAMS).
    """

    n_exc: int = 40
    n_inh: int = 10
    p_connect: float = 0.2
    w_ee: float = 0.02
    w_ei: float = 0.10
    w_ie: float = 0.20
    w_ii: float = 0.10
    tau_exc: float = 2.0
    tau_inh: float = 8.0
    e_exc: float = 0.0
    e_inh: float = -75.0
    drive_exc_mean: float = 8.0
    drive_exc_std: float = 0.5
    drive_inh_mean: float = 0.0
    drive_inh_std: float = 0.0
    noise_std: float = 0.0
    seed: int = 1
    dt: float = 0.02
    params: HHParams = DEFAULT_PARAMS


@dataclass
class NetworkResult:
    """Recorded output and raster data from a network simulation.

    Parameters
    ----------
    config : NetworkConfig
        Configuration used for the run.
    t_max : float
        Simulation duration in ms.
    spike_times : np.ndarray
        Array of spike times in ms, sorted in ascending order.
    spike_ids : np.ndarray
        Array of spiking neuron indices corresponding to spike_times.
    t_rec : np.ndarray
        Subsampled recording time array of shape (n_rec,) in ms.
    v_rec : np.ndarray
        Membrane voltage traces of shape (n_rec, n_exc + n_inh) in mV.
    """

    config: NetworkConfig
    t_max: float
    spike_times: np.ndarray
    spike_ids: np.ndarray
    t_rec: np.ndarray
    v_rec: np.ndarray


def build_connectivity(config: NetworkConfig, rng: np.random.Generator) -> np.ndarray:
    """Build directed synaptic weight matrix W[post, pre] with zero diagonal.

    Parameters
    ----------
    config : NetworkConfig
        Network configuration.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Synaptic weight matrix of shape (N, N) in mS/cm^2.
    """
    n_exc = config.n_exc
    n_inh = config.n_inh
    n_total = n_exc + n_inh

    conn = rng.random((n_total, n_total)) < config.p_connect
    np.fill_diagonal(conn, False)

    w_mat = np.zeros((n_total, n_total), dtype=float)
    w_mat[:n_exc, :n_exc] = config.w_ee
    w_mat[n_exc:, :n_exc] = config.w_ei
    w_mat[:n_exc, n_exc:] = config.w_ie
    w_mat[n_exc:, n_exc:] = config.w_ii

    return w_mat * conn


def simulate_network(
    config: NetworkConfig,
    t_max: float = 500.0,
    record_every: int = 5,
) -> NetworkResult:
    """Simulate coupled E-I network using vectorised fourth-order Runge-Kutta.

    Parameters
    ----------
    config : NetworkConfig
        Network parameters.
    t_max : float, optional
        Simulation duration in ms (default 500.0).
    record_every : int, optional
        Subsampling interval in time steps (default 5).

    Returns
    -------
    NetworkResult
        Spike trains and subsampled membrane potential traces.
    """
    rng = np.random.default_rng(config.seed)
    w_mat = build_connectivity(config, rng)

    n_exc = config.n_exc
    n_inh = config.n_inh
    n_total = n_exc + n_inh

    # Random draws order: connectivity -> constant drives -> initial voltages
    drive = np.concatenate([
        rng.normal(config.drive_exc_mean, config.drive_exc_std, n_exc),
        rng.normal(config.drive_inh_mean, config.drive_inh_std, n_inh),
    ])

    v_state = -65.0 + rng.uniform(0.0, 5.0, n_total)
    m_inf, h_inf, n_inf = steady_state(-65.0)
    m_state = np.full(n_total, float(m_inf), dtype=float)
    h_state = np.full(n_total, float(h_inf), dtype=float)
    n_state = np.full(n_total, float(n_inf), dtype=float)

    g_e = np.zeros(n_total, dtype=float)
    g_i = np.zeros(n_total, dtype=float)

    dt = config.dt
    half_dt = 0.5 * dt
    n_steps = int(round(t_max / dt))

    decay_e = np.exp(-dt / config.tau_exc)
    decay_i = np.exp(-dt / config.tau_inh)
    phi = config.params.phi()
    c_m = config.params.C

    # Setup recording arrays
    recorded_steps = list(range(0, n_steps + 1, record_every))
    n_rec = len(recorded_steps)
    t_rec = np.array([s * dt for s in recorded_steps], dtype=float)
    v_rec = np.zeros((n_rec, n_total), dtype=float)
    v_rec[0] = v_state

    spikes_list: list[tuple[float, int]] = []
    rec_idx = 1

    for step_idx in range(n_steps):
        t_curr = step_idx * dt
        noise = (
            rng.standard_normal(n_total) * config.noise_std
            if config.noise_std > 0.0
            else 0.0
        )

        def _derivatives(v_s, m_s, h_s, n_s):
            i_syn = -g_e * (v_s - config.e_exc) - g_i * (v_s - config.e_inh)
            i_total = drive + noise + i_syn
            i_na, i_k, i_l = ionic_currents(v_s, m_s, h_s, n_s, config.params)
            dv = (i_total - i_na - i_k - i_l) / c_m
            am, bm, ah, bh, an, bn = alpha_beta(v_s)
            dm = phi * (am * (1.0 - m_s) - bm * m_s)
            dh = phi * (ah * (1.0 - h_s) - bh * h_s)
            dn = phi * (an * (1.0 - n_s) - bn * n_s)
            return dv, dm, dh, dn

        # Four stages of classical RK4
        dv1, dm1, dh1, dn1 = _derivatives(v_state, m_state, h_state, n_state)
        dv2, dm2, dh2, dn2 = _derivatives(
            v_state + half_dt * dv1,
            m_state + half_dt * dm1,
            h_state + half_dt * dh1,
            n_state + half_dt * dn1,
        )
        dv3, dm3, dh3, dn3 = _derivatives(
            v_state + half_dt * dv2,
            m_state + half_dt * dm2,
            h_state + half_dt * dh2,
            n_state + half_dt * dn2,
        )
        dv4, dm4, dh4, dn4 = _derivatives(
            v_state + dt * dv3,
            m_state + dt * dm3,
            h_state + dt * dh3,
            n_state + dt * dn3,
        )

        v_new = v_state + (dt / 6.0) * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4)
        m_new = m_state + (dt / 6.0) * (dm1 + 2.0 * dm2 + 2.0 * dm3 + dm4)
        h_new = h_state + (dt / 6.0) * (dh1 + 2.0 * dh2 + 2.0 * dh3 + dh4)
        n_new = n_state + (dt / 6.0) * (dn1 + 2.0 * dn2 + 2.0 * dn3 + dn4)

        # Detect threshold upward crossings: V_prev < 0 <= V_new
        spiked_mask = (v_state < 0.0) & (v_new >= 0.0)
        spiked_indices = np.where(spiked_mask)[0]

        for s_id in spiked_indices:
            denom = v_new[s_id] - v_state[s_id]
            frac = (0.0 - v_state[s_id]) / denom if denom != 0.0 else 0.0
            spk_t = t_curr + frac * dt
            spikes_list.append((float(spk_t), int(s_id)))

        # Decay synaptic conductances
        g_e *= decay_e
        g_i *= decay_i

        # Distribute synaptic inputs from presynaptic spikes
        spiked_exc = spiked_indices[spiked_indices < n_exc]
        spiked_inh = spiked_indices[spiked_indices >= n_exc]

        if len(spiked_exc) > 0:
            g_e += np.sum(w_mat[:, spiked_exc], axis=1)
        if len(spiked_inh) > 0:
            g_i += np.sum(w_mat[:, spiked_inh], axis=1)

        v_state = v_new
        m_state = m_new
        h_state = h_new
        n_state = n_new

        step_next = step_idx + 1
        if step_next % record_every == 0 and rec_idx < n_rec:
            v_rec[rec_idx] = v_state
            rec_idx += 1

    if spikes_list:
        spikes_list.sort(key=lambda item: item[0])
        spike_times = np.array([item[0] for item in spikes_list], dtype=float)
        spike_ids = np.array([item[1] for item in spikes_list], dtype=int)
    else:
        spike_times = np.array([], dtype=float)
        spike_ids = np.array([], dtype=int)

    return NetworkResult(
        config=config,
        t_max=float(t_max),
        spike_times=spike_times,
        spike_ids=spike_ids,
        t_rec=t_rec,
        v_rec=v_rec,
    )


def population_rates(
    result: NetworkResult,
    t_start: float = 100.0,
) -> tuple[float, float]:
    """Calculate average firing rate per neuron in excitatory and inhibitory populations.

    Parameters
    ----------
    result : NetworkResult
        Simulation outcome.
    t_start : float, optional
        Transient cutoff time in ms (default 100.0).

    Returns
    -------
    tuple of (float, float)
        (exc_rate_hz, inh_rate_hz). Returns NaN if population size is 0.
    """
    span_s = (result.t_max - t_start) / 1000.0
    if span_s <= 0.0:
        return 0.0, 0.0

    mask = result.spike_times >= t_start
    kept_ids = result.spike_ids[mask]

    n_exc = result.config.n_exc
    n_inh = result.config.n_inh

    if n_exc > 0:
        exc_count = int(np.sum(kept_ids < n_exc))
        exc_rate = float(exc_count / (n_exc * span_s))
    else:
        exc_rate = float(np.nan)

    if n_inh > 0:
        inh_count = int(np.sum(kept_ids >= n_exc))
        inh_rate = float(inh_count / (n_inh * span_s))
    else:
        inh_rate = float(np.nan)

    return exc_rate, inh_rate


def synchrony_chi(
    result: NetworkResult,
    t_start: float = 100.0,
) -> float:
    """Calculate Golomb-Hansel synchrony parameter chi from membrane voltage traces.

    chi = sqrt(var_over_time(mean_over_neurons(V)) / mean_over_neurons(var_over_time(V_i))).
    Takes values in [0, 1]: exactly 1 corresponds to fully synchronous activity,
    while independent unsynchronised firing scales as approximately 1 / sqrt(N).

    Parameters
    ----------
    result : NetworkResult
        Simulation outcome.
    t_start : float, optional
        Analysis start time in ms (default 100.0).

    Returns
    -------
    float
        Dimensionless synchrony index chi, or NaN if denominator is zero.
    """
    mask = result.t_rec >= t_start
    v_sub = result.v_rec[mask]

    if v_sub.shape[0] < 2 or v_sub.shape[1] == 0:
        return float(np.nan)

    v_pop_mean = np.mean(v_sub, axis=1)
    var_pop = np.var(v_pop_mean, ddof=0)

    var_individual = np.var(v_sub, axis=0, ddof=0)
    mean_var_ind = np.mean(var_individual)

    if mean_var_ind <= 0.0 or np.isnan(mean_var_ind):
        return float(np.nan)

    chi2 = var_pop / mean_var_ind
    return float(np.sqrt(max(0.0, chi2)))


def population_spectrum(
    result: NetworkResult,
    population: str = "exc",
    t_start: float = 100.0,
    bin_ms: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute power spectrum of population spike count time series.

    Parameters
    ----------
    result : NetworkResult
        Simulation outcome.
    population : str, optional
        Target population: 'exc' or 'inh' (default 'exc').
    t_start : float, optional
        Transient window cutoff time in ms (default 100.0).
    bin_ms : float, optional
        Histogram bin width in ms (default 1.0).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (frequencies_hz, power).
    """
    if population not in ("exc", "inh"):
        raise ValueError(f"population must be 'exc' or 'inh', got {population}")

    n_exc = result.config.n_exc
    if population == "exc":
        pop_mask = result.spike_ids < n_exc
    else:
        pop_mask = result.spike_ids >= n_exc

    pop_spikes = result.spike_times[pop_mask]
    total_span = result.t_max - t_start
    n_bins = int(np.floor(total_span / bin_ms))

    if n_bins <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    bin_edges = t_start + np.arange(n_bins + 1) * bin_ms
    counts, _ = np.histogram(pop_spikes, bins=bin_edges)

    centered_counts = counts.astype(float) - np.mean(counts)
    power = np.abs(np.fft.rfft(centered_counts)) ** 2
    freqs = np.fft.rfftfreq(n_bins, d=bin_ms / 1000.0)

    return freqs, power


def peak_frequency(
    freqs: np.ndarray,
    power: np.ndarray,
    f_min: float = 5.0,
    f_max: float = 200.0,
) -> float:
    """Identify dominant oscillation frequency within a defined frequency band.

    Parameters
    ----------
    freqs : np.ndarray
        Array of frequencies in Hz.
    power : np.ndarray
        Array of spectral power estimates.
    f_min : float, optional
        Lower frequency cutoff in Hz (default 5.0).
    f_max : float, optional
        Upper frequency cutoff in Hz (default 200.0).

    Returns
    -------
    float
        Peak frequency in Hz, or NaN if no frequencies lie in the search range.
    """
    f_arr = np.asarray(freqs, dtype=float)
    p_arr = np.asarray(power, dtype=float)

    mask = (f_arr >= f_min) & (f_arr <= f_max)
    if not np.any(mask):
        return float(np.nan)

    sub_f = f_arr[mask]
    sub_p = p_arr[mask]
    peak_idx = int(np.argmax(sub_p))
    return float(sub_f[peak_idx])