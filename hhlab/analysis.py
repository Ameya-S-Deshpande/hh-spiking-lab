"""Analysis tools for action potentials, firing rates, excitability, and depolarisation block.

Provides peak detection, spike shape metrics, f-I curve characterisation,
and bisection searches for single-spike and repetitive firing thresholds.
"""

from __future__ import annotations

import numpy as np

from hhlab.neuron import Trace, simulate
from hhlab.params import DEFAULT_PARAMS, HHParams
from hhlab.stimuli import step


_DEFAULT_MIN_ISI_MS = 2.0


def _upward_crossings(
    t: np.ndarray,
    V: np.ndarray,
    threshold: float,
    min_isi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sample indices, interpolated times) of kept upward threshold crossings.

    A crossing is a sample pair with V[i] < threshold <= V[i+1]. The crossing time is found by
    linear interpolation. A crossing closer than ``min_isi`` ms to the previously kept one is
    dropped, which prevents double counting on broad or noisy peaks.
    """
    t_arr = np.asarray(t, dtype=float)
    v_arr = np.asarray(V, dtype=float)
    if v_arr.size < 2:
        return np.array([], dtype=int), np.array([], dtype=float)
    idx = np.flatnonzero((v_arr[:-1] < threshold) & (v_arr[1:] >= threshold))
    if idx.size == 0:
        return idx, np.array([], dtype=float)
    dv = v_arr[idx + 1] - v_arr[idx]
    frac = np.where(dv != 0.0, (threshold - v_arr[idx]) / np.where(dv != 0.0, dv, 1.0), 0.0)
    times = t_arr[idx] + frac * (t_arr[idx + 1] - t_arr[idx])
    kept_i: list[int] = []
    kept_t: list[float] = []
    for i, tc in zip(idx.tolist(), times.tolist()):
        if not kept_t or tc - kept_t[-1] >= min_isi:
            kept_i.append(i)
            kept_t.append(tc)
    return np.array(kept_i, dtype=int), np.array(kept_t, dtype=float)


def detect_spikes(
    t: np.ndarray,
    V: np.ndarray,
    threshold: float = 0.0,
    min_isi: float = 2.0,
) -> np.ndarray:
    """Detect upward threshold crossings with linear interpolation and refractory lockout.

    Parameters
    ----------
    t : np.ndarray
        Time array in ms.
    V : np.ndarray
        Voltage array in mV.
    threshold : float, optional
        Detection threshold in mV (default 0.0).
    min_isi : float, optional
        Minimum inter-spike interval lockout in ms (default 2.0).

    Returns
    -------
    np.ndarray
        Upward-crossing spike times in ms.
    """
    _, times = _upward_crossings(t, V, threshold, min_isi)
    return times


def _spike_window(
    t: np.ndarray,
    V: np.ndarray,
    spike_index: int,
    threshold: float = 0.0,
) -> tuple[int, int, int] | None:
    """Return bounding sample indices and peak index for a detected spike.

    Parameters
    ----------
    t : np.ndarray
        Time array in ms.
    V : np.ndarray
        Voltage array in mV.
    spike_index : int
        0-based index of the target spike.
    threshold : float, optional
        Upward crossing threshold in mV (default 0.0).

    Returns
    -------
    tuple of (int, int, int) or None
        (i_start, i_end, i_peak), or None if spike_index is out of range.
    """
    v_arr = np.asarray(V, dtype=float)
    n = len(v_arr)
    kept_indices, _ = _upward_crossings(t, v_arr, threshold, _DEFAULT_MIN_ISI_MS)
    kept_indices = kept_indices.tolist()

    if spike_index < 0 or spike_index >= len(kept_indices):
        return None

    i_start = kept_indices[spike_index]
    i_end = kept_indices[spike_index + 1] if (spike_index + 1 < len(kept_indices)) else n
    i_peak = i_start + int(np.argmax(v_arr[i_start:i_end]))
    return i_start, i_end, i_peak


def spike_peak(t: np.ndarray, V: np.ndarray, spike_index: int = 0) -> float:
    """Calculate the peak membrane voltage of a selected action potential.

    Parameters
    ----------
    t : np.ndarray
        Time array in ms.
    V : np.ndarray
        Voltage array in mV.
    spike_index : int, optional
        0-based spike index (default 0).

    Returns
    -------
    float
        Peak membrane potential in mV, or NaN if spike is missing.
    """
    win = _spike_window(t, V, spike_index)
    if win is None:
        return float(np.nan)
    _, _, i_peak = win
    return float(np.asarray(V, dtype=float)[i_peak])


def spike_width(
    t: np.ndarray,
    V: np.ndarray,
    spike_index: int = 0,
    baseline: float | None = None,
) -> float:
    """Calculate full-width at half-maximum (FWHM) of a selected spike.

    Parameters
    ----------
    t : np.ndarray
        Time array in ms.
    V : np.ndarray
        Voltage array in mV.
    spike_index : int, optional
        0-based spike index (default 0).
    baseline : float, optional
        Baseline voltage in mV. If None, defaults to V[0].

    Returns
    -------
    float
        Spike width in ms, or NaN if crossing is not contained in the spike window.
    """
    win = _spike_window(t, V, spike_index)
    if win is None:
        return float(np.nan)
    i_start, i_end, i_peak = win
    v_arr = np.asarray(V, dtype=float)
    t_arr = np.asarray(t, dtype=float)

    base = float(v_arr[0]) if baseline is None else float(baseline)
    peak = float(v_arr[i_peak])
    half_level = 0.5 * (peak + base)

    # Walk backward from peak to locate upward crossing of half_level
    t_up: float | None = None
    for j in range(i_peak - 1, -1, -1):
        if (v_arr[j] <= half_level < v_arr[j + 1]) or (v_arr[j] < half_level <= v_arr[j + 1]):
            denom = v_arr[j + 1] - v_arr[j]
            frac = (half_level - v_arr[j]) / denom if denom != 0.0 else 0.0
            t_up = float(t_arr[j] + frac * (t_arr[j + 1] - t_arr[j]))
            break

    # Walk forward from peak within spike window to locate downward crossing
    t_down: float | None = None
    for j in range(i_peak, i_end - 1):
        if (v_arr[j] >= half_level > v_arr[j + 1]) or (v_arr[j] > half_level >= v_arr[j + 1]):
            denom = v_arr[j + 1] - v_arr[j]
            frac = (half_level - v_arr[j]) / denom if denom != 0.0 else 0.0
            t_down = float(t_arr[j] + frac * (t_arr[j + 1] - t_arr[j]))
            break

    if t_up is None or t_down is None:
        return float(np.nan)

    return float(t_down - t_up)


def afterhyperpolarisation(t: np.ndarray, V: np.ndarray, spike_index: int = 0) -> float:
    """Calculate minimum potential in the undershoot window after the spike peak.

    Parameters
    ----------
    t : np.ndarray
        Time array in ms.
    V : np.ndarray
        Voltage array in mV.
    spike_index : int, optional
        0-based spike index (default 0).

    Returns
    -------
    float
        Afterhyperpolarisation minimum in mV, or NaN if spike is missing.
    """
    win = _spike_window(t, V, spike_index)
    if win is None:
        return float(np.nan)
    _, i_end, i_peak = win
    v_arr = np.asarray(V, dtype=float)
    if i_peak >= i_end:
        return float(v_arr[i_peak])
    return float(np.min(v_arr[i_peak:i_end]))


def firing_rate(t: np.ndarray, V: np.ndarray, t_start: float, t_end: float) -> float:
    """Compute mean action potential firing rate over a specified time window.

    Parameters
    ----------
    t : np.ndarray
        Time array in ms.
    V : np.ndarray
        Voltage array in mV.
    t_start : float
        Interval start time in ms (inclusive).
    t_end : float
        Interval end time in ms (exclusive).

    Returns
    -------
    float
        Firing rate in Hz.
    """
    t_st = float(t_start)
    t_en = float(t_end)
    if t_en <= t_st:
        return 0.0
    spikes = detect_spikes(t, V)
    n_spikes = int(np.sum((spikes >= t_st) & (spikes < t_en)))
    return float(n_spikes / (t_en - t_st) * 1000.0)


def run_step(
    amplitude: float,
    params: HHParams = DEFAULT_PARAMS,
    t_pre: float = 50.0,
    duration: float = 200.0,
    t_post: float = 50.0,
    dt: float = 0.01,
) -> Trace:
    """Simulate single-compartment response to a rectangular current step.

    Parameters
    ----------
    amplitude : float
        Current pulse amplitude in uA/cm^2.
    params : HHParams, optional
        Neuron parameters (default DEFAULT_PARAMS).
    t_pre : float, optional
        Pre-stimulus baseline duration in ms (default 50.0).
    duration : float, optional
        Stimulus pulse duration in ms (default 200.0).
    t_post : float, optional
        Post-stimulus recovery duration in ms (default 50.0).
    dt : float, optional
        Solver time step in ms (default 0.01).

    Returns
    -------
    Trace
        Simulated trajectory.
    """
    t_total = float(t_pre + duration + t_post)
    stim = step(amplitude, float(t_pre), float(t_pre + duration))
    return simulate(stim, t_max=t_total, dt=dt, params=params)


def find_rheobase(
    params: HHParams = DEFAULT_PARAMS,
    i_lo: float = 0.0,
    i_hi: float = 20.0,
    tol: float = 0.01,
    t_pre: float = 50.0,
    duration: float = 200.0,
) -> float:
    """Find the single-spike threshold (rheobase) via bisection.

    Parameters
    ----------
    params : HHParams, optional
        Neuron parameters.
    i_lo : float, optional
        Lower current bound in uA/cm^2 (must yield zero spikes).
    i_hi : float, optional
        Upper current bound in uA/cm^2 (must yield >= 1 spike).
    tol : float, optional
        Convergence tolerance in uA/cm^2 (default 0.01).
    t_pre : float, optional
        Pre-step duration in ms (default 50.0).
    duration : float, optional
        Step duration in ms (default 200.0).

    Returns
    -------
    float
        Rheobase current density in uA/cm^2.
    """
    lo = float(i_lo)
    hi = float(i_hi)

    tr_lo = run_step(lo, params=params, t_pre=t_pre, duration=duration)
    if len(detect_spikes(tr_lo.t, tr_lo.V)) >= 1:
        raise ValueError(f"Spike already present at lower bound i_lo={lo}")

    tr_hi = run_step(hi, params=params, t_pre=t_pre, duration=duration)
    if len(detect_spikes(tr_hi.t, tr_hi.V)) == 0:
        raise ValueError(f"No spike generated at upper bound i_hi={hi}")

    while (hi - lo) >= tol:
        mid = 0.5 * (lo + hi)
        tr_mid = run_step(mid, params=params, t_pre=t_pre, duration=duration)
        if len(detect_spikes(tr_mid.t, tr_mid.V)) >= 1:
            hi = mid
        else:
            lo = mid

    return float(hi)


def find_repetitive_onset(
    params: HHParams = DEFAULT_PARAMS,
    i_lo: float = 2.0,
    i_hi: float = 20.0,
    tol: float = 0.05,
    min_spikes: int = 10,
    duration: float = 1000.0,
) -> float:
    """Find onset of repetitive sustained firing via bisection over a 1000 ms step.

    Repetitive firing onset is distinct from rheobase (the single-spike threshold).
    The gap between them is the primary signature of class 2 neural excitability.

    Parameters
    ----------
    params : HHParams, optional
        Neuron parameters.
    i_lo : float, optional
        Lower current bound in uA/cm^2.
    i_hi : float, optional
        Upper current bound in uA/cm^2.
    tol : float, optional
        Convergence tolerance in uA/cm^2 (default 0.05).
    min_spikes : int, optional
        Spike threshold count for sustained firing (default 10).
    duration : float, optional
        Step duration in ms (default 1000.0).

    Returns
    -------
    float
        Repetitive onset current density in uA/cm^2.
    """
    lo = float(i_lo)
    hi = float(i_hi)
    t_pre = 50.0

    tr_lo = run_step(lo, params=params, t_pre=t_pre, duration=duration, t_post=0.0)
    spikes_lo = detect_spikes(tr_lo.t, tr_lo.V)
    count_lo = int(np.sum((spikes_lo >= t_pre) & (spikes_lo <= t_pre + duration)))
    if count_lo >= min_spikes:
        raise ValueError(f"Repetitive firing already active at i_lo={lo}")

    tr_hi = run_step(hi, params=params, t_pre=t_pre, duration=duration, t_post=0.0)
    spikes_hi = detect_spikes(tr_hi.t, tr_hi.V)
    count_hi = int(np.sum((spikes_hi >= t_pre) & (spikes_hi <= t_pre + duration)))
    if count_hi < min_spikes:
        raise ValueError(f"Insufficient spiking at upper bound i_hi={hi}")

    while (hi - lo) >= tol:
        mid = 0.5 * (lo + hi)
        tr_mid = run_step(mid, params=params, t_pre=t_pre, duration=duration, t_post=0.0)
        spikes_mid = detect_spikes(tr_mid.t, tr_mid.V)
        count_mid = int(np.sum((spikes_mid >= t_pre) & (spikes_mid <= t_pre + duration)))
        if count_mid >= min_spikes:
            hi = mid
        else:
            lo = mid

    return float(hi)


def fi_curve(
    currents,
    params: HHParams = DEFAULT_PARAMS,
    t_pre: float = 50.0,
    duration: float = 500.0,
    transient: float = 100.0,
) -> dict[str, np.ndarray]:
    """Compute frequency-current (f-I) relationship across an array of stimulus currents.

    Parameters
    ----------
    currents : array_like
        Injected current amplitudes in uA/cm^2.
    params : HHParams, optional
        Neuron parameters.
    t_pre : float, optional
        Pre-stimulus duration in ms (default 50.0).
    duration : float, optional
        Pulse duration in ms (default 500.0).
    transient : float, optional
        Transient window ignored at pulse start in ms (default 100.0).

    Returns
    -------
    dict[str, np.ndarray]
        Arrays with keys 'current', 'rate_hz', 'n_spikes', 'v_max'.
    """
    curr_arr = np.asarray(currents, dtype=float)
    rates = []
    n_spikes_list = []
    v_max_list = []

    for amp in curr_arr:
        tr = run_step(float(amp), params=params, t_pre=t_pre, duration=duration, t_post=0.0)
        rate = firing_rate(tr.t, tr.V, t_pre + transient, t_pre + duration)
        spikes = detect_spikes(tr.t, tr.V)
        n_spk = int(np.sum((spikes >= t_pre) & (spikes <= t_pre + duration)))
        v_max = float(np.max(tr.V))

        rates.append(rate)
        n_spikes_list.append(n_spk)
        v_max_list.append(v_max)

    return {
        "current": curr_arr,
        "rate_hz": np.array(rates, dtype=float),
        "n_spikes": np.array(n_spikes_list, dtype=int),
        "v_max": np.array(v_max_list, dtype=float),
    }


def detect_block_onset(
    currents,
    rates_hz,
    drop_fraction: float = 0.5,
) -> float:
    """Detect current at onset of depolarisation block where firing collapses.

    Parameters
    ----------
    currents : array_like
        Tested stimulus currents in uA/cm^2.
    rates_hz : array_like
        Corresponding firing rates in Hz.
    drop_fraction : float, optional
        Fraction of peak firing rate defining collapse (default 0.5).

    Returns
    -------
    float
        Current density at block onset in uA/cm^2, or NaN if unobserved.
    """
    currs = np.asarray(currents, dtype=float)
    rates = np.asarray(rates_hz, dtype=float)
    if len(rates) == 0:
        return float(np.nan)

    peak_idx = int(np.argmax(rates))
    cutoff = float(drop_fraction) * rates[peak_idx]

    for idx in range(peak_idx + 1, len(currs)):
        if rates[idx] < cutoff:
            return float(currs[idx])

    return float(np.nan)