"""Tests for electrophysiological spike detection, shape metrics, and excitability analysis.

Contains fast synthetic waveform verification and slower full-model simulation tests.
Tests running simulations of 1 second or longer are decorated with @pytest.mark.slow.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hhlab.analysis import (
    afterhyperpolarisation,
    detect_block_onset,
    detect_spikes,
    fi_curve,
    find_repetitive_onset,
    find_rheobase,
    firing_rate,
    run_step,
    spike_peak,
    spike_width,
)
from hhlab.params import DEFAULT_PARAMS


def _gauss(
    t: np.ndarray,
    centres: float | list[float] | tuple[float, ...],
    sigma: float,
    amp: float = 105.0,
    base: float = -65.0,
) -> np.ndarray:
    """Generate synthetic potential trace summing Gaussian spikes on a baseline."""
    if isinstance(centres, (int, float)):
        c_list = [float(centres)]
    else:
        c_list = [float(c) for c in centres]

    v = np.full_like(t, base, dtype=float)
    for c in c_list:
        v += amp * np.exp(-((t - c) ** 2) / (2.0 * sigma**2))
    return v


def test_detect_spikes_times():
    """Verify linear interpolation accuracy on an isolated synthetic Gaussian spike."""
    t = np.arange(0, 60, 0.01)
    v = _gauss(t, 20.0, 0.5)
    spikes = detect_spikes(t, v)
    assert len(spikes) == 1
    expected = 20.0 - 0.4897
    assert abs(spikes[0] - expected) < 0.005


def test_min_isi():
    """Verify refractory period lockout prevents double-counting close spikes."""
    t = np.arange(0, 60, 0.01)
    v = _gauss(t, [20.0, 21.0], 0.15)
    spikes_1 = detect_spikes(t, v, min_isi=2.0)
    spikes_2 = detect_spikes(t, v, min_isi=0.5)
    assert len(spikes_1) == 1
    assert len(spikes_2) == 2


def test_peak_and_width():
    """Verify peak voltage and half-width estimation on a Gaussian benchmark."""
    t = np.arange(0, 60, 0.01)
    v = _gauss(t, 20.0, 0.5)
    peak = spike_peak(t, v, 0)
    width = spike_width(t, v, 0)
    assert abs(peak - 40.0) < 0.01
    assert abs(width - 1.177) < 0.01


def test_second_spike_indexing():
    """Verify spike selection by index and NaN return for out-of-range indices."""
    t = np.arange(0, 60, 0.01)
    v = _gauss(t, [10.0, 20.0, 30.0], 0.5)
    assert abs(spike_peak(t, v, 2) - 40.0) < 0.01
    assert np.isnan(spike_peak(t, v, 5))
    assert np.isnan(spike_width(t, v, 5))
    assert np.isnan(afterhyperpolarisation(t, v, 5))


def test_ahp():
    """Verify detection of post-spike afterhyperpolarisation minimum."""
    t = np.arange(0, 60, 0.01)
    v = _gauss(t, 20.0, 0.5) - 8.0 * np.exp(-((t - 25.0) ** 2) / 8.0)
    ahp = afterhyperpolarisation(t, v, 0)
    assert abs(ahp - (-73.0)) < 0.05


def test_firing_rate_window():
    """Verify firing rate calculation with exact start-inclusive, end-exclusive windowing."""
    t = np.arange(0, 60, 0.01)
    v = _gauss(t, [10.0, 20.0, 30.0], 0.5)
    r1 = firing_rate(t, v, 0.0, 60.0)
    r2 = firing_rate(t, v, 15.0, 35.0)
    r3 = firing_rate(t, v, 20.0, 30.0)
    assert abs(r1 - 50.0) < 1e-9
    assert abs(r2 - 100.0) < 1e-9
    assert abs(r3 - 100.0) < 1e-9


def test_detect_block_onset():
    """Verify depolarisation block current identification from firing rate vectors."""
    currents = np.arange(0.0, 81.0, 10.0)
    rates = [0, 10, 60, 100, 120, 115, 60, 3, 0]
    block_curr = detect_block_onset(currents, rates)
    assert block_curr == 70.0

    mono_rates = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    assert np.isnan(detect_block_onset(currents, mono_rates))


def test_run_step_reference():
    """Verify single-compartment model action potential peak and duration at 10 uA/cm^2."""
    tr = run_step(10.0)
    peak = spike_peak(tr.t, tr.V, 0)
    width = spike_width(tr.t, tr.V, 0)
    assert 35.0 <= peak <= 45.0
    assert 1.2 <= width <= 1.8


def test_width_shrinks_with_temperature():
    """Verify that spike duration narrows substantially when temperature increases to 18.5 °C."""
    params_warm = DEFAULT_PARAMS.with_(temp_C=18.5)
    tr = run_step(10.0, params_warm, t_pre=5.0, duration=40.0, t_post=5.0)
    width = spike_width(tr.t, tr.V, 0)
    assert 0.3 <= width <= 0.55


@pytest.mark.slow
def test_rheobase():
    """Verify single-spike rheobase threshold and exception handling for invalid bounds."""
    rheo = find_rheobase()
    assert 2.1 <= rheo <= 2.4
    with pytest.raises(ValueError):
        find_rheobase(i_hi=1.0)


@pytest.mark.slow
def test_repetitive_onset():
    """Verify onset of repetitive firing is strictly higher than rheobase (class 2)."""
    rheo = find_rheobase()
    onset = find_repetitive_onset()
    assert 6.0 <= onset <= 6.8
    assert onset > rheo


@pytest.mark.slow
def test_fi_curve():
    """Verify f-I curve rates and dictionary format across stimulus levels."""
    currents = [0.0, 10.0, 50.0]
    res = fi_curve(currents)
    for key in ("current", "rate_hz", "n_spikes", "v_max"):
        assert key in res
        assert len(res[key]) == len(currents)
    assert res["rate_hz"][0] == 0.0
    assert 60.0 <= res["rate_hz"][1] <= 80.0
    assert 100.0 <= res["rate_hz"][2] <= 135.0


@pytest.mark.slow
def test_depolarisation_block():
    """Verify collapse of repetitive spiking under high current drive."""
    currents = np.arange(0.0, 81.0, 10.0)
    res = fi_curve(currents)
    idx_70 = int(np.where(currents == 70.0)[0][0])
    assert res["rate_hz"][idx_70] < 10.0
    block_curr = detect_block_onset(res["current"], res["rate_hz"])
    assert 60.0 < block_curr <= 70.0