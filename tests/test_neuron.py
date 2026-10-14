"""Tests for single-compartment Hodgkin-Huxley dynamics, channels, and stimuli.

Verifies mathematical consistency, numerical solvers, and biophysical properties.
Does not import hhlab.analysis.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hhlab.channels import alpha_beta, resting_state, steady_state
from hhlab.neuron import derivatives, rk4_simulate, simulate
from hhlab.params import DEFAULT_PARAMS, HHParams
from hhlab.stimuli import constant, pulse_train, ramp, step, total


def _crossings(t: np.ndarray, V: np.ndarray, thr: float = 0.0) -> np.ndarray:
    """Return linearly interpolated upward-crossing times across thr."""
    crossing_times = []
    for i in range(len(V) - 1):
        if V[i] < thr <= V[i + 1]:
            frac = (thr - V[i]) / (V[i + 1] - V[i])
            crossing_times.append(t[i] + frac * (t[i + 1] - t[i]))
    return np.array(crossing_times, dtype=float)


def _run(
    amplitude: float,
    params: HHParams = DEFAULT_PARAMS,
    t_max: float = 50.0,
    t_on: float = 5.0,
    t_off: float = 45.0,
):
    """Run standard step simulation benchmark helper."""
    return simulate(step(amplitude, t_on, t_off), t_max=t_max, params=params)


def test_steady_state_at_rest():
    """Verify resting gate steady-state values match reference values."""
    m_inf, h_inf, n_inf = steady_state(-65.0)
    assert abs(m_inf - 0.0529) < 0.0005
    assert abs(h_inf - 0.5961) < 0.0005
    assert abs(n_inf - 0.3177) < 0.0005


def test_singularities_are_finite():
    """Verify removable singularities in rate equations evaluate without divergence."""
    rates_55 = alpha_beta(-55.0)
    assert abs(rates_55[4] - 0.1) < 1e-9

    rates_40 = alpha_beta(-40.0)
    assert abs(rates_40[0] - 1.0) < 1e-9

    v_arr = np.array([-55.0, -40.0, -65.0])
    arr_rates = alpha_beta(v_arr)
    assert len(arr_rates) == 6
    for r in arr_rates:
        assert isinstance(r, np.ndarray)
        assert len(r) == 3
        assert np.all(np.isfinite(r))


def test_rest_is_stable():
    """Verify stability of the resting state under zero injected current."""
    tr = simulate(constant(0.0), t_max=100.0)
    assert np.all(np.abs(tr.V - (-65.0)) < 0.1)

    dydt = derivatives(0.0, resting_state(), constant(0.0), DEFAULT_PARAMS)
    assert abs(dydt[0]) < 0.05  # true rest is within 0.1 mV of -65, so dV/dt is tiny but not zero
    assert all(abs(val) < 1e-9 for val in dydt[1:])  # gates sit exactly at steady state


def test_action_potential_shape():
    """Verify peak height, latency, and afterhyperpolarisation of a 10 uA/cm^2 spike."""
    tr = _run(10.0)
    peak_idx = int(np.argmax(tr.V))
    peak_v = tr.V[peak_idx]
    assert 35.0 <= peak_v <= 45.0

    crossings = _crossings(tr.t, tr.V, 0.0)
    assert len(crossings) > 0
    assert 5.5 <= crossings[0] <= 9.0

    min_v_after_peak = np.min(tr.V[peak_idx:])
    assert min_v_after_peak < -72.0


def test_conductance_peaks():
    """Verify maximum active sodium and potassium conductances during action potential."""
    tr = _run(10.0)
    conds = tr.conductances()
    max_gna = np.max(conds["gNa"])
    max_gk = np.max(conds["gK"])
    assert 25.0 <= max_gna <= 40.0
    assert 10.0 <= max_gk <= 15.0


def test_currents_balance():
    """Verify Kirchhoff current law C*dV/dt = I_stim - sum(I_ion) along the trajectory."""
    tr = _run(10.0)
    currs = tr.currents()
    i_ion = currs["I_Na"] + currs["I_K"] + currs["I_L"]
    rhs = tr.I_stim - i_ion
    dv_dt = np.gradient(tr.V, tr.t)
    c_dv_dt = tr.params.C * dv_dt

    mask = (tr.t >= 10.0) & (tr.t <= 40.0)
    diff = np.abs(rhs[mask] - c_dv_dt[mask])
    assert np.all(diff < 0.5)


def test_rk4_matches_lsoda():
    """Verify agreement of fixed-step RK4 solver with adaptive LSODA solution."""
    tr_lsoda = _run(10.0)
    tr_rk4 = rk4_simulate(step(10.0, 5.0, 45.0), 50.0, dt=0.005)

    assert len(tr_rk4.t) == int(round(50.0 / 0.005)) + 1

    crossings_lsoda = _crossings(tr_lsoda.t, tr_lsoda.V, 0.0)
    crossings_rk4 = _crossings(tr_rk4.t, tr_rk4.V, 0.0)

    assert len(crossings_lsoda) > 0
    assert len(crossings_rk4) > 0
    assert abs(crossings_lsoda[0] - crossings_rk4[0]) < 0.02


def test_ttx_like_blocks_spikes():
    """Verify complete block of action potentials when sodium conductance is eliminated."""
    params = DEFAULT_PARAMS.with_(gNa=0.0)
    for amp in (10.0, 50.0):
        tr = _run(amp, params=params)
        crossings = _crossings(tr.t, tr.V, 0.0)
        assert len(crossings) == 0
        assert np.max(tr.V) < -25.0


def test_tea_like_plateau():
    """Verify prolonged depolarisation plateau when potassium channels are blocked."""
    params = DEFAULT_PARAMS.with_(gK=0.0)
    tr = _run(10.0, params=params)
    idx = int(np.argmin(np.abs(tr.t - 30.0)))
    assert 3.0 <= tr.V[idx] <= 13.0


def test_heat_block():
    """Verify action potential suppression at elevated temperature (25 °C)."""
    params_hot = DEFAULT_PARAMS.with_(temp_C=25.0)
    tr_hot = _run(10.0, params=params_hot)
    assert len(_crossings(tr_hot.t, tr_hot.V, 0.0)) == 0

    params_ref = DEFAULT_PARAMS.with_(temp_C=6.3)
    tr_ref = _run(10.0, params=params_ref)
    assert len(_crossings(tr_ref.t, tr_ref.V, 0.0)) >= 1


def test_temperature_speeds_up_spike():
    """Verify shorter spike onset latency at higher temperatures due to accelerated rates."""
    p_cold = DEFAULT_PARAMS.with_(temp_C=6.3)
    p_warm = DEFAULT_PARAMS.with_(temp_C=18.5)

    tr_cold = _run(20.0, params=p_cold)
    tr_warm = _run(20.0, params=p_warm)

    c_cold = _crossings(tr_cold.t, tr_cold.V, 0.0)
    c_warm = _crossings(tr_warm.t, tr_warm.V, 0.0)

    latency_cold = c_cold[0] - 5.0
    latency_warm = c_warm[0] - 5.0
    assert latency_warm < latency_cold


def test_stimuli():
    """Verify waveform generation and strict time boundary checks for all stimulus types."""
    s = step(10.0, 5.0, 45.0)
    assert s(4.99) == 0.0
    assert s(5.0) == 10.0
    assert s(44.99) == 10.0
    assert s(45.0) == 0.0

    pt = pulse_train(10.0, 5.0, 1.0, 10.0, 3)
    assert pt(4.99) == 0.0
    assert pt(5.0) == 10.0
    assert pt(5.99) == 10.0
    assert pt(6.0) == 0.0
    assert pt(15.0) == 10.0
    assert pt(16.0) == 0.0
    assert pt(25.0) == 10.0
    assert pt(26.0) == 0.0
    assert pt(35.0) == 0.0

    r = ramp(2.0, 12.0, 10.0, 20.0)
    assert r(9.99) == 0.0
    assert r(10.0) == 2.0
    assert abs(r(15.0) - 7.0) < 1e-9
    assert r(20.0) == 0.0

    tot = total(step(5.0, 0.0, 10.0), step(3.0, 5.0, 15.0))
    assert tot(2.0) == 5.0
    assert tot(7.0) == 8.0
    assert tot(12.0) == 3.0
    assert tot(16.0) == 0.0


def test_params():
    """Verify Q10 scaling factor, immutability, and with_ method behaviour."""
    p = DEFAULT_PARAMS
    assert abs(p.phi() - 1.0) < 1e-9

    p_warm = p.with_(temp_C=18.5)
    assert abs((p_warm.phi() - 3.8189) / 3.8189) < 1e-3
    assert p is not p_warm
    assert p.temp_C == 6.3

    with pytest.raises(dataclasses.FrozenInstanceError):
        p.gNa = 100.0  # type: ignore