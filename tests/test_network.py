"""Tests for excitatory-inhibitory network connectivity, dynamics, and population metrics.

Verifies connectivity construction, deterministic random seeding, synchrony calculation,
population firing rates, spectral analysis, and numerical convergence.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hhlab.network import (
    NetworkConfig,
    NetworkResult,
    build_connectivity,
    peak_frequency,
    population_rates,
    population_spectrum,
    simulate_network,
    synchrony_chi,
)

SMALL = NetworkConfig(n_exc=16, n_inh=4, seed=3)


def test_connectivity_structure():
    """Verify weight matrix shape, zero diagonal, non-negative weights, and block values."""
    cfg = NetworkConfig()
    rng = np.random.default_rng(cfg.seed)
    w_mat = build_connectivity(cfg, rng)

    assert w_mat.shape == (50, 50)
    assert np.all(np.diag(w_mat) == 0.0)
    assert np.all(w_mat >= 0.0)

    frac_nonzero = np.count_nonzero(w_mat) / w_mat.size
    assert 0.15 <= frac_nonzero <= 0.25

    ee_block = w_mat[: cfg.n_exc, : cfg.n_exc]
    ee_nonzero = ee_block[ee_block > 0.0]
    assert np.all(np.isclose(ee_nonzero, 0.02))

    ie_block = w_mat[: cfg.n_exc, cfg.n_exc :]
    ie_nonzero = ie_block[ie_block > 0.0]
    assert np.all(np.isclose(ie_nonzero, 0.20))


def test_connectivity_seed():
    """Verify identical random seeds generate identical matrices, and different seeds differ."""
    cfg1 = NetworkConfig(seed=42)
    cfg2 = NetworkConfig(seed=42)
    cfg3 = NetworkConfig(seed=43)

    w1 = build_connectivity(cfg1, np.random.default_rng(cfg1.seed))
    w2 = build_connectivity(cfg2, np.random.default_rng(cfg2.seed))
    w3 = build_connectivity(cfg3, np.random.default_rng(cfg3.seed))

    assert np.array_equal(w1, w2)
    assert not np.array_equal(w1, w3)


def test_no_drive_no_spikes():
    """Verify zero input drive produces zero spikes and voltages settle to rest."""
    cfg = dataclasses.replace(SMALL, drive_exc_mean=0.0, drive_exc_std=0.0)
    res = simulate_network(cfg, t_max=100.0)

    assert len(res.spike_times) == 0
    # Following initial out-of-phase potential relaxation, all neurons settle to rest
    assert np.all(np.abs(res.v_rec[res.t_rec >= 20.0] - (-65.0)) < 0.5)


def test_result_structure():
    """Verify NetworkResult attributes, ascending times, ID ranges, and time grids."""
    res = simulate_network(SMALL, t_max=100.0)

    assert np.all(np.diff(res.spike_times) >= 0.0)
    assert np.all(res.spike_ids >= 0)
    assert np.all(res.spike_ids < (SMALL.n_exc + SMALL.n_inh))
    assert res.v_rec.shape == (len(res.t_rec), 20)
    assert abs((res.t_rec[1] - res.t_rec[0]) - (res.config.dt * 5)) < 1e-9


def test_determinism():
    """Verify network simulation reproducibility under identical initial configurations."""
    res1 = simulate_network(SMALL, t_max=100.0)
    res2 = simulate_network(SMALL, t_max=100.0)

    assert np.array_equal(res1.spike_times, res2.spike_times)
    assert np.array_equal(res1.spike_ids, res2.spike_ids)


def test_uncoupled_neurons_fire_at_single_cell_rate():
    """Verify uncoupled network neurons fire within expected single-cell rate bounds."""
    cfg = NetworkConfig(
        n_exc=16,
        n_inh=0,
        w_ee=0.0,
        w_ei=0.0,
        w_ie=0.0,
        w_ii=0.0,
        drive_exc_mean=10.0,
        drive_exc_std=0.0,
        seed=1,
    )
    res = simulate_network(cfg, t_max=400.0)
    exc_rate, _ = population_rates(res, t_start=100.0)
    assert 55.0 <= exc_rate <= 80.0


def test_inhibition_lowers_excitatory_rate():
    """Verify increasing I -> E synaptic weight systematically suppresses excitatory firing."""
    rates = []
    for w_val in (0.0, 0.2, 0.5):
        cfg = dataclasses.replace(SMALL, w_ie=w_val)
        res = simulate_network(cfg, t_max=250.0)
        exc_rate, _ = population_rates(res, t_start=50.0)
        rates.append(exc_rate)

    assert rates[0] > rates[1] > rates[2]
    assert (rates[0] - rates[2]) > 8.0


def test_chi_limits():
    """Verify Golomb-Hansel synchrony index on synthetic perfectly synchronous and noisy signals."""
    cfg = NetworkConfig(n_exc=40, n_inh=10)
    t_rec = np.linspace(0.0, 200.0, 2000)

    rng = np.random.default_rng(0)
    v_single = rng.standard_normal(2000)
    v_identical = np.tile(v_single[:, np.newaxis], (1, 50))
    res_identical = NetworkResult(
        config=cfg,
        t_max=200.0,
        spike_times=np.array([], dtype=float),
        spike_ids=np.array([], dtype=int),
        t_rec=t_rec,
        v_rec=v_identical,
    )
    chi_identical = synchrony_chi(res_identical, t_start=0.0)
    assert abs(chi_identical - 1.0) < 1e-9

    v_independent = rng.standard_normal((2000, 50))
    res_independent = NetworkResult(
        config=cfg,
        t_max=200.0,
        spike_times=np.array([], dtype=float),
        spike_ids=np.array([], dtype=int),
        t_rec=t_rec,
        v_rec=v_independent,
    )
    chi_independent = synchrony_chi(res_independent, t_start=0.0)
    assert chi_independent < 0.4


def test_spectrum_peak():
    """Verify FFT spectrum identification of 50 Hz periodic spike trains."""
    spikes = np.arange(100.0, 1001.0, 20.0)
    ids = np.zeros(len(spikes), dtype=int)
    t_rec = np.linspace(0.0, 1000.0, 1001)
    v_rec = np.zeros((1001, 50), dtype=float)

    res = NetworkResult(
        config=NetworkConfig(n_exc=40, n_inh=10),
        t_max=1000.0,
        spike_times=spikes,
        spike_ids=ids,
        t_rec=t_rec,
        v_rec=v_rec,
    )

    freqs, power = population_spectrum(res, "exc", t_start=100.0)
    peak_f = peak_frequency(freqs, power)
    assert abs(peak_f - 50.0) < 3.0

    with pytest.raises(ValueError):
        population_spectrum(res, "bad")


@pytest.mark.slow
def test_timestep_convergence():
    """Verify time step convergence between dt = 0.02 ms and dt = 0.01 ms."""
    cfg1 = dataclasses.replace(SMALL, dt=0.02)
    cfg2 = dataclasses.replace(SMALL, dt=0.01)

    res1 = simulate_network(cfg1, t_max=250.0)
    res2 = simulate_network(cfg2, t_max=250.0)

    n1 = len(res1.spike_times)
    n2 = len(res2.spike_times)
    diff_frac = abs(n1 - n2) / max(n1, n2)
    assert diff_frac <= 0.05


@pytest.mark.slow
def test_default_network_reference():
    """Verify default 50-neuron network matches baseline reference rate and synchrony ranges."""
    cfg = NetworkConfig()
    res = simulate_network(cfg, t_max=500.0)

    exc_rate, _ = population_rates(res, t_start=100.0)
    assert 35.0 <= exc_rate <= 65.0

    chi = synchrony_chi(res, t_start=100.0)
    assert 0.2 < chi < 0.95

    freqs, power = population_spectrum(res, "exc", t_start=100.0)
    peak_f = peak_frequency(freqs, power)
    assert np.isfinite(peak_f)