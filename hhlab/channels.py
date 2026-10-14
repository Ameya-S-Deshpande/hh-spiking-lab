"""Hodgkin-Huxley voltage-dependent channel kinetics and ionic currents.

Rate constants are evaluated at the reference temperature T_REF_C = 6.3 °C.
Temperature scaling via phi is applied in derivative calculations, not in this module.
"""

from __future__ import annotations

import numpy as np

from hhlab.params import HHParams, V_REST


def _x_over_one_minus_exp(x: float | np.ndarray, k: float) -> float | np.ndarray:
    """Evaluate x / (1 - exp(-x / k)) with series expansion at x = 0.

    Parameters
    ----------
    x : float or np.ndarray
        Voltage displacement in mV.
    k : float
        Exponential denominator slope factor in mV.

    Returns
    -------
    float or np.ndarray
        Evaluated value with removable singularity removed at x = 0.
    """
    is_scalar = np.isscalar(x)
    x_arr = np.asarray(x, dtype=float)
    u = x_arr / k

    # Near singularity |x/k| < 1e-6, use Taylor series: k * (1 - x / (2 * k))
    mask = np.abs(u) < 1e-6
    safe_u = np.where(mask, 1.0, u)
    direct = (k * safe_u) / (1.0 - np.exp(-safe_u))
    series = k * (1.0 - u / 2.0)
    result = np.where(mask, series, direct)

    if is_scalar:
        return float(result.item())
    return result


def alpha_beta(
    V: float | np.ndarray,
) -> tuple[
    float | np.ndarray,
    float | np.ndarray,
    float | np.ndarray,
    float | np.ndarray,
    float | np.ndarray,
    float | np.ndarray,
]:
    """Calculate forward and backward gating transition rates at T_REF_C.

    Parameters
    ----------
    V : float or np.ndarray
        Membrane potential in mV.

    Returns
    -------
    tuple
        (alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n) in 1/ms.
    """
    alpha_m = 0.1 * _x_over_one_minus_exp(V + 40.0, 10.0)
    beta_m = 4.0 * np.exp(-(V + 65.0) / 18.0)

    alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
    beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    alpha_n = 0.01 * _x_over_one_minus_exp(V + 55.0, 10.0)
    beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)

    return alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n


def steady_state(
    V: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """Calculate gating variable steady-state activation and inactivation.

    Parameters
    ----------
    V : float or np.ndarray
        Membrane potential in mV.

    Returns
    -------
    tuple
        Steady-state values (m_inf, h_inf, n_inf), dimensionless in [0, 1].
    """
    am, bm, ah, bh, an, bn = alpha_beta(V)
    m_inf = am / (am + bm)
    h_inf = ah / (ah + bh)
    n_inf = an / (an + bn)
    return m_inf, h_inf, n_inf


def time_constants(
    V: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """Calculate gating variable time constants at T_REF_C.

    Parameters
    ----------
    V : float or np.ndarray
        Membrane potential in mV.

    Returns
    -------
    tuple
        Time constants (tau_m, tau_h, tau_n) in ms.
    """
    am, bm, ah, bh, an, bn = alpha_beta(V)
    tau_m = 1.0 / (am + bm)
    tau_h = 1.0 / (ah + bh)
    tau_n = 1.0 / (an + bn)
    return tau_m, tau_h, tau_n


def resting_state(V: float = V_REST) -> np.ndarray:
    """Calculate state vector [V, m, h, n] at steady rest.

    Parameters
    ----------
    V : float, optional
        Resting membrane potential in mV (default V_REST = -65.0 mV).

    Returns
    -------
    np.ndarray
        State array of shape (4,) containing [V, m_inf, h_inf, n_inf].
    """
    m_inf, h_inf, n_inf = steady_state(V)
    return np.array([float(V), float(m_inf), float(h_inf), float(n_inf)], dtype=float)


def ionic_currents(
    V: float | np.ndarray,
    m: float | np.ndarray,
    h: float | np.ndarray,
    n: float | np.ndarray,
    params: HHParams,
) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """Compute individual ionic transmembrane currents (outward positive convention).

    Gating powers m**3 * h and n**4 arise from independent gates in series:
    three activation m gates and one inactivation h gate for sodium,
    and four activation n gates for delayed-rectifier potassium.

    Parameters
    ----------
    V : float or np.ndarray
        Membrane potential in mV.
    m : float or np.ndarray
        Sodium activation gating variable.
    h : float or np.ndarray
        Sodium inactivation gating variable.
    n : float or np.ndarray
        Potassium activation gating variable.
    params : HHParams
        Neuron channel parameters.

    Returns
    -------
    tuple
        Currents (I_Na, I_K, I_L) in uA/cm^2.
    """
    i_na = params.gNa * (m**3) * h * (V - params.ENa)
    i_k = params.gK * (n**4) * (V - params.EK)
    i_l = params.gL * (V - params.EL)
    return i_na, i_k, i_l