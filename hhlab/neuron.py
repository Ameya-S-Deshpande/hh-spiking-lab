"""Hodgkin-Huxley single-compartment numerical integration.

LSODA is used as the primary solver because the dynamics become mildly stiff during the upstroke.
A classical fixed-step RK4 solver is retained to independently cross-check and verify trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from hhlab.channels import alpha_beta, ionic_currents, resting_state
from hhlab.params import DEFAULT_PARAMS, HHParams
from hhlab.stimuli import Stimulus, sample


@dataclass
class Trace:
    """Simulation trajectory of a single Hodgkin-Huxley compartment.

    Parameters
    ----------
    t : np.ndarray
        Time points in ms.
    V : np.ndarray
        Membrane potentials in mV.
    m : np.ndarray
        Sodium activation gating trajectory.
    h : np.ndarray
        Sodium inactivation gating trajectory.
    n : np.ndarray
        Potassium activation gating trajectory.
    I_stim : np.ndarray
        Sampled stimulus current density in uA/cm^2.
    params : HHParams
        Parameter instance used for this simulation.
    """

    t: np.ndarray
    V: np.ndarray
    m: np.ndarray
    h: np.ndarray
    n: np.ndarray
    I_stim: np.ndarray
    params: HHParams

    def currents(self) -> dict[str, np.ndarray]:
        """Compute ionic currents along the simulation trajectory.

        Returns
        -------
        dict[str, np.ndarray]
            Currents with keys 'I_Na', 'I_K', 'I_L' in uA/cm^2 (outward positive).
        """
        i_na, i_k, i_l = ionic_currents(self.V, self.m, self.h, self.n, self.params)
        return {"I_Na": i_na, "I_K": i_k, "I_L": i_l}

    def conductances(self) -> dict[str, np.ndarray]:
        """Compute active conductances along the simulation trajectory.

        Returns
        -------
        dict[str, np.ndarray]
            Conductances with keys 'gNa' and 'gK' in mS/cm^2.
        """
        return {
            "gNa": self.params.gNa * (self.m**3) * self.h,
            "gK": self.params.gK * (self.n**4),
        }


def derivatives(t: float, y, stimulus: Stimulus, params: HHParams) -> list[float]:
    """Calculate state vector time derivatives [dV/dt, dm/dt, dh/dt, dn/dt].

    Parameters
    ----------
    t : float
        Current time in ms.
    y : sequence of float
        State vector [V, m, h, n] in [mV, 1, 1, 1].
    stimulus : Stimulus
        Current stimulus function returning uA/cm^2.
    params : HHParams
        Hodgkin-Huxley model parameters.

    Returns
    -------
    list[float]
        Time derivatives [dV, dm, dh, dn] in [mV/ms, 1/ms, 1/ms, 1/ms].
    """
    v_val, m_val, h_val, n_val = y
    i_na, i_k, i_l = ionic_currents(v_val, m_val, h_val, n_val, params)
    i_stim = stimulus(float(t))
    dv = (i_stim - i_na - i_k - i_l) / params.C

    am, bm, ah, bh, an, bn = alpha_beta(v_val)
    phi = params.phi()
    dm = phi * (am * (1.0 - m_val) - bm * m_val)
    dh = phi * (ah * (1.0 - h_val) - bh * h_val)
    dn = phi * (an * (1.0 - n_val) - bn * n_val)

    return [float(dv), float(dm), float(dh), float(dn)]


def simulate(
    stimulus: Stimulus,
    t_max: float = 100.0,
    dt: float = 0.01,
    params: HHParams = DEFAULT_PARAMS,
    y0=None,
) -> Trace:
    """Simulate single-compartment dynamics using the adaptive SciPy LSODA solver.

    Parameters
    ----------
    stimulus : Stimulus
        Current stimulus callable.
    t_max : float, optional
        Simulation duration in ms (default 100.0).
    dt : float, optional
        Output evaluation time step in ms (default 0.01).
    params : HHParams, optional
        Neuron parameters (default DEFAULT_PARAMS).
    y0 : sequence of float, optional
        Initial state [V, m, h, n]. If None, defaults to resting_state().

    Returns
    -------
    Trace
        Simulated voltage, gating trajectories, and stimulus samples.
    """
    n_pts = int(round(t_max / dt)) + 1
    t_eval = np.linspace(0.0, t_max, n_pts)
    init_state = resting_state() if y0 is None else np.asarray(y0, dtype=float)

    # max_step=0.1 stops the adaptive solver from stepping over a short stimulus pulse
    sol = solve_ivp(
        lambda t_val, y_val: derivatives(t_val, y_val, stimulus, params),
        (0.0, float(t_max)),
        init_state,
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
        max_step=0.1,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    i_stim = sample(stimulus, t_eval)
    return Trace(
        t=t_eval,
        V=sol.y[0],
        m=sol.y[1],
        h=sol.y[2],
        n=sol.y[3],
        I_stim=i_stim,
        params=params,
    )


def rk4_simulate(
    stimulus: Stimulus,
    t_max: float = 100.0,
    dt: float = 0.005,
    params: HHParams = DEFAULT_PARAMS,
    y0=None,
) -> Trace:
    """Simulate single-compartment dynamics using classical fixed-step Runge-Kutta 4.

    Parameters
    ----------
    stimulus : Stimulus
        Current stimulus callable.
    t_max : float, optional
        Simulation duration in ms (default 100.0).
    dt : float, optional
        Integration time step in ms (default 0.005).
    params : HHParams, optional
        Neuron parameters (default DEFAULT_PARAMS).
    y0 : sequence of float, optional
        Initial state [V, m, h, n]. If None, defaults to resting_state().

    Returns
    -------
    Trace
        Simulated voltage, gating trajectories, and stimulus samples.
    """
    n_steps = int(round(t_max / dt))
    t = np.linspace(0.0, t_max, n_steps + 1)
    init_state = resting_state() if y0 is None else np.asarray(y0, dtype=float)

    y = np.zeros((n_steps + 1, 4), dtype=float)
    y[0] = init_state

    half_dt = 0.5 * dt
    for k in range(n_steps):
        t_k = t[k]
        y_k = y[k]

        k1 = np.asarray(derivatives(t_k, y_k, stimulus, params), dtype=float)
        k2 = np.asarray(
            derivatives(t_k + half_dt, y_k + half_dt * k1, stimulus, params), dtype=float
        )
        k3 = np.asarray(
            derivatives(t_k + half_dt, y_k + half_dt * k2, stimulus, params), dtype=float
        )
        k4 = np.asarray(derivatives(t_k + dt, y_k + dt * k3, stimulus, params), dtype=float)

        y[k + 1] = y_k + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    i_stim = sample(stimulus, t)
    return Trace(
        t=t,
        V=y[:, 0],
        m=y[:, 1],
        h=y[:, 2],
        n=y[:, 3],
        I_stim=i_stim,
        params=params,
    )