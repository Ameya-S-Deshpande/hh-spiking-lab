"""Injected stimulus current protocols for Hodgkin-Huxley simulations.

Every protocol constructor returns a closure f(t: float) -> float representing
applied current density in uA/cm^2 as a function of time in ms.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

Stimulus = Callable[[float], float]


def constant(amplitude: float) -> Stimulus:
    """Create a constant current stimulus protocol.

    Parameters
    ----------
    amplitude : float
        Current density in uA/cm^2.

    Returns
    -------
    Stimulus
        Stimulus closure evaluating current at time t in ms.
    """
    amp = float(amplitude)

    def _stim(t: float) -> float:
        return amp

    return _stim


def step(amplitude: float, t_on: float, t_off: float) -> Stimulus:
    """Create a rectangular current step protocol.

    Parameters
    ----------
    amplitude : float
        Current density during the step in uA/cm^2.
    t_on : float
        Onset time in ms (inclusive).
    t_off : float
        Offset time in ms (exclusive).

    Returns
    -------
    Stimulus
        Stimulus closure evaluating current at time t in ms.
    """
    amp = float(amplitude)
    ton = float(t_on)
    toff = float(t_off)

    def _stim(t: float) -> float:
        if ton <= t < toff:
            return amp
        return 0.0

    return _stim


def pulse_train(
    amplitude: float,
    t_first: float,
    width: float,
    period: float,
    n_pulses: int,
) -> Stimulus:
    """Create a train of rectangular pulses.

    Pulse k (for k from 0 to n_pulses - 1) is active on the interval:
    t_first + k * period <= t < t_first + k * period + width.

    Parameters
    ----------
    amplitude : float
        Pulse amplitude in uA/cm^2.
    t_first : float
        Onset time of the first pulse in ms.
    width : float
        Duration of each pulse in ms.
    period : float
        Inter-pulse interval period in ms.
    n_pulses : int
        Total number of pulses.

    Returns
    -------
    Stimulus
        Stimulus closure evaluating current at time t in ms.
    """
    amp = float(amplitude)
    t1 = float(t_first)
    w = float(width)
    p = float(period)
    n = int(n_pulses)

    def _stim(t: float) -> float:
        if n <= 0 or t < t1:
            return 0.0
        rel = t - t1
        if p > 0.0:
            k = int(rel // p)
            if 0 <= k < n:
                pulse_start = t1 + k * p
                if pulse_start <= t < pulse_start + w:
                    return amp
        return 0.0

    return _stim


def ramp(i_start: float, i_end: float, t_on: float, t_off: float) -> Stimulus:
    """Create a linearly ramped current stimulus protocol.

    Parameters
    ----------
    i_start : float
        Initial current density at t_on in uA/cm^2.
    i_end : float
        Final current density at t_off in uA/cm^2.
    t_on : float
        Onset time in ms (inclusive).
    t_off : float
        Offset time in ms (exclusive).

    Returns
    -------
    Stimulus
        Stimulus closure evaluating current at time t in ms.
    """
    i0 = float(i_start)
    i1 = float(i_end)
    ton = float(t_on)
    toff = float(t_off)

    def _stim(t: float) -> float:
        if ton <= t < toff:
            return float(i0 + (i1 - i0) * (t - ton) / (toff - ton))
        return 0.0

    return _stim


def total(*stimuli: Stimulus) -> Stimulus:
    """Create a composite stimulus returning the linear superposition of stimuli.

    Parameters
    ----------
    *stimuli : Stimulus
        One or more stimulus callables.

    Returns
    -------
    Stimulus
        Stimulus closure evaluating the summed current at time t in ms.
    """
    stims = list(stimuli)

    def _stim(t: float) -> float:
        return float(sum(s(float(t)) for s in stims))

    return _stim


def sample(stimulus: Stimulus, t: np.ndarray) -> np.ndarray:
    """Evaluate a stimulus closure over a 1D array of time points.

    Parameters
    ----------
    stimulus : Stimulus
        Current stimulus function.
    t : np.ndarray
        Time points in ms.

    Returns
    -------
    np.ndarray
        Sampled stimulus current densities in uA/cm^2.
    """
    return np.array([stimulus(float(x)) for x in t], dtype=float)