"""Modern convention Hodgkin-Huxley parameters.

This is the modern convention in which the resting potential is -65 mV (Hodgkin and Huxley's
1952 paper measured voltage relative to rest). The parameters describe the squid giant axon
at 6.3 °C. Temperature acts only through the kinetic factor phi.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

T_REF_C: float = 6.3
V_REST: float = -65.0


@dataclass(frozen=True)
class HHParams:
    """Electrophysiological parameters for single-compartment Hodgkin-Huxley dynamics.

    Parameters
    ----------
    C : float
        Membrane capacitance in uF/cm^2.
    gNa : float
        Maximal sodium conductance in mS/cm^2.
    gK : float
        Maximal potassium conductance in mS/cm^2.
    gL : float
        Leak conductance in mS/cm^2.
    ENa : float
        Sodium reversal potential in mV.
    EK : float
        Potassium reversal potential in mV.
    EL : float
        Leak reversal potential in mV.
    temp_C : float
        Experimental temperature in °C.
    q10 : float
        Kinetic temperature coefficient Q10.
    """

    C: float = 1.0
    gNa: float = 120.0
    gK: float = 36.0
    gL: float = 0.3
    ENa: float = 50.0
    EK: float = -77.0
    EL: float = -54.387
    temp_C: float = 6.3
    q10: float = 3.0

    def phi(self) -> float:
        """Compute the kinetic acceleration factor for gating rates.

        Returns
        -------
        float
            Dimensionless scaling factor phi.
        """
        return self.q10 ** ((self.temp_C - T_REF_C) / 10.0)

    def with_(self, **changes) -> HHParams:
        """Return a copy of the parameter set with replaced fields.

        Parameters
        ----------
        **changes : Any
            Parameter fields to replace.

        Returns
        -------
        HHParams
            Updated parameter instance.
        """
        return replace(self, **changes)


DEFAULT_PARAMS = HHParams()