# Theory and Biophysical Foundations of the Hodgkin–Huxley Model

This guide covers the biophysical foundations, mathematical formulation, numerical methods,
and network architecture implemented in this lab. It is written to serve as a comprehensive
conceptual reference.

---

## 1. The Neuron as an Electrical Circuit

The neuronal cell membrane functions as an electrical capacitor consisting of a thin insulating
lipid bilayer (approximately 3 to 4 nanometres thick) separating two conductive aqueous
electrolytic solutions: the intracellular cytoplasm and the extracellular interstitial fluid.
Embedded within this lipid matrix are specialized macromolecular channel proteins through which
specific ions can selectively pass.

In electrical circuit terms, the membrane is represented as:
1. A membrane capacitance `C` that stores electrical charge across the bilayer.
2. Three parallel conductive branches corresponding to independent transmembrane ionic pathways:
   a voltage-gated sodium conductance, a voltage-gated potassium conductance, and a non-gated
   passive leak conductance.
3. Electromotive batteries connected in series with each conductive branch, representing the
   electrochemical equilibrium (reversal) potentials `ENa`, `EK`, and `EL`.
4. An externally applied stimulus current density `I_stim`.

By applying Kirchhoff's current conservation law to an isopotential membrane patch, the total
membrane capacitive displacement current balances the sum of the injected stimulus current and
the transmembrane ionic currents:

`C dV/dt = I_stim - I_Na - I_K - I_L`

Sign convention and units:
* `V`: Membrane potential in millivolts (mV).
* `t`: Time in milliseconds (ms).
* `C`: Specific membrane capacitance in microfarads per square centimetre (µF/cm²).
* `I_stim`: Injected current density in microamperes per square centimetre (µA/cm²). Injected
  depolarising current into the cell interior is taken as positive.
* `I_Na`, `I_K`, `I_L`: Transmembrane ionic currents in µA/cm². We use the modern physiological
  convention where outward positive charge movement from the cytoplasm to the extracellular
  bath is positive.

---

## 2. The Currents and Parameter Table

Each ionic current is governed by Ohm's law applied across the net electrical driving force,
defined as the difference between the membrane potential `V` and that ion's reversal potential:

`I_Na = gNa * m³ * h * (V - ENa)`
`I_K  = gK * n⁴ * (V - EK)`
`I_L  = gL * (V - EL)`

### Standard Parameter Set (`HHParams`)

| Parameter | Biophysical Meaning | Value | Units |
|---|---|---|---|
| `C` | Specific membrane capacitance | 1.0 | µF/cm² |
| `gNa` | Maximal sodium conductance density | 120.0 | mS/cm² |
| `gK` | Maximal potassium conductance density | 36.0 | mS/cm² |
| `gL` | Passive leak conductance density | 0.3 | mS/cm² |
| `ENa` | Sodium reversal potential | +50.0 | mV |
| `EK` | Potassium reversal potential | -77.0 | mV |
| `EL` | Leak reversal potential | -54.387 | mV |
| `temp_C` | Reference base temperature | 6.3 | °C |
| `q10` | Temperature kinetic coefficient | 3.0 | dimensionless |

### Origin of Reversal Potentials (The Nernst Equation)
The reversal potential for an ionic species is the membrane voltage at which outward electrical
drift exactly balances inward chemical concentration gradient diffusion. For an ion species with
valence `z` at absolute temperature `T`, the equilibrium potential is determined by the Nernst
equation: `E_ion = (R * T / (z * F)) * ln([ion]_out / [ion]_in)`, where `R` is the universal gas
constant, `F` is the Faraday constant, and brackets represent ionic molar concentrations. In the
squid giant axon, active sodium-potassium ATPase exchange pumps maintain high intracellular
potassium and low intracellular sodium. This chemical imbalance makes `ENa` strongly positive
(+50.0 mV) and `EK` strongly negative (-77.0 mV). The leak reversal potential `EL` (-54.387 mV)
reflects passive chloride and residual ion permeability and is tuned to maintain electrical
stability at the resting potential (-65.0 mV).

---

## 3. Gating Variables and Kinetic Equations

Hodgkin and Huxley conceptualised voltage-dependent ionic conductances by hypothesising that
charged sub-molecular "particles" or "gates" within the channel pore transition between closed
and open states in response to the electric field across the membrane.

Let `x` represent the probability that a gating particle is in the permissive (open) state,
or equivalently the fraction of open gates in a large population:
* `1 - x`: Probability that a gate is in the non-permissive (closed) state.
* `α(V)`: Voltage-dependent rate constant for transitioning from closed to open (units 1/ms).
* `β(V)`: Voltage-dependent rate constant for transitioning from open to closed (units 1/ms).

The transition kinetics obey first-order reaction dynamics:

`dx/dt = α(V) * (1 - x) - β(V) * x`

Equivalently, this can be written in terms of an asymptotic steady-state activation `x_inf` and
a voltage-dependent relaxation time constant `τ`:

`τ(V) * dx/dt = x_inf(V) - x`

where:
`x_inf(V) = α(V) / (α(V) + β(V))`
`τ(V) = 1.0 / (α(V) + β(V))`

### The Six Empirical Rate Equations (at 6.3 °C)

For sodium activation (`m`):
* `α_m(V) = 0.1 * (V + 40) / (1 - exp(-(V + 40) / 10))`
* `β_m(V) = 4.0 * exp(-(V + 65) / 18)`

For sodium inactivation (`h`):
* `α_h(V) = 0.07 * exp(-(V + 65) / 20)`
* `β_h(V) = 1.0 / (1 + exp(-(V + 35) / 10))`

For delayed-rectifier potassium activation (`n`):
* `α_n(V) = 0.01 * (V + 55) / (1 - exp(-(V + 55) / 10))`
* `β_n(V) = 0.125 * exp(-(V + 65) / 80)`

### Why `m³ * h` and `n⁴`?
In their original voltage-clamp experiments, Hodgkin and Huxley observed that upon sudden
depolarisation, potassium currents rose with an initial sigmoidal delay before plateauing. A
single first-order variable `n` produces a purely exponential rise with maximal slope at t = 0.
To fit the sigmoidal lag, they hypothesised that potassium channels open only when four identical,
independent gating subunits are all simultaneously in their open positions, yielding open
probability `n⁴`. Similarly, sodium conductance turns on with an inflection and subsequently
decays, which they matched by postulating three independent fast activating `m` gates and one
slower inactivating `h` gate operating in series, yielding open probability `m³ * h`.

### Removable Singularities in Rate Equations
The mathematical expressions for `α_m(V)` at `V = -40 mV` and `α_n(V)` at `V = -55 mV` contain
terms of the form `f(x) = x / (1 - exp(-x / k))`. When `x = 0`, the expression evaluates to the
indeterminate form `0 / 0`. Applying L'Hôpital's rule shows that the mathematical limit as `x`
approaches 0 is finite and equals `k`:
`limit_{x -> 0} x / (1 - exp(-x / k)) = limit_{x -> 0} 1 / ((1 / k) * exp(-x / k)) = k`.

In floating-point hardware, evaluating `0 / 0` triggers a zero-division warning and produces NaN.
The codebase resolves this analytically using a helper `_x_over_one_minus_exp(x, k)`. For
`|x / k| < 1e-6`, it substitutes the first-order Taylor expansion `k * (1 - x / (2 * k))`.
To prevent division warnings during vectorized evaluation, `np.where` substitutes a safe nonzero
dummy argument inside the exponential before the branch selection occurs.

---

## 4. Voltage Convention

In their landmark 1952 paper, Hodgkin and Huxley defined membrane potential relative to the
resting potential, setting resting membrane potential as `v = 0 mV`, with depolarisation taken
as negative in their original sign notation.

In modern electrophysiology and throughout this codebase, membrane potential is referenced
to the extracellular bath ground:
* The resting membrane potential of the squid giant axon is `V_REST = -65.0 mV`.
* Depolarisation corresponds to increasingly positive potentials (`V > -65 mV`).
* Hyperpolarisation corresponds to increasingly negative potentials (`V < -65 mV`).

All empirical equations, reversal potentials, and state vectors in this project are strictly
formulated according to this modern `-65.0 mV` absolute convention.

---

## 5. Temperature Scaling and Q10

Biochemical gating rates are driven by conformational changes in channel protein domains,
which accelerate at higher temperatures according to Arrhenius kinetics.

The rate constants were originally measured at `T_REF_C = 6.3 °C`. For any experimental
temperature `T` (in °C), the kinetic transition rates `α` and `β` are accelerated by a
dimensionless factor `φ` (phi):

`φ = Q10^((T - 6.3) / 10.0)`

With `Q10 = 3.0`, every 10 °C increase in temperature triples the transition rates of all
six gating equations:
* `dx/dt = φ * (α_x(V) * (1 - x) - β_x(V) * x)`
* Equivalently, relaxation time constants shorten: `τ_effective(V) = τ(V) / φ`.

Physical scope and limitations of temperature modelling:
* The model captures kinetic acceleration: channels open and close much faster at higher
  temperatures, which dramatically narrows the action potential duration.
* The model does NOT scale maximum conductances (`gNa`, `gK`, `gL`) or reversal potentials with
  temperature (which in biological systems exhibit modest temperature dependence via Nernst's
  absolute temperature factor `T`).

---

## 6. Anatomy of One Action Potential

An action potential is an all-or-none electrical impulse driven by nonlinear, voltage-dependent
feedback loops between membrane voltage and ion conductances. The sequence unfolds in distinct
phases:

1. Sub-threshold depolarisation: An outward injected stimulus current `I_stim` charges the
   membrane capacitance, driving `V` upward from `-65 mV`.
2. Regenerative sodium activation: As `V` crosses the threshold around `-55 mV`, the fast sodium
   activation gate `m` opens rapidly (`τ_m` is short, approximately 0.1 to 0.5 ms). Inward
   positive sodium flux (`I_Na < 0`) depolarises the membrane further. This depolarisation recruits
   additional sodium channels, unleashing explosive regenerative positive feedback (`V -> m ->
   gNa -> inward current -> V`).
3. Spike peak: The rapid sodium influx drives `V` toward the sodium reversal potential `ENa = +50
   mV`. The membrane potential peaks around +40 mV, falling short of `ENa` because sodium
   inactivation commences and potassium activation begins.
4. Repolarisation: Two delayed mechanisms terminate the upstroke:
   * Sodium inactivation: The slower inactivation gate `h` closes (`τ_h` is longer, about 1 to 8.5 ms depending on voltage),
     shutting down inward sodium current despite persistent depolarisation.
   * Potassium activation: The delayed-rectifier gate `n` opens (`τ_n` is about 1 to 5.5 ms depending on voltage), generating
     a massive outward potassium current (`I_K > 0`) that evacuates positive charge and rapidly
     repolarises the membrane.
5. Afterhyperpolarisation (undershoot): Potassium channel deactivation is slow. When `V` returns
   to `-65 mV`, `n` remains elevated well above its resting level (`n > 0.3177`). This excess
   potassium conductance pulls the membrane potential toward `EK = -77.0 mV`, producing an
   undershoot reaching below `-72 mV`.
6. Refractoriness and recovery: During the absolute refractory period (coinciding with the spike
   and early undershoot), `h` is close to zero and `n` is high; no amount of current can trigger a spike. During
   the relative refractory period, `h` slowly resets toward its resting value (~0.60) while `n`
   relaxes back to ~0.32, gradually restoring resting excitability.

Hierarchy of kinetic timescales:
* Fast variable: `m` (sodium activation, tracks voltage almost instantaneously).
* Slow variables: `h` (sodium inactivation) and `n` (potassium activation).

---

## 7. Rheobase and Repetitive Firing

Two distinct current thresholds define neural excitation:
* Single-spike rheobase: The minimum sustained current amplitude required to evoke a single
  action potential during a prolonged current pulse.
* Onset of repetitive firing: The minimum sustained current required to maintain continuous,
  indefinite repetitive spike firing.

In the Hodgkin–Huxley model, repetitive firing emerges via a subcritical Hopf bifurcation:
* When injected current is elevated, the stable resting equilibrium loses stability not via a
  gradual frequency-zero onset (saddle-node on invariant circle / SNIC), but by shedding stability
  to a high-amplitude limit cycle.
* As a result, repetitive firing cannot occur at arbitrary low frequencies; when firing turns on,
  it jumps discontinuously to a non-zero firing rate (in this implementation about 55 Hz just above onset, see experiment 2).
* Because a single spike can be triggered dynamically before the system settles into silence,
  the single-spike rheobase (around 2.2 µA/cm²) is substantially lower than the sustained repetitive
  firing threshold (around 6.3 µA/cm²).
* This substantial gap and discontinuous frequency jump define Class 2 excitability, as classified
  by Alan Hodgkin (1948) and modern dynamical systems theory (see Izhikevich 2007; Dayan & Abbott
  2001).

---

## 8. Depolarisation Block

When a neuron is subjected to very strong depolarising current drive, repetitive firing does not
continue to increase indefinitely; instead, firing collapses, and the cell becomes trapped in
a depolarised plateau (depolarisation block).

Biophysical mechanism:
* Heavy continuous current injection holds the membrane potential at a persistently depolarised
  level (`V > -40 mV`).
* At these sustained positive voltages, the steady-state sodium inactivation variable `h_inf(V)`
  collapses to near zero.
* Because the membrane never repolarises into the negative range where the recovery rate `α_h`
  is large enough to re-open `h` gates, sodium channels stay inactivated.
* Deprived of available sodium channels, regenerative spike initiation fails, leaving the cell
  electrically quiescent at a depolarised sub-threshold plateau.

---

## 9. Pharmacological Channel Blockers

Neurotoxins that specifically target ionic channel pores are modeled by modifying maximal
conductance parameters:
* Tetrodotoxin (TTX): A potent neurotoxin found in pufferfish that physically blocks the
  extracellular pore of voltage-gated sodium channels. In the model, TTX is simulated by setting
  `gNa = 0`. Under TTX, the membrane behaves as a passive RC circuit with leak conductance;
  depolarising steps produce only a passive, RC-like charging response (slightly shaped by the slowly activating potassium channels) and no action potential.
* Tetraethylammonium (TEA): An organic cation that occludes the intracellular pore of
  delayed-rectifier potassium channels. In the model, TEA is simulated by setting `gK = 0`.
  Without potassium channels to repolarise the cell, an initial sodium spike fires normally,
  after which the cell is trapped at a protracted, depolarised plateau near +5 to +10 mV
  where the stimulus current is balanced by the leak current and the small residual sodium current that has not yet inactivated.

Limitations of binary conductance deletion:
* Real pharmacological blockers exhibit dose-dependent, reversible binding kinetics described
  by Michaelis–Menten or Hill equations, and may exert voltage-dependent or use-dependent
  blockade rather than static, all-or-none elimination.

---

## 10. The Excitatory–Inhibitory Network

The lab implements a recurrent network comprising `N = n_exc + n_inh` single-compartment neurons
(40 excitatory and 10 inhibitory cells in the default configuration).

### Conductance-Based Synaptic Dynamics
Synapses are modeled as transient conductance increases with exponential decay:

`dg_e/dt = -g_e / tau_exc`
`dg_i/dt = -g_i / tau_inh`

When presynaptic neuron `j` fires an action potential (upward zero crossing), the postsynaptic
conductance of target neuron `k` undergoes an instantaneous jump:
`g_e,k -> g_e,k + W_kj` (if neuron `j` is excitatory)
`g_i,k -> g_i,k + W_kj` (if neuron `j` is inhibitory)

The total synaptic current entering postsynaptic cell `k` is:

`I_syn = -g_e,k * (V_k - E_exc) - g_i,k * (V_k - E_inh)`

Reversal potentials:
* `E_exc = 0.0 mV`: Excitatory AMPA-like reversal potential, depolarising relative to rest.
* `E_inh = -75.0 mV`: Inhibitory GABA_A-like reversal potential. Because `E_inh` lies close to
  the potassium reversal potential and is more negative than the resting potential (-65 mV),
  inhibitory synaptic activation drives hyperpolarising outward current, reducing excitability.

### Connectivity Matrix
The network topology is an Erdős–Rényi random directed graph with connection probability
`p_connect = 0.2` and zero diagonal (no self-connections). The weight matrix `W[post, pre]`
is partitioned into four sub-blocks:
* E -> E: `w_ee = 0.02 mS/cm²`
* E -> I: `w_ei = 0.10 mS/cm²`
* I -> E: `w_ie = 0.20 mS/cm²`
* I -> I: `w_ii = 0.10 mS/cm²`

### Golomb–Hansel Synchrony Measure (`χ`)
Network-wide voltage synchrony is quantified using the metric introduced by Golomb and Hansel
(2000):

`χ = sqrt(Var(mean_i(V_i(t))) / mean_i(Var(V_i(t))))`

Interpretation:
* `χ = 1.0`: Perfect synchrony across all neurons (individual voltages fluctuate identically).
* `χ ≈ 1 / sqrt(N)`: Completely asynchronous, independent firing across `N` neurons.

### Population Spectrum
Population rhythmic activity is analyzed by binning excitatory spike counts into 1 ms bins,
subtracting the temporal mean, computing the one-sided real fast Fourier transform (rFFT),
and calculating spectral power `|rFFT(counts)|²`. Dominant rhythm frequencies are identified
by finding the maximum spectral peak between 5 Hz and 200 Hz.

---

## 11. Numerical Methods

### Why LSODA for Single-Neuron Simulations?
The Hodgkin–Huxley equations are mathematically stiff during the rapid action potential upstroke
(where `dm/dt` accelerates over sub-millisecond intervals) and non-stiff during slow inter-spike
recovery intervals. The Adams/BDF solver with automatic stiffness detection (SciPy's `LSODA`)
dynamically switches between Adams predictor-corrector methods for non-stiff regimes and backward
differentiation formulas (BDF) for stiff regimes. Setting `max_step = 0.1 ms` prevents the
adaptive stepper from skipping over brief current pulses.

### Classical Runge–Kutta 4 (RK4) Cross-Check
To verify LSODA trajectories against an independent numerical method, the lab includes a
fixed-step 4th-order Runge–Kutta solver:
* Evaluates four derivative slopes per step: `k1 = f(t, y)`, `k2 = f(t + dt/2, y + dt/2 * k1)`,
  `k3 = f(t + dt/2, y + dt/2 * k2)`, and `k4 = f(t + dt, y + dt * k3)`.
* Truncation error per step is fifth-order `O(dt⁵)`, giving global accuracy `O(dt⁴)`.
* Used to confirm that spike peak timing, latencies, and amplitudes are independent of solver
  implementation details.

### Fixed-Step Vectorised RK4 for Networks
Simulating 50 coupled neurons with adaptive ODE integrators creates excessive overhead because
asynchronous spikes force the adaptive step size to collapse constantly. The entire 50-neuron
state is advanced simultaneously as 1D NumPy arrays using fixed-step RK4 with `dt = 0.02 ms`.
Numerical convergence is demonstrated by verifying that halving `dt` to 0.01 ms changes total
spike counts by less than 5%.

### Spike Detection
Spikes are identified at upward crossings across `V = 0.0 mV` (`V[i] < 0 <= V[i+1]`). The precise
crossing time is linearly interpolated within the interval:
`t_cross = t[i] + ((0.0 - V[i]) / (V[i+1] - V[i])) * (t[i+1] - t[i])`.
A 2.0 ms lockout window prevents spurious double-counting during broad peaks.

---

## 12. Limitations of the Model

* Single-compartment assumption: Assumes an isopotential point neuron without dendritic arbor,
  soma, axon hillock, or axonal transmission cables.
* Squid giant axon physiology: Channel kinetics, resting potential (-65 mV), and gating functions
  represent cold-water invertebrate unmyelinated axons rather than warm mammalian neurons.
* Limited channel repertoire: Does not include persistent sodium currents, low-threshold
  T-type calcium currents, high-threshold L-type calcium channels, A-type potassium channels,
  or calcium-activated potassium currents (SK/BK channels).
* Absence of dynamic ion concentrations: Intracellular and extracellular concentrations are
  assumed fixed; ion accumulation and pump exhaustion are omitted.
* Simplified network architecture: The network assumes identical neurons, random Erdős–Rényi
  connectivity, and instantaneous synaptic conductance jumps with single-step transmission delays.

---

## 13. Questions to Expect and How to Answer Them

**1. Why does the membrane potential overshoot to positive values during an action potential?**
During the upstroke, opening of sodium channels drives sodium conductance far above potassium
conductance. The membrane potential is pulled toward the positive sodium reversal potential
(+50 mV), peaking around +40 mV before inactivation intervenes.

**2. What mechanism terminates the action potential spike?**
The spike is terminated by the dual action of delayed sodium channel inactivation (the `h` gate
closes) and delayed-rectifier potassium activation (the `n` gate opens). This rapidly quenches
inward sodium flux and accelerates outward potassium repolarisation current.

**3. What is rheobase?**
Rheobase is the minimum continuous current density required to elicit a single action potential
during a long depolarising step. Below this threshold, passive leak and outward potassium currents
subdue depolarisation before regenerative sodium activation can ignite.

**4. Why does an afterhyperpolarisation undershoot occur after the spike?**
Potassium channel deactivation is slow relative to repolarisation. As the membrane returns to rest,
delayed-rectifier potassium channels remain open above their baseline resting state, pulling the
voltage closer to the negative potassium reversal potential (-77 mV).

**5. How does increasing temperature affect action potential half-width?**
Higher temperatures scale the kinetic factor `φ = Q10^((T - 6.3)/10)`, shortening the time
constants of all gating transitions. Because sodium inactivation and potassium activation turn
on much faster, repolarisation occurs rapidly, drastically narrowing spike duration.

**6. How did you verify the accuracy of your numerical solver?**
The primary adaptive solver (LSODA) was independently cross-checked against a custom fixed-step
fourth-order Runge–Kutta (RK4) integration. Spike peak times, half-widths, and waveforms between
the two solvers agree within tight tolerances (first crossing discrepancy under 0.02 ms).

**7. Why implement the model from scratch rather than using NEURON or Brian?**
Writing it myself forced me to understand every term: the rate functions and their singularities, the
solver choice, and how to check the code against known behaviour. A simulator such as NEURON or Brian
is the right tool for large or multi-compartment models; this project is a learning implementation,
and the tests and the RK4 cross-check are how I trust it.

**8. What enhancements would you implement next to extend the project?**
Natural next steps include incorporating a multi-compartment cable model to study spatial action
potential propagation and velocity, adding A-type potassium currents to explore Class 1
excitability, and introducing short-term synaptic plasticity into the recurrent network.

**9. What is a key limitation of the 50-neuron network model?**
The network consists of identical point neurons connected with random Erdős–Rényi topology and
negligible axonal conduction delays. Real cortical networks feature heterogeneous cell types,
structured distance-dependent architectures, and finite conduction delays.

**10. What did you learn from running these numerical experiments?**
*Draft from my own runs; rewrite it in your own words before an interview.* One spike needs only
about 2.2 µA/cm², but sustained firing needs about 6.3 µA/cm² and starts at roughly 55 Hz, so the
neuron has no slow-firing regime (class 2). Very strong drive (about 65 µA/cm² and above) stops
the firing altogether (depolarisation block). Blocking sodium removes spikes and blocking
potassium leaves a depolarised plateau after one spike. Warming speeds the kinetics, narrowing the
spike from about 1.5 to 0.4 ms between 6.3 and 18.5 °C, but at 25 °C the spike fails at the same
drive (heat block in this model). In the network, stronger inhibition lowered the excitatory rate
from about 62 to about 31 Hz as the I→E weight rose from 0 to 0.5, and also lowered synchrony,
though the network stays strongly rhythmic.

---

## 14. Glossary of Fifteen Electrophysiological Terms

1. **Action potential**: A rapid, regenerative all-or-none electrical impulse generated by
   voltage-gated ion channels that propagates along excitable membranes.
2. **Activation gate**: A channel protein conformational subunit (e.g., `m` for sodium, `n` for
   potassium) that opens upon membrane depolarisation.
3. **Inactivation gate**: A channel protein subunit (e.g., `h` for sodium) that closes upon
   sustained depolarisation, terminating ionic conductance.
4. **Conductance**: The ease with which electric current flows through channel pores across the
   membrane, measured in millisiemens per square centimetre (mS/cm²).
5. **Reversal potential**: The transmembrane voltage at which the electrical and chemical driving
   forces for a specific ion species are in exact equilibrium.
6. **Membrane capacitance**: The electrical charge storage capacity of the lipid bilayer per unit
   area, typically 1.0 µF/cm².
7. **Depolarisation block**: The cessation of action potential firing under strong sustained
   depolarising drive caused by persistent sodium channel inactivation.
8. **Rheobase**: The minimal continuous stimulus current density required to trigger an action
   potential.
9. **Class 2 excitability**: Neuronal firing behaviour where repetitive spiking initiates at a
   discontinuous non-zero frequency via a Hopf bifurcation.
10. **Refractory period**: The recovery phase following an action potential during which the
    generation of subsequent spikes is impossible (absolute) or requires elevated drive (relative).
11. **Afterhyperpolarisation (AHP)**: The transient hyperpolarising undershoot below resting
    potential that follows an action potential, driven by slowly closing potassium channels.
12. **Q10 temperature coefficient**: The factor by which a biological reaction or gating rate
    accelerates when the temperature is elevated by 10 °C.
13. **Synchrony parameter (χ)**: A normalized statistical metric measuring the coherence of
    fluctuations across a population of neurons, spanning from 1 (perfect synchrony) to 1/sqrt(N).
14. **Subcritical Hopf bifurcation**: A mathematical boundary in dynamical systems where a stable
    equilibrium loses stability and abruptly jumps to large-amplitude limit-cycle oscillations.
15. **Removable singularity**: A mathematical point where a function is undefined in form (0/0)
    but possesses a well-behaved, finite limiting value accessible via L'Hôpital's rule.

---

## References

* Hodgkin, A. L. and Huxley, A. F. (1952), A quantitative description of membrane current and
  its application to conduction and excitation in nerve, J. Physiol. 117, 500–544.
* Dayan, P. and Abbott, L. F. (2001), Theoretical Neuroscience, MIT Press.
* Gerstner, W., Kistler, W. M., Naud, R. and Paninski, L. (2014), Neuronal Dynamics,
  Cambridge University Press.
* Izhikevich, E. M. (2007), Dynamical Systems in Neuroscience, MIT Press.
* Golomb, D. and Hansel, D. (2000), The number of synaptic inputs and the synchrony of large,
  sparse neuronal networks, Neural Computation 12, 1095–1113.