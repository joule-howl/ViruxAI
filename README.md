# ViruxAI – Neuromorphic AI Engine

A complete neuromorphic AI system built from biological neuron models up to
embedded hardware deployment. The project covers the full pipeline: single-neuron
mathematics, spiking neural networks (SNNs), structural optimisation, FPGA
implementation, and real-time edge perception.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Modules](#running-the-modules)
- [Module Details](#module-details)
- [Experimental Results](#experimental-results)
- [References](#references)

---

## Architecture Overview

```
DVS Camera
    │
    ▼
Event Accumulator ──→ Event Frame [H×W]
    │
    ▼
SNN Detector (LIF layers)
    │
    ▼
Kalman Tracker ──→ (cx, cy, confidence)
    │
    ▼
Output / Control Signal
```

The entire processing pipeline is **event-driven** — computation is triggered
only by sensory changes, mirroring the biological neural reflex mechanism.
This approach achieves up to **99.6% energy savings** over equivalent
CNN-based edge systems in our simulation experiments.

---

## Project Structure

```
ViruxAI/
├── LIF/
│   ├── lif_neuron.py           # LIF single-neuron simulation
│   ├── lif_voltage_traces.png  # Membrane potential at various current levels
│   ├── fi_curve.png            # F-I curve (firing rate vs. stimulus)
│   └── lif_varied_input.png    # Response to time-varying input
│
├── SNN_STDP/
│   ├── snn_stdp.py             # SNN + STDP learning
│   ├── spike_encoding.png      # MNIST → spike train encoding
│   ├── stdp_weights.png        # Weight distribution before / after learning
│   ├── receptive_fields.png    # Hidden-layer receptive fields
│   └── raster_before_after.png # Spike activity before / after STDP
│
├── Optimization/
│   ├── optimization.py         # Pruning + State Space Model (SSM)
│   ├── lava_sim.py             # Neuromorphic Process Runtime (Process/Port/Runtime engine)
│   ├── rwkv_mini.py            # Mini RWKV (non-Transformer architecture)
│   ├── optimization_results.png
│   ├── lava_simulation.png
│   └── rwkv_comparison.png
│
├── FPGA/
│   ├── lif_neuron.v            # RTL Verilog – 16-bit fixed-point LIF neuron
│   ├── snn_core.v              # RTL Verilog – SNN Core top-level
│   ├── fpga_simulation.py      # Fixed-point Python simulation
│   └── fpga_simulation.png
│
├── EdgeAI/
│   ├── edge_ai.py              # DVS + SNN Detector + Kalman Tracker
│   └── edge_ai_results.png
│
├── data/                       # MNIST dataset (auto-downloaded)
├── TASK.txt
└── README.md
```

---

## Requirements

| Package | Minimum version |
|---|---|
| Python | 3.10+ |
| NumPy | 1.24+ |
| Matplotlib | 3.7+ |
| SciPy | 1.10+ |
| PyTorch | 2.0+ |
| snnTorch | 0.9+ |
| torchvision | 0.15+ |

---

## Installation

```bash
# Clone or extract the project
cd ViruxAI

# Install all dependencies
pip install numpy matplotlib scipy torch torchvision snntorch
```

The MNIST dataset is downloaded automatically the first time `SNN_STDP/snn_stdp.py` is run.

---

## Running the Modules

Run each module independently from the project root:

```bash
# LIF neuron simulation
python LIF/lif_neuron.py

# SNN + STDP learning on MNIST
python SNN_STDP/snn_stdp.py

# SNN pruning + SSM optimisation
python Optimization/optimization.py

# Neuromorphic Process Runtime (dataflow engine)
python Optimization/lava_sim.py

# Mini RWKV
python Optimization/rwkv_mini.py

# FPGA fixed-point simulation
python FPGA/fpga_simulation.py

# Edge AI: DVS + SNN + Kalman tracking
python EdgeAI/edge_ai.py
```

Each script saves its output plots as `.png` files in the corresponding directory.

---

## Module Details

### LIF – Leaky Integrate-and-Fire Neuron

**File:** `LIF/lif_neuron.py`

Implements the LIF membrane equation using the explicit Euler method:

$$\tau \frac{dV}{dt} = -(V - V_{rest}) + R \cdot I(t)$$

When $V \geq V_{th}$, the neuron fires an action potential and resets to
$V_{reset}$, then enters an absolute refractory period $t_{ref}$.

| Parameter | Value |
|---|---|
| $\tau$ (time constant) | 20 ms |
| $V_{rest}$ | −70 mV |
| $V_{th}$ (threshold) | −55 mV |
| $V_{reset}$ | −75 mV |
| $t_{ref}$ | 2 ms |

**Outputs:**
- `lif_voltage_traces.png` — Membrane potential for four input current levels
- `fi_curve.png` — Linear increase of firing rate with stimulus intensity
- `lif_varied_input.png` — Response to sinusoidal + random-pulse input

---

### SNN_STDP – Spiking Neural Network & STDP Learning

**File:** `SNN_STDP/snn_stdp.py`

Builds a two-layer SNN (784 → 64 LIF neurons) and trains it with **STDP** —
a gradient-free, local learning rule that mirrors the Hebbian plasticity
observed in biological synapses.

**Poisson Rate Coding:**

Each pixel $p \in [0, 1]$ is encoded into an independent Bernoulli spike
train over $T = 100$ timesteps:

$$s_k(t) \sim \text{Bernoulli}(p_k), \quad t = 1, \dots, T$$

**STDP Rule:**

$$\Delta w = \begin{cases}
  A_+ \cdot e^{-\Delta t / \tau_+} & \Delta t > 0 \text{ (pre before post)} \\
  A_- \cdot e^{+\Delta t / \tau_-} & \Delta t < 0 \text{ (post before pre)}
\end{cases}$$

| Parameter | Value |
|---|---|
| $A_+$ | 0.005 |
| $A_-$ | −0.003 |
| $\tau_+, \tau_-$ | 15 timesteps |
| Epochs | 20 |
| Samples / epoch | 200 |

**Result:** Mean weight increases from **0.15 → 0.37** after training.
Receptive fields develop localised patterns from MNIST statistics.

---

### Optimization – SNN Structure Optimisation

#### Pruning & Sparse Computation

**File:** `Optimization/optimization.py`

Prunes all connections where $|w| < \theta$ and represents the sparse matrix
in SciPy CSC format for true sparse matrix multiplication.

| Threshold $\theta$ | Sparsity | Connections remaining | RAM saved |
|---|---|---|---|
| 0.00 | 0.0% | 50,176 | 0% |
| 0.15 | 74.5% | 12,807 | 23.2% |
| 0.30 | 89.6% | 5,240 | 68.4% |

#### State Space Model (SSM / Mamba-style)

Linear time-invariant recurrence with **constant** hidden state size:

$$h[t] = A \cdot h[t-1] + B \cdot x[t], \quad y[t] = C \cdot h[t] + D \cdot x[t]$$

| Sequence Length T | Transformer RAM | SSM RAM | Ratio |
|---|---|---|---|
| 500 | 125.0 KB | 0.06 KB | 2,000× |
| 2,000 | 500.0 KB | 0.06 KB | 8,000× |
| 5,000 | 1,250.0 KB | 0.06 KB | 20,000× |

#### Neuromorphic Process Runtime

**File:** `Optimization/lava_sim.py`

Implements a lightweight **Process / Port / Runtime** execution engine for
multi-layer SNNs. Each computational unit (Process) communicates through
typed zero-copy channels (Ports) and is driven by a central time-step
scheduler. The engine runs a 3-layer network (128 → 64 → 32) with
dedicated `SpikeMonitor` processes for offline analysis.

Key design choices:
- `Port.recv()` returns a direct buffer view — no per-call allocation
- `SpikeMonitor` uses a pre-allocated numpy array instead of `list.append()`
- All state is explicit and inspectable, enabling clean debugging

#### Mini RWKV

**File:** `Optimization/rwkv_mini.py`

Two-layer RWKV with WKV recurrent formulation:

$$\text{num}[t] = e^{-w} \cdot \text{num}[t-1] + e^{k[t]} \cdot v[t]$$
$$\text{wkv}[t] = \frac{e^{u+k[t]} \cdot v[t] + \text{num}[t-1]}{e^{u+k[t]} + \text{den}[t-1]}$$

Hidden state: **1,536 bytes constant** (d_model=64, 2 layers).
At T=2,000, RWKV uses **666×** less memory than a Transformer KV-cache.

---

### FPGA – Fixed-Point Hardware Simulation

**Files:** `FPGA/fpga_simulation.py`, `FPGA/lif_neuron.v`, `FPGA/snn_core.v`

Simulates the Verilog RTL behaviour in Python using Q8.7 16-bit
signed fixed-point arithmetic. Adopts a **Non-Von Neumann** design:
weight BRAM is co-located with the MAC unit, eliminating the
memory-bandwidth bottleneck of conventional architectures.

**Estimated FPGA resources (Xilinx Artix-7, 64 neurons):**

| Resource | Count |
|---|---|
| LUT | 9,600 |
| Flip-Flop | 4,800 |
| BRAM | 8 |
| DSP | 128 |

**Power comparison (128 → 64 neurons, 500 timesteps):**

| Platform | Power | Saving |
|---|---|---|
| FPGA Artix-7 | ~530 mW | — |
| CPU Core i5 | ~3,500 mW | **84.9%** |

---

### EdgeAI – Event-based Vision & Object Tracking

**File:** `EdgeAI/edge_ai.py`

End-to-end system for detecting and tracking fast-moving objects using a
DVS event camera and an embedded SNN.

| Class | Role |
|---|---|
| `DVSCamera` | DVS simulator — fires events only on log-intensity changes |
| `EventAccumulator` | Aggregates events in a time window → event frame |
| `SNNDetector` | 3-layer LIF network — detects object position from event frame |
| `KalmanTracker` | 4-state Kalman filter — estimates position and velocity |

**Simulation results (150 frames, 64×64 sensor):**

| Metric | Value |
|---|---|
| Mean inference latency | ~9.6 ms |
| Tracking MAE | ~47 px |
| Energy saving vs. CNN edge | **99.6%** |

---

## Experimental Results

| Module | Key Metric | Result |
|---|---|---|
| LIF Neuron | Minimum firing threshold $I_{th}$ | ~1.85 nA |
| LIF Neuron | Firing rate range | 0 – 100+ Hz |
| STDP | Weight after training | 0.15 → 0.37 |
| Pruning (θ=0.15) | Sparsity / RAM saved | 74.5% / 23% |
| SSM vs. Transformer (T=5,000) | Memory ratio | 20,000× less |
| RWKV vs. Transformer (T=2,000) | Memory ratio | 666× less |
| FPGA vs. CPU | Power saving | 84.9% |
| Edge AI vs. CNN | Energy saving | 99.6% |

---

## References

- Mahowald, M. & Douglas, R. (1991). *A silicon neuron.* Nature, 354, 515–518.
- Gerstner, W. & Kistler, W. (2002). *Spiking Neuron Models.* Cambridge University Press.
- Bi, G. & Poo, M. (1998). *Synaptic modifications in cultured hippocampal neurons.* Journal of Neuroscience.
- Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752.
- Peng, B. et al. (2023). *RWKV: Reinventing RNNs for the Transformer Era.* arXiv:2305.13048.
- snnTorch documentation: https://snntorch.readthedocs.io/
- Gallego, G. et al. (2020). *Event-based Vision: A Survey.* IEEE TPAMI.
