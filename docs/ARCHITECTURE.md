# AnyBody (`A://`): First-Principles Biophysical Architecture

<p align="center">
  <img src="../assets/logos/anybody_logo.svg" alt="AnyBody Logo (A://)" width="420"/>
</p>

## 1. Architectural Thesis: Replacing Optimization Loops with Differential Equations

Conventional humanoid and multi-morphology robotics stacks depend on three fragile abstractions:
1. **Hand-tuned joint-space PID controllers** ($K_p, K_d$) or body-specific trajectory optimizers that fight external perturbations with stiff mechanical impedance ($\approx 1.22\text{ kW}$ continuous power draw).
2. **Dense backpropagation-through-time (BPTT) or PPO policies** requiring millions of reward-shaped simulation rollouts per task and morphology.
3. **Hardcoded kinematic lookup tables** that break immediately when transferred across morphologies (e.g., from a 19-DOF Unitree H1 humanoid to a 44-DOF whole-body humanoid with 5-finger hands, a 12-DOF quadruped dog, or a 7-DOF industrial arm).

**AnyBody** (`A://`) eliminates all three. Instead of training a monolithic neural network to output joint angles, AnyBody implements a **4-layer, 4,134,584-neuron, 25,107,740-synapse (`25.11M`) spiking human motor connectome in pure Rust (`#![forbid(unsafe_code)]`)**, coupled at $1,000\text{ Hz}$ ($\Delta t = 1\text{ ms}$) to antagonistic Hill-type muscle-tendon units and self-organized through prenatal motor babbling.

---

## 2. Exact Connectome Scale & Memory Layout (`crates/subsystems/src/lib.rs`)

All neural populations use 64-byte cache-line-aligned Structure-of-Arrays (`#[repr(C, align(64))]` in [`crates/neuro_core/src/lib.rs`](../crates/neuro_core/src/lib.rs)) with zero heap allocations inside the $1,000\text{ Hz}$ loop. The connectome supports three runtime scale profiles (`ScaleProfile` in [`crates/subsystems/src/lib.rs`](../crates/subsystems/src/lib.rs)):

| Subsystem Layer | Cell Types & Biophysical Specializations | `FullScaleHuman` Neurons | `FullScaleHuman` Synapses | Plasticity & Learning Rule |
| :--- | :--- | ---: | ---: | :--- |
| **Layer 1: Prefrontal & Vision (`meta_prefrontal.rs`)** | Meta `Muse-Spark-1.3` / `Muse-Open` + `SAM-3.1` (`ego_head_cam` $116^\circ$ FOV RGB-D) + Parietal 3D Working Memory | Executive / Retinal Stream | Cortical Apical Context (`14-D` Attractor Bus) | Zero-shot natural-language & visual task-graph synthesis + autonomous self-chaining |
| **Layer 1B: Basal Ganglia (`basal_ganglia.rs`)** | Striatal $D_1$ (Go), $D_2$ (No-Go), $\text{GPe}$, $\text{STN}$ (Hyperdirect brake), $\text{GPi/SNr}$ tonic inhibition, $\text{SNc}$ dopamine | `704` (`7` channels $\times$ `88` pools + `SNc`) | `1,936` | Dopamine RPE ($\delta_{\text{DA}}$) 3-factor cortico-striatal gating |
| **Layer 2: Neocortex (`cortex.rs` + `neuro_core`)** | `2,500` minicolumns ($50\times 50$ grid) $\times$ `6` cells: **3-Compartment `L5` Pyramidal** + `L2/3`, `L4`, `PV`, `SST`, `VIP` | `15,000` (`2,500` $\times$ `6` cells) | `390,036` | **4-Timescale Synapse Bank**: Short-term depression/facilitation, `1,000 ms` **BTSP** plateau trace, Oja/STDP, and **Fusi cascade metaplasticity** |
| **Layer 3: Cerebellum (`cerebellum.rs`)** | `88` microzones $\times$ (`46,800` Granule + Golgi + MLI + Purkinje + Deep Cerebellar Nuclei) | `4,118,752` | `24,710,400` | $\text{mGluR1}$ ($\tau = 45\text{ ms}$, discrete peak at $44\text{ ms}$) delayed eligibility + Kawato Feedback-Error Learning (Purkinje LTD) |
| **Layer 4: Spinal Cord & Plant (`spinal.rs` + `body_physics`)** | `14-D` $I_{\text{NaP}}$ half-center CPG + `128` $\text{Ia/Ib}$ Renshaw interneurons + `88` $\alpha/\gamma$ motor pools $\to$ `88` Hill muscles (`44` DoFs) | `128` (+ `88` $\alpha/\gamma$ pools) | `5,368` | Sensory-gated CPG entrainment + $\alpha\text{--}\gamma$ co-activation + emergent cross-bridge stiffness |
| **Total (`ScaleProfile::FullScaleHuman`)** | **Complete 4-Layer Spiking Motor Connectome** | **`4,134,584`** | **`25,107,740` (`25.11M`)** | **100% Local Biophysical Plasticity (Zero BPTT, Zero PID)** |

* **Interactive Real-Time Profile (`ScaleProfile::InteractiveRealTime`):** `184,864` neurons and `1,178,540` (`1.18M`) synapses (`2,048` granules/microzone, $24\times 24 = 576$ cortical minicolumns), stepping faster than real time ($1,000\text{ Hz}$) on a single CPU thread.
* **High-Precision Real-Time Profile (`ScaleProfile::HighPrecisionRealTime`):** `727,840` neurons and `4,492,280` (`4.49M`) synapses (`8,192` granules/microzone, $32\times 32 = 1,024$ cortical minicolumns).
* **Fast Verification Profile (`ScaleProfile::FastVerification`):** `25,248` neurons and `182,444` (`0.18M`) synapses (`256` granules/microzone, $16\times 16 = 256$ cortical minicolumns) for rapid CI regression testing.

---

## 3. Layer-by-Layer Mathematical Formulation

### 3.1 Layer 2: Three-Compartment L5 Pyramidal Neurons & Larkum BAC Multiplexing (`crates/neuro_core/src/lib.rs`)

Standard point neurons cannot separate *bottom-up sensory reality* from *top-down executive intent* because both streams sum linearly at the soma. In [`PyramidalPopulationSoA`](../crates/neuro_core/src/lib.rs#L178-L355), every Layer-5 Thick-Tufted Pyramidal Technique (`L5 PT`) neuron is modeled with **three coupled electrotonic compartments**:
1. **Perisomatic Compartment ($V_s$):** Receives fast perisomatic basket-cell (`PV+`) shunting inhibition ($g_{\text{PV}}$) and integrates axial currents from the basal and apical dendrites:
   $$C_s \frac{dV_s}{dt} = -g_L(V_s - E_L) - g_{\text{PV}}(V_s - E_{\text{GABA}}) + g_{\text{ax,b}}(V_b - V_s) + g_{\text{ax,a}}(V_a - V_s) - w_{\text{adapt}}$$
2. **Basal Dendritic Compartment ($V_b$):** Receives bottom-up thalamic/proprioceptive ascending fibers ($\text{Ia}$ spindle velocity, $\text{Ib}$ Golgi tendon force, joint angles) with voltage-dependent $\text{Mg}^{2+}$ unblock of **NMDA plateau conductances** (`basal_nmda_plateau_ms`).
3. **Apical Tuft Compartment ($V_a$):** Receives top-down contextual goal attractors from the Prefrontal Cortex and Basal Ganglia (`Layer 1`), gated by Martinotti (`SST+`) dendritic inhibition and disinhibited by `VIP+` interneurons (`VIP -> SST` disinhibition in [`InhibitoryRoutingPool`](../crates/neuro_core/src/lib.rs#L359-L448)):
   $$C_a \frac{dV_a}{dt} = -g_{L,a}(V_a - E_L) + I_{\text{apical}} - g_{\text{SST}}(V_a - E_{\text{GABA}}) + g_{\text{Ca,HVA}}(V_a) \cdot (E_{\text{Ca}} - V_a) + g_{\text{bAP}} \cdot S_{\text{soma}}(t - 1\text{ ms})$$

* **Backpropagating Action Potential Activated Calcium (`BAC`) Bursting:** When a basal sensory input triggers a single somatic spike $S_{\text{soma}}(t) = 1$, a backpropagating action potential ($g_{\text{bAP}}$) sweeps into the apical tuft. If and only if the apical tuft is simultaneously depolarized by top-down Meta `Muse` / Basal Ganglia intent (and disinhibited by `VIP+` interneurons), $V_a$ crosses the high-voltage-activated $\text{Ca}^{2+}$ threshold ($-38\text{ mV}$), igniting a $25\text{ ms}$ dendritic $\text{Ca}^{2+}$ plateau (`apical_ca_plateau_ms`) that drives a **high-frequency somatic burst (3–4 spikes at $>200\text{ Hz}$)**.
* **Behavioral Time-Scale Synaptic Plasticity (`BTSP`) & 4-Timescale Cascade (`FourTimescaleSynapseBank`):** Apical $\text{Ca}^{2+}$ plateaus open a $1,000\text{ ms}$ eligibility trace (`btsp_eligibility_trace`) and consolidate into Fusi-style metaplastic hidden states (`metaplastic_state`), preventing catastrophic forgetting across multi-skill sequences (**`0.00%` degradation** verified in `Test 0.2`).

---

### 3.2 Layer 1: Prenatal Motor Babbling & Somatotopic Homunculus (`crates/subsystems/src/cortex.rs`)

How does a single brain control a **44-DOF whole-body humanoid with two 5-finger dexterous hands** (`crates/body_physics/src/lib.rs`), a **19-DOF Unitree H1 humanoid**, a **12-DOF Unitree Go2 / Spot quadruped**, or a **7-DOF Franka Panda arm** (`data/body_trunks/`) without knowing which motor wire connects to which limb?

During **Stage 1 (`run_prenatal_development`)**, the spinal cord emits spontaneous uncorrelated motor twitches across all muscle channels (`NUM_MUSCLES = 88` across `NUM_DOF = 44` joints in 7 kinematic chains: Left Leg, Right Leg, Torso, Left Arm, Right Arm, Left 5-Finger Hand, Right 5-Finger Hand). Returning proprioceptive spindle bursts ($\text{Ia}, \text{Ib}$) project onto the 2D cortical sheet. Through local lateral Mexican-hat excitation/inhibition and Oja-normalized Hebbian STDP:
$$\Delta W_{ij} = \eta \, y_i \left( x_j - y_i W_{ij} \right)$$
adjacent cortical minicolumns lock onto physically coupled kinematic chains. Across `1,500` prenatal babbling episodes (`Test 1.1`), **Topological Disorder (mean kinematic tree distance between neighboring cortical columns)** drops from **`6.78` (`t=0`, random wiring) to `3.13` (`t=1500`)** — a **`53.8%` reduction in topological disorder (`2.17x` topographic organization gain)** with **zero kinematic labels**.

---

### 3.3 Layer 3: Cerebellar $\text{mGluR1}$ Delay Cancellation (`crates/subsystems/src/cerebellum.rs`)

Biological and physical hardware loops suffer from peripheral sensory conduction delays ($\tau_d \approx 30\text{ ms}$). A naive reactive feedback loop oscillates violently because corrective torques arrive $30\text{ ms}$ too late.

In [`CerebellarSubsystem`](../crates/subsystems/src/cerebellum.rs), Mossy Fibers ($\text{MF}$) carrying efference copies of cortical/CPG commands expand into a high-dimensional basis across **Granule Cells (`46,800` per microzone at `FullScaleHuman`)**. At the Parallel Fiber $\to$ Purkinje Cell ($\text{PF}\to\text{PC}$) synapse, glutamate binding activates the metabotropic **$\text{mGluR1} \to \text{IP}_3 \to \text{Ca}^{2+}$ second-messenger cascade**, modeled as a dual-exponential eligibility filter (`Mglur1EligibilityTrace` in [`crates/neuro_core/src/lib.rs`](../crates/neuro_core/src/lib.rs#L454-L511)):
$$h(t) = \frac{t}{\tau_{\text{mGluR1}}^2} \exp\!\left(-\frac{t}{\tau_{\text{mGluR1}}}\right), \qquad \tau_{\text{mGluR1}} = 45.0\text{ ms}\quad (\text{discrete peak at } t = 44\text{ ms})$$

When a delayed vestibular or spinal slip error arrives via the **Climbing Fiber ($\text{CF}$)** $30\text{ ms}$–$45\text{ ms}$ *after* the causal motor command, the $\text{PF}$ eligibility trace $e_{\text{PF}}(t)$ is at its exact peak (`Test 0.3`). Climbing Fiber complex spikes induce **Kawato Feedback-Error Learning (`FEL`)** via Long-Term Depression (`LTD`):
$$\Delta w_{\text{PF}\to\text{PC}} = -\eta_{\text{LTD}} \cdot e_{\text{PF}}(t) \cdot \left(S_{\text{CF}}(t) - S_0\right)$$
Because Purkinje cells inhibit the **Deep Cerebellar Nuclei ($\text{DCN}$)**, Purkinje LTD *disinhibits* the $\text{DCN}$ $30\text{ ms}$ **ahead** of the disturbance, reducing postural tracking RMS error by **`-16.9%` (`0.3950 -> 0.3282`)** and cutting emergency spinal reflex effort by **`-58.9%`** (`Stage 2`).

---

### 3.4 Layer 4: Spinal CPG, $\alpha\text{--}\gamma$ Co-Activation & 88 Hill-Type Muscles (`crates/body_physics/src/lib.rs`)

At the bottom of the hierarchy, AnyBody contains **zero PID controllers**:
* **Spinal Half-Center CPG (`crates/subsystems/src/spinal.rs`):** Rhythm generation is driven by persistent sodium ($I_{\text{NaP}}$) half-center oscillators with mutual glycinergic inhibition (`InaP`). Stance-to-swing phase transitions are **100% sensory-gated**: swing cannot initiate until ipsilateral Golgi Tendon Organs ($\text{Ib}$) verify unloading and contralateral load receptors confirm weight acceptance.
* **88 Antagonistic Hill-Type Muscle-Tendon Units (`crates/body_physics/src/lib.rs`):** Each of the `44` degrees of freedom (`NUM_DOF = 44`: 6 legL + 6 legR + 3 waist + 4 armL + 4 armR + 1 neck + 10 handL 5-finger + 10 handR 5-finger) is driven by an antagonistic flexor/extensor pair (`NUM_MUSCLES = 88`):
  $$F_{\text{muscle}} = F_{\max} \left[ a(t) \cdot f_L(\tilde{l}_m) \cdot f_V(\tilde{v}_m) + f_{\text{PE}}(\tilde{l}_m) \right]$$
  where $a(t)$ obeys asymmetric $\text{Ca}^{2+}$ excitation-contraction kinetics ($\tau_{\text{act}} = 12\text{ ms}, \tau_{\text{deact}} = 36\text{ ms}$).
* **Emergent Physical Stiffness Without Torque (`test_emergent_co_contraction_stiffness_without_torque`):** Co-contracting agonist and antagonist muscles equally yields **$0.000\text{ N}\cdot\text{m}$ net torque** while increasing physical joint impedance $\partial\tau/\partial\theta$ by **`>15x`** purely from actin-myosin cross-bridge force-length curvature.

---

### 3.5 Layer 1 Executive: Meta `Muse` + `SAM-3.1` Closed-Loop Retinal Grounding (`crates/subsystems/src/meta_prefrontal.rs` & `src/api/dual_client.py`)

Higher-level reasoning and perception operate asynchronously above the $1,000\text{ Hz}$ spinal-cortical loop:
1. **First-Person Retinal Stream (`SAM-3.1` on `ego_head_cam`):** Projects 3D workspace objects through the humanoid's moving head camera (`ego_head_cam`, $116^\circ \times 86^\circ$ FOV, pitch/yaw transformed at $1,000\text{ Hz}$). Objects outside the camera frustum cannot be targeted until the Prefrontal Cortex issues an active head/torso visual search sweep (`Stage 5.5` & `5.7`).
2. **Executive Task Compilation (`Meta Muse-Spark-1.3` / `Muse-Open` via Meta API & Baseten):** Natural-language or voice instructions (including arbitrary **negations**, e.g., *"Ignore the Red Box; only put the Blue Ball and Green Cylinder into the Green Basket"*) are compiled into a sequence of `14-D` cortical attractor states (`MotorStageCommand`) and an `agentic_target_mask`.
3. **Self-Chaining Agentic Loop (`Stage 5.7`):** For open-ended goals (*"Clean all the things in the basket"*), the Prefrontal Cortex autonomously re-queries the visual working memory after each object delivery (`4` delivery cycles + `1` final $360^\circ$ visual verification sweep = `5` autonomous executive cycles) and halts automatically when zero target objects remain outside the basket.
