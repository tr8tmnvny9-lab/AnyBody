<p align="center">
  <img src="assets/logos/anybody_logo.svg" alt="AnyBody Logo (A://)" width="540"/>
</p>

<h1 align="center">AnyBody (<code>A://</code>)</h1>

<p align="center">
  <strong>Universal Body-Agnostic &amp; Task-Agnostic Neuromorphic Motor Brain in Pure Rust</strong><br/>
  <em>25.11M Spiking Synapses · 4.13M Neurons · 1,000 Hz Closed-Loop Biomechanics · Meta Muse + SAM-3.1 Prefrontal Grounding · Zero PID Controllers</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Language-Pure_Rust_2021_(%23!%5Bforbid(unsafe__code)%5D)-0052CC?style=for-the-badge&logo=rust" alt="Pure Rust"/>
  <img src="https://img.shields.io/badge/Connectome-25.11M_Synapses_%7C_4.13M_Neurons-7C3AED?style=for-the-badge" alt="25.11M Synapses"/>
  <img src="https://img.shields.io/badge/Meta_Models-Muse--Spark--1.3_%7C_Muse--Open_%7C_SAM--3.1-0081FB?style=for-the-badge&logo=meta" alt="Meta Foundation Models"/>
  <img src="https://img.shields.io/badge/Efficiency-2.89x_Performance_Per_Watt-10B981?style=for-the-badge" alt="2.89x Performance Per Watt"/>
  <img src="https://img.shields.io/badge/Verification-6_Stages_%2B_146_Tests_PASS-059669?style=for-the-badge" alt="Verification Pass"/>
</p>

---

## Executive Overview

Modern humanoid and multi-morphology robotics stacks are trapped between two brittle extremes:
1. **Hand-tuned joint-space PID controllers ($K_p, K_d$)** that fight gravity and external contacts with rigid co-contraction, burning **`~1,221 W`** continuously and failing immediately when transferred to a new body topology.
2. **Reward-engineered reinforcement learning (PPO / BPTT)** that requires millions of body-specific simulation rollouts per skill and collapses when given compositional natural-language instructions or negations.

**AnyBody (`A://`)** replaces both with a **first-principles, 4-layer spiking human motor connectome (`4,134,584` neurons, `25,107,740` synapses)** written in 100% safe, zero-dependency Rust (`crates/`) and coupled at **`1,000 Hz` ($\Delta t = 1\text{ ms}$)** to **88 antagonistic Hill-type muscle-tendon units (`44` whole-body & 5-finger dexterous hand DoFs)** as well as multi-morphology MuJoCo bodies (**Unitree H1 `19-DOF`**, **Unitree G1 `43-DOF` with 5-finger hands**, **Unitree Go2 / Spot `12-DOF` quadrupeds**, and **Franka Panda `7-DOF` arm**).

<p align="center">
  <img src="assets/figures/design_snap_02_architecture_brain.png" alt="AnyBody 4-Layer 25.11M-Synapse Architecture X-Ray" width="96%"/>
</p>

---

## Key Breakthroughs & Falsifiable Empirical Results

Every number in this repository is deterministically reproduced from source in ~110 seconds via `cargo test --release && cargo run --release -p v3_runner --bin verify_all`:

| Architectural Pillar | Biological Mechanism (`crates/`) | Falsifiable Benchmark & Empirical Result |
| :--- | :--- | :--- |
| **1. Zero-Cheat Falsifiable Physics** | 88 Hill-type muscle-tendon units (`ca_activation` $\tau = 12/36\text{ ms}$) across 44 DoFs; zero joint-angle PID servos (`crates/body_physics`) | **Untrained 0-weight brain physically collapses under gravity in `0.179 s`** (`Audit 4A`), proving zero hidden PD loops or virtual strings exist. Learned connectome walks at **`1.76 m/s`** (`Score = 99.6/100`, `0.00000 cm` bone stretch error). |
| **2. Body-Agnostic Prenatal Self-Organization** | Spontaneous spinal motor babbling (`1,500` twitches) + Oja-normalized Hebbian STDP on a 2D cortical sheet (`crates/subsystems/src/cortex.rs`) | Self-sorts a somatotopic homunculus across 44 DoFs / 7 kinematic chains with **zero kinematic labels**: **Topological Disorder drops `6.78 -> 3.13` (`-53.8%` disorder / `2.17x` topographic organization gain)**. |
| **3. 3-Compartment Cortical BAC Multiplexing** | Perisomatic ($V_s$), Basal NMDA ($V_b$), and Apical $\text{Ca}^{2+}$ tuft ($V_a$) `L5` pyramidal cells + `PV/SST/VIP` disinhibition + 4-timescale BTSP/Fusi synapses (`crates/neuro_core`) | Coincident basal sensory input + top-down apical Meta `Muse` context ignites $25\text{ ms}$ dendritic $\text{Ca}^{2+}$ plateaus (`3-spike BAC bursts`). Sequential skill learning exhibits **`0.00%` catastrophic forgetting** (`Test 0.2`). |
| **4. Cerebellar $\text{mGluR1}$ Delay Cancellation** | `88` microzones $\times$ `46,800` Granule cells (`24.71M` synapses) with second-order $\text{mGluR1} \to \text{IP}_3 \to \text{Ca}^{2+}$ eligibility ($\tau = 45\text{ ms}$, discrete peak at `44 ms`) (`cerebellum.rs`) | Cancels **`30 ms` peripheral sensory conduction delay** in real time via Kawato Feedback-Error Learning (Purkinje LTD), reducing postural RMS error by **`-16.9%`** and emergency spinal reflex effort by **`-58.9%`** (`Stage 2`). |
| **5. `2.89x` Performance-per-Watt Advantage** | Event-driven sparse spiking connectome + compliant Hill-type muscle pendulum dynamics vs. continuous 1 kHz dense stiff-PD servo (`Stage 4` & `Stage 5B`) | **`84.2%` spike sparsity** (`26.6 mW` neuromorphic compute vs. `42.84 W` dense baseline). Total system power drops from **`1,220.7 W` (stiff PD) to `422.0 W` (AnyBody)** — a **`2.89x` (`+189%`) performance-per-watt gain** (`798.7 W` saved). |
| **6. Meta `Muse` + `SAM-3.1` Closed-Loop Agency** | First-person `ego_head_cam` ($116^\circ \times 86^\circ$ FOV) segmented by `SAM-3.1` + `Muse-Spark-1.3` / `Muse-Open` self-chaining task compiler (`meta_prefrontal.rs` & `src/api/dual_client.py`) | Zero-shot execution of **7 closed-loop 3D tasks (`Tests 5.3–5.9`)**: out-of-FOV `134°` visual search, autonomous 4-object room cleanup (`5` self-chaining LLM cycles + $360^\circ$ audit), selective sorting, and **negation filtering** (*"Clean everything except the red box"*). |

<p align="center">
  <img src="assets/figures/design_snap_05_demo3_agentic_ui.png" alt="AnyBody Closed-Loop Meta Muse + SAM-3.1 Autonomous Cleanup" width="96%"/>
</p>

---

## System Architecture (`25.11M` Synapses Across 4 Layers)

```mermaid
flowchart TB
    subgraph L1["Layer 1: Prefrontal Executive & Retinal Vision (Meta Foundation Models + Basal Ganglia)"]
        MUSE["Meta Muse-Spark-1.3 & Muse-Open<br/>Natural Language, Negation & Self-Chaining Task Graph"]
        SAM["Meta SAM-3.1 on ego_head_cam (116°x86° FOV)<br/>First-Person RGB-D Segmentation -> 3D Retinal Centroids"]
        BG["Basal Ganglia Action Gate (704 Neurons)<br/>Striatal D1 (Go) / D2 (No-Go) / STN Hyperdirect Brake / SNc Dopamine"]
        MUSE --> BG
        SAM --> BG
    end

    subgraph L2["Layer 2: Somatotopic Neocortex (15,000 Neurons · 390,036 Synapses · 2,500 Minicolumns)"]
        L5["3-Compartment L5 Pyramidal Neurons<br/>Soma (Vs) + Basal NMDA (Vb) + Apical Ca2+ Tuft (Va BAC Bursting)"]
        INH["Canonical PV / SST / VIP Interneuron Routing<br/>VIP -> SST Disinhibition Gates Top-Down Apical Context"]
        SYN["4-Timescale Synapse Bank<br/>Short-Term Depression · 1,000ms BTSP Trace · Oja/STDP · Fusi Metaplasticity"]
        INH --> L5
        L5 --> SYN
    end

    subgraph L3["Layer 3: Predictive Cerebellum (4,118,752 Neurons · 24,710,400 Synapses · 88 Microzones)"]
        GC["4.12M Granule Cells (Sparse 4-Modality Expansion)<br/>45ms mGluR1 -> IP3 -> Ca2+ Delayed Eligibility Kernel (Peak @ 44ms)"]
        PC["Purkinje Cells & Deep Cerebellar Nuclei (DCN)<br/>Kawato Feedback-Error Learning Cancels 30ms Sensory Delay"]
        GC --> PC
    end

    subgraph L4["Layer 4: Spinal Cord CPG & Biomechanical Plant (128 Interneurons · 88 α/γ Pools · 88 Hill Muscles)"]
        CPG["14-D Persistent-Sodium (INaP) Half-Center CPG<br/>100% Sensory-Gated Stance/Swing Transitions + 25ms Palmar Slip Reflex"]
        HILL["88 Antagonistic Hill-Type Muscle-Tendon Units -> 44 DoFs<br/>Ca2+ Excitation-Contraction (τ=12/36ms) + Emergent Cross-Bridge Stiffness"]
        CPG --> HILL
    end

    BG -->|"14-D Apical Context + Thalamic Gate"| L2
    SAM -->|"Parietal 3D Working Memory"| L2
    L2 -->|"Corticospinal Drive + Efference Copy"| L3
    L2 -->|"Descending Voluntary Drive"| L4
    L3 -->|"30ms Predictive DCN Correction"| L4
    HILL -->|"30ms Delayed Ia Spindle, Ib Golgi & Vestibular Stream"| L2
    HILL -->|"30ms Delayed Climbing Fiber Error"| L3
```

### Scale Profiles (`ScaleProfile` in [`crates/subsystems/src/lib.rs`](crates/subsystems/src/lib.rs))

| Scale Profile | Granules / Microzone | Cortical Minicolumns | Total Neurons | Total Synapses | Typical Use Case |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **`FullScaleHuman`** | `46,800` ($\times 88$) | $50 \times 50 = 2,500$ | **`4,134,584`** | **`25,107,740` (`25.11M`)** | Full-scale human cerebellar & cortical connectome benchmark (`--full-scale`) |
| **`HighPrecisionRealTime`** | `8,192` ($\times 88$) | $32 \times 32 = 1,024$ | **`727,840`** | **`4,492,280` (`4.49M`)** | Sub-millisecond sparse CSR high-precision real-time control |
| **`InteractiveRealTime`** | `2,048` ($\times 88$) | $24 \times 24 = 576$ | **`184,864`** | **`1,178,540` (`1.18M`)** | Default 1,000 Hz interactive 3D viewer & Stage 4/5A/5B benchmark |
| **`FastVerification`** | `256` ($\times 88$) | $16 \times 16 = 256$ | **`25,248`** | **`182,444` (`0.18M`)** | Rapid unit testing & Stage 5C 7-prompt closed-loop CI verification |

---

## Quick Start & 1-Command Verification

### Prerequisites
* **Rust Toolchain:** `rustup` (`cargo` / `rustc >= 1.75`, zero external C/C++ libraries required for the entire 6-stage Rust verification suite).
* **Optional Python Environment:** Python `3.10+` (`pip install -r requirements.txt`) for the multi-morphology MuJoCo test suite (`pytest`) and 4K UHD video rendering scripts (`scripts/video/`).

### 1. Run All Rust Unit Tests & the 6-Stage Neurodevelopmental Verification Suite
```bash
# Run fast unit tests across neuro_core, body_physics, and subsystems (8 unit tests)
cargo test --release

# Run the complete 6-Stage Biophysical, Wattage & Closed-Loop Agentic Verification Suite (~110s)
# (Pass `-- --full-scale` to run Stage 4 & Stage 5 on the full 25.11M-synapse profile)
cargo run --release -p v3_runner --bin verify_all
```

### 2. Launch the Live 1,000 Hz Interactive 3D Humanoid & Single-Neuron Web Viewers
```bash
cargo run --release -p v3_runner --bin live_viewer
```
Then open in your browser:
* **`http://127.0.0.1:8421`** — **Live 3D Humanoid & Spiking Telemetry Studio** ([`docs/diagnostics/live_3d_humanoid_viewer.html`](docs/diagnostics/live_3d_humanoid_viewer.html)): type any natural-language prompt into the live Meta `Muse` + `SAM-3.1` prompt bar, toggle between `NeuromorphicHuman` (`422 W`) and `ConventionalDensePdMlp` (`1,221 W`), or trigger external force pushes in real time.
* **`http://127.0.0.1:8421/neuron`** — **Interactive 3-Compartment Neuron & 4-Timescale Synapse Explainer** ([`docs/diagnostics/human_neuron_synapse_explainer.html`](docs/diagnostics/human_neuron_synapse_explainer.html)).
* **Publication Figures:** See [`docs/diagnostics/seven_neuron_print_figure.html`](docs/diagnostics/seven_neuron_print_figure.html) and [`docs/diagnostics/seven_neuron_signal_propagation.pdf`](docs/diagnostics/seven_neuron_signal_propagation.pdf) for the 7-neuron biophysical signal-propagation diagrams.

### 3. Run the 138 Python Multi-Morphology & Meta AI Bridge Tests
```bash
pytest
```

### 4. Configure Live Meta AI & Baseten API Keys (Optional)
Copy `.env.example` to `.env` to enable live cloud calls to **Meta AI (`muse-spark-1.3`, `muse-open`, `sam-3.1`, `muse-wonder`)** and **Baseten** (when unset, `MetaPrefrontalSubsystem` and `DualAIClient` automatically use the deterministic biophysical task compiler so CI and offline verification always succeed):
```bash
cp .env.example .env
```

---

## Stage-by-Stage Empirical Telemetry (`verify_all`)

Running `cargo run --release -p v3_runner --bin verify_all` generates six publication-grade SVG telemetry plots in both `assets/figures/` and `docs/diagnostics/`:

### Stage 0: 3-Compartment `L5` Pyramidal BAC Multiplexing, `PV/SST/VIP` Routing & `mGluR1` Eligibility
* **`Test 0.1`:** Basal-only sensory input produces `4` single spikes (`0` bursts); coincident Basal + Apical input triggers `4` dendritic $\text{Ca}^{2+}$ plateaus (`10` spikes, `3` bursts).
* **`Test 0.2`:** `VIP -> SST` disinhibition gates BTSP + Fusi metaplastic consolidation across sequential tasks with **`0.00%` catastrophic forgetting**.
* **`Test 0.3`:** Second-order $\text{mGluR1}$ intracellular trace ($\tau = 45.0\text{ ms}$) rises from `0.000` at $t=0$ to its exact discrete peak at **`t = 44 ms`**.

<p align="center">
  <img src="assets/figures/stage0_electrophysiology.svg" alt="Stage 0 Electrophysiology Telemetry" width="92%"/>
</p>

### Stage 1: Prenatal Motor Babbling & Somatotopic Homunculus Self-Organization
* **`Test 1.1`:** Across `1,500` spontaneous motor babbling twitches across `44` DoFs (`88` muscles), cortical Oja/STDP self-organization reduces **Topological Disorder from `6.78` (`t=0`) to `3.13` (`t=1500`)** (**`-53.8%` disorder / `2.17x` topographic organization gain**).

<p align="center">
  <img src="assets/figures/stage1_homunculus_map.svg" alt="Stage 1 Somatotopic Homunculus Self-Organization" width="92%"/>
</p>

### Stage 2: Cerebellar Kawato Feedback-Error Learning Under `30 ms` Peripheral Delay
* **`Test 2.1`:** Under a `30 ms` sensory delay and sinusoidal floor tilt, Purkinje parallel-fiber LTD reduces postural RMS tracking error by **`-16.9%` (`0.3950 -> 0.3282`)** and cuts emergency spinal reflex effort by **`-58.9%`**.

<p align="center">
  <img src="assets/figures/stage2_cerebellar_fel_balance.svg" alt="Stage 2 Cerebellar Delay Cancellation" width="92%"/>
</p>

### Stage 3: Spinal CPG Locomotion, Basal Ganglia Gating & `25 ms` Tactile Slip Reflex
* **`Test 3.1`:** Sensory-gated $I_{\text{NaP}}$ half-center CPG sustains stable bipedal locomotion across `5` alternating footfalls (`1.93 m` forward progression).
* **`Test 3.2`:** Under a sudden $+2.0\text{ N}$ tangential load surge on a grasped object, the spinal/cortical tactile slip reflex boosts palmar grip force from `26.5 N` to **`42.9 N` (`+16.4 N` reflex boost)** in `<25 ms` with zero object drop.

<p align="center">
  <img src="assets/figures/stage3_locomotion_and_grasp.svg" alt="Stage 3 Spinal CPG & Tactile Slip Reflex" width="92%"/>
</p>

### Stage 4: Unpinned 3D Rigid-Bone Locomotion, Gravity Collapse Proof & `2.89x` Wattage Benchmark
* **`Audit 4A` (Falsifiable Gravity Collapse):** Untrained `0-weight` connectome collapses under gravity at **`t = 0.179 s`** (`Pelvis Z: 0.88 m -> 0.18 m`).
* **`Audit 4B` (Trained 3D Walk & Wattage Proof):** Walks `5.29 m` at **`1.76 m/s`** (`Score = 99.6/100`, `0.00000 cm` bone stretch error, `84.2%` spike sparsity) at **`422.0 W` total system power** vs. **`1,220.7 W`** for conventional stiff-PD control (**`2.89x` performance-per-watt advantage**).

<p align="center">
  <img src="assets/figures/stage4_3d_humanoid_gait.svg" alt="Stage 4 3D Humanoid Gait & Wattage Proof" width="92%"/>
</p>

### Stage 5: Whole-Body Multi-Skill Gauntlet, Scaling Law & 7 Closed-Loop Agentic Tasks (`Meta Muse` + `SAM-3.1`)
* **`Stage 5A` & `5B`:** Survives a `32 N` lateral push + `2.0 kg` bimanual load pickup + `35%` muscle fatigue (`-66.9%` cerebellar LTD weight adaptation) and verifies sub-millisecond sparse scaling from `0.18M` to **`25.11M` synapses**.
* **`Stage 5C` (`Tests 5.3–5.9`):** Verifies 7 closed-loop natural-language + `SAM-3.1` first-person camera tasks:
  * **`Test 5.3`:** *"Walk to the box, pick up the box, stand up, turn left, and lift it overhead"* (`Box lifted to Z = 1.60 m`, `SAM-3.1` retinal error `0.0 cm`).
  * **`Test 5.4`:** *"Kick high with right leg, then bow forward and wave both hands"* (`Right foot kick Z = 0.49 m`, `Bow pitch = 16.9°`, `Wave Z = 1.49 m`).
  * **`Test 5.5`:** *"Find the green cylinder and lift it overhead"* (`134°` out-of-FOV behind robot $\to$ `123.5°` active head/torso search $\to$ `SAM-3.1` lock $\to$ `Z = 1.48 m`).
  * **`Test 5.6`:** *"Clean all the things in the basket"* (`4/4` objects autonomously discovered and delivered across `5` self-chaining LLM cycles + $360^\circ$ verification scan).
  * **`Test 5.7`:** *"Only put the blue ball and green cylinder into the basket; leave the red box and gold crate on the floor"* (`Selective sorting 100% verified`).
  * **`Test 5.8`:** *"Clean everything except the red box into the basket"* (`Negation filtering 100% verified`: `Blue, Green, Gold` in basket; `Red Box` untouched).
  * **`Test 5.9`:** *"Walk past the blue ball without touching it, pick up the heavy gold crate, place it in the green basket, and wave both hands overhead"* (`Dynamic obstacle sidestep + 1.4 kg crate delivery + overhead celebration`).

<p align="center">
  <img src="assets/figures/stage5_stress_and_scaling.svg" alt="Stage 5 Multi-Skill Gauntlet & 25.11M Synapse Scaling" width="92%"/>
</p>

---

## Repository Structure

```text
AnyBody/
├── Cargo.toml                           # Pure-Rust workspace manifest (zero external runtime crates)
├── requirements.txt                     # Optional Python/MuJoCo/video dependencies
├── pytest.ini                           # Pytest configuration for 138 multi-morphology & API tests
├── .env.example                         # Template for Meta AI (Muse / SAM-3.1) & Baseten API keys
├── crates/
│   ├── neuro_core/src/lib.rs            # 3-Compartment L5 Pyramidal SoA, PV/SST/VIP, 4-Timescale BTSP/Fusi, mGluR1
│   ├── body_physics/src/
│   │   ├── lib.rs                       # 44-DOF / 88-Muscle Hill-type biomechanics & lock-free NeuralBodyBus
│   │   └── kinematics_3d.rs             # Unpinned 3D rigid-bone humanoid plant, 5-object workspace & ego_head_cam
│   ├── subsystems/src/
│   │   ├── lib.rs                       # WholeBodyNeuralController (FullScaleHuman = 4.13M neurons, 25.11M synapses)
│   │   ├── meta_prefrontal.rs           # Layer 1: Meta Muse + SAM-3.1 task compiler, negation parser & self-chaining loop
│   │   ├── basal_ganglia.rs             # Layer 1B: Striatal D1/D2, STN hyperdirect brake, GPi/SNr & SNc dopamine
│   │   ├── cortex.rs                    # Layer 2: Somatotopic homunculus babbling, S1/M1/PMv & Parietal 3D working memory
│   │   ├── cerebellum.rs                # Layer 3: 88 microzones x 46,800 Granule cells, Purkinje LTD & DCN predictive drive
│   │   └── spinal.rs                    # Layer 4: 14-D INaP half-center CPG, α-γ co-activation & Ia/Ib reflexes
│   ├── telemetry_viz/src/lib.rs         # Zero-dependency SVG scientific figure generator (Stages 0–5)
│   └── v3_runner/src/
│       ├── main.rs                      # `verify_all`: Master 6-Stage verification & wattage benchmark binary
│       └── bin/
│           ├── live_viewer.rs           # `live_viewer`: 1,000 Hz interactive 3D web server (http://127.0.0.1:8421)
│           └── telemetry_dumper.rs      # `telemetry_dumper`: 1,000 Hz JSONL trajectory exporter for 4K MuJoCo rendering
├── src/                                 # Python / MuJoCo Multi-Morphology & Dual Meta+Baseten Inference Bridge
│   ├── api/                             # `DualAIClient` (Meta Muse / SAM-3.1 + Baseten) & token budget guardrails
│   ├── brain/                           # Body-agnostic `BodyTrunk` persistence, `PerceptionSAM` & task compiler
│   └── physics/                         # Blind `BodyInspector`, MuJoCo Menagerie arena & 5 robot morphologies
├── data/
│   ├── body_trunks/                     # Discovered body trunks (Unitree H1, G1 5-Finger, Go2, Spot, Franka Panda)
│   └── v3_diamond_benchmark_report.json # Empirical multi-morphology zero-cheat benchmark report
├── tests/                               # 138 Python/MuJoCo unit & integration tests (`pytest`)
├── scripts/video/                       # 4K UHD 60fps MuJoCo + vector keynote rendering pipeline
├── assets/
│   ├── figures/                         # Architectural X-Ray snapshots & Stage 0–5 SVG telemetry plots
│   └── logos/                           # `A://` AnyBody SVG vector mark & partner badges
└── docs/
    ├── ARCHITECTURE.md                  # Full biophysical equations & 25.11M-synapse layer census
    ├── FALSIFIABILITY_PROOFS.md         # Zero-weight gravity collapse & zero-cheat proofs
    ├── SUBMISSION_ANSWERS.md            # Official Meta Global AI Developer Hackathon submission answers
    └── diagnostics/                     # Interactive 3D HTML viewer, single-neuron explainer & 7-neuron PDF figure
```

---

## Acknowledgments & Creator

* **Creator & Systems Architect:** **Oscar Kinnala** 🇫🇮
* **Compute & Foundation Model Partners:** Built for the **Meta Global AI Developer Hackathon** (co-hosted by **Meta** and **Cerebral Valley**), powered by **Meta `Muse-Spark-1.3`**, **`Muse-Open`**, **`SAM-3.1`**, and **`Muse-Wonder`**, with special thanks to **Baseten** for sponsoring `$100` (`100€`) in low-latency inference credits.
* **License:** [MIT License](LICENSE)
