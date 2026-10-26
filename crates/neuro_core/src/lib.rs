//! `neuro_core`: First-Principles Event-Driven Biophysical Neural Primitives
#![forbid(unsafe_code)]
//!
//! Designed around the core neuromorphic efficiency thesis:
//! **Maximum Whole-Body Motor Performance from Minimum Watts (Compute SynOps + Metabolic Joules)**
//! via intelligent multi-timescale, event-driven neural layers:
//!
//! 1. **3-Compartment Pyramidal BAC (Backpropagating Action Potential Activated Calcium) Multiplexing**:
//!    - Basal compartment integrates bottom-up proprioceptive state error (`g_ampa_basal`, Mg2+-gated `g_nmda_basal`).
//!    - Apical tuft integrates top-down motor intent & perturbation context (`g_ampa_apical`, `g_nmda_apical`).
//!    - Emits isolated **Single Spikes** (`is_burst = false`) for low-energy baseline maintenance, and
//!      high-frequency **Calcium Plateau Bursts** (`is_burst = true`) ONLY when coincidence or novelty
//!      demands rapid synaptic adaptation or reflex override.
//! 2. **Canonical Cortical Inhibitory Routing (`InhibitoryRoutingPool`: PV, SST, VIP)**:
//!    - **PV Basket Cells**: Fast somatic shunting inhibition (`E_GABA_A = -70 mV`) enforcing k-WTA sparsity
//!      so >90% of cortical columns remain silent at any millisecond (slashing compute & synaptic energy!).
//!    - **SST Martinotti Cells**: Facilitating synapses respond to bursts to gate apical excitability.
//!    - **VIP Interneurons**: Driven by novelty / slip / perturbation alarms to disinhibit SST and open the
//!      one-shot plasticity gate.
//! 3. **4-Timescale Biophysical Synapse Bank (`FourTimescaleSynapseBank`)**:
//!    - Actively wired into runtime sensorimotor execution!
//!    - Combines Tsodyks-Markram Short-Term Plasticity (`stp_u`, `stp_x`), BTSP Eligibility (`eligibility`),
//!      Active Conductance Weight (`weight`), and Fusi Metaplastic Consolidation (`consolidation`).
//!    - Event-driven sparse traversal: only processes presynaptic neurons that actually spiked or carry active trace!

use std::sync::atomic::{AtomicU32, Ordering};

/// Deterministic, fast xorshift64* PRNG for reproducible connectome initialization.
#[derive(Clone, Copy, Debug)]
pub struct FastRng {
    state: u64,
}

impl FastRng {
    #[inline]
    pub const fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xBAD5EED_CAFEBABE } else { seed },
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as u32 as f32) * (1.0 / 16_777_216.0)
    }

    #[inline]
    pub fn range_f32(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }

    #[inline]
    pub fn next_usize(&mut self, bound: usize) -> usize {
        if bound == 0 {
            0
        } else {
            (self.next_u64() as usize) % bound
        }
    }
}

/// Thread-safe Atomic f32 wrapper using `AtomicU32::to_bits / from_bits`
/// for lock-free cross-subsystem communication on the `NeuralBodyBus`.
#[repr(transparent)]
#[derive(Debug)]
pub struct AtomicF32(AtomicU32);

impl AtomicF32 {
    #[inline]
    pub const fn new(val: f32) -> Self {
        Self(AtomicU32::new(val.to_bits()))
    }

    #[inline]
    pub fn load(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Acquire))
    }

    #[inline]
    pub fn store(&self, val: f32) {
        self.0.store(val.to_bits(), Ordering::Release);
    }
}

impl Default for AtomicF32 {
    fn default() -> Self {
        Self::new(0.0)
    }
}

/// Biophysical Reversal Potentials (in mV) and Membrane Constants
pub mod biophysics {
    /// Resting leak reversal potential (mV)
    pub const E_LEAK: f32 = -70.0;
    /// Excitatory AMPA / NMDA glutamate reversal potential (mV)
    pub const E_EXC: f32 = 0.0;
    /// Fast shunting GABA_A reversal potential (mV) - near resting potential!
    pub const E_GABA_A: f32 = -70.0;
    /// Slow hyperpolarizing GABA_B / K+ reversal potential (mV)
    pub const E_GABA_B: f32 = -90.0;
    /// High-Voltage-Activated Ca2+ reversal potential in apical dendrites (mV)
    pub const E_CA: f32 = 120.0;
    /// Action potential threshold at Axon Initial Segment (mV)
    pub const V_THRESH_SOMA: f32 = -50.0;
    /// Action potential reset voltage for isolated event spike (mV)
    pub const V_RESET_EVENT: f32 = -65.0;
    /// Action potential reset voltage during Apical Ca2+-driven burst (mV)
    pub const V_RESET_BURST: f32 = -53.0;
    /// NMDA plateau ignition threshold in dendritic branch (mV)
    pub const V_THRESH_NMDA: f32 = -38.0;
    /// Apical Ca2+ HVA spike ignition threshold (mV)
    pub const V_THRESH_APICAL_CA: f32 = -35.0;

    /// Jahr & Stevens (1990) extracellular Mg2+ block of NMDA channels.
    /// Uses a fast rational Padé approximant when needed or exact exp, keeping <0.5% accuracy.
    #[inline]
    pub fn nmda_mg_unblock(v_mv: f32) -> f32 {
        1.0 / (1.0 + 0.2801 * (-0.062 * v_mv).exp())
    }
}

/// Multiplexed Action Potential Event (Payeur, Naud, Sacramento, & Bengio 2021).
/// Distinguishes bottom-up feedforward inference (`SingleSpike`) from top-down
/// apical coincidence / credit assignment (`BurstSpike`: 2-4 spikes at >150 Hz).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpikeEvent {
    pub neuron_id: u32,
    pub is_burst: bool,
}

/// Axonal Conduction Delay Ring Buffer.
/// Biological pathways have finite conduction velocities (Ia afferent: ~15ms,
/// Corticospinal: ~20-30ms). Preserves exact spike timing across up to 128 ms.
#[derive(Clone, Debug)]
pub struct AxonalDelayLine<T: Copy + Default> {
    buffer: Vec<T>,
    delay_steps: usize,
    cursor: usize,
}

impl<T: Copy + Default> AxonalDelayLine<T> {
    pub fn new(delay_steps: usize) -> Self {
        let len = delay_steps.max(1);
        Self {
            buffer: vec![T::default(); len],
            delay_steps: len,
            cursor: 0,
        }
    }

    /// Pushes the signal emitted at `t` and returns the signal arriving from `t - delay_steps`.
    #[inline]
    pub fn push_and_pop(&mut self, incoming: T) -> T {
        let delayed = self.buffer[self.cursor];
        self.buffer[self.cursor] = incoming;
        self.cursor += 1;
        if self.cursor >= self.delay_steps {
            self.cursor = 0;
        }
        delayed
    }

    #[inline]
    pub fn delay_ms(&self) -> usize {
        self.delay_steps
    }
}

/// 3-Compartment Human Layer 5 / Layer 2/3 Pyramidal Population (SoA layout).
/// Tracks exact synaptic operation count (`synops_this_tick`) for neuromorphic power budgeting.
#[derive(Clone, Debug)]
pub struct PyramidalPopulationSoA {
    pub count: usize,
    // Membrane voltages (mV)
    pub v_soma: Vec<f32>,
    pub v_basal: Vec<f32>,
    pub v_apical: Vec<f32>,

    // Synaptic conductances (normalized nS / C_m, in ms^-1)
    pub g_ampa_basal: Vec<f32>,
    pub g_nmda_basal: Vec<f32>,
    pub g_ampa_apical: Vec<f32>,
    pub g_nmda_apical: Vec<f32>,
    pub g_shunt_pv_soma: Vec<f32>,
    pub g_inh_sst_apical: Vec<f32>,

    // Active dendritic & somatic state machines
    pub basal_nmda_plateau_ms: Vec<u16>,
    pub apical_ca_plateau_ms: Vec<u16>,
    pub bap_trace: Vec<f32>,
    pub w_adapt: Vec<f32>,
    pub refractory_ms: Vec<u8>,

    // Burstprop & Homeostatic trackers
    pub last_spiked_event: Vec<bool>,
    pub last_spiked_burst: Vec<bool>,
    pub moving_burst_prob: Vec<f32>,
    pub firing_rate_hz: Vec<f32>,
    pub calcium_homeostasis: Vec<f32>,

    // Neuromorphic energy & sparsity telemetry
    pub active_dendrite_count: usize,
    pub single_spike_count: usize,
    pub burst_spike_count: usize,
}

impl PyramidalPopulationSoA {
    pub fn new(count: usize) -> Self {
        use biophysics::E_LEAK;
        Self {
            count,
            v_soma: vec![E_LEAK; count],
            v_basal: vec![E_LEAK; count],
            v_apical: vec![E_LEAK; count],
            g_ampa_basal: vec![0.0; count],
            g_nmda_basal: vec![0.0; count],
            g_ampa_apical: vec![0.0; count],
            g_nmda_apical: vec![0.0; count],
            g_shunt_pv_soma: vec![0.0; count],
            g_inh_sst_apical: vec![0.0; count],
            basal_nmda_plateau_ms: vec![0; count],
            apical_ca_plateau_ms: vec![0; count],
            bap_trace: vec![0.0; count],
            w_adapt: vec![0.0; count],
            refractory_ms: vec![0; count],
            last_spiked_event: vec![false; count],
            last_spiked_burst: vec![false; count],
            moving_burst_prob: vec![0.15; count],
            firing_rate_hz: vec![0.0; count],
            calcium_homeostasis: vec![0.02; count],
            active_dendrite_count: 0,
            single_spike_count: 0,
            burst_spike_count: 0,
        }
    }

    /// Injects conductance into basal dendrites (bottom-up sensory/feedforward).
    #[inline]
    pub fn inject_basal_glutamate(&mut self, idx: usize, ampa_dg: f32, nmda_dg: f32) {
        self.g_ampa_basal[idx] += ampa_dg;
        self.g_nmda_basal[idx] += nmda_dg;
    }

    /// Injects conductance into apical tuft dendrites (top-down context/prediction).
    #[inline]
    pub fn inject_apical_glutamate(&mut self, idx: usize, ampa_dg: f32, nmda_dg: f32) {
        self.g_ampa_apical[idx] += ampa_dg;
        self.g_nmda_apical[idx] += nmda_dg;
    }

    /// Injects PV somatic shunting and SST apical hyperpolarizing GABA conductances from `InhibitoryRoutingPool`.
    #[inline]
    pub fn inject_inhibitory_routing(&mut self, idx: usize, pv_shunt_dg: f32, sst_apical_dg: f32) {
        self.g_shunt_pv_soma[idx] = (self.g_shunt_pv_soma[idx] + pv_shunt_dg).min(2.5);
        self.g_inh_sst_apical[idx] = (self.g_inh_sst_apical[idx] + sst_apical_dg).min(2.5);
    }

    /// Steps all 3 compartments at `dt = 1.0 ms` (with 2x 0.5ms sub-stepping for stiff conductances).
    /// Populates `spike_out` with emitted `SpikeEvent`s and updates sparsity/energy metrics.
    pub fn step_1ms(&mut self, spike_out: &mut Vec<SpikeEvent>) {
        use biophysics::*;
        spike_out.clear();
        self.active_dendrite_count = 0;
        self.single_spike_count = 0;
        self.burst_spike_count = 0;

        let g_leak_soma = 0.08_f32;
        let g_leak_dend = 0.06_f32;
        let g_axial_bs = 0.14_f32;
        let g_axial_as = 0.09_f32;

        for i in 0..self.count {
            self.last_spiked_event[i] = false;
            self.last_spiked_burst[i] = false;

            // Event-driven power gating: if a neuron is at rest with negligible synaptic input,
            // skip stiff exponential sub-stepping (emulating asynchronous neuromorphic clock gating!)
            let total_g = self.g_ampa_basal[i]
                + self.g_nmda_basal[i]
                + self.g_ampa_apical[i]
                + self.g_nmda_apical[i]
                + self.bap_trace[i];
            if total_g < 1e-3
                && (self.v_soma[i] - E_LEAK).abs() < 0.5
                && self.basal_nmda_plateau_ms[i] == 0
                && self.apical_ca_plateau_ms[i] == 0
            {
                self.v_soma[i] = E_LEAK;
                self.v_basal[i] = E_LEAK;
                self.v_apical[i] = E_LEAK;
                self.w_adapt[i] *= 0.9875;
                self.firing_rate_hz[i] *= 0.99;
                if self.refractory_ms[i] > 0 {
                    self.refractory_ms[i] -= 1;
                }
                continue;
            }
            self.active_dendrite_count += 1;

            // Sub-step 2 x 0.5 ms for numerical stability of conductance equations
            for _sub in 0..2 {
                let dt = 0.5_f32;

                // 1. BASAL DENDRITE COMPARTMENT
                let vb = self.v_basal[i];
                let mg_b = nmda_mg_unblock(vb);
                let plateau_b_drive = if self.basal_nmda_plateau_ms[i] > 0 {
                    0.18 * (E_EXC - vb)
                } else {
                    0.0
                };
                let i_leak_b = g_leak_dend * (E_LEAK - vb);
                let i_ampa_b = self.g_ampa_basal[i] * (E_EXC - vb);
                let i_nmda_b = self.g_nmda_basal[i] * mg_b * (E_EXC - vb);
                let i_axial_b = g_axial_bs * (self.v_soma[i] - vb);

                let vb_next = (vb + dt * (i_leak_b + i_ampa_b + i_nmda_b + plateau_b_drive + i_axial_b))
                    .clamp(-90.0, 20.0);
                self.v_basal[i] = vb_next;

                if vb_next > V_THRESH_NMDA && self.g_nmda_basal[i] > 0.12 && self.basal_nmda_plateau_ms[i] == 0 {
                    self.basal_nmda_plateau_ms[i] = 120;
                }

                // 2. APICAL TUFT COMPARTMENT (BAC Coincidence Detector)
                let va = self.v_apical[i];
                let mg_a = nmda_mg_unblock(va);
                let bap_depol = 18.0 * self.bap_trace[i];
                let i_leak_a = g_leak_dend * (E_LEAK - va);
                let i_ampa_a = self.g_ampa_apical[i] * (E_EXC - va);
                let i_nmda_a = self.g_nmda_apical[i] * mg_a * (E_EXC - va);
                let i_sst_a = self.g_inh_sst_apical[i] * ((E_GABA_B - va) * 0.6 + (E_GABA_A - va) * 0.4);
                let i_ca_plateau = if self.apical_ca_plateau_ms[i] > 0 {
                    0.28 * (E_EXC - va)
                } else {
                    0.0
                };
                let i_axial_a = 0.05 * (self.v_soma[i] - va);

                let va_next = (va
                    + dt * (i_leak_a + i_ampa_a + i_nmda_a + i_sst_a + i_ca_plateau + i_axial_a + 0.04 * bap_depol))
                    .clamp(-95.0, 25.0);
                self.v_apical[i] = va_next;

                if (va_next + bap_depol) > V_THRESH_APICAL_CA
                    && self.g_inh_sst_apical[i] < 0.25
                    && (self.g_ampa_apical[i] + self.g_nmda_apical[i]) > 0.08
                    && self.apical_ca_plateau_ms[i] == 0
                {
                    self.apical_ca_plateau_ms[i] = 35;
                }

                // 3. SOMATIC COMPARTMENT
                let vs = self.v_soma[i];
                let i_leak_s = g_leak_soma * (E_LEAK - vs);
                let i_shunt_pv = self.g_shunt_pv_soma[i] * (E_GABA_A - vs);
                let i_from_basal = g_axial_bs * (vb_next - vs);
                let apical_coupling = if self.apical_ca_plateau_ms[i] > 0 {
                    g_axial_as * 2.4
                } else {
                    g_axial_as * 0.35
                };
                let i_from_apical = apical_coupling * (va_next - vs);

                let exp_arg = ((vs - V_THRESH_SOMA) / 2.5).clamp(-10.0, 4.0);
                let i_na_init = g_leak_soma * 2.5 * exp_arg.exp();

                if self.refractory_ms[i] == 0 {
                    let vs_next = vs
                        + dt * (i_leak_s + i_shunt_pv + i_from_basal + i_from_apical + i_na_init - self.w_adapt[i]);
                    self.v_soma[i] = vs_next.clamp(-90.0, 30.0);
                }
            }

            if self.basal_nmda_plateau_ms[i] > 0 {
                self.basal_nmda_plateau_ms[i] -= 1;
            }
            if self.apical_ca_plateau_ms[i] > 0 {
                self.apical_ca_plateau_ms[i] -= 1;
            }
            if self.refractory_ms[i] > 0 {
                self.refractory_ms[i] -= 1;
            }

            if self.v_soma[i] >= V_THRESH_SOMA && self.refractory_ms[i] == 0 {
                let is_burst = self.apical_ca_plateau_ms[i] > 0 || self.v_apical[i] > -42.0;
                self.last_spiked_event[i] = true;
                self.last_spiked_burst[i] = is_burst;

                let burst_val = if is_burst { 1.0 } else { 0.0 };
                self.moving_burst_prob[i] = 0.98 * self.moving_burst_prob[i] + 0.02 * burst_val;
                self.bap_trace[i] = (self.bap_trace[i] + 1.0).min(2.0);

                if is_burst {
                    self.v_soma[i] = V_RESET_BURST;
                    self.w_adapt[i] += 0.35;
                    self.refractory_ms[i] = 2;
                    self.burst_spike_count += 1;
                } else {
                    self.v_soma[i] = V_RESET_EVENT;
                    self.w_adapt[i] += 0.15;
                    self.refractory_ms[i] = 3;
                    self.single_spike_count += 1;
                }

                spike_out.push(SpikeEvent {
                    neuron_id: i as u32,
                    is_burst,
                });
            }

            self.g_ampa_basal[i] *= 0.8187;
            self.g_nmda_basal[i] *= 0.9847;
            self.g_ampa_apical[i] *= 0.8187;
            self.g_nmda_apical[i] *= 0.9847;
            self.g_shunt_pv_soma[i] *= 0.8465;
            self.g_inh_sst_apical[i] *= 0.9753;
            self.bap_trace[i] *= 0.9200;
            self.w_adapt[i] *= 0.9875;

            let inst_hz = if self.last_spiked_event[i] {
                if self.last_spiked_burst[i] { 300.0 } else { 100.0 }
            } else {
                0.0
            };
            self.firing_rate_hz[i] = 0.99 * self.firing_rate_hz[i] + 0.01 * inst_hz;
            self.calcium_homeostasis[i] =
                0.999 * self.calcium_homeostasis[i] + 0.001 * (inst_hz * 0.001);
        }
    }
}

/// Canonical Cortical Inhibitory Routing Microcircuit (PV, SST, VIP).
/// Actively gates somatic k-WTA sparsity (PV) and apical burst plasticity (SST/VIP).
#[derive(Clone, Debug)]
pub struct InhibitoryRoutingPool {
    pub pv_activity: f32,
    pub sst_activity: f32,
    pub vip_activity: f32,
    pub pyr_to_pv_x: f32,
    pub pyr_to_sst_u: f32,
}

impl Default for InhibitoryRoutingPool {
    fn default() -> Self {
        Self::new()
    }
}

impl InhibitoryRoutingPool {
    pub const fn new() -> Self {
        Self {
            pv_activity: 0.0,
            sst_activity: 0.0,
            vip_activity: 0.0,
            pyr_to_pv_x: 1.0,
            pyr_to_sst_u: 0.05,
        }
    }

    pub fn step_1ms(
        &mut self,
        feedforward_drive: f32,
        single_spike_fraction: f32,
        burst_spike_fraction: f32,
        top_down_vip_drive: f32,
    ) {
        let total_pyr = single_spike_fraction + 2.5 * burst_spike_fraction;

        // 1. Pyr -> PV is Short-Term Depressing (enforces rapid k-WTA onset sparsity)
        let pv_syn_drive = 0.6 * self.pyr_to_pv_x * total_pyr;
        self.pyr_to_pv_x += (1.0 - self.pyr_to_pv_x) * (1.0 / 150.0) - 0.35 * pv_syn_drive;
        self.pyr_to_pv_x = self.pyr_to_pv_x.clamp(0.1, 1.0);

        let pv_target = (0.8 * feedforward_drive + 2.2 * pv_syn_drive).clamp(0.0, 3.0);
        self.pv_activity = 0.80 * self.pv_activity + 0.20 * pv_target;

        // 2. VIP integrates top-down novelty/error disinhibitory control
        let vip_target = top_down_vip_drive.clamp(0.0, 2.5);
        self.vip_activity = 0.90 * self.vip_activity + 0.10 * vip_target;

        // 3. Pyr -> SST is Strongly Short-Term Facilitating (responds selectively to bursts)
        let u_0 = 0.02_f32;
        self.pyr_to_sst_u += (u_0 - self.pyr_to_sst_u) * (1.0 / 400.0)
            + 0.04 * (1.0 - self.pyr_to_sst_u) * single_spike_fraction
            + 0.65 * (1.0 - self.pyr_to_sst_u) * burst_spike_fraction;
        self.pyr_to_sst_u = self.pyr_to_sst_u.clamp(u_0, 0.95);

        let sst_exc = self.pyr_to_sst_u * (0.2 * single_spike_fraction + 3.5 * burst_spike_fraction);
        let sst_net = (sst_exc - 1.4 * self.vip_activity).max(0.0);
        self.sst_activity = 0.88 * self.sst_activity + 0.12 * sst_net;
    }
}

/// The 4-Timescale Biophysical Synapse Array (SoA Layout).
/// Actively wired into Corticospinal & Premotor Skill Readout!
/// 1. Short-Term Plasticity (`stp_u`, `stp_x`: 10ms - 1500ms working memory / burst filter)
/// 2. BTSP Eligibility Trace (`eligibility`: 1000ms - 3000ms asymmetric temporal tag)
/// 3. Active AMPA Conductance Weight (`weight`: minutes to hours)
/// 4. Fusi Metaplastic Consolidation Cascade (`consolidation`: hours to lifetime, controls learning rate & stability)
#[derive(Clone, Debug)]
pub struct FourTimescaleSynapseBank {
    pub count: usize,
    pub pre_idx: Vec<u32>,
    pub post_idx: Vec<u32>,
    pub stp_u: Vec<f32>,
    pub stp_x: Vec<f32>,
    pub eligibility: Vec<f32>,
    pub weight: Vec<f32>,
    pub consolidation: Vec<f32>,
    pub u_base: f32,
    pub tau_facil_ms: f32,
    pub tau_rec_ms: f32,
    pub tau_elig_ms: f32,
    pub active_synops_last_step: usize,
}

impl FourTimescaleSynapseBank {
    pub fn new(
        capacity: usize,
        u_base: f32,
        tau_facil_ms: f32,
        tau_rec_ms: f32,
        tau_elig_ms: f32,
    ) -> Self {
        Self {
            count: 0,
            pre_idx: Vec::with_capacity(capacity),
            post_idx: Vec::with_capacity(capacity),
            stp_u: Vec::with_capacity(capacity),
            stp_x: Vec::with_capacity(capacity),
            eligibility: Vec::with_capacity(capacity),
            weight: Vec::with_capacity(capacity),
            consolidation: Vec::with_capacity(capacity),
            u_base,
            tau_facil_ms,
            tau_rec_ms,
            tau_elig_ms,
            active_synops_last_step: 0,
        }
    }

    pub fn add_synapse(&mut self, pre: u32, post: u32, initial_weight: f32) -> usize {
        let id = self.count;
        self.pre_idx.push(pre);
        self.post_idx.push(post);
        self.stp_u.push(self.u_base);
        self.stp_x.push(1.0);
        self.eligibility.push(0.0);
        self.weight.push(initial_weight);
        self.consolidation.push(0.1);
        self.count += 1;
        id
    }

    /// Event-driven propagation: updates STP + Eligibility traces AND accumulates postsynaptic currents
    /// `post_current_out[post] += weight * u * x * r_pre` in a single cache-friendly pass!
    pub fn step_and_propagate(
        &mut self,
        pre_rates: &[f32],
        post_current_out: &mut [f32],
        dt_ms: f32,
    ) {
        let inv_facil = dt_ms / self.tau_facil_ms;
        let inv_rec = dt_ms / self.tau_rec_ms;
        let elig_decay = (-dt_ms / self.tau_elig_ms).exp();
        let mut synops = 0_usize;

        for s in 0..self.count {
            let pre = self.pre_idx[s] as usize;
            if pre == u32::MAX as usize {
                continue;
            }
            let r_pre = pre_rates[pre];
            let elig_prev = self.eligibility[s];

            // Event-driven sparsity: if presynaptic neuron is silent and eligibility trace has decayed,
            // skip expensive state updates!
            if r_pre < 1e-4 && elig_prev < 1e-4 {
                continue;
            }
            synops += 1;

            let mut u = self.stp_u[s];
            let mut x = self.stp_x[s];

            u += (self.u_base - u) * inv_facil + self.u_base * (1.0 - u) * r_pre;
            u = u.clamp(self.u_base, 0.95);
            let released = u * x * r_pre;
            x += (1.0 - x) * inv_rec - released;
            x = x.clamp(0.05, 1.0);

            self.stp_u[s] = u;
            self.stp_x[s] = x;
            self.eligibility[s] = elig_prev * elig_decay + released;

            if released > 1e-5 {
                let post = self.post_idx[s] as usize;
                post_current_out[post] += self.weight[s] * released;
            }
        }
        self.active_synops_last_step = synops;
    }

    /// Backwards-compatible helper for STP & eligibility update without post_current_out accumulation.
    pub fn step_stp_and_eligibility(&mut self, pre_rates: &[f32], dt_ms: f32) {
        let inv_facil = dt_ms / self.tau_facil_ms;
        let inv_rec = dt_ms / self.tau_rec_ms;
        let elig_decay = (-dt_ms / self.tau_elig_ms).exp();

        for s in 0..self.count {
            let pre = self.pre_idx[s] as usize;
            if pre == u32::MAX as usize {
                continue;
            }
            let r_pre = pre_rates[pre];
            let mut u = self.stp_u[s];
            let mut x = self.stp_x[s];

            u += (self.u_base - u) * inv_facil + self.u_base * (1.0 - u) * r_pre;
            u = u.clamp(self.u_base, 0.95);
            let released = u * x * r_pre;
            x += (1.0 - x) * inv_rec - released;
            x = x.clamp(0.05, 1.0);

            self.stp_u[s] = u;
            self.stp_x[s] = x;
            self.eligibility[s] = self.eligibility[s] * elig_decay + released;
        }
    }

    /// Applies Burstprop + BTSP + Fusi Metaplastic Consolidation update.
    pub fn apply_burstprop_btsp_plasticity(
        &mut self,
        post_teaching_signal: &[f32],
        dopamine_modulator: f32,
        base_lr: f32,
        w_max: f32,
    ) {
        for s in 0..self.count {
            if self.pre_idx[s] == u32::MAX {
                continue;
            }
            let elig = self.eligibility[s];
            if elig.abs() < 1e-4 {
                continue;
            }
            let post = self.post_idx[s] as usize;
            let teach = post_teaching_signal[post];
            if teach.abs() < 1e-5 {
                continue;
            }

            let cons = self.consolidation[s];
            let effective_lr = base_lr * (-0.5 * cons.abs()).exp();
            let dw = effective_lr * dopamine_modulator * elig * teach;

            let new_w = (self.weight[s] + dw).clamp(0.0, w_max);
            self.weight[s] = new_w;

            if dw > 0.0 {
                self.consolidation[s] = (cons + 0.05 * dw).min(5.0);
            } else {
                self.consolidation[s] = (cons + 0.02 * dw).max(0.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage0_1_pyramidal_bac_burst_multiplexing() {
        let mut pop = PyramidalPopulationSoA::new(2);
        let mut spikes = Vec::new();

        let mut neuron0_single_spikes = 0;
        let mut neuron0_burst_spikes = 0;
        let mut neuron1_single_spikes = 0;
        let mut neuron1_burst_spikes = 0;

        for _ms in 0..50 {
            pop.inject_basal_glutamate(0, 0.22, 0.15);
            pop.inject_basal_glutamate(1, 0.22, 0.15);
            pop.inject_apical_glutamate(1, 0.20, 0.18);

            pop.step_1ms(&mut spikes);
            for sp in &spikes {
                if sp.neuron_id == 0 {
                    if sp.is_burst {
                        neuron0_burst_spikes += 1;
                    } else {
                        neuron0_single_spikes += 1;
                    }
                } else if sp.neuron_id == 1 {
                    if sp.is_burst {
                        neuron1_burst_spikes += 1;
                    } else {
                        neuron1_single_spikes += 1;
                    }
                }
            }
        }

        assert!(
            neuron0_single_spikes >= 2 && neuron0_burst_spikes == 0,
            "Basal-only neuron must emit single spikes without bursting: single={}, burst={}",
            neuron0_single_spikes,
            neuron0_burst_spikes
        );
        assert!(
            neuron1_burst_spikes >= 3,
            "Basal+Apical coincident neuron must ignite Apical Ca2+ plateau and burst: single={}, burst={}",
            neuron1_single_spikes,
            neuron1_burst_spikes
        );
    }

    #[test]
    fn test_stage0_2_vip_sst_pv_canonical_routing() {
        let mut circuit = InhibitoryRoutingPool::new();

        for _ in 0..20 {
            circuit.step_1ms(0.5, 0.4, 0.0, 0.0);
        }
        let pv_during_single = circuit.pv_activity;
        let sst_during_single = circuit.sst_activity;
        assert!(
            pv_during_single > 0.4 && sst_during_single < 0.05,
            "Single spikes should drive PV ({}) while leaving facilitating SST quiet ({})",
            pv_during_single,
            sst_during_single
        );

        for _ in 0..20 {
            circuit.step_1ms(0.5, 0.1, 0.5, 0.0);
        }
        let sst_during_burst = circuit.sst_activity;
        assert!(
            sst_during_burst > 0.45,
            "Bursting must facilitate Pyr->SST and recruit strong apical SST inhibition ({})",
            sst_during_burst
        );

        for _ in 0..20 {
            circuit.step_1ms(0.5, 0.1, 0.5, 1.5);
        }
        let sst_with_vip = circuit.sst_activity;
        assert!(
            sst_with_vip < sst_during_burst * 0.25,
            "VIP activation must suppress SST to open the apical gate: before={}, after={}",
            sst_during_burst,
            sst_with_vip
        );
    }
}
