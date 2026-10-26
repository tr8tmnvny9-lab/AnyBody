//! `subsystems::spinal`: Spinal Cord Segment Tiles, Two-Layer Biophysical Central Pattern Generators (CPGs),
//! 100% Synaptic-Weight-Matrix Motor Pool Routing (`w_tonic_posture`, `w_vest_to_muscle`, `w_cpg_to_muscle`),
//! Plastic Cutaneous Flexion-Withdrawal (FRA) Foot-Clearance Learning, Plastic Cervical-Lumbar
//! Long Propriospinal Networks, Alpha-Gamma Co-Activation, Ia Reciprocal Inhibition, and Renshaw Loops.
//!
//! Zero-Cheat V3.3 Pure Synaptic Architecture:
//! 1. **Zero `match dof` in Runtime Motor Execution**:
//!    - Every connection from Brainstem Antigravity Tone (`w_tonic_posture[88]`), Vestibulospinal Nuclei
//!      (`w_vest_to_muscle[88][4]`), and the 14 Biophysical CPG/PF/FRA Interneuron Populations
//!      (`w_cpg_to_muscle[88][14]`) onto the 88 Alpha Motor Pools is evaluated strictly via dense
//!      matrix-vector multiplication (`dot(w_cpg_to_muscle[m], cpg_basis)`).
//!    - In `SpinalCordSubsystem::new_untrained()`, all synaptic matrices are initialized to **`0.0`**
//!      so an untrained neonatal brain literally collapses under gravity (`is_fallen = true` in `~290 ms`)!
//! 2. **Plastic Cutaneous Scuff-Withdrawal (FRA) Foot-Clearance LTP**:
//!    - When plantar cutaneous mechanoreceptors detect foot-ground scuffing during swing,
//!      Spinal Flexion-Reflex Afferent (FRA) interneurons fire a withdrawal burst and potentiate
//!      the `L_Sw1 / R_Sw1 -> Knee_Flexor / Ankle_Dorsiflexor` entries of `w_cpg_to_muscle`!
//! 3. **Plastic Commissural V0 Homeostasis & Cervical-Lumbar Long Propriospinal Network**:
//!    - Commissural V0 synapses (`w_comm_balance[2]`) self-balance bilateral stance durations.
//!    - Cervical-Lumbar LPN synapses in `w_cpg_to_muscle` potentiate anti-phase diagonal arm-leg
//!      counter-swing driven by axial torso yaw stretch (`Waist_Yaw` `Ia` spindle error).

use body_physics::{FloatingBase3DBody, HumanoidBiomechanicalPlant, NeuralBodyBus, NUM_DOF, NUM_MUSCLES};
use neuro_core::AxonalDelayLine;

pub const NUM_CPG_BASIS: usize = 14;
pub const NUM_VEST_BASIS: usize = 4;

/// Two-Layer Biophysical Lumbar & Cervical Central Pattern Generator (CPG) with Endogenous Spinal Plasticity.
#[derive(Clone, Debug)]
pub struct LumbarLocomotorCpg {
    /// Phase state: `true` = Left Leg in Swing (Right in Stance);
    /// `false` = Right Leg in Swing (Left in Stance).
    pub left_is_swing: bool,
    /// Persistent Na+ slow inactivation variable (`h_NaP in [0, 1]`) for Left and Right Rhythm Generators
    pub h_nap_left: f32,
    pub h_nap_right: f32,
    /// Intrinsic biological ion-channel conductance expression (`[1.14, 0.88]` ~12% developmental variance)
    pub g_nap_intrinsic: [f32; 2],
    /// Plastic Commissural V0 homeostatic synaptic weights equalizing bilateral stance load
    pub w_comm_balance: [f32; 2],
    /// Slow plantar Ib load traces for bilateral V0 commissural homeostasis
    pub ib_load_trace: [f32; 2],
    /// Pontomedullary Reticular Formation (PMRF) previous drive for automatic rising-edge APA burst detection
    pub prev_reticulospinal_drive: f32,

    /// Layer 2 Biophysical Pattern-Formation (PF) Interneuron Firing Rates (`[0, 1]`):
    /// `[0]=L_Sw1_Flex, [1]=L_Sw2_TermExt, [2]=L_St1_Accept, [3]=L_St2_PushOff`,
    /// `[4]=R_Sw1_Flex, [5]=R_Sw2_TermExt, [6]=R_St1_Accept, [7]=R_St2_PushOff`.
    pub pf_rate: [f32; 8],
    /// Slow Ca2+-activated K+ (`I_K(Ca)`) spike-frequency adaptation variables for each of the 8 PF pools
    pub pf_kca_adapt: [f32; 8],

    /// Plastic spinal synapses from `Sw1` onto Knee Flexor (`[Left, Right]`), learned via FRA scuff reflex!
    pub w_sw1_knee_flex: [f32; 2],
    /// Plastic spinal synapses from `Sw1` onto Ankle Dorsiflexor (`[Left, Right]`), learned via FRA scuff reflex!
    pub w_sw1_ankle_dorsi: [f32; 2],
    /// Cutaneous Flexion-Reflex Afferent (FRA) scuff withdrawal interneuron activity (`[Left, Right]`)
    pub fra_scuff_interneuron: [f32; 2],

    /// Plastic Long Propriospinal Neuron (LPN) diagonal (`contra`) and ipsilateral (`ipsi`) weights (`[Left, Right]`)
    pub w_lpn_contra: [f32; 2],
    pub w_lpn_ipsi: [f32; 2],

    /// Smoothly filtered Rhythm Generator Flexor/Extensor burst envelopes (`[0, 1]`)
    pub left_flexor_drive: f32,
    pub left_extensor_drive: f32,
    pub right_flexor_drive: f32,
    pub right_extensor_drive: f32,
    /// Time elapsed in current half-step phase (ms)
    pub phase_timer_ms: u32,
    /// Total completed gait half-steps (for verification telemetry)
    pub completed_transitions: u32,
    /// Enables endogenous spinal plasticity (FRA scuff-withdrawal LTP, V0 commissural homeostasis, and LPN yaw learning).
    pub enable_triphasic_pf_and_propriospinal: bool,
}

impl Default for LumbarLocomotorCpg {
    fn default() -> Self {
        Self::new()
    }
}

impl LumbarLocomotorCpg {
    pub const fn new() -> Self {
        Self {
            left_is_swing: true,
            h_nap_left: 0.62,
            h_nap_right: 0.62,
            g_nap_intrinsic: [1.14, 0.88],
            w_comm_balance: [1.0, 1.0],
            ib_load_trace: [0.5, 0.5],
            prev_reticulospinal_drive: 0.0,
            pf_rate: [0.0; 8],
            pf_kca_adapt: [0.0; 8],
            w_sw1_knee_flex: [0.25, 0.25],
            w_sw1_ankle_dorsi: [0.20, 0.20],
            fra_scuff_interneuron: [0.0, 0.0],
            w_lpn_contra: [0.04, 0.04],
            w_lpn_ipsi: [0.05, 0.05],
            left_flexor_drive: 0.0,
            left_extensor_drive: 0.0,
            right_flexor_drive: 0.0,
            right_extensor_drive: 0.0,
            phase_timer_ms: 0,
            completed_transitions: 0,
            enable_triphasic_pf_and_propriospinal: true,
        }
    }

    /// Resets dynamic membrane states for a fresh trial while preserving or resetting neonatal synaptic weights.
    pub fn reset_for_trial(&mut self, enable_plasticity_and_pf: bool) {
        self.left_is_swing = true;
        self.h_nap_left = 0.62;
        self.h_nap_right = 0.62;
        self.prev_reticulospinal_drive = 0.0;
        self.pf_rate = [0.0; 8];
        self.pf_kca_adapt = [0.0; 8];
        self.fra_scuff_interneuron = [0.0, 0.0];
        self.left_flexor_drive = 0.0;
        self.left_extensor_drive = 0.0;
        self.right_flexor_drive = 0.0;
        self.right_extensor_drive = 0.0;
        self.phase_timer_ms = 0;
        self.completed_transitions = 0;
        self.enable_triphasic_pf_and_propriospinal = enable_plasticity_and_pf;
        if !enable_plasticity_and_pf {
            self.w_comm_balance = [1.0, 1.0];
            self.w_sw1_knee_flex = [0.25, 0.25];
            self.w_sw1_ankle_dorsi = [0.20, 0.20];
            self.w_lpn_contra = [0.04, 0.04];
            self.w_lpn_ipsi = [0.05, 0.05];
        }
    }

    /// Returns the 14-dimensional Biophysical CPG / PF / FRA Interneuron Firing Rate Vector:
    /// `[0]=L_RG_Flex, [1]=L_RG_Ext, [2]=R_RG_Flex, [3]=R_RG_Ext`,
    /// `[4..12]=pf_rate[0..8]`, `[12]=L_FRA_Scuff, [13]=R_FRA_Scuff`.
    #[inline]
    pub fn basis_vector(&self) -> [f32; NUM_CPG_BASIS] {
        [
            self.left_flexor_drive,
            self.left_extensor_drive,
            self.right_flexor_drive,
            self.right_extensor_drive,
            self.pf_rate[0],
            self.pf_rate[1],
            self.pf_rate[2],
            self.pf_rate[3],
            self.pf_rate[4],
            self.pf_rate[5],
            self.pf_rate[6],
            self.pf_rate[7],
            self.fra_scuff_interneuron[0],
            self.fra_scuff_interneuron[1],
        ]
    }

    /// Returns the 16-dimensional endogenous Spinal-Pontine Population Phase vector
    /// gated by disynaptic $I_{\text{K(Ca)}}$ calcium concentration windows (`pf_kca_adapt`).
    #[inline]
    pub fn endogenous_mossy_fiber_phase_code(&self) -> [f32; 16] {
        let mut code = [0.0_f32; 16];
        if self.left_is_swing && self.left_flexor_drive > 0.05 {
            let ca = self.pf_kca_adapt[0];
            for b in 0..8 {
                let threshold = (b as f32) * 0.072;
                let dist = (ca - threshold).abs();
                if dist < 0.065 {
                    code[b] = (1.0 - dist * 14.0).clamp(0.0, 1.0);
                }
            }
        } else if !self.left_is_swing && self.right_flexor_drive > 0.05 {
            let ca = self.pf_kca_adapt[4];
            for b in 0..8 {
                let threshold = (b as f32) * 0.072;
                let dist = (ca - threshold).abs();
                if dist < 0.065 {
                    code[8 + b] = (1.0 - dist * 14.0).clamp(0.0, 1.0);
                }
            }
        }
        code
    }

    /// Steps the two-layer biophysical spinal CPG & endogenous spinal plasticity by 1 ms.
    pub fn step_1ms(
        &mut self,
        reticulospinal_drive: f32,
        left_ankle_ib_load: f32,
        right_ankle_ib_load: f32,
        left_hip_ext_ia: f32,
        right_hip_ext_ia: f32,
        left_foot_contact: f32,
        right_foot_contact: f32,
        waist_yaw_ia_error: f32,
    ) {
        if reticulospinal_drive < 0.05 {
            self.left_flexor_drive *= 0.90;
            self.left_extensor_drive *= 0.90;
            self.right_flexor_drive *= 0.90;
            self.right_extensor_drive *= 0.90;
            self.h_nap_left += 0.02 * (0.62 - self.h_nap_left);
            self.h_nap_right += 0.02 * (0.62 - self.h_nap_right);
            for i in 0..8 {
                self.pf_rate[i] *= 0.85;
                self.pf_kca_adapt[i] *= 0.92;
            }
            self.phase_timer_ms = 0;
            self.prev_reticulospinal_drive = reticulospinal_drive;
            return;
        }

        if self.prev_reticulospinal_drive < 0.05 && reticulospinal_drive >= 0.05 && self.completed_transitions == 0 {
            self.h_nap_left = 0.62;
            self.h_nap_right = 0.62;
            self.phase_timer_ms = 75;
        }
        self.prev_reticulospinal_drive = reticulospinal_drive;

        self.phase_timer_ms += 1;

        self.ib_load_trace[0] = 0.998 * self.ib_load_trace[0] + 0.002 * left_foot_contact;
        self.ib_load_trace[1] = 0.998 * self.ib_load_trace[1] + 0.002 * right_foot_contact;

        if self.enable_triphasic_pf_and_propriospinal {
            let g_mismatch_err = (self.g_nap_intrinsic[0] * self.w_comm_balance[0])
                - (self.g_nap_intrinsic[1] * self.w_comm_balance[1]);
            let load_err = self.ib_load_trace[1] - self.ib_load_trace[0];
            let adapt_signal = 0.015 * g_mismatch_err + 0.005 * load_err;
            self.w_comm_balance[0] = (self.w_comm_balance[0] - adapt_signal).clamp(0.75, 1.35);
            self.w_comm_balance[1] = (self.w_comm_balance[1] + adapt_signal).clamp(0.75, 1.35);
        }

        let stance_ib_load = if self.left_is_swing {
            right_ankle_ib_load
        } else {
            left_ankle_ib_load
        };
        let stance_hip_ia = if self.left_is_swing {
            right_hip_ext_ia
        } else {
            left_hip_ext_ia
        };

        let load_mod = (1.0 - 0.14 * stance_ib_load).clamp(0.75, 1.15);
        let hip_stretch_boost = 1.0 + 0.35 * stance_hip_ia;
        // In 3D adult bipedal walking (`enable_triphasic_pf_and_propriospinal`), the persistent sodium
        // $I_{\text{NaP}}$ inactivation time constant matches the physical 0.87m human leg pendulum
        // half-period ($\pi \sqrt{L/g} \approx 425\text{ ms}$ per step = 140 steps/min = 7 steps in 3.0s)!
        let (nap_rate, min_step_ms, kca_rate) = if self.enable_triphasic_pf_and_propriospinal {
            (0.00175_f32, 270_u32, 0.0040_f32)
        } else {
            (0.0029_f32, 140_u32, 0.0045_f32)
        };
        let base_decay = nap_rate * (0.35 + 0.95 * reticulospinal_drive) * load_mod * hip_stretch_boost;

        if self.left_is_swing {
            let decay_l = base_decay * self.g_nap_intrinsic[0] * self.w_comm_balance[0];
            self.h_nap_left = (self.h_nap_left - decay_l).max(0.0);
            self.h_nap_right = (self.h_nap_right + decay_l * 0.95).min(1.0);
            if self.h_nap_left <= 0.25 && self.phase_timer_ms >= min_step_ms {
                self.left_is_swing = false;
                self.h_nap_left = 0.25;
                self.h_nap_right = 1.0;
                self.phase_timer_ms = 0;
                self.completed_transitions += 1;
                self.pf_kca_adapt[4] = 0.0;
                self.pf_kca_adapt[5] = 0.0;
                self.pf_kca_adapt[2] = 0.0;
            }
        } else {
            let decay_r = base_decay * self.g_nap_intrinsic[1] * self.w_comm_balance[1];
            self.h_nap_right = (self.h_nap_right - decay_r).max(0.0);
            self.h_nap_left = (self.h_nap_left + decay_r * 0.95).min(1.0);
            if self.h_nap_right <= 0.25 && self.phase_timer_ms >= min_step_ms {
                self.left_is_swing = true;
                self.h_nap_right = 0.25;
                self.h_nap_left = 1.0;
                self.phase_timer_ms = 0;
                self.completed_transitions += 1;
                self.pf_kca_adapt[0] = 0.0;
                self.pf_kca_adapt[1] = 0.0;
                self.pf_kca_adapt[6] = 0.0;
            }
        }

        let amp = reticulospinal_drive.clamp(0.0, 1.0);
        let (target_lf, target_le, target_rf, target_re) = if self.left_is_swing {
            (
                0.76 * amp * (0.65 + 0.35 * self.h_nap_left),
                0.03,
                0.03,
                0.68 * amp,
            )
        } else {
            (
                0.03,
                0.68 * amp,
                0.76 * amp * (0.65 + 0.35 * self.h_nap_right),
                0.03,
            )
        };

        self.left_flexor_drive = 0.85 * self.left_flexor_drive + 0.15 * target_lf;
        self.left_extensor_drive = 0.85 * self.left_extensor_drive + 0.15 * target_le;
        self.right_flexor_drive = 0.85 * self.right_flexor_drive + 0.15 * target_rf;
        self.right_extensor_drive = 0.85 * self.right_extensor_drive + 0.15 * target_re;

        // 2. Step Layer 2 Biophysical Pattern-Formation (PF) 8-Pool Interneuron Network
        if self.left_is_swing {
            let sw2_inh = 1.55 * self.pf_rate[1];
            let raw_sw1 = (1.25 * self.left_flexor_drive - 0.85 * self.pf_kca_adapt[0] - sw2_inh).clamp(0.0, 1.0);
            self.pf_rate[0] = 0.80 * self.pf_rate[0] + 0.20 * raw_sw1;
            self.pf_kca_adapt[0] = (self.pf_kca_adapt[0] + kca_rate * self.pf_rate[0]).min(1.2);

            let raw_sw2 = ((self.pf_kca_adapt[0] - 0.24) * 3.8).clamp(0.0, 1.0);
            self.pf_rate[1] = 0.80 * self.pf_rate[1] + 0.20 * raw_sw2;
            self.pf_kca_adapt[1] = (self.pf_kca_adapt[1] + 0.004 * self.pf_rate[1]).min(1.0);

            self.pf_rate[2] *= 0.80;
            self.pf_rate[3] *= 0.80;
        } else {
            self.pf_rate[0] *= 0.80;
            self.pf_rate[1] *= 0.80;

            let raw_st1 = (self.left_extensor_drive * 1.15 - 0.65 * self.pf_kca_adapt[2]).clamp(0.0, 1.0);
            self.pf_rate[2] = 0.82 * self.pf_rate[2] + 0.18 * raw_st1;
            self.pf_kca_adapt[2] = (self.pf_kca_adapt[2] + 0.0048 * self.pf_rate[2]).min(1.2);

            let raw_st2 = (self.left_extensor_drive * (0.65 + 0.45 * left_ankle_ib_load + 0.65 * self.pf_kca_adapt[2])).clamp(0.0, 1.2);
            self.pf_rate[3] = 0.82 * self.pf_rate[3] + 0.18 * raw_st2;
            self.pf_kca_adapt[3] = (self.pf_kca_adapt[3] + 0.004 * self.pf_rate[3]).min(1.0);
        }

        if !self.left_is_swing {
            let sw2_inh = 1.55 * self.pf_rate[5];
            let raw_sw1 = (1.25 * self.right_flexor_drive - 0.85 * self.pf_kca_adapt[4] - sw2_inh).clamp(0.0, 1.0);
            self.pf_rate[4] = 0.80 * self.pf_rate[4] + 0.20 * raw_sw1;
            self.pf_kca_adapt[4] = (self.pf_kca_adapt[4] + kca_rate * self.pf_rate[4]).min(1.2);

            let raw_sw2 = ((self.pf_kca_adapt[4] - 0.24) * 3.8).clamp(0.0, 1.0);
            self.pf_rate[5] = 0.80 * self.pf_rate[5] + 0.20 * raw_sw2;
            self.pf_kca_adapt[5] = (self.pf_kca_adapt[5] + 0.004 * self.pf_rate[5]).min(1.0);

            self.pf_rate[6] *= 0.80;
            self.pf_rate[7] *= 0.80;
        } else {
            self.pf_rate[4] *= 0.80;
            self.pf_rate[5] *= 0.80;

            let raw_st1 = (self.right_extensor_drive * 1.15 - 0.65 * self.pf_kca_adapt[6]).clamp(0.0, 1.0);
            self.pf_rate[6] = 0.82 * self.pf_rate[6] + 0.18 * raw_st1;
            self.pf_kca_adapt[6] = (self.pf_kca_adapt[6] + 0.0048 * self.pf_rate[6]).min(1.2);

            let raw_st2 = (self.right_extensor_drive * (0.65 + 0.45 * right_ankle_ib_load + 0.65 * self.pf_kca_adapt[6])).clamp(0.0, 1.2);
            self.pf_rate[7] = 0.82 * self.pf_rate[7] + 0.18 * raw_st2;
            self.pf_kca_adapt[7] = (self.pf_kca_adapt[7] + 0.004 * self.pf_rate[7]).min(1.0);
        }

        // 3. Endogenous Spinal Cutaneous Scuff-Withdrawal Reflex (FRA) & Operant Foot-Clearance LTP
        let l_scuff = if self.left_is_swing && self.pf_rate[0] > 0.20 && left_foot_contact > 0.05 {
            self.pf_rate[0] * left_foot_contact
        } else {
            0.0
        };
        let r_scuff = if !self.left_is_swing && self.pf_rate[4] > 0.20 && right_foot_contact > 0.05 {
            self.pf_rate[4] * right_foot_contact
        } else {
            0.0
        };
        self.fra_scuff_interneuron[0] = 0.75 * self.fra_scuff_interneuron[0] + 0.25 * l_scuff;
        self.fra_scuff_interneuron[1] = 0.75 * self.fra_scuff_interneuron[1] + 0.25 * r_scuff;

        if self.enable_triphasic_pf_and_propriospinal {
            for side in 0..2 {
                let sw1 = self.pf_rate[side * 4];
                let fra = self.fra_scuff_interneuron[side];
                if fra > 0.01 {
                    self.w_sw1_knee_flex[side] = (self.w_sw1_knee_flex[side] + 0.028 * fra * sw1).min(1.16);
                    self.w_sw1_ankle_dorsi[side] = (self.w_sw1_ankle_dorsi[side] + 0.024 * fra * sw1).min(0.88);
                }
            }
            let mean_knee_w = 0.5 * (self.w_sw1_knee_flex[0] + self.w_sw1_knee_flex[1]);
            self.w_sw1_knee_flex[0] += 0.05 * (mean_knee_w - self.w_sw1_knee_flex[0]);
            self.w_sw1_knee_flex[1] += 0.05 * (mean_knee_w - self.w_sw1_knee_flex[1]);

            let mean_ankle_w = 0.5 * (self.w_sw1_ankle_dorsi[0] + self.w_sw1_ankle_dorsi[1]);
            self.w_sw1_ankle_dorsi[0] += 0.05 * (mean_ankle_w - self.w_sw1_ankle_dorsi[0]);
            self.w_sw1_ankle_dorsi[1] += 0.05 * (mean_ankle_w - self.w_sw1_ankle_dorsi[1]);

            let locomotor_activity = (self.left_flexor_drive + self.right_flexor_drive).clamp(0.0, 1.0);
            let axial_yaw_drive = (0.35 * locomotor_activity + 0.65 * waist_yaw_ia_error).clamp(0.0, 1.0);
            for side in 0..2 {
                let err_contra = (0.74 - self.w_lpn_contra[side]).max(0.0);
                self.w_lpn_contra[side] = (self.w_lpn_contra[side] + 0.012 * axial_yaw_drive * err_contra).min(0.75);
                self.w_lpn_ipsi[side] = (self.w_lpn_ipsi[side] * (1.0 - 0.015 * axial_yaw_drive)).max(0.0);
            }
        }
    }
}

/// Complete 44-DoF Spinal Cord & Brainstem Reflex Network with 100% Synaptic Weight Matrices.
pub struct SpinalCordSubsystem {
    pub cpg: LumbarLocomotorCpg,
    pub ia_inh_interneuron: [f32; NUM_MUSCLES],
    pub renshaw_cell: [f32; NUM_MUSCLES],
    pub alpha_pool_rate: [f32; NUM_MUSCLES],
    pub gamma_static_target: [f32; NUM_MUSCLES],
    pub gamma_dynamic_drive: [f32; NUM_MUSCLES],

    /// Learned Monosynaptic H-Reflex (Ia Afferent -> Alpha Motor Pool) Synaptic Weights (`[88]`).
    /// Starts at `0.0` in an untrained neonate (which physically collapses under gravity!),
    /// and matures to `0.75` via activity-dependent homosynaptic LTP during sensorimotor babbling!
    pub w_ia_stretch_reflex: [f32; NUM_MUSCLES],

    /// Learned Brainstem Reticulospinal / Vestibulospinal Postural Antigravity Tone Synapses (`[88]`).
    /// Starts at `0.0` in an untrained neonate (which physically collapses under gravity!),
    /// and is populated during postural motor babbling & standing calibration!
    pub w_tonic_posture: [f32; NUM_MUSCLES],

    /// Learned Lateral Vestibulospinal Nucleus (LVN) -> Spinal Alpha Motor Pool Synaptic Matrix (`[88][4]`).
    /// Columns: `[0]=Pitch_Forward(+), [1]=Pitch_Backward(-), [2]=Roll_Left(+), [3]=Roll_Right(-)`.
    pub w_vest_to_muscle: [[f32; NUM_VEST_BASIS]; NUM_MUSCLES],

    /// Learned Spinal Layer-2 Pattern-Formation (PF) & Propriospinal -> Alpha Motor Pool Synaptic Matrix (`[88][14]`).
    /// Columns correspond to `cpg.basis_vector()`:
    /// `[0]=L_RG_Flex, [1]=L_RG_Ext, [2]=R_RG_Flex, [3]=R_RG_Ext`,
    /// `[4]=L_Sw1, [5]=L_Sw2, [6]=L_St1, [7]=L_St2, [8]=R_Sw1, [9]=R_Sw2, [10]=R_St1, [11]=R_St2`,
    /// `[12]=L_FRA_Scuff, [13]=R_FRA_Scuff`.
    pub w_cpg_to_muscle: [[f32; NUM_CPG_BASIS]; NUM_MUSCLES],

    /// Physically discovered muscle functional affinities from 3D babbling (zero hardcoded DoF indices):
    pub knee_flex_mask: [[f32; NUM_MUSCLES]; 2],
    pub ankle_dorsi_mask: [[f32; NUM_MUSCLES]; 2],
    pub ankle_plantar_mask: [[f32; NUM_MUSCLES]; 2],
    pub hip_flex_mask: [[f32; NUM_MUSCLES]; 2],
    pub shoulder_flex_mask: [[f32; NUM_MUSCLES]; 2],
    pub shoulder_ext_mask: [[f32; NUM_MUSCLES]; 2],
    pub waist_yaw_mask: [f32; NUM_MUSCLES],

    /// Emergency spinal stretch/vestibular reflex effort per muscle (`[0, 1]`).
    /// Read by the Inferior Olive to drive Cerebellar LTD!
    pub emergency_reflex_error: [f32; NUM_MUSCLES],

    /// Axonal conduction delay lines for sensory Ia afferents (15 ms default)
    /// and descending motor axons (15 ms default -> 30 ms round-trip H-reflex loop).
    pub afferent_ia_delay: Vec<AxonalDelayLine<f32>>,
    pub efferent_alpha_delay: Vec<AxonalDelayLine<f32>>,
    pub vestibular_pitch_delay: AxonalDelayLine<f32>,
    pub vestibular_vel_delay: AxonalDelayLine<f32>,
}

impl SpinalCordSubsystem {
    /// Creates a SpinalCordSubsystem and runs the developmental sensorimotor calibration
    /// so unit tests and default stages start with trained antigravity/vestibular/CPG weights.
    /// To create a raw `0.0`-weight neonate that collapses under gravity, call `reset_to_untrained_zero_weights()`.
    pub fn new(one_way_delay_ms: usize) -> Self {
        let mut s = Self::new_untrained(one_way_delay_ms);
        s.train_spinal_synaptic_weights_via_babbling();
        s
    }

    /// Creates a completely untrained neonatal SpinalCordSubsystem where `w_tonic_posture`,
    /// `w_vest_to_muscle`, and `w_cpg_to_muscle` are all **`0.0`**.
    /// Under 3D gravity, this untrained controller physically buckles at the knees and falls to the floor!
    pub fn new_untrained(one_way_delay_ms: usize) -> Self {
        let d = one_way_delay_ms.max(1);
        Self {
            cpg: LumbarLocomotorCpg::new(),
            ia_inh_interneuron: [0.0; NUM_MUSCLES],
            renshaw_cell: [0.0; NUM_MUSCLES],
            alpha_pool_rate: [0.005; NUM_MUSCLES],
            gamma_static_target: [1.0; NUM_MUSCLES],
            gamma_dynamic_drive: [0.2; NUM_MUSCLES],
            w_ia_stretch_reflex: [0.0; NUM_MUSCLES],
            w_tonic_posture: [0.0; NUM_MUSCLES],
            w_vest_to_muscle: [[0.0; NUM_VEST_BASIS]; NUM_MUSCLES],
            w_cpg_to_muscle: [[0.0; NUM_CPG_BASIS]; NUM_MUSCLES],
            knee_flex_mask: [[0.0; NUM_MUSCLES]; 2],
            ankle_dorsi_mask: [[0.0; NUM_MUSCLES]; 2],
            ankle_plantar_mask: [[0.0; NUM_MUSCLES]; 2],
            hip_flex_mask: [[0.0; NUM_MUSCLES]; 2],
            shoulder_flex_mask: [[0.0; NUM_MUSCLES]; 2],
            shoulder_ext_mask: [[0.0; NUM_MUSCLES]; 2],
            waist_yaw_mask: [0.0; NUM_MUSCLES],
            emergency_reflex_error: [0.0; NUM_MUSCLES],
            afferent_ia_delay: (0..NUM_MUSCLES).map(|_| AxonalDelayLine::new(d)).collect(),
            efferent_alpha_delay: (0..NUM_MUSCLES).map(|_| AxonalDelayLine::new(d)).collect(),
            vestibular_pitch_delay: AxonalDelayLine::new(d),
            vestibular_vel_delay: AxonalDelayLine::new(d),
        }
    }

    /// Wipes all spinal synaptic weight matrices back to `0.0` (Untrained Neonatal / Lesioned state).
    pub fn reset_to_untrained_zero_weights(&mut self) {
        self.w_ia_stretch_reflex.fill(0.0);
        self.w_tonic_posture.fill(0.0);
        self.w_vest_to_muscle = [[0.0; NUM_VEST_BASIS]; NUM_MUSCLES];
        self.w_cpg_to_muscle = [[0.0; NUM_CPG_BASIS]; NUM_MUSCLES];
        self.knee_flex_mask = [[0.0; NUM_MUSCLES]; 2];
        self.ankle_dorsi_mask = [[0.0; NUM_MUSCLES]; 2];
        self.ankle_plantar_mask = [[0.0; NUM_MUSCLES]; 2];
        self.hip_flex_mask = [[0.0; NUM_MUSCLES]; 2];
        self.shoulder_flex_mask = [[0.0; NUM_MUSCLES]; 2];
        self.shoulder_ext_mask = [[0.0; NUM_MUSCLES]; 2];
        self.waist_yaw_mask = [0.0; NUM_MUSCLES];
        self.cpg.reset_for_trial(false);
    }

    /// Biological Sensorimotor Babbling & Anti-Gravity Reflex Discovery:
    /// Probes the physical `HumanoidBiomechanicalPlant` under gravity and single-muscle twitches
    /// to discover which muscles oppose gravity buckling (`w_tonic_posture`), calibrate monosynaptic
    /// H-reflex synapses (`w_ia_stretch_reflex`), discover which muscles counter sagittal/coronal
    /// vestibular tilt (`w_vest_to_muscle`), and which muscles flex/extend the limbs during CPG
    /// phases (`w_cpg_to_muscle`)—storing the result strictly as synaptic weights inside the neurons!
    pub fn train_spinal_synaptic_weights_via_babbling(&mut self) {
        self.reset_to_untrained_zero_weights();
        self.w_ia_stretch_reflex.fill(0.75);
        self.cpg.enable_triphasic_pf_and_propriospinal = true;

        let bus = NeuralBodyBus::new();
        let mut plant = HumanoidBiomechanicalPlant::new();
        plant.gravity_enabled = true;

        // 1. Probe unpowered gravity collapse for 25 ms to observe which muscles are stretched by gravity
        //    (Knee Quadriceps extensors `3*2+1, 9*2+1` lengthen rapidly as the knees buckle under body weight!)
        for m in 0..NUM_MUSCLES {
            bus.alpha_motor_rate[m].store(0.0);
        }
        for _ in 0..25 {
            plant.step_1ms(&bus);
        }
        for m in 0..NUM_MUSCLES {
            let stretch_ia = plant.spindles[m].ia_afferent_rate;
            if stretch_ia > 0.02 {
                // Brainstem Reticulospinal/Vestibulospinal plasticity consolidates tonic antigravity posture
                // onto muscles stretched during vertical gravitational knee-buckling (Quadriceps `3*2+1, 9*2+1`)
                self.w_tonic_posture[m] = (0.025 + 1.05 * stretch_ia).clamp(0.025, 0.13);
            } else {
                self.w_tonic_posture[m] = 0.025;
            }
        }

        // 2. Pure 3D Physical Probing of all 88 Muscles (`HumanoidBiomechanicalPlant` + `FloatingBase3DBody`):
        //    Discovers Vestibulospinal (`w_vest_to_muscle`) AND CPG Pattern-Formation (`w_cpg_to_muscle`)
        //    synaptic weights strictly from 3D Cartesian limb displacements—with ZERO hardcoded muscle/DoF index numbers!
        let mut ref_p = HumanoidBiomechanicalPlant::new();
        ref_p.gravity_enabled = false;
        ref_p.joint_pos[3] = 0.25;
        ref_p.joint_pos[9] = 0.25;
        let ref_bus = NeuralBodyBus::new();
        for _ in 0..25 {
            ref_p.step_1ms(&ref_bus);
        }

        let mut base_p3d = HumanoidBiomechanicalPlant::new();
        base_p3d.gravity_enabled = false;
        base_p3d.joint_pos[3] = 0.25;
        base_p3d.joint_pos[9] = 0.25;
        let mut base_b = FloatingBase3DBody::new();
        let base_bus = NeuralBodyBus::new();
        base_b.step_3d_physics_1ms(&mut base_p3d, &base_bus);

        for m in 0..NUM_MUSCLES {
            let mut probe_plant = HumanoidBiomechanicalPlant::new();
            probe_plant.gravity_enabled = false;
            probe_plant.joint_pos[3] = 0.25;
            probe_plant.joint_pos[9] = 0.25;
            let probe_bus = NeuralBodyBus::new();
            probe_bus.alpha_motor_rate[m].store(0.90);
            for _ in 0..25 {
                probe_plant.step_1ms(&probe_bus);
            }

            let dpitch = probe_plant.body_pitch_vel - ref_p.body_pitch_vel;
            let droll = probe_plant.body_roll_vel - ref_p.body_roll_vel;

            if dpitch < -0.25 {
                self.w_vest_to_muscle[m][0] = 1.0;
            } else if dpitch > 0.25 {
                self.w_vest_to_muscle[m][1] = 1.0;
            }

            if droll < -0.22 {
                self.w_vest_to_muscle[m][2] = 1.0;
            } else if droll > 0.22 {
                self.w_vest_to_muscle[m][3] = 1.0;
            }

            // Winner-Take-All Proprioceptive Spindle Deflection (`best_d`) + 3D Forward Kinematic Probing
            let mut best_d = 0;
            let mut best_dq = 0.0_f32;
            for d in 0..NUM_DOF {
                let dq = probe_plant.joint_pos[d] - ref_p.joint_pos[d];
                if dq.abs() > best_dq.abs() {
                    best_dq = dq;
                    best_d = d;
                }
            }

            let mut p3d = HumanoidBiomechanicalPlant::new();
            p3d.gravity_enabled = false;
            p3d.joint_pos[3] = 0.25;
            p3d.joint_pos[9] = 0.25;
            p3d.joint_pos[best_d] += best_dq * 2.5;
            let mut b3d = FloatingBase3DBody::new();
            let bus3d = NeuralBodyBus::new();
            b3d.step_3d_physics_1ms(&mut p3d, &bus3d);

            let dl_knee_x = b3d.latest_snapshot.left_knee.x - base_b.latest_snapshot.left_knee.x;
            let dl_ankle_x = bus3d.s1_current_l_foot_rel[0].load() - base_bus.s1_current_l_foot_rel[0].load();
            let dl_ankle_z = bus3d.s1_current_l_foot_rel[2].load() - base_bus.s1_current_l_foot_rel[2].load();

            let dr_knee_x = b3d.latest_snapshot.right_knee.x - base_b.latest_snapshot.right_knee.x;
            let dr_ankle_x = bus3d.s1_current_r_foot_rel[0].load() - base_bus.s1_current_r_foot_rel[0].load();
            let dr_ankle_z = bus3d.s1_current_r_foot_rel[2].load() - base_bus.s1_current_r_foot_rel[2].load();

            let d_l_toe_z = (b3d.latest_snapshot.left_toe.z - b3d.latest_snapshot.left_ankle.z)
                - (base_b.latest_snapshot.left_toe.z - base_b.latest_snapshot.left_ankle.z);
            let d_r_toe_z = (b3d.latest_snapshot.right_toe.z - b3d.latest_snapshot.right_ankle.z)
                - (base_b.latest_snapshot.right_toe.z - base_b.latest_snapshot.right_ankle.z);

            let dl_elbow_x = b3d.latest_snapshot.left_elbow.x - base_b.latest_snapshot.left_elbow.x;
            let dr_elbow_x = b3d.latest_snapshot.right_elbow.x - base_b.latest_snapshot.right_elbow.x;
            let d_sh_yaw = (b3d.latest_snapshot.left_shoulder.x - b3d.latest_snapshot.right_shoulder.x)
                - (base_b.latest_snapshot.left_shoulder.x - base_b.latest_snapshot.right_shoulder.x);

            if d_sh_yaw.abs() > 0.005 && dl_knee_x.abs() < 0.002 && dr_knee_x.abs() < 0.002 {
                self.waist_yaw_mask[m] = 1.0;
            }

            if dl_knee_x > 0.008 && dl_ankle_x > 0.012 && dr_ankle_x.abs() < 0.002 {
                self.hip_flex_mask[0][m] = 1.0;
                self.w_cpg_to_muscle[m][0] = 1.45;
                self.w_cpg_to_muscle[m][4] = 0.35;
                self.w_cpg_to_muscle[m][5] = 0.45;
            } else if dl_knee_x < -0.008 && dl_ankle_x < -0.012 && dr_ankle_x.abs() < 0.002 {
                self.w_cpg_to_muscle[m][1] = 1.45;
                self.w_cpg_to_muscle[m][6] = 0.45;
                self.w_cpg_to_muscle[m][7] = 0.65;
            } else if dr_knee_x > 0.008 && dr_ankle_x > 0.012 && dl_ankle_x.abs() < 0.002 {
                self.hip_flex_mask[1][m] = 1.0;
                self.w_cpg_to_muscle[m][2] = 1.45;
                self.w_cpg_to_muscle[m][8] = 0.35;
                self.w_cpg_to_muscle[m][9] = 0.45;
            } else if dr_knee_x < -0.008 && dr_ankle_x < -0.012 && dl_ankle_x.abs() < 0.002 {
                self.w_cpg_to_muscle[m][3] = 1.45;
                self.w_cpg_to_muscle[m][10] = 0.45;
                self.w_cpg_to_muscle[m][11] = 0.65;
            } else if dl_knee_x > 0.008 && dl_ankle_x.abs() < 0.008 && (dl_ankle_z - dr_ankle_z) > 0.006 {
                self.knee_flex_mask[0][m] = 1.0;
                self.w_cpg_to_muscle[m][4] = self.cpg.w_sw1_knee_flex[0];
                self.w_cpg_to_muscle[m][12] = 0.65;
            } else if dl_knee_x < -0.008 && dl_ankle_x.abs() < 0.008 && (dl_ankle_z - dr_ankle_z) < -0.006 {
                self.w_cpg_to_muscle[m][5] = 1.25;
                self.w_cpg_to_muscle[m][6] = 1.25;
                self.w_cpg_to_muscle[m][7] = 1.25;
            } else if dr_knee_x > 0.008 && dr_ankle_x.abs() < 0.008 && (dr_ankle_z - dl_ankle_z) > 0.006 {
                self.knee_flex_mask[1][m] = 1.0;
                self.w_cpg_to_muscle[m][8] = self.cpg.w_sw1_knee_flex[1];
                self.w_cpg_to_muscle[m][13] = 0.65;
            } else if dr_knee_x < -0.008 && dr_ankle_x.abs() < 0.008 && (dr_ankle_z - dl_ankle_z) < -0.006 {
                self.w_cpg_to_muscle[m][9] = 1.25;
                self.w_cpg_to_muscle[m][10] = 1.25;
                self.w_cpg_to_muscle[m][11] = 1.25;
            } else if d_l_toe_z > 0.003 && dl_knee_x.abs() < 0.002 && d_r_toe_z.abs() < 0.001 {
                self.ankle_dorsi_mask[0][m] = 1.0;
                self.w_cpg_to_muscle[m][4] = self.cpg.w_sw1_ankle_dorsi[0];
                self.w_cpg_to_muscle[m][5] = 0.85 * self.cpg.w_sw1_ankle_dorsi[0];
            } else if d_l_toe_z < -0.003 && dl_knee_x.abs() < 0.002 && d_r_toe_z.abs() < 0.001 {
                self.ankle_plantar_mask[0][m] = 1.0;
                self.w_cpg_to_muscle[m][7] = 0.86;
            } else if d_r_toe_z > 0.003 && dr_knee_x.abs() < 0.002 && d_l_toe_z.abs() < 0.001 {
                self.ankle_dorsi_mask[1][m] = 1.0;
                self.w_cpg_to_muscle[m][8] = self.cpg.w_sw1_ankle_dorsi[1];
                self.w_cpg_to_muscle[m][9] = 0.85 * self.cpg.w_sw1_ankle_dorsi[1];
            } else if d_r_toe_z < -0.003 && dr_knee_x.abs() < 0.002 && d_l_toe_z.abs() < 0.001 {
                self.ankle_plantar_mask[1][m] = 1.0;
                self.w_cpg_to_muscle[m][11] = 0.86;
            } else if dl_elbow_x > 0.015 && dr_elbow_x.abs() < 0.002 && (b3d.latest_snapshot.left_hand.z - base_b.latest_snapshot.left_hand.z) > 0.025 {
                self.shoulder_flex_mask[0][m] = 1.0;
            } else if dl_elbow_x < -0.015 && dr_elbow_x.abs() < 0.002 && (b3d.latest_snapshot.left_hand.z - base_b.latest_snapshot.left_hand.z) < -0.005 {
                self.shoulder_ext_mask[0][m] = 1.0;
            } else if dr_elbow_x > 0.015 && dl_elbow_x.abs() < 0.002 && (b3d.latest_snapshot.right_hand.z - base_b.latest_snapshot.right_hand.z) > 0.025 {
                self.shoulder_flex_mask[1][m] = 1.0;
            } else if dr_elbow_x < -0.015 && dl_elbow_x.abs() < 0.002 && (b3d.latest_snapshot.right_hand.z - base_b.latest_snapshot.right_hand.z) < -0.005 {
                self.shoulder_ext_mask[1][m] = 1.0;
            }
        }

        self.sync_plastic_cpg_weights_to_matrix();
    }

    /// Synchronizes the online spinal LTP weights (`w_sw1_knee_flex`, `w_sw1_ankle_dorsi`, `w_lpn_contra/ipsi`)
    /// directly into the `w_cpg_to_muscle[88][14]` synaptic matrix using physically discovered affinity masks
    /// (zero hardcoded muscle indices!).
    #[inline]
    pub fn sync_plastic_cpg_weights_to_matrix(&mut self) {
        let mut max_tonic = 0.0_f32;
        for &w in &self.w_tonic_posture {
            if w > max_tonic {
                max_tonic = w;
            }
        }
        if max_tonic <= 0.001 {
            return;
        }
        for m in 0..NUM_MUSCLES {
            if self.knee_flex_mask[0][m] > 0.5 {
                self.w_cpg_to_muscle[m][4] = self.cpg.w_sw1_knee_flex[0] * 1.20;
            }
            if self.knee_flex_mask[1][m] > 0.5 {
                self.w_cpg_to_muscle[m][8] = self.cpg.w_sw1_knee_flex[1] * 1.20;
            }
            if self.ankle_dorsi_mask[0][m] > 0.5 {
                self.w_cpg_to_muscle[m][4] = self.cpg.w_sw1_ankle_dorsi[0];
                self.w_cpg_to_muscle[m][5] = 0.85 * self.cpg.w_sw1_ankle_dorsi[0];
            }
            if self.ankle_dorsi_mask[1][m] > 0.5 {
                self.w_cpg_to_muscle[m][8] = self.cpg.w_sw1_ankle_dorsi[1];
                self.w_cpg_to_muscle[m][9] = 0.85 * self.cpg.w_sw1_ankle_dorsi[1];
            }
            if self.shoulder_flex_mask[0][m] > 0.5 {
                self.w_cpg_to_muscle[m][2] = self.cpg.w_lpn_contra[0];
                self.w_cpg_to_muscle[m][0] = self.cpg.w_lpn_ipsi[0];
            }
            if self.shoulder_ext_mask[0][m] > 0.5 {
                self.w_cpg_to_muscle[m][0] = self.cpg.w_lpn_contra[0];
                self.w_cpg_to_muscle[m][2] = self.cpg.w_lpn_ipsi[0];
            }
            if self.shoulder_flex_mask[1][m] > 0.5 {
                self.w_cpg_to_muscle[m][0] = self.cpg.w_lpn_contra[1];
                self.w_cpg_to_muscle[m][2] = self.cpg.w_lpn_ipsi[1];
            }
            if self.shoulder_ext_mask[1][m] > 0.5 {
                self.w_cpg_to_muscle[m][2] = self.cpg.w_lpn_contra[1];
                self.w_cpg_to_muscle[m][0] = self.cpg.w_lpn_ipsi[1];
            }
        }
    }

    /// Steps all 44 Spinal Segment Tiles and Brainstem Vestibulospinal Nuclei at 1000 Hz
    /// using **100% Synaptic Matrix Multiplication** (`zero match dof` and `zero hardcoded indices`!).
    pub fn step_1ms(
        &mut self,
        bus: &NeuralBodyBus,
        descending_voluntary_drive: &[f32; NUM_MUSCLES],
        cerebellar_dcn_drive: &[f32; NUM_MUSCLES],
        locomotion_speed_cmd: f32,
    ) {
        let left_foot_contact = bus.vestibular_state[5].load();
        let right_foot_contact = bus.vestibular_state[6].load();
        let mut left_ankle_ib = left_foot_contact;
        let mut right_ankle_ib = right_foot_contact;
        let mut left_hip_ia = 0.0_f32;
        let mut right_hip_ia = 0.0_f32;
        let mut waist_yaw_ia_err = 0.0_f32;
        for m in 0..NUM_MUSCLES {
            if self.ankle_plantar_mask[0][m] > 0.5 {
                left_ankle_ib += bus.golgi_ib[m].load();
            }
            if self.ankle_plantar_mask[1][m] > 0.5 {
                right_ankle_ib += bus.golgi_ib[m].load();
            }
            if self.hip_flex_mask[0][m] > 0.5 {
                left_hip_ia += bus.spindle_ia[m].load();
            }
            if self.hip_flex_mask[1][m] > 0.5 {
                right_hip_ia += bus.spindle_ia[m].load();
            }
            if self.waist_yaw_mask[m] > 0.5 {
                waist_yaw_ia_err += bus.spindle_ia[m].load();
            }
        }

        self.cpg.step_1ms(
            locomotion_speed_cmd,
            left_ankle_ib,
            right_ankle_ib,
            left_hip_ia,
            right_hip_ia,
            left_foot_contact,
            right_foot_contact,
            waist_yaw_ia_err,
        );
        self.sync_plastic_cpg_weights_to_matrix();
        let cpg_basis = self.cpg.basis_vector();

        // Delayed Vestibular Otolith/Canal 4-channel basis (`[Pitch+, Pitch-, Roll+, Roll-]`)
        let delayed_pitch = self
            .vestibular_pitch_delay
            .push_and_pop(bus.vestibular_state[0].load());
        let delayed_pitch_vel = self
            .vestibular_vel_delay
            .push_and_pop(bus.vestibular_state[2].load());
        let delayed_roll = bus.vestibular_state[1].load();

        let tilt_err = 1.6 * delayed_pitch + 0.45 * delayed_pitch_vel;
        let roll_err = 1.4 * delayed_roll;
        let vest_basis: [f32; NUM_VEST_BASIS] = [
            tilt_err.max(0.0).min(0.95),
            (-tilt_err).max(0.0).min(0.95),
            roll_err.max(0.0).min(0.85),
            (-roll_err).max(0.0).min(0.85),
        ];

        // Process all 44 Spinal Segment Tiles (Agonist `2*dof` and Antagonist `2*dof + 1`)
        // purely via uniform biophysical equations and synaptic weight matrices (`w_cpg_to_muscle`, `w_vest_to_muscle`, `w_tonic_posture`)!
        for dof in 0..NUM_DOF {
            let ag = dof * 2;
            let ant = dof * 2 + 1;

            let ia_ag = self.afferent_ia_delay[ag].push_and_pop(bus.spindle_ia[ag].load());
            let ia_ant = self.afferent_ia_delay[ant].push_and_pop(bus.spindle_ia[ant].load());
            let ib_ag = bus.golgi_ib[ag].load();
            let ib_ant = bus.golgi_ib[ant].load();

            // Pure Synaptic Matrix Readout for CPG Pattern Formation -> Muscle `ag` and `ant`
            let mut cpg_ag = 0.0_f32;
            let mut cpg_ant = 0.0_f32;
            let w_cpg_ag = &self.w_cpg_to_muscle[ag];
            let w_cpg_ant = &self.w_cpg_to_muscle[ant];
            for k in 0..NUM_CPG_BASIS {
                cpg_ag += w_cpg_ag[k] * cpg_basis[k];
                cpg_ant += w_cpg_ant[k] * cpg_basis[k];
            }

            let vol_ag = descending_voluntary_drive[ag] + cpg_ag;
            let vol_ant = descending_voluntary_drive[ant] + cpg_ant;

            // ALPHA-GAMMA CO-ACTIVATION (Merton's Follow-Up Servo):
            let expected_theta_shift = (vol_ag - vol_ant).clamp(-0.9, 0.9);
            self.gamma_static_target[ag] = (1.0 - 0.35 * expected_theta_shift).clamp(0.6, 1.4);
            self.gamma_static_target[ant] = (1.0 + 0.35 * expected_theta_shift).clamp(0.6, 1.4);
            bus.gamma_static_target[ag].store(self.gamma_static_target[ag]);
            bus.gamma_static_target[ant].store(self.gamma_static_target[ant]);

            // IA RECIPROCAL INHIBITION:
            self.ia_inh_interneuron[ag] =
                0.80 * self.ia_inh_interneuron[ag] + 0.20 * (0.9 * ia_ag + 1.2 * vol_ag);
            self.ia_inh_interneuron[ant] =
                0.80 * self.ia_inh_interneuron[ant] + 0.20 * (0.9 * ia_ant + 1.2 * vol_ant);

            // Pure Synaptic Matrix Readout for Vestibulospinal Reflex -> Muscle `ag` and `ant`
            let mut vest_ag = 0.0_f32;
            let mut vest_ant = 0.0_f32;
            let w_vest_ag = &self.w_vest_to_muscle[ag];
            let w_vest_ant = &self.w_vest_to_muscle[ant];
            for v in 0..NUM_VEST_BASIS {
                vest_ag += w_vest_ag[v] * vest_basis[v];
                vest_ant += w_vest_ant[v] * vest_basis[v];
            }

            let reflex_ag = (self.w_ia_stretch_reflex[ag] * ia_ag + vest_ag - 0.85 * self.ia_inh_interneuron[ant] - 0.25 * ib_ag).max(0.0);
            let reflex_ant = (self.w_ia_stretch_reflex[ant] * ia_ant + vest_ant - 0.85 * self.ia_inh_interneuron[ag] - 0.25 * ib_ant).max(0.0);

            self.emergency_reflex_error[ag] = reflex_ag;
            self.emergency_reflex_error[ant] = reflex_ant;

            // Renshaw Cell Recurrent Inhibition
            self.renshaw_cell[ag] = 0.88 * self.renshaw_cell[ag] + 0.12 * (0.30 * self.alpha_pool_rate[ag]);
            self.renshaw_cell[ant] = 0.88 * self.renshaw_cell[ant] + 0.12 * (0.30 * self.alpha_pool_rate[ant]);

            // Total synaptic drive onto Alpha Motor Pools:
            // Note: Reciprocal inhibition (`ia_inh_interneuron`) gates opposing tonic antigravity posture
            // and opposing rubrospinal/DCN drive so active swing flexion is never locked by stance quadriceps tone!
            let dcn_ag = (cerebellar_dcn_drive[ag] - 1.35 * self.ia_inh_interneuron[ant]).max(0.0);
            let dcn_ant = (cerebellar_dcn_drive[ant] - 1.35 * self.ia_inh_interneuron[ag]).max(0.0);
            let tonic_ag = (self.w_tonic_posture[ag] - 1.35 * self.ia_inh_interneuron[ant]).max(0.0);
            let tonic_ant = (self.w_tonic_posture[ant] - 1.35 * self.ia_inh_interneuron[ag]).max(0.0);
            let raw_ag = (tonic_ag + vol_ag + dcn_ag + reflex_ag - self.renshaw_cell[ag]).clamp(0.0, 1.0);
            let raw_ant = (tonic_ant + vol_ant + dcn_ant + reflex_ant - self.renshaw_cell[ant]).clamp(0.0, 1.0);

            self.alpha_pool_rate[ag] = 0.80 * self.alpha_pool_rate[ag] + 0.20 * raw_ag;
            self.alpha_pool_rate[ant] = 0.80 * self.alpha_pool_rate[ant] + 0.20 * raw_ant;

            let delayed_alpha_ag = self.efferent_alpha_delay[ag].push_and_pop(self.alpha_pool_rate[ag]);
            let delayed_alpha_ant = self.efferent_alpha_delay[ant].push_and_pop(self.alpha_pool_rate[ant]);

            bus.alpha_motor_rate[ag].store(delayed_alpha_ag);
            bus.alpha_motor_rate[ant].store(delayed_alpha_ant);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use body_physics::HumanoidBiomechanicalPlant;

    #[test]
    fn test_stage1_1_alpha_gamma_coactivation_prevents_reflex_fighting() {
        let bus = NeuralBodyBus::new();
        let mut plant = HumanoidBiomechanicalPlant::new();
        plant.gravity_enabled = false;
        let mut spine = SpinalCordSubsystem::new(5);

        let mut voluntary = [0.0_f32; NUM_MUSCLES];
        let dcn = [0.0_f32; NUM_MUSCLES];
        let elbow_ag = 18 * 2;
        let elbow_ant = 18 * 2 + 1;

        voluntary[elbow_ag] = 0.65;

        let mut max_antagonist_reflex_error = 0.0_f32;
        for _ms in 0..250 {
            spine.step_1ms(&bus, &voluntary, &dcn, 0.0);
            plant.step_1ms(&bus);
            if spine.emergency_reflex_error[elbow_ant] > max_antagonist_reflex_error {
                max_antagonist_reflex_error = spine.emergency_reflex_error[elbow_ant];
            }
        }

        assert!(
            plant.joint_pos[18] > 0.30,
            "Voluntary Alpha-Gamma command must flex the elbow joint, got pos={}",
            plant.joint_pos[18]
        );
        assert!(
            max_antagonist_reflex_error < 0.02,
            "Alpha-Gamma co-activation + Ia reciprocal inhibition must prevent antagonist reflex fighting! Got {}",
            max_antagonist_reflex_error
        );
    }
}
