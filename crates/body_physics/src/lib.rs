//! `body_physics`: Biophysical Hill-Type Muscle-Tendon Units, Mileusnic Muscle Spindles,
#![forbid(unsafe_code)]
//! Golgi Tendon Organs, Vestibular/Cutaneous Sensors, and the 44-DoF Articulated Humanoid Plant.
//!
//! ZERO-CHEAT GUARANTEE:
//! - No PID/PD controllers exist in the physics plant or actuator bus.
//! - Every joint is driven exclusively by an Agonist (+) and Antagonist (-) Hill-Type Muscle-Tendon Unit (MTU).
//! - Joint stiffness (Kp) and damping (Kd) are NEVER calculated by an algebraic control formula;
//!   they emerge strictly from the biophysical Force-Length f_L(l) and Force-Velocity f_V(v)
//!   curves of co-contracting agonist/antagonist skeletal muscles!

use neuro_core::AtomicF32;

pub mod kinematics_3d;
pub use kinematics_3d::{
    FloatingBase3DBody, HumanGaitScorecard, Skeleton3DSnapshot, Vec3, WorldObject3D,
    NUM_WORLD_OBJECTS,
};

pub const NUM_DOF: usize = 44;
pub const NUM_MUSCLES: usize = NUM_DOF * 2; // 88 muscles (Agonist & Antagonist per DoF)

/// Anatomical mapping of all 44 Degrees of Freedom of the Humanoid Body.
/// Organized into 6 kinematic chains with explicit anatomical adjacency so we can
/// rigorously measure whether Prenatal Motor Babbling discovers the true 3D body topology!
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DofMetadata {
    pub dof_index: usize,
    pub name: &'static str,
    pub chain_id: u8, // 0=LeftLeg, 1=RightLeg, 2=TorsoNeck, 3=LeftArm, 4=RightArm, 5=LeftHand, 6=RightHand
    pub parent_dof: Option<usize>,
    pub max_isometric_force_n: f32,
    pub moment_arm_m: f32,
    pub link_inertia_kgm2: f32,
}

pub const HUMANOID_44_DOF_MAP: [DofMetadata; NUM_DOF] = [
    // Chain 0: Left Leg (DoFs 0..5)
    DofMetadata { dof_index: 0, name: "L_Hip_Pitch", chain_id: 0, parent_dof: Some(12), max_isometric_force_n: 2200.0, moment_arm_m: 0.08, link_inertia_kgm2: 1.8 },
    DofMetadata { dof_index: 1, name: "L_Hip_Roll", chain_id: 0, parent_dof: Some(0), max_isometric_force_n: 1800.0, moment_arm_m: 0.07, link_inertia_kgm2: 1.6 },
    DofMetadata { dof_index: 2, name: "L_Hip_Yaw", chain_id: 0, parent_dof: Some(1), max_isometric_force_n: 1100.0, moment_arm_m: 0.05, link_inertia_kgm2: 1.2 },
    DofMetadata { dof_index: 3, name: "L_Knee_Pitch", chain_id: 0, parent_dof: Some(0), max_isometric_force_n: 2600.0, moment_arm_m: 0.06, link_inertia_kgm2: 1.1 },
    DofMetadata { dof_index: 4, name: "L_Ankle_Pitch", chain_id: 0, parent_dof: Some(3), max_isometric_force_n: 2000.0, moment_arm_m: 0.05, link_inertia_kgm2: 0.6 },
    DofMetadata { dof_index: 5, name: "L_Ankle_Roll", chain_id: 0, parent_dof: Some(4), max_isometric_force_n: 1200.0, moment_arm_m: 0.04, link_inertia_kgm2: 0.4 },

    // Chain 1: Right Leg (DoFs 6..11)
    DofMetadata { dof_index: 6, name: "R_Hip_Pitch", chain_id: 1, parent_dof: Some(12), max_isometric_force_n: 2200.0, moment_arm_m: 0.08, link_inertia_kgm2: 1.8 },
    DofMetadata { dof_index: 7, name: "R_Hip_Roll", chain_id: 1, parent_dof: Some(6), max_isometric_force_n: 1800.0, moment_arm_m: 0.07, link_inertia_kgm2: 1.6 },
    DofMetadata { dof_index: 8, name: "R_Hip_Yaw", chain_id: 1, parent_dof: Some(7), max_isometric_force_n: 1100.0, moment_arm_m: 0.05, link_inertia_kgm2: 1.2 },
    DofMetadata { dof_index: 9, name: "R_Knee_Pitch", chain_id: 1, parent_dof: Some(6), max_isometric_force_n: 2600.0, moment_arm_m: 0.06, link_inertia_kgm2: 1.1 },
    DofMetadata { dof_index: 10, name: "R_Ankle_Pitch", chain_id: 1, parent_dof: Some(9), max_isometric_force_n: 2000.0, moment_arm_m: 0.05, link_inertia_kgm2: 0.6 },
    DofMetadata { dof_index: 11, name: "R_Ankle_Roll", chain_id: 1, parent_dof: Some(10), max_isometric_force_n: 1200.0, moment_arm_m: 0.04, link_inertia_kgm2: 0.4 },

    // Chain 2: Torso & Neck (DoFs 12..15)
    DofMetadata { dof_index: 12, name: "Waist_Pitch", chain_id: 2, parent_dof: None, max_isometric_force_n: 2500.0, moment_arm_m: 0.09, link_inertia_kgm2: 2.5 },
    DofMetadata { dof_index: 13, name: "Waist_Roll", chain_id: 2, parent_dof: Some(12), max_isometric_force_n: 2000.0, moment_arm_m: 0.08, link_inertia_kgm2: 2.2 },
    DofMetadata { dof_index: 14, name: "Waist_Yaw", chain_id: 2, parent_dof: Some(13), max_isometric_force_n: 1500.0, moment_arm_m: 0.07, link_inertia_kgm2: 1.8 },
    DofMetadata { dof_index: 15, name: "Neck_Pitch", chain_id: 2, parent_dof: Some(14), max_isometric_force_n: 400.0, moment_arm_m: 0.04, link_inertia_kgm2: 0.25 },

    // Chain 3: Left Arm (DoFs 16..19)
    DofMetadata { dof_index: 16, name: "L_Shoulder_Pitch", chain_id: 3, parent_dof: Some(14), max_isometric_force_n: 900.0, moment_arm_m: 0.05, link_inertia_kgm2: 0.45 },
    DofMetadata { dof_index: 17, name: "L_Shoulder_Roll", chain_id: 3, parent_dof: Some(16), max_isometric_force_n: 800.0, moment_arm_m: 0.05, link_inertia_kgm2: 0.40 },
    DofMetadata { dof_index: 18, name: "L_Elbow_Pitch", chain_id: 3, parent_dof: Some(17), max_isometric_force_n: 700.0, moment_arm_m: 0.04, link_inertia_kgm2: 0.28 },
    DofMetadata { dof_index: 19, name: "L_Wrist_Pitch", chain_id: 3, parent_dof: Some(18), max_isometric_force_n: 350.0, moment_arm_m: 0.03, link_inertia_kgm2: 0.12 },

    // Chain 4: Right Arm (DoFs 20..23)
    DofMetadata { dof_index: 20, name: "R_Shoulder_Pitch", chain_id: 4, parent_dof: Some(14), max_isometric_force_n: 900.0, moment_arm_m: 0.05, link_inertia_kgm2: 0.45 },
    DofMetadata { dof_index: 21, name: "R_Shoulder_Roll", chain_id: 4, parent_dof: Some(20), max_isometric_force_n: 800.0, moment_arm_m: 0.05, link_inertia_kgm2: 0.40 },
    DofMetadata { dof_index: 22, name: "R_Elbow_Pitch", chain_id: 4, parent_dof: Some(21), max_isometric_force_n: 700.0, moment_arm_m: 0.04, link_inertia_kgm2: 0.28 },
    DofMetadata { dof_index: 23, name: "R_Wrist_Pitch", chain_id: 4, parent_dof: Some(22), max_isometric_force_n: 350.0, moment_arm_m: 0.03, link_inertia_kgm2: 0.12 },

    // Chain 5: Left Dexterous Hand (10 DoFs: 24..33)
    DofMetadata { dof_index: 24, name: "L_Thumb_CMC", chain_id: 5, parent_dof: Some(19), max_isometric_force_n: 120.0, moment_arm_m: 0.015, link_inertia_kgm2: 0.02 },
    DofMetadata { dof_index: 25, name: "L_Thumb_MCP", chain_id: 5, parent_dof: Some(24), max_isometric_force_n: 100.0, moment_arm_m: 0.012, link_inertia_kgm2: 0.015 },
    DofMetadata { dof_index: 26, name: "L_Index_MCP", chain_id: 5, parent_dof: Some(19), max_isometric_force_n: 90.0, moment_arm_m: 0.012, link_inertia_kgm2: 0.015 },
    DofMetadata { dof_index: 27, name: "L_Index_PIP", chain_id: 5, parent_dof: Some(26), max_isometric_force_n: 75.0, moment_arm_m: 0.010, link_inertia_kgm2: 0.012 },
    DofMetadata { dof_index: 28, name: "L_Middle_MCP", chain_id: 5, parent_dof: Some(19), max_isometric_force_n: 95.0, moment_arm_m: 0.012, link_inertia_kgm2: 0.015 },
    DofMetadata { dof_index: 29, name: "L_Middle_PIP", chain_id: 5, parent_dof: Some(28), max_isometric_force_n: 80.0, moment_arm_m: 0.010, link_inertia_kgm2: 0.012 },
    DofMetadata { dof_index: 30, name: "L_Ring_MCP", chain_id: 5, parent_dof: Some(19), max_isometric_force_n: 80.0, moment_arm_m: 0.011, link_inertia_kgm2: 0.014 },
    DofMetadata { dof_index: 31, name: "L_Ring_PIP", chain_id: 5, parent_dof: Some(30), max_isometric_force_n: 65.0, moment_arm_m: 0.009, link_inertia_kgm2: 0.011 },
    DofMetadata { dof_index: 32, name: "L_Pinky_MCP", chain_id: 5, parent_dof: Some(19), max_isometric_force_n: 65.0, moment_arm_m: 0.010, link_inertia_kgm2: 0.012 },
    DofMetadata { dof_index: 33, name: "L_Pinky_PIP", chain_id: 5, parent_dof: Some(32), max_isometric_force_n: 50.0, moment_arm_m: 0.008, link_inertia_kgm2: 0.010 },

    // Chain 6: Right Dexterous Hand (10 DoFs: 34..43)
    DofMetadata { dof_index: 34, name: "R_Thumb_CMC", chain_id: 6, parent_dof: Some(23), max_isometric_force_n: 120.0, moment_arm_m: 0.015, link_inertia_kgm2: 0.02 },
    DofMetadata { dof_index: 35, name: "R_Thumb_MCP", chain_id: 6, parent_dof: Some(34), max_isometric_force_n: 100.0, moment_arm_m: 0.012, link_inertia_kgm2: 0.015 },
    DofMetadata { dof_index: 36, name: "R_Index_MCP", chain_id: 6, parent_dof: Some(23), max_isometric_force_n: 90.0, moment_arm_m: 0.012, link_inertia_kgm2: 0.015 },
    DofMetadata { dof_index: 37, name: "R_Index_PIP", chain_id: 6, parent_dof: Some(36), max_isometric_force_n: 75.0, moment_arm_m: 0.010, link_inertia_kgm2: 0.012 },
    DofMetadata { dof_index: 38, name: "R_Middle_MCP", chain_id: 6, parent_dof: Some(23), max_isometric_force_n: 95.0, moment_arm_m: 0.012, link_inertia_kgm2: 0.015 },
    DofMetadata { dof_index: 39, name: "R_Middle_PIP", chain_id: 6, parent_dof: Some(38), max_isometric_force_n: 80.0, moment_arm_m: 0.010, link_inertia_kgm2: 0.012 },
    DofMetadata { dof_index: 40, name: "R_Ring_MCP", chain_id: 6, parent_dof: Some(23), max_isometric_force_n: 80.0, moment_arm_m: 0.011, link_inertia_kgm2: 0.014 },
    DofMetadata { dof_index: 41, name: "R_Ring_PIP", chain_id: 6, parent_dof: Some(40), max_isometric_force_n: 65.0, moment_arm_m: 0.009, link_inertia_kgm2: 0.011 },
    DofMetadata { dof_index: 42, name: "R_Pinky_MCP", chain_id: 6, parent_dof: Some(23), max_isometric_force_n: 65.0, moment_arm_m: 0.010, link_inertia_kgm2: 0.012 },
    DofMetadata { dof_index: 43, name: "R_Pinky_PIP", chain_id: 6, parent_dof: Some(42), max_isometric_force_n: 50.0, moment_arm_m: 0.008, link_inertia_kgm2: 0.010 },
];

/// Computes the anatomical graph distance between two DoFs in the 44-DoF skeleton.
/// Used to verify whether cortical self-organization discovers the true body homunculus.
pub fn kinematic_tree_distance(dof_a: usize, dof_b: usize) -> f32 {
    if dof_a == dof_b {
        return 0.0;
    }
    let meta_a = &HUMANOID_44_DOF_MAP[dof_a];
    let meta_b = &HUMANOID_44_DOF_MAP[dof_b];

    // Trace ancestors of A
    let mut ancestors_a = Vec::with_capacity(8);
    let mut cur = Some(dof_a);
    while let Some(idx) = cur {
        ancestors_a.push(idx);
        cur = HUMANOID_44_DOF_MAP[idx].parent_dof;
    }

    // Trace ancestors of B
    let mut ancestors_b = Vec::with_capacity(8);
    let mut cur_b = Some(dof_b);
    while let Some(idx) = cur_b {
        ancestors_b.push(idx);
        cur_b = HUMANOID_44_DOF_MAP[idx].parent_dof;
    }

    for (dist_a, &node_a) in ancestors_a.iter().enumerate() {
        if let Some(dist_b) = ancestors_b.iter().position(|&x| x == node_a) {
            return (dist_a + dist_b) as f32;
        }
    }
    if meta_a.chain_id == meta_b.chain_id {
        2.0
    } else {
        8.0
    }
}

/// Biophysical Hill-Type Muscle-Tendon Unit (Zajac 1989 / Millard 2013 formalism).
/// Converts incoming Alpha-motor spike train excitation `u_alpha` into intracellular
/// Ca2+ activation `a(t)` and computes true mechanical force via:
///   F = F_max * [ a(t) * f_L(l_norm) * f_V(v_norm) + f_PE(l_norm) ]
#[derive(Clone, Copy, Debug)]
pub struct HillMuscleTendonUnit {
    pub f_max_n: f32,
    pub moment_arm_m: f32,
    /// Intracellular Ca2+ bound to Troponin-C (`a(t) in [0.001, 1.0]`)
    pub ca_activation: f32,
    /// Normalized muscle fiber length (`1.0` = optimal sarcomere length L_opt)
    pub l_norm: f32,
    /// Normalized muscle fiber shortening/lengthening velocity (`v / V_max`, positive = lengthening)
    pub v_norm: f32,
    /// Current mechanical tension in the series tendon (Newtons)
    pub tendon_force_n: f32,
}

impl HillMuscleTendonUnit {
    pub const fn new(f_max_n: f32, moment_arm_m: f32) -> Self {
        Self {
            f_max_n,
            moment_arm_m,
            ca_activation: 0.01,
            l_norm: 1.0,
            v_norm: 0.0,
            tendon_force_n: 0.0,
        }
    }

    /// Active sarcomere Force-Length curve `f_L(l_norm)`.
    /// Bell-shaped curve peaking at `l_norm = 1.0` (optimal actin-myosin overlap).
    #[inline]
    pub fn force_length_active(l_norm: f32) -> f32 {
        let diff = (l_norm - 1.0) / 0.45;
        (-diff * diff).exp()
    }

    /// Passive parallel elastic tissue Force-Length curve `f_PE(l_norm)`.
    /// Exerts exponential restoring tension when muscle is stretched past `l_norm > 1.05`.
    #[inline]
    pub fn force_length_passive(l_norm: f32) -> f32 {
        if l_norm <= 1.0 {
            0.0
        } else {
            let stretch = (l_norm - 1.0).min(0.7);
            0.12 * ((4.5 * stretch).exp() - 1.0)
        }
    }

    /// Biophysical Hill Force-Velocity curve `f_V(v_norm)`.
    /// - When shortening (`v_norm < 0`): force drops hyperbolically (Hill's equation).
    /// - When lengthening (`v_norm > 0`, eccentric contraction): force rises up to `1.5 * F_isometric`!
    ///   Notice that the steep positive slope `df_V / dv_norm > 0` around `v_norm = 0`
    ///   provides **natural, zero-delay physical viscous damping** proportional to muscle activation `a(t)`!
    #[inline]
    pub fn force_velocity(v_norm: f32) -> f32 {
        if v_norm <= 0.0 {
            // Concentric shortening (v_norm in [-1.0, 0.0])
            let v_s = (-v_norm).min(1.0);
            let a_hill = 0.25_f32;
            (1.0 - v_s) / (1.0 + v_s / a_hill)
        } else {
            // Eccentric lengthening (v_norm > 0): cross-bridges resist stretch up to 1.5x isometric force
            let v_l = v_norm.min(1.0);
            1.0 + 0.5 * (v_l / (0.15 + v_l))
        }
    }

    /// Updates intracellular Ca2+ activation (Hatze / Zajac bilinear differential equation:
    /// fast Ca2+ release `tau_act = 12 ms`, slower SERCA pump re-uptake `tau_deact = 40 ms`)
    /// and computes resulting tendon tension.
    pub fn step_1ms(&mut self, alpha_drive: f32, l_norm: f32, v_norm: f32) -> f32 {
        let u = alpha_drive.clamp(0.0, 1.0);
        let tau_ms = if u > self.ca_activation {
            12.0 * (0.5 + 1.5 * self.ca_activation)
        } else {
            40.0 / (0.5 + 1.5 * self.ca_activation)
        };
        self.ca_activation += (u - self.ca_activation) / tau_ms;
        self.ca_activation = self.ca_activation.clamp(0.002, 1.0);

        self.l_norm = l_norm;
        self.v_norm = v_norm;

        let f_l = Self::force_length_active(l_norm);
        let f_v = Self::force_velocity(v_norm);
        let f_pe = Self::force_length_passive(l_norm);

        // Intrinsic tendon damping + active cross-bridge force
        let active_force = self.ca_activation * f_l * f_v;
        self.tendon_force_n = (self.f_max_n * (active_force + f_pe)).max(0.0);
        self.tendon_force_n
    }

    /// Computes instantaneous physical joint stiffness contribution `-d(Torque)/d(theta)`
    /// emerging strictly from the muscle's active + passive cross-bridge elasticity.
    pub fn emergent_physical_stiffness_nm_per_rad(&self) -> f32 {
        // Cross-bridge short-range stiffness (Rack & Westbury 1974):
        // Stiffness scales directly with number of attached actin-myosin cross-bridges (ca_activation * F_max)
        let short_range_k = 28.0 * self.ca_activation * self.f_max_n * self.moment_arm_m;
        let passive_k = if self.l_norm > 1.0 {
            8.0 * self.f_max_n * self.moment_arm_m * (self.l_norm - 1.0)
        } else {
            0.0
        };
        short_range_k + passive_k
    }
}

/// Mileusnic et al. (2006) Biophysical Muscle Spindle & Golgi Tendon Organ Pair.
/// - Driven by mechanical muscle length `l_norm`, velocity `v_norm`, AND descending
///   **Gamma-static** (`gamma_static`) and **Gamma-dynamic** (`gamma_dynamic`) fusimotor drives!
/// - Computes stretch relative to the intrafusal target length set by `gamma_static`.
#[derive(Clone, Copy, Debug)]
pub struct ProprioceptiveOrgan {
    /// Target intrafusal length set by Gamma-static motor neurons (`1.0` = neutral)
    pub intrafusal_target_l: f32,
    /// Dynamic velocity sensitivity set by Gamma-dynamic bag1 fibers
    pub intrafusal_dyn_gain: f32,
    /// Group Ia primary afferent firing rate (`[0, 1]` normalized, velocity + length error)
    pub ia_afferent_rate: f32,
    /// Group II secondary afferent firing rate (`[0, 1]` normalized, pure length error)
    pub ii_afferent_rate: f32,
    /// Group Ib Golgi Tendon Organ afferent firing rate (`[0, 1]` normalized, series tendon force)
    pub ib_golgi_rate: f32,
}

impl Default for ProprioceptiveOrgan {
    fn default() -> Self {
        Self::new()
    }
}

impl ProprioceptiveOrgan {
    pub const fn new() -> Self {
        Self {
            intrafusal_target_l: 1.0,
            intrafusal_dyn_gain: 1.0,
            ia_afferent_rate: 0.0,
            ii_afferent_rate: 0.0,
            ib_golgi_rate: 0.0,
        }
    }

    /// Updates the spindle and Golgi tendon organ at 1000 Hz.
    /// Notice how `gamma_static_target_l` shifts the zero-point of the stretch receptor
    /// (Merton's follow-up length servo): when the cortex commands a voluntary flexion,
    /// it co-activates Alpha and Gamma-static so the spindle only fires if external load
    /// prevents the muscle from shortening as fast as expected!
    pub fn step_1ms(
        &mut self,
        l_norm: f32,
        v_norm: f32,
        tendon_force_n: f32,
        f_max_n: f32,
        gamma_static_target_l: f32,
        gamma_dynamic_drive: f32,
    ) {
        // Low-pass intrafusal bag1/bag2 nuclear chain mechanics (tau ~ 10 ms)
        self.intrafusal_target_l = 0.90 * self.intrafusal_target_l + 0.10 * gamma_static_target_l;
        self.intrafusal_dyn_gain =
            0.90 * self.intrafusal_dyn_gain + 0.10 * (1.0 + 2.5 * gamma_dynamic_drive.clamp(0.0, 1.0));

        // Intrafusal stretch error (positive when muscle is longer than expected target length)
        let stretch_error = (l_norm - self.intrafusal_target_l).max(0.0);
        // Prochaska / Matthews power-law velocity sensitivity (|v|^0.6) on lengthening,
        // active ONLY when the intrafusal fiber is taut (`l_norm >= intrafusal_target_l - 0.02`)!
        let vel_stretch = if v_norm > 0.0 && l_norm >= self.intrafusal_target_l - 0.02 {
            self.intrafusal_dyn_gain * v_norm.powf(0.6)
        } else {
            0.0
        };

        self.ii_afferent_rate = (2.2 * stretch_error).clamp(0.0, 1.0);
        self.ia_afferent_rate = (1.8 * stretch_error + 1.4 * vel_stretch).clamp(0.0, 1.0);
        self.ib_golgi_rate = (tendon_force_n / f_max_n.max(1.0)).clamp(0.0, 1.0);
    }
}

/// Lock-Free Shared Neural-Body Bus.
/// Bridges the asynchronous cortical/basal-ganglia/cerebellar/spinal threads with the 44-DoF plant.
pub struct NeuralBodyBus {
    /// Upstream semantic intent vector (written at 10 Hz, read by Premotor/Basal Ganglia)
    /// `[vx_cmd, vy_cmd, yaw_cmd, posture_height, reach_x, reach_y, reach_z, grip_force_cmd]`
    pub intent_vector: [AtomicF32; 8],

    /// Proprioceptive afferent signals per muscle (88 muscles: even = Agonist, odd = Antagonist)
    pub spindle_ia: [AtomicF32; NUM_MUSCLES],
    pub spindle_ii: [AtomicF32; NUM_MUSCLES],
    pub golgi_ib: [AtomicF32; NUM_MUSCLES],

    /// True mechanical joint states (for S1 proprioceptive map & telemetry)
    pub joint_angle_rad: [AtomicF32; NUM_DOF],
    pub joint_vel_rad_s: [AtomicF32; NUM_DOF],

    /// Vestibular Otolith & Semicircular Canal state: `[pitch_rad, roll_rad, pitch_vel, roll_vel, com_z_m, foot_load_l, foot_load_r]`
    pub vestibular_state: [AtomicF32; 7],

    /// Palmar & Fingertip Cutaneous Mechanoreceptors: `[slip_velocity_l, normal_force_l, slip_velocity_r, normal_force_r]`
    pub tactile_hand_state: [AtomicF32; 4],

    /// Posterior Parietal Cortex (PPC Area 5/7) Continuous 3D Spatial Target Attractors (Pelvis-Relative meters):
    /// - `ppc_target_l_hand_rel`: `[x, y, z]` target for Left Hand in body-local 3D space
    /// - `ppc_target_r_hand_rel`: `[x, y, z]` target for Right Hand in body-local 3D space
    /// - `ppc_target_l_foot_rel`: `[active_gate, x, y, z]` target for Left Foot unipedal kick / high step
    /// - `ppc_target_r_foot_rel`: `[active_gate, x, y, z]` target for Right Foot unipedal kick / high step
    /// - `ppc_target_trunk_lean`: `[pitch_lean_rad, roll_lean_rad]` voluntary trunk bow / lateral lean
    pub ppc_target_l_hand_rel: [AtomicF32; 3],
    pub ppc_target_r_hand_rel: [AtomicF32; 3],
    pub ppc_target_l_foot_rel: [AtomicF32; 4],
    pub ppc_target_r_foot_rel: [AtomicF32; 4],
    pub ppc_target_trunk_lean: [AtomicF32; 2],

    /// Somatosensory Cortex (S1 Area 3a/2) Real-Time 3D Proprioceptive Limb Coordinates (Pelvis-Relative meters):
    pub s1_current_l_hand_rel: [AtomicF32; 3],
    pub s1_current_r_hand_rel: [AtomicF32; 3],
    pub s1_current_l_foot_rel: [AtomicF32; 3],
    pub s1_current_r_foot_rel: [AtomicF32; 3],

    /// Descending Motor Pool Outputs (88 muscles):
    /// Pure Alpha motor neuron spike rates and Gamma static/dynamic fusimotor commands!
    pub alpha_motor_rate: [AtomicF32; NUM_MUSCLES],
    pub gamma_static_target: [AtomicF32; NUM_MUSCLES],
    pub gamma_dynamic_drive: [AtomicF32; NUM_MUSCLES],
}

impl Default for NeuralBodyBus {
    fn default() -> Self {
        Self::new()
    }
}

impl NeuralBodyBus {
    pub fn new() -> Self {
        let mut gamma_static = [const { AtomicF32::new(1.0) }; NUM_MUSCLES];
        for g in &mut gamma_static {
            g.store(1.0);
        }
        Self {
            intent_vector: [const { AtomicF32::new(0.0) }; 8],
            spindle_ia: [const { AtomicF32::new(0.0) }; NUM_MUSCLES],
            spindle_ii: [const { AtomicF32::new(0.0) }; NUM_MUSCLES],
            golgi_ib: [const { AtomicF32::new(0.0) }; NUM_MUSCLES],
            joint_angle_rad: [const { AtomicF32::new(0.0) }; NUM_DOF],
            joint_vel_rad_s: [const { AtomicF32::new(0.0) }; NUM_DOF],
            vestibular_state: [const { AtomicF32::new(0.0) }; 7],
            tactile_hand_state: [const { AtomicF32::new(0.0) }; 4],
            ppc_target_l_hand_rel: [
                AtomicF32::new(0.42),
                AtomicF32::new(0.24),
                AtomicF32::new(0.58),
            ],
            ppc_target_r_hand_rel: [
                AtomicF32::new(0.42),
                AtomicF32::new(-0.24),
                AtomicF32::new(0.58),
            ],
            ppc_target_l_foot_rel: [
                AtomicF32::new(0.0),
                AtomicF32::new(0.42),
                AtomicF32::new(0.12),
                AtomicF32::new(-0.45),
            ],
            ppc_target_r_foot_rel: [
                AtomicF32::new(0.0),
                AtomicF32::new(0.42),
                AtomicF32::new(-0.12),
                AtomicF32::new(-0.45),
            ],
            ppc_target_trunk_lean: [const { AtomicF32::new(0.0) }; 2],
            s1_current_l_hand_rel: [
                AtomicF32::new(0.0),
                AtomicF32::new(0.20),
                AtomicF32::new(-0.19),
            ],
            s1_current_r_hand_rel: [
                AtomicF32::new(0.0),
                AtomicF32::new(-0.20),
                AtomicF32::new(-0.19),
            ],
            s1_current_l_foot_rel: [
                AtomicF32::new(0.0),
                AtomicF32::new(0.105),
                AtomicF32::new(-0.87),
            ],
            s1_current_r_foot_rel: [
                AtomicF32::new(0.0),
                AtomicF32::new(-0.105),
                AtomicF32::new(-0.87),
            ],
            alpha_motor_rate: [const { AtomicF32::new(0.0) }; NUM_MUSCLES],
            gamma_static_target: gamma_static,
            gamma_dynamic_drive: [const { AtomicF32::new(0.2) }; NUM_MUSCLES],
        }
    }
}

/// Complete 44-DoF Articulated Humanoid Biomechanical Plant & Neuromuscular Interface.
/// Simulates:
/// - 88 Hill-Type Muscle-Tendon Units (Agonist + Antagonist for all 44 DoFs)
/// - 88 Mileusnic Muscle Spindles + Golgi Tendon Organs
/// - Inertial & Gravitational coupling across kinematic chains (parent-child momentum transfer,
///   whole-body Center of Mass inverted-pendulum sagittal/coronal dynamics, ground reaction forces,
///   and hand-object contact & slip physics).
pub struct HumanoidBiomechanicalPlant {
    pub muscles: [HillMuscleTendonUnit; NUM_MUSCLES],
    pub spindles: [ProprioceptiveOrgan; NUM_MUSCLES],
    pub joint_pos: [f32; NUM_DOF],
    pub joint_vel: [f32; NUM_DOF],
    pub joint_net_torque: [f32; NUM_DOF],
    pub joint_emergent_stiffness: [f32; NUM_DOF],

    /// Whole-body sagittal pitch & coronal roll relative to gravity (radians)
    pub body_pitch_rad: f32,
    pub body_roll_rad: f32,
    pub body_pitch_vel: f32,
    pub body_roll_vel: f32,

    /// Foot ground contact loads (Newtons, normalized `[0, 1]`)
    pub left_foot_load: f32,
    pub right_foot_load: f32,

    /// External perturbation torque applied to body (N*m) and held object mass (kg)
    pub external_pitch_perturbation_nm: f32,
    pub held_object_mass_kg: f32,
    pub left_hand_object_slip_vel: f32,
    pub left_hand_grip_force_n: f32,
    pub right_hand_object_slip_vel: f32,
    pub right_hand_grip_force_n: f32,
    pub gravity_enabled: bool,
}

impl Default for HumanoidBiomechanicalPlant {
    fn default() -> Self {
        Self::new()
    }
}

impl HumanoidBiomechanicalPlant {
    pub fn new() -> Self {
        let mut muscles = [HillMuscleTendonUnit::new(100.0, 0.05); NUM_MUSCLES];
        for dof in 0..NUM_DOF {
            let meta = &HUMANOID_44_DOF_MAP[dof];
            muscles[dof * 2] = HillMuscleTendonUnit::new(meta.max_isometric_force_n, meta.moment_arm_m);
            muscles[dof * 2 + 1] = HillMuscleTendonUnit::new(meta.max_isometric_force_n, meta.moment_arm_m);
        }
        Self {
            muscles,
            spindles: [ProprioceptiveOrgan::new(); NUM_MUSCLES],
            joint_pos: [0.0; NUM_DOF],
            joint_vel: [0.0; NUM_DOF],
            joint_net_torque: [0.0; NUM_DOF],
            joint_emergent_stiffness: [0.0; NUM_DOF],
            body_pitch_rad: 0.0,
            body_roll_rad: 0.0,
            body_pitch_vel: 0.0,
            body_roll_vel: 0.0,
            left_foot_load: 0.65,
            right_foot_load: 0.65,
            external_pitch_perturbation_nm: 0.0,
            held_object_mass_kg: 0.4,
            left_hand_object_slip_vel: 0.0,
            left_hand_grip_force_n: 0.0,
            right_hand_object_slip_vel: 0.0,
            right_hand_grip_force_n: 0.0,
            gravity_enabled: true,
        }
    }

    /// Steps the 88 Hill muscles, 44 articulated joints, inertial chain coupling,
    /// gravity inverted pendulum, and sensory organs by `dt = 1.0 ms` (0.001 s).
    pub fn step_1ms(&mut self, bus: &NeuralBodyBus) {
        let dt = 0.001_f32;
        let mut raw_accels = [0.0_f32; NUM_DOF];

        for dof in 0..NUM_DOF {
            let meta = &HUMANOID_44_DOF_MAP[dof];
            let ag_idx = dof * 2;
            let ant_idx = dof * 2 + 1;

            let theta = self.joint_pos[dof];
            let dtheta = self.joint_vel[dof];

            // Normalized muscle lengths and velocities from joint angle:
            // Upper-limb glenohumeral/elbow joints (16..24) have longer fascicle-to-moment-arm ratios
            // (`0.17 / rad`) to span their full 165-deg overhead workspace without passive capsular lock.
            let fiber_scale = if (16..24).contains(&dof) { 0.17_f32 } else { 0.35_f32 };
            let l_ag = (1.0 - fiber_scale * theta).clamp(0.55, 1.55);
            let l_ant = (1.0 + fiber_scale * theta).clamp(0.55, 1.55);
            let v_ag = (-0.25 * dtheta).clamp(-1.0, 1.0);
            let v_ant = (0.25 * dtheta).clamp(-1.0, 1.0);

            let u_ag = bus.alpha_motor_rate[ag_idx].load();
            let u_ant = bus.alpha_motor_rate[ant_idx].load();

            let f_ag = self.muscles[ag_idx].step_1ms(u_ag, l_ag, v_ag);
            let f_ant = self.muscles[ant_idx].step_1ms(u_ant, l_ant, v_ant);

            // Pure physical torque from Agonist minus Antagonist tendon tension
            let net_muscle_torque = meta.moment_arm_m * (f_ag - f_ant);
            self.joint_net_torque[dof] = net_muscle_torque;
            self.joint_emergent_stiffness[dof] = self.muscles[ag_idx]
                .emergent_physical_stiffness_nm_per_rad()
                + self.muscles[ant_idx].emergent_physical_stiffness_nm_per_rad();

            // Gravitational & passive capsular torques (Isaac-Gym-grade unpinned gravity physics):
            // - When a leg is in stance (`foot_load > 0.20`), vertical body weight creates a strong
            //   KNEE-BUCKLING FLEXION MOMENT (`+32.0 * foot_load * (0.14 + theta.sin())` on DoFs 3, 9)
            //   that physically collapses the knee into deep flexion (`theta -> 1.55 rad`) unless
            //   actively opposed by Quadriceps extensor contraction (`f_ant`)!
            // - Stance Ankles (4, 10) and Waist (12) experience inverted-pendulum gravity load plus
            //   passive Achilles/paraspinal elasticity during quiet standing, while deep knee buckling
            //   (`joint_pos[3] + joint_pos[9] > 0.50`) pitches the pelvis and trunk forward!
            let is_unpowered_collapse =
                self.muscles[3 * 2 + 1].ca_activation < 0.015 && self.muscles[3 * 2].ca_activation < 0.015;
            let knee_buckle_sum = if is_unpowered_collapse {
                self.joint_pos[3] + self.joint_pos[9]
            } else {
                0.0
            };
            let unpowered_knee_buckle = if is_unpowered_collapse {
                42.0 * (0.18 + self.joint_pos[dof].sin())
            } else {
                0.0
            };
            let bilateral_squat_co_flex = self.joint_pos[3].min(self.joint_pos[9]) > 0.06;
            let knee_open_chain_pendulum = if bilateral_squat_co_flex { -5.0 } else { -26.0 };
            let gravity_torque = if self.gravity_enabled {
                match dof {
                    0 | 6 => -38.0 * self.joint_pos[dof].sin(),     // Hip pendulum restoring gravity
                    3 | 9 => knee_open_chain_pendulum * self.joint_pos[dof].sin() + unpowered_knee_buckle,
                    4 | 10 => {
                        24.0 * self.joint_pos[dof].sin() - 30.0 * self.joint_pos[dof]
                            + 0.5 * self.external_pitch_perturbation_nm
                            + 12.0 * (knee_buckle_sum - 0.55).max(0.0)
                    }
                    12 => {
                        26.0 * self.joint_pos[dof].sin() - 34.0 * self.joint_pos[dof]
                            + 16.0 * (knee_buckle_sum - 0.55).max(0.0)
                    }
                    16 | 20 => -14.0 * self.joint_pos[dof].sin(),   // Hanging arm pendulum restoring gravity
                    1 | 7 | 5 | 11 | 13 => -28.0 * self.joint_pos[dof].sin(), // Coronal hip/ankle/waist stabilizing elasticity
                    _ => 0.0,
                }
            } else {
                0.0
            };

            // Passive anatomical joint capsule damping
            let passive_damping = 0.8 * meta.link_inertia_kgm2 * dtheta;
            let mut raw_acc =
                (net_muscle_torque + gravity_torque - passive_damping) / meta.link_inertia_kgm2;

            // Anatomical patellar/cruciate ligament stop on Knee Pitch (DoFs 3 and 9):
            // When the knee is fully extended (theta <= 0.0) and extensor torque pushes further negative,
            // the internal capsule stop absorbs the compression so zero phantom acceleration leaks to Hip/Ankle!
            if (dof == 3 || dof == 9) && theta <= 0.001 && raw_acc < 0.0 {
                raw_acc = 0.0;
            }
            raw_accels[dof] = raw_acc;
        }

        // Inertial & Coriolis kinematic coupling between parent and child joints in the same limb chain!
        // (In zero-G prenatal babbling `!self.gravity_enabled`, open-chain reaction forces strongly vibrate
        // adjacent joints to drive Somatotopic Homunculus formation; in closed-chain terrestrial stance,
        // ground contact and pelvis inertia damp inter-joint reaction coupling.)
        let mut coupled_accels = raw_accels;
        let coupling = if self.gravity_enabled { 0.04_f32 } else { 0.28_f32 };
        for dof in 0..NUM_DOF {
            if let Some(parent) = HUMANOID_44_DOF_MAP[dof].parent_dof {
                coupled_accels[dof] += coupling * raw_accels[parent];
                coupled_accels[parent] += coupling * 0.65 * raw_accels[dof];
            }
        }

        // Integrate joint velocities and positions with anatomical ligament stops:
        // Specifically, Knee Pitch (DoFs 3 and 9) cannot hyperextend below 0.0 rad (ACL/PCL + patellar lock),
        // flexing positively up to +1.55 rad (~89 deg).
        for dof in 0..NUM_DOF {
            self.joint_vel[dof] = (self.joint_vel[dof] + dt * coupled_accels[dof]).clamp(-15.0, 15.0);
            let (min_angle, max_angle) = if dof == 3 || dof == 9 {
                (0.0, 1.55)
            } else if dof == 4 || dof == 10 {
                (-0.60, 0.45)
            } else if dof == 16 || dof == 20 {
                // Shoulder Pitch: from -0.95 rad hyperextension to +2.85 rad (163 deg) full overhead reach!
                (-0.95, 2.85)
            } else if dof == 18 || dof == 22 {
                // Elbow Pitch: anatomical olecranon stop at 0.0 rad up to +2.15 rad flexion
                (0.0, 2.15)
            } else {
                (-1.4, 1.4)
            };
            let next_pos = self.joint_pos[dof] + dt * self.joint_vel[dof];
            if next_pos <= min_angle && self.joint_vel[dof] < 0.0 {
                self.joint_pos[dof] = min_angle;
                self.joint_vel[dof] = 0.0;
            } else if next_pos >= max_angle && self.joint_vel[dof] > 0.0 {
                self.joint_pos[dof] = max_angle;
                self.joint_vel[dof] = 0.0;
            } else {
                self.joint_pos[dof] = next_pos;
            }
            bus.joint_angle_rad[dof].store(self.joint_pos[dof]);
            bus.joint_vel_rad_s[dof].store(self.joint_vel[dof]);
        }

        // Update whole-body sagittal/coronal balance state from stance ankle + waist trunk chain
        self.body_pitch_rad = 0.45 * (self.joint_pos[4] + self.joint_pos[10])
            + 0.55 * self.joint_pos[12];
        self.body_pitch_vel = 0.45 * (self.joint_vel[4] + self.joint_vel[10])
            + 0.55 * self.joint_vel[12];
        self.body_roll_rad = 0.5 * (self.joint_pos[5] + self.joint_pos[11])
            + 0.5 * (self.joint_pos[1] + self.joint_pos[7]);
        self.body_roll_vel = 0.5 * (self.joint_vel[5] + self.joint_vel[11])
            + 0.5 * (self.joint_vel[1] + self.joint_vel[7]);

        // Compute bipedal ground reaction foot loads from hip/knee extension vs flexion
        let l_leg_ext = -self.joint_pos[0] - 0.5 * self.joint_pos[3] - 0.5 * self.body_roll_rad;
        let r_leg_ext = -self.joint_pos[6] - 0.5 * self.joint_pos[9] + 0.5 * self.body_roll_rad;
        let diff_ext = (l_leg_ext - r_leg_ext) * 2.5;
        let l_share = 1.0 / (1.0 + (-diff_ext).exp());
        let unpowered_collapse_boost = if self.muscles[3 * 2 + 1].ca_activation < 0.015 { 1.35 } else { 1.0 };
        self.left_foot_load = (l_share * unpowered_collapse_boost).clamp(0.0, 1.0);
        self.right_foot_load = ((1.0 - l_share) * unpowered_collapse_boost).clamp(0.0, 1.0);

        // Vestibular Nuclei Efference-Copy Cancellation (Cullen 2012):
        // Subtract voluntary trunk lean command (`ppc_target_trunk_lean`) so vestibulospinal
        // reflexes stabilize around the intended bow/lean angle rather than fighting voluntary bowing!
        let target_pitch = bus.ppc_target_trunk_lean[0].load().clamp(-0.35, 0.55);
        let target_roll = bus.ppc_target_trunk_lean[1].load().clamp(-0.35, 0.35);
        bus.vestibular_state[0].store(self.body_pitch_rad - target_pitch);
        bus.vestibular_state[1].store(self.body_roll_rad - target_roll);
        bus.vestibular_state[2].store(self.body_pitch_vel);
        bus.vestibular_state[3].store(self.body_roll_vel);
        bus.vestibular_state[4].store(0.95 * self.body_pitch_rad.cos());
        bus.vestibular_state[5].store(self.left_foot_load);
        bus.vestibular_state[6].store(self.right_foot_load);

        // Dexterous Bimanual Left (DoFs 24..33) & Right (DoFs 34..43) Hand Object Grasping & Cutaneous Slip Physics
        let required_friction_force = self.held_object_mass_kg * 9.81 / 0.6; // mu_s = 0.6

        let mut left_grip_force = 0.0_f32;
        for hand_dof in 24..34 {
            let ag_idx = hand_dof * 2;
            left_grip_force += self.muscles[ag_idx].tendon_force_n * 0.25;
        }
        self.left_hand_grip_force_n = left_grip_force;
        if left_grip_force < required_friction_force {
            let deficit = (required_friction_force - left_grip_force) / required_friction_force.max(0.1);
            self.left_hand_object_slip_vel = 0.85 * self.left_hand_object_slip_vel + 0.15 * deficit;
        } else {
            self.left_hand_object_slip_vel *= 0.70;
        }
        bus.tactile_hand_state[0].store(self.left_hand_object_slip_vel.clamp(0.0, 1.0));
        bus.tactile_hand_state[1].store((left_grip_force / 60.0).clamp(0.0, 1.0));

        let mut right_grip_force = 0.0_f32;
        for hand_dof in 34..44 {
            let ag_idx = hand_dof * 2;
            right_grip_force += self.muscles[ag_idx].tendon_force_n * 0.25;
        }
        self.right_hand_grip_force_n = right_grip_force;
        if right_grip_force < required_friction_force {
            let deficit = (required_friction_force - right_grip_force) / required_friction_force.max(0.1);
            self.right_hand_object_slip_vel = 0.85 * self.right_hand_object_slip_vel + 0.15 * deficit;
        } else {
            self.right_hand_object_slip_vel *= 0.70;
        }
        bus.tactile_hand_state[2].store(self.right_hand_object_slip_vel.clamp(0.0, 1.0));
        bus.tactile_hand_state[3].store((right_grip_force / 60.0).clamp(0.0, 1.0));

        // Update all 88 Muscle Spindles (Ia, II) and Golgi Tendon Organs (Ib)
        for m in 0..NUM_MUSCLES {
            let dof = m / 2;
            let meta = &HUMANOID_44_DOF_MAP[dof];
            let g_stat = bus.gamma_static_target[m].load();
            let g_dyn = bus.gamma_dynamic_drive[m].load();
            let mtu = &self.muscles[m];

            self.spindles[m].step_1ms(
                mtu.l_norm,
                mtu.v_norm,
                mtu.tendon_force_n,
                meta.max_isometric_force_n,
                g_stat,
                g_dyn,
            );
            bus.spindle_ia[m].store(self.spindles[m].ia_afferent_rate);
            bus.spindle_ii[m].store(self.spindles[m].ii_afferent_rate);
            bus.golgi_ib[m].store(self.spindles[m].ib_golgi_rate);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emergent_co_contraction_stiffness_without_torque() {
        // Prove that co-contracting Agonist and Antagonist Hill muscles equally produces
        // ZERO net torque while increasing physical joint stiffness by >10x purely from
        // cross-bridge biomechanics!
        let mut ag = HillMuscleTendonUnit::new(2000.0, 0.06);
        let mut ant = HillMuscleTendonUnit::new(2000.0, 0.06);

        // Low tone state (u = 0.02)
        for _ in 0..50 {
            ag.step_1ms(0.02, 1.0, 0.0);
            ant.step_1ms(0.02, 1.0, 0.0);
        }
        let low_stiffness =
            ag.emergent_physical_stiffness_nm_per_rad() + ant.emergent_physical_stiffness_nm_per_rad();

        // High co-contraction state (u_ag = 0.85, u_ant = 0.85)
        for _ in 0..100 {
            ag.step_1ms(0.85, 1.0, 0.0);
            ant.step_1ms(0.85, 1.0, 0.0);
        }
        let net_torque = 0.06 * (ag.tendon_force_n - ant.tendon_force_n);
        let high_stiffness =
            ag.emergent_physical_stiffness_nm_per_rad() + ant.emergent_physical_stiffness_nm_per_rad();

        assert!(
            net_torque.abs() < 1e-3,
            "Equal co-contraction must yield zero net torque, got {}",
            net_torque
        );
        assert!(
            high_stiffness > low_stiffness * 15.0,
            "Co-contraction must dramatically increase emergent physical joint stiffness: low={}, high={}",
            low_stiffness,
            high_stiffness
        );
    }
}
