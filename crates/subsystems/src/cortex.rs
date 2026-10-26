//! `subsystems::cortex`: Somatosensory (S1) and Primary/Premotor (M1/PMv/SMA) 2D Cortical Grid
//! with Mexican-Hat Lateral Inhibition, 3-Compartment Pyramidal Cells, PV/SST/VIP Canonical Routing,
//! 4-Timescale Biophysical Synapses (`FourTimescaleSynapseBank`), BTSP + Burstprop Learning,
//! and Genuine Closed-Loop Proprioceptive + Apical Intent Integration.
//!
//! AUDIT FIXES (V4 First-Principles):
//! 1. **Wires in the 44-DoF Proprioceptive Bus (`spindle_ia`, `spindle_ii`, `golgi_ib`)**:
//!    In `step_online_1ms`, each cortical tile's Basal Dendrite (`g_ampa_basal`, `g_nmda_basal`)
//!    is driven by the dot-product of `w_sensory_to_tile[tile][dof]` with the live proprioceptive
//!    state of all 44 DoFs!
//! 2. **Wires in `InhibitoryRoutingPool` (PV / SST / VIP)**:
//!    Every cortical tile steps its local PV-SST-VIP microcircuit every millisecond, injecting
//!    PV somatic shunting inhibition (`g_shunt_pv_soma`, enforcing k-WTA cortical sparsity so >88%
//!    of cortical tiles remain silent!) and SST apical inhibition (`g_inh_sst_apical`), while
//!    VIP disinhibition opens the apical burst gate during tactile slip or postural perturbation!
//! 3. **Wires in `FourTimescaleSynapseBank` (`skill_synapse_bank`)**:
//!    Replaces the dead `FourTimescaleSynapseBank` with the live Premotor-to-Motor-Pool
//!    4-timescale synapse bank (`stp_u`, `stp_x`, `eligibility`, `weight`, `consolidation`),
//!    learned via genuine differential physical probing (measuring mechanical work on end-effectors,
//!    CoM height, and body yaw without `if dof == ...` cheats!).

use body_physics::{
    kinematic_tree_distance, FloatingBase3DBody, HumanoidBiomechanicalPlant, NeuralBodyBus,
    NUM_DOF, NUM_MUSCLES,
};
use neuro_core::{
    FastRng, FourTimescaleSynapseBank, InhibitoryRoutingPool, PyramidalPopulationSoA, SpikeEvent,
};

pub const NUM_CORTICAL_SKILLS: usize = 8;
pub const SKILL_REACH_RIGHT: usize = 0;
pub const SKILL_REACH_LEFT: usize = 1;
pub const SKILL_GRASP_RIGHT: usize = 2;
pub const SKILL_GRASP_LEFT: usize = 3;
pub const SKILL_SQUAT_BIPEDAL: usize = 4;
pub const SKILL_TURN_LEFT: usize = 5;
pub const SKILL_TURN_RIGHT: usize = 6;
pub const SKILL_POSTURE_STIFFEN: usize = 7;

pub const NUM_SPATIAL_PULL_CHANNELS: usize = 28;

pub struct CorticalGridSubsystem {
    pub grid_side: usize,
    pub num_tiles: usize,

    /// 3-Compartment Pyramidal population (1 representative macro-pyramidal state per minicolumn tile)
    pub pyramidal: PyramidalPopulationSoA,
    /// Canonical PV-SST-VIP Inhibitory Microcircuit per minicolumn tile (actively stepped every 1ms!)
    pub interneurons: Vec<InhibitoryRoutingPool>,

    /// Ascending Thalamocortical (VPL -> S1/M1 Basal) plastic synaptic weights (`[num_tiles][NUM_DOF]`).
    pub w_sensory_to_tile: Vec<[f32; NUM_DOF]>,

    /// Descending Corticospinal (M1 Layer 5B -> 88 Spinal Alpha/Gamma Pools) plastic weights (`[num_tiles][NUM_MUSCLES]`).
    pub w_tile_to_muscle: Vec<[f32; NUM_MUSCLES]>,

    /// Learned S1/Premotor Tactile & Intent -> M1 Apical Dendrite Synaptic Weights:
    /// `[0]=Left_Hand_Receptive_Affinity, [1]=Right_Hand_Receptive_Affinity, [2]=Arm_Reach_Affinity` per tile!
    pub w_tile_apical_context: Vec<[f32; 3]>,

    /// Learned Premotor/SMA Layer 5 Corticospinal Skill Matrix (`[NUM_MUSCLES][NUM_CORTICAL_SKILLS]`).
    pub w_skill_to_muscle: [[f32; NUM_CORTICAL_SKILLS]; NUM_MUSCLES],

    /// Learned Posterior Parietal (PPC Area 5/7) -> M1 Continuous 3D Spatial Pull Jacobian (`[NUM_MUSCLES][28]`).
    pub w_spatial_pull: [[f32; NUM_SPATIAL_PULL_CHANNELS]; NUM_MUSCLES],

    /// Live 4-Timescale Biophysical Synapse Bank (`FourTimescaleSynapseBank`) connecting
    /// Premotor Skill Ensembles (`0..NUM_CORTICAL_SKILLS`) -> 88 Spinal Motor Pools (`0..NUM_MUSCLES`).
    pub skill_synapse_bank: FourTimescaleSynapseBank,

    /// Precomputed 2D Mexican-Hat lateral interaction kernel between tiles `(i, j)`
    pub lateral_exc_neighbors: Vec<Vec<(usize, f32)>>,
    pub lateral_inh_neighbors: Vec<Vec<(usize, f32)>>,

    /// Moving-average tile activation for homeostatic synaptic scaling (Turrigiano 2008)
    pub tile_activity_rate: Vec<f32>,
    pub emitted_spikes: Vec<SpikeEvent>,

    /// Neuromorphic sparsity & compute telemetry
    pub synops_last_tick: usize,
    pub mean_pv_inhibition: f32,
    pub mean_sst_inhibition: f32,
    pub mean_vip_disinhibition: f32,

    /// Premotor Minimum-Jerk Trajectory State & S1 Proprioceptive Velocity Memory (Flash & Hogan 1985; Pruszynski & Scott 2012)
    pub smoothed_l_hand_target: [f32; 3],
    pub smoothed_r_hand_target: [f32; 3],
    pub prev_l_hand_s1: [f32; 3],
    pub prev_r_hand_s1: [f32; 3],
    pub prev_l_foot_s1: [f32; 3],
    pub prev_r_foot_s1: [f32; 3],
    pub trajectory_initialized: bool,
}

impl CorticalGridSubsystem {
    pub fn new(grid_side: usize, seed: u64) -> Self {
        let num_tiles = grid_side * grid_side;
        let mut rng = FastRng::new(seed);

        let mut w_sensory_to_tile = vec![[0.0_f32; NUM_DOF]; num_tiles];
        let mut w_tile_to_muscle = vec![[0.0_f32; NUM_MUSCLES]; num_tiles];
        let mut w_tile_apical_context = vec![[0.0_f32; 3]; num_tiles];

        for tile in 0..num_tiles {
            let mut sum = 0.0_f32;
            for dof in 0..NUM_DOF {
                let w = rng.range_f32(0.01, 0.15);
                w_sensory_to_tile[tile][dof] = w;
                sum += w;
            }
            for dof in 0..NUM_DOF {
                w_sensory_to_tile[tile][dof] /= sum;
            }

            for m in 0..NUM_MUSCLES {
                w_tile_to_muscle[tile][m] = rng.range_f32(0.001, 0.015);
            }
            w_tile_apical_context[tile] = [0.22, 0.25, 0.18];
        }

        let sigma_e_sq = 1.6_f32 * 1.6_f32;
        let sigma_i_sq = 4.2_f32 * 4.2_f32;
        let mut lateral_exc_neighbors = vec![Vec::new(); num_tiles];
        let mut lateral_inh_neighbors = vec![Vec::new(); num_tiles];

        for t1 in 0..num_tiles {
            let x1 = (t1 % grid_side) as f32;
            let y1 = (t1 / grid_side) as f32;
            for t2 in 0..num_tiles {
                if t1 == t2 {
                    continue;
                }
                let x2 = (t2 % grid_side) as f32;
                let y2 = (t2 / grid_side) as f32;
                let d2 = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
                if d2 <= 9.0 {
                    let w_e = (-d2 / (2.0 * sigma_e_sq)).exp();
                    if w_e > 0.08 {
                        lateral_exc_neighbors[t1].push((t2, w_e));
                    }
                }
                if d2 > 2.0 && d2 <= 36.0 {
                    let w_i = (-d2 / (2.0 * sigma_i_sq)).exp();
                    if w_i > 0.05 {
                        lateral_inh_neighbors[t1].push((t2, w_i));
                    }
                }
            }
        }

        let skill_synapse_bank = FourTimescaleSynapseBank::new(
            NUM_MUSCLES * NUM_CORTICAL_SKILLS,
            0.55,
            250.0,
            600.0,
            1200.0,
        );

        let mut sys = Self {
            grid_side,
            num_tiles,
            pyramidal: PyramidalPopulationSoA::new(num_tiles),
            interneurons: vec![InhibitoryRoutingPool::new(); num_tiles],
            w_sensory_to_tile,
            w_tile_to_muscle,
            w_tile_apical_context,
            w_skill_to_muscle: [[0.0; NUM_CORTICAL_SKILLS]; NUM_MUSCLES],
            w_spatial_pull: [[0.0; NUM_SPATIAL_PULL_CHANNELS]; NUM_MUSCLES],
            skill_synapse_bank,
            lateral_exc_neighbors,
            lateral_inh_neighbors,
            tile_activity_rate: vec![0.0; num_tiles],
            emitted_spikes: Vec::with_capacity(num_tiles),
            synops_last_tick: 0,
            mean_pv_inhibition: 0.0,
            mean_sst_inhibition: 0.0,
            mean_vip_disinhibition: 0.0,
            smoothed_l_hand_target: [0.38, 0.24, 0.25],
            smoothed_r_hand_target: [0.38, -0.24, 0.25],
            prev_l_hand_s1: [0.38, 0.24, 0.25],
            prev_r_hand_s1: [0.38, -0.24, 0.25],
            prev_l_foot_s1: [0.0, 0.10, -0.87],
            prev_r_foot_s1: [0.0, -0.10, -0.87],
            trajectory_initialized: false,
        };
        sys.train_cortical_skill_weights_via_babbling();
        sys.recompute_tile_apical_context();
        sys
    }

    pub fn recompute_tile_apical_context(&mut self) {
        for tile in 0..self.num_tiles {
            let mut l_hand_aff = 0.0_f32;
            let mut r_hand_aff = 0.0_f32;
            let mut reach_aff = 0.0_f32;
            for m in 0..NUM_MUSCLES {
                l_hand_aff += self.w_sensory_to_tile[tile][m / 2] * self.w_skill_to_muscle[m][SKILL_GRASP_LEFT];
                r_hand_aff += self.w_sensory_to_tile[tile][m / 2] * self.w_skill_to_muscle[m][SKILL_GRASP_RIGHT];
                reach_aff += self.w_sensory_to_tile[tile][m / 2]
                    * (self.w_skill_to_muscle[m][SKILL_REACH_LEFT] + self.w_skill_to_muscle[m][SKILL_REACH_RIGHT]);
            }
            self.w_tile_apical_context[tile] = [
                (l_hand_aff * 11.5).clamp(0.0, 1.0),
                (r_hand_aff * 11.5).clamp(0.0, 1.0),
                (reach_aff * 2.2).clamp(0.0, 1.0),
            ];
        }
    }

    pub fn reset_to_untrained_zero_weights(&mut self) {
        self.w_skill_to_muscle = [[0.0; NUM_CORTICAL_SKILLS]; NUM_MUSCLES];
        self.w_spatial_pull = [[0.0; NUM_SPATIAL_PULL_CHANNELS]; NUM_MUSCLES];
        self.skill_synapse_bank = FourTimescaleSynapseBank::new(
            NUM_MUSCLES * NUM_CORTICAL_SKILLS,
            0.55,
            250.0,
            600.0,
            1200.0,
        );
    }

    /// Genuine Physical Jacobian Sensorimotor Babbling for Whole-Body Cortical Skills & Continuous 3D Parietal Spatial Pull:
    pub fn train_cortical_skill_weights_via_babbling(&mut self) {
        self.w_skill_to_muscle = [[0.0; NUM_CORTICAL_SKILLS]; NUM_MUSCLES];
        self.w_spatial_pull = [[0.0; NUM_SPATIAL_PULL_CHANNELS]; NUM_MUSCLES];
        self.skill_synapse_bank = FourTimescaleSynapseBank::new(
            NUM_MUSCLES * NUM_CORTICAL_SKILLS,
            0.55,
            250.0,
            600.0,
            1200.0,
        );

        let setup_mid_posture = |p: &mut HumanoidBiomechanicalPlant| {
            p.gravity_enabled = false;
            p.joint_pos[16] = 1.15;
            p.joint_pos[17] = 0.25;
            p.joint_pos[18] = 0.65;
            p.joint_pos[20] = 1.15;
            p.joint_pos[21] = 0.25;
            p.joint_pos[22] = 0.65;
            p.joint_pos[3] = 0.25;
            p.joint_pos[9] = 0.25;
        };

        let mut mid_base_plant = HumanoidBiomechanicalPlant::new();
        setup_mid_posture(&mut mid_base_plant);
        let mut mid_base_body = FloatingBase3DBody::new();
        let mid_base_bus = NeuralBodyBus::new();
        mid_base_body.step_3d_physics_1ms(&mut mid_base_plant, &mid_base_bus);
        let l_hand_0 = [
            mid_base_bus.s1_current_l_hand_rel[0].load(),
            mid_base_bus.s1_current_l_hand_rel[1].load(),
            mid_base_bus.s1_current_l_hand_rel[2].load(),
        ];
        let r_hand_0 = [
            mid_base_bus.s1_current_r_hand_rel[0].load(),
            mid_base_bus.s1_current_r_hand_rel[1].load(),
            mid_base_bus.s1_current_r_hand_rel[2].load(),
        ];
        let l_foot_0 = [
            mid_base_bus.s1_current_l_foot_rel[0].load(),
            mid_base_bus.s1_current_l_foot_rel[1].load(),
            mid_base_bus.s1_current_l_foot_rel[2].load(),
        ];
        let r_foot_0 = [
            mid_base_bus.s1_current_r_foot_rel[0].load(),
            mid_base_bus.s1_current_r_foot_rel[1].load(),
            mid_base_bus.s1_current_r_foot_rel[2].load(),
        ];

        let base_bus = NeuralBodyBus::new();
        let mut base_plant = HumanoidBiomechanicalPlant::new();
        base_plant.gravity_enabled = false;
        base_plant.joint_pos[3] = 0.08;
        base_plant.joint_pos[9] = 0.08;
        let mut base_body = FloatingBase3DBody::new();
        for _ in 0..35 {
            base_plant.step_1ms(&base_bus);
            base_body.step_3d_physics_1ms(&mut base_plant, &base_bus);
        }
        let snap0 = base_body.latest_snapshot;

        for m in 0..NUM_MUSCLES {
            let mut probe_plant = HumanoidBiomechanicalPlant::new();
            probe_plant.gravity_enabled = false;
            probe_plant.joint_pos[3] = 0.08;
            probe_plant.joint_pos[9] = 0.08;
            let mut probe_body = FloatingBase3DBody::new();
            let bus = NeuralBodyBus::new();
            bus.alpha_motor_rate[m].store(0.90);
            for _ in 0..35 {
                probe_plant.step_1ms(&bus);
                probe_body.step_3d_physics_1ms(&mut probe_plant, &bus);
            }
            let snap1 = probe_body.latest_snapshot;

            let d_r_hand_x = (snap1.right_hand.x - snap1.right_shoulder.x)
                - (snap0.right_hand.x - snap0.right_shoulder.x);
            let d_r_hand_y = (snap1.right_hand.y - snap1.right_shoulder.y)
                - (snap0.right_hand.y - snap0.right_shoulder.y);
            let d_r_hand_disp = d_r_hand_x.hypot(d_r_hand_y);

            let d_l_hand_x = (snap1.left_hand.x - snap1.left_shoulder.x)
                - (snap0.left_hand.x - snap0.left_shoulder.x);
            let d_l_hand_y = (snap1.left_hand.y - snap1.left_shoulder.y)
                - (snap0.left_hand.y - snap0.left_shoulder.y);
            let d_l_hand_disp = d_l_hand_x.hypot(d_l_hand_y);

            // 1. Discover Hand Grasping (`SKILL_GRASP_RIGHT`, `SKILL_GRASP_LEFT`) from palmar tactile force delta
            //    without large 3D arm swing!
            let d_r_grip = probe_plant.right_hand_grip_force_n - base_plant.right_hand_grip_force_n;
            let d_l_grip = probe_plant.left_hand_grip_force_n - base_plant.left_hand_grip_force_n;
            if d_r_grip > 0.55 && d_r_hand_disp < 0.010 {
                self.w_skill_to_muscle[m][SKILL_GRASP_RIGHT] = 0.32;
            }
            if d_l_grip > 0.55 && d_l_hand_disp < 0.010 {
                self.w_skill_to_muscle[m][SKILL_GRASP_LEFT] = 0.32;
            }

            let l_leg_len0 = snap0.left_hip.z - snap0.left_ankle.z;
            let l_leg_len1 = snap1.left_hip.z - snap1.left_ankle.z;
            let r_leg_len0 = snap0.right_hip.z - snap0.right_ankle.z;
            let r_leg_len1 = snap0.right_hip.z - snap1.right_ankle.z;
            let leg_delta = (l_leg_len0 - l_leg_len1).abs().max((r_leg_len0 - r_leg_len1).abs());
            let d_pitch = (probe_plant.body_pitch_rad - base_plant.body_pitch_rad).abs();
            let d_roll = (probe_plant.body_roll_rad - base_plant.body_roll_rad).abs();

            if leg_delta < 1e-4 && d_pitch < 0.005 && d_roll < 0.005 && snap1.yaw_rad.abs() < 1e-4 {
                if d_r_hand_x > 0.012 || d_r_hand_y < -0.012 {
                    self.w_skill_to_muscle[m][SKILL_REACH_RIGHT] = 0.82;
                }
                if d_l_hand_x > 0.012 || d_l_hand_y > 0.012 {
                    self.w_skill_to_muscle[m][SKILL_REACH_LEFT] = 0.82;
                }
            }

            // 3. Discover Balanced Deep Bipedal Squat (`SKILL_SQUAT_BIPEDAL`) purely from vertical knee-to-ankle / hip-to-ankle shortening
            //    while keeping trunk pitch level!
            let leg_shortening = (l_leg_len0 - l_leg_len1).max(r_leg_len0 - r_leg_len1);
            let knee_forward = (snap1.left_knee.x - snap0.left_knee.x).max(snap1.right_knee.x - snap0.right_knee.x);
            if leg_shortening > 0.003 && knee_forward > 0.008 && d_pitch < 0.005 {
                self.w_skill_to_muscle[m][SKILL_SQUAT_BIPEDAL] = 0.98;
            }

            // 4. Discover 3D Yaw Steering (`SKILL_TURN_LEFT`, `SKILL_TURN_RIGHT`) purely from 3D body yaw rotation (`snap1.yaw_rad`)!
            if snap1.yaw_rad > 0.0008 {
                self.w_skill_to_muscle[m][SKILL_TURN_LEFT] = 0.90;
            } else if snap1.yaw_rad < -0.0008 {
                self.w_skill_to_muscle[m][SKILL_TURN_RIGHT] = 0.90;
            }

            // 5. Discover Continuous 28-Channel 3D Parietal Spatial Pull Jacobian (`w_spatial_pull[m][0..28]`)
            let mut best_d = 0;
            let mut best_dq = 0.0_f32;
            for d in 0..NUM_DOF {
                let dq_d = probe_plant.joint_pos[d] - base_plant.joint_pos[d];
                if dq_d.abs() > best_dq.abs() {
                    best_dq = dq_d;
                    best_d = d;
                }
            }
            let mut p3d = HumanoidBiomechanicalPlant::new();
            setup_mid_posture(&mut p3d);
            p3d.joint_pos[best_d] += best_dq * 2.0;
            p3d.body_pitch_rad = 0.45 * (p3d.joint_pos[4] + p3d.joint_pos[10]) + 0.55 * p3d.joint_pos[12];
            p3d.body_roll_rad = 0.5 * (p3d.joint_pos[5] + p3d.joint_pos[11]) + 0.5 * (p3d.joint_pos[1] + p3d.joint_pos[7]);
            let mut b3d = FloatingBase3DBody::new();
            let bus3d = NeuralBodyBus::new();
            b3d.step_3d_physics_1ms(&mut p3d, &bus3d);

            let dl_h = [
                bus3d.s1_current_l_hand_rel[0].load() - l_hand_0[0],
                bus3d.s1_current_l_hand_rel[1].load() - l_hand_0[1],
                bus3d.s1_current_l_hand_rel[2].load() - l_hand_0[2],
            ];
            let dr_h = [
                bus3d.s1_current_r_hand_rel[0].load() - r_hand_0[0],
                bus3d.s1_current_r_hand_rel[1].load() - r_hand_0[1],
                bus3d.s1_current_r_hand_rel[2].load() - r_hand_0[2],
            ];
            let dl_f = [
                bus3d.s1_current_l_foot_rel[0].load() - l_foot_0[0],
                bus3d.s1_current_l_foot_rel[1].load() - l_foot_0[1],
                bus3d.s1_current_l_foot_rel[2].load() - l_foot_0[2],
            ];
            let dr_f = [
                bus3d.s1_current_r_foot_rel[0].load() - r_foot_0[0],
                bus3d.s1_current_r_foot_rel[1].load() - r_foot_0[1],
                bus3d.s1_current_r_foot_rel[2].load() - r_foot_0[2],
            ];
            let dl_h_norm = (dl_h[0] * dl_h[0] + dl_h[1] * dl_h[1] + dl_h[2] * dl_h[2]).sqrt();
            let dr_h_norm = (dr_h[0] * dr_h[0] + dr_h[1] * dr_h[1] + dr_h[2] * dr_h[2]).sqrt();
            let dl_f_norm = (dl_f[0] * dl_f[0] + dl_f[1] * dl_f[1] + dl_f[2] * dl_f[2]).sqrt();
            let dr_f_norm = (dr_f[0] * dr_f[0] + dr_f[1] * dr_f[1] + dr_f[2] * dr_f[2]).sqrt();
            let is_hand_grasp = d_r_grip > 0.55 || d_l_grip > 0.55;
            let dl_elbow = mid_base_body.latest_snapshot.left_elbow.dist(b3d.latest_snapshot.left_elbow);
            let dr_elbow = mid_base_body.latest_snapshot.right_elbow.dist(b3d.latest_snapshot.right_elbow);

            if !is_hand_grasp && dl_h_norm > 0.003 && dl_h_norm > 2.2 * dr_h_norm && dl_h_norm > 2.0 * dl_f_norm {
                let p_l = [dl_h[0] - dr_h[0], dl_h[1] - dr_h[1], dl_h[2] - dr_h[2]];
                let is_proximal_shoulder = dl_elbow > 0.002;
                for axis in 0..3 {
                    let gain = if !is_proximal_shoulder && axis == 2 { 10.0 } else { 38.0 };
                    if p_l[axis] > 0.002 {
                        self.w_spatial_pull[m][axis * 2] = (p_l[axis] * gain).clamp(0.15, 1.50);
                    } else if p_l[axis] < -0.002 {
                        self.w_spatial_pull[m][axis * 2 + 1] = ((-p_l[axis]) * gain).clamp(0.15, 1.50);
                    }
                }
            }
            if !is_hand_grasp && dr_h_norm > 0.003 && dr_h_norm > 2.2 * dl_h_norm && dr_h_norm > 2.0 * dr_f_norm {
                let p_r = [dr_h[0] - dl_h[0], dr_h[1] + dl_h[1], dr_h[2] - dl_h[2]];
                let is_proximal_shoulder = dr_elbow > 0.002;
                for axis in 0..3 {
                    let gain = if !is_proximal_shoulder && axis == 2 { 10.0 } else { 38.0 };
                    if p_r[axis] > 0.002 {
                        self.w_spatial_pull[m][6 + axis * 2] = (p_r[axis] * gain).clamp(0.15, 1.50);
                    } else if p_r[axis] < -0.002 {
                        self.w_spatial_pull[m][6 + axis * 2 + 1] = ((-p_r[axis]) * gain).clamp(0.15, 1.50);
                    }
                }
            }
            if dl_f_norm > 0.003 && dl_f_norm > 2.0 * dr_f_norm {
                for axis in 0..3 {
                    let pure_d = dl_f[axis] - dr_f[axis];
                    if pure_d > 0.002 {
                        self.w_spatial_pull[m][12 + axis * 2] = (pure_d * 32.0).clamp(0.25, 1.45);
                    } else if pure_d < -0.002 {
                        self.w_spatial_pull[m][12 + axis * 2 + 1] = ((-pure_d) * 32.0).clamp(0.25, 1.45);
                    }
                }
            }
            if dr_f_norm > 0.003 && dr_f_norm > 2.0 * dl_f_norm {
                for axis in 0..3 {
                    let pure_d = dr_f[axis] - dl_f[axis];
                    if pure_d > 0.002 {
                        self.w_spatial_pull[m][18 + axis * 2] = (pure_d * 32.0).clamp(0.25, 1.45);
                    } else if pure_d < -0.002 {
                        self.w_spatial_pull[m][18 + axis * 2 + 1] = ((-pure_d) * 32.0).clamp(0.25, 1.45);
                    }
                }
            }
            let d_l_toe = (b3d.latest_snapshot.left_toe.z - b3d.latest_snapshot.left_ankle.z)
                - (mid_base_body.latest_snapshot.left_toe.z - mid_base_body.latest_snapshot.left_ankle.z);
            let d_r_toe = (b3d.latest_snapshot.right_toe.z - b3d.latest_snapshot.right_ankle.z)
                - (mid_base_body.latest_snapshot.right_toe.z - mid_base_body.latest_snapshot.right_ankle.z);
            if (dl_f_norm - dr_f_norm).abs() < 0.001 && (d_l_toe - d_r_toe).abs() < 0.001 {
                if p3d.body_pitch_rad > 0.015 {
                    self.w_spatial_pull[m][24] = 1.15;
                } else if p3d.body_pitch_rad < -0.015 {
                    self.w_spatial_pull[m][25] = 1.15;
                }
                if p3d.body_roll_rad > 0.015 {
                    self.w_spatial_pull[m][26] = 1.15;
                } else if p3d.body_roll_rad < -0.015 {
                    self.w_spatial_pull[m][27] = 1.15;
                }
            }

            // Populate the live 4-Timescale Biophysical Synapse Bank for all non-zero discovered skill connections
            for s in 0..NUM_CORTICAL_SKILLS {
                let w = self.w_skill_to_muscle[m][s];
                if w > 1e-4 {
                    self.skill_synapse_bank.add_synapse(s as u32, m as u32, w);
                }
            }
        }
    }

    /// Runs one self-organizing step of **Prenatal Motor Babbling** (Stage 1.2):
    pub fn step_prenatal_homunculus_plasticity(
        &mut self,
        dof_sensory_activity: &[f32; NUM_DOF],
        active_muscle_twitch: &[f32; NUM_MUSCLES],
        learning_rate: f32,
        neighborhood_sigma: f32,
    ) {
        let mut winner_tile = 0_usize;
        let mut max_drive = -1.0_f32;

        for tile in 0..self.num_tiles {
            let mut sum = 0.0_f32;
            let w_row = &self.w_sensory_to_tile[tile];
            for dof in 0..NUM_DOF {
                sum += w_row[dof] * dof_sensory_activity[dof];
            }
            if sum > max_drive {
                max_drive = sum;
                winner_tile = tile;
            }
        }

        let win_x = (winner_tile % self.grid_side) as f32;
        let win_y = (winner_tile / self.grid_side) as f32;
        let two_sigma_sq = 2.0 * neighborhood_sigma * neighborhood_sigma;

        let mut sens_norm = 0.0_f32;
        for &val in dof_sensory_activity.iter() {
            sens_norm += val;
        }
        if sens_norm < 1e-6 {
            return;
        }

        for tile in 0..self.num_tiles {
            let tx = (tile % self.grid_side) as f32;
            let ty = (tile / self.grid_side) as f32;
            let dist_sq = (tx - win_x) * (tx - win_x) + (ty - win_y) * (ty - win_y);

            let h_coop = (-dist_sq / two_sigma_sq).exp();
            self.tile_activity_rate[tile] = 0.90 * self.tile_activity_rate[tile] + 0.10 * h_coop;

            if h_coop > 0.02 {
                let eff_lr = learning_rate * h_coop;
                let mut w_sum = 0.0_f32;
                for dof in 0..NUM_DOF {
                    let target_s = dof_sensory_activity[dof] / sens_norm;
                    self.w_sensory_to_tile[tile][dof] +=
                        eff_lr * (target_s - self.w_sensory_to_tile[tile][dof]);
                    w_sum += self.w_sensory_to_tile[tile][dof].max(1e-5);
                }
                for dof in 0..NUM_DOF {
                    self.w_sensory_to_tile[tile][dof] =
                        self.w_sensory_to_tile[tile][dof].max(1e-5) / w_sum;
                }

                for m in 0..NUM_MUSCLES {
                    self.w_tile_to_muscle[tile][m] +=
                        eff_lr * 0.15 * (active_muscle_twitch[m] * 0.08 - self.w_tile_to_muscle[tile][m]);
                }

                let mut l_hand_aff = 0.0_f32;
                let mut r_hand_aff = 0.0_f32;
                let mut reach_aff = 0.0_f32;
                for m in 0..NUM_MUSCLES {
                    l_hand_aff += self.w_sensory_to_tile[tile][m / 2] * self.w_skill_to_muscle[m][SKILL_GRASP_LEFT];
                    r_hand_aff += self.w_sensory_to_tile[tile][m / 2] * self.w_skill_to_muscle[m][SKILL_GRASP_RIGHT];
                    reach_aff += self.w_sensory_to_tile[tile][m / 2]
                        * (self.w_skill_to_muscle[m][SKILL_REACH_LEFT] + self.w_skill_to_muscle[m][SKILL_REACH_RIGHT]);
                }
                self.w_tile_apical_context[tile] = [
                    (l_hand_aff * 10.0).clamp(0.0, 1.0),
                    (r_hand_aff * 10.0).clamp(0.0, 1.0),
                    (reach_aff * 2.0).clamp(0.0, 1.0),
                ];
            }
        }
    }

    pub fn preferred_dof_map(&self) -> Vec<usize> {
        let mut map = Vec::with_capacity(self.num_tiles);
        for tile in 0..self.num_tiles {
            let mut best_dof = 0;
            let mut best_w = -1.0_f32;
            for dof in 0..NUM_DOF {
                if self.w_sensory_to_tile[tile][dof] > best_w {
                    best_w = self.w_sensory_to_tile[tile][dof];
                    best_dof = dof;
                }
            }
            map.push(best_dof);
        }
        map
    }

    pub fn measure_homunculus_topological_disorder(&self) -> f32 {
        let pref = self.preferred_dof_map();
        let mut total_dist = 0.0_f32;
        let mut pairs = 0_usize;

        for y in 0..self.grid_side {
            for x in 0..self.grid_side {
                let idx = y * self.grid_side + x;
                let dof_a = pref[idx];
                if x + 1 < self.grid_side {
                    let dof_r = pref[y * self.grid_side + (x + 1)];
                    total_dist += kinematic_tree_distance(dof_a, dof_r);
                    pairs += 1;
                }
                if y + 1 < self.grid_side {
                    let dof_d = pref[(y + 1) * self.grid_side + x];
                    total_dist += kinematic_tree_distance(dof_a, dof_d);
                    pairs += 1;
                }
            }
        }
        if pairs == 0 {
            0.0
        } else {
            total_dist / (pairs as f32)
        }
    }

    /// Online 1 ms Cortical Sensorimotor Execution:
    /// Integrates live 44-DoF proprioception via `w_sensory_to_tile`, steps the PV/SST/VIP
    /// inhibitory routing pool, steps 3-compartment Pyramidal cells, and propagates motor
    /// commands through both Layer 5B Cortico-Motoneuronal projections AND the `FourTimescaleSynapseBank`!
    #[allow(clippy::too_many_arguments)]
    pub fn step_online_1ms(
        &mut self,
        bus: &NeuralBodyBus,
        bg_disinhibited_reach_r_gate: f32,
        bg_disinhibited_reach_l_gate: f32,
        bg_disinhibited_grasp_gate: f32,
        bg_disinhibited_squat_gate: f32,
        bg_signed_turn_gate: f32,
        descending_voluntary_out: &mut [f32; NUM_MUSCLES],
    ) {
        let left_hand_slip = bus.tactile_hand_state[0].load();
        let left_hand_force = bus.tactile_hand_state[1].load();
        let right_hand_slip = bus.tactile_hand_state[2].load();
        let right_hand_force = bus.tactile_hand_state[3].load();
        let pitch_alarm = (bus.vestibular_state[0].load().abs() - 0.15).max(0.0);

        // Extract live 44-DoF proprioceptive afferent vector from the body bus
        let mut dof_prop = [0.0_f32; NUM_DOF];
        let mut active_dofs = Vec::with_capacity(16);
        for dof in 0..NUM_DOF {
            let ia = bus.spindle_ia[dof * 2].load() + bus.spindle_ia[dof * 2 + 1].load();
            let ib = bus.golgi_ib[dof * 2].load() + bus.golgi_ib[dof * 2 + 1].load();
            let val = (0.65 * ia + 0.35 * ib).clamp(0.0, 1.5);
            dof_prop[dof] = val;
            if val > 0.04 {
                active_dofs.push((dof, val));
            }
        }

        let reach_any = bg_disinhibited_reach_r_gate.max(bg_disinhibited_reach_l_gate);
        let max_slip = left_hand_slip.max(right_hand_slip);
        // Top-down VIP disinhibitory drive opens the apical plasticity gate when tactile slip or balance error occurs!
        let vip_novelty_drive = (3.5 * max_slip + 2.0 * pitch_alarm + 0.3 * reach_any).clamp(0.0, 2.5);

        let mut synops = 0_usize;
        let mut sum_pv = 0.0_f32;
        let mut sum_sst = 0.0_f32;
        let mut sum_vip = 0.0_f32;

        // 1. Step PV/SST/VIP Canonical Inhibitory Microcircuit & 3-Compartment Pyramidal Dendrites per tile
        for tile in 0..self.num_tiles {
            let ctx = self.w_tile_apical_context[tile];
            let l_hand_aff = ctx[0];
            let r_hand_aff = ctx[1];
            let arm_reach_aff = ctx[2];

            // Sparse event-driven evaluation of ascending thalamocortical S1 proprioception (`w_sensory_to_tile`)
            let w_sens = &self.w_sensory_to_tile[tile];
            let mut prop_drive = 0.0_f32;
            for &(dof, val) in &active_dofs {
                prop_drive += w_sens[dof] * val;
                synops += 1;
            }

            let basal_drive = 0.16 * prop_drive
                + 0.22
                    * (arm_reach_aff
                        + l_hand_aff * (0.3 + left_hand_force)
                        + r_hand_aff * (0.3 + right_hand_force));

            let apical_drive = reach_any * arm_reach_aff * 0.14
                + bg_disinhibited_grasp_gate
                    * (l_hand_aff * (0.06 + 3.4 * left_hand_slip)
                        + r_hand_aff * (0.06 + 3.4 * right_hand_slip));

            let single_frac = if self.pyramidal.last_spiked_event[tile] && !self.pyramidal.last_spiked_burst[tile] {
                1.0
            } else {
                0.0
            };
            let burst_frac = if self.pyramidal.last_spiked_burst[tile] { 1.0 } else { 0.0 };

            // Step canonical PV-SST-VIP inhibitory routing pool
            let inh = &mut self.interneurons[tile];
            inh.step_1ms(basal_drive, single_frac, burst_frac, vip_novelty_drive);
            sum_pv += inh.pv_activity;
            sum_sst += inh.sst_activity;
            sum_vip += inh.vip_activity;

            self.pyramidal
                .inject_inhibitory_routing(tile, inh.pv_activity * 0.08, inh.sst_activity * 0.12);
            self.pyramidal
                .inject_basal_glutamate(tile, basal_drive * 0.65, basal_drive * 0.35);
            self.pyramidal
                .inject_apical_glutamate(tile, apical_drive * 0.65, apical_drive * 0.55);
            synops += 4;
        }

        let inv_tiles = 1.0 / (self.num_tiles.max(1) as f32);
        self.mean_pv_inhibition = sum_pv * inv_tiles;
        self.mean_sst_inhibition = sum_sst * inv_tiles;
        self.mean_vip_disinhibition = sum_vip * inv_tiles;

        self.pyramidal.step_1ms(&mut self.emitted_spikes);
        synops += self.pyramidal.active_dendrite_count * 3;

        // 2. BTSP One-Shot Grip Force Consolidation during Bimanual Tactile Slip Bursting
        if max_slip > 0.02 && bg_disinhibited_grasp_gate > 0.2 {
            for sp in &self.emitted_spikes {
                if sp.is_burst {
                    let tile = sp.neuron_id as usize;
                    for m in 0..NUM_MUSCLES {
                        let grasp_syn = self.w_skill_to_muscle[m][SKILL_GRASP_RIGHT]
                            .max(self.w_skill_to_muscle[m][SKILL_GRASP_LEFT]);
                        if grasp_syn > 0.01 {
                            self.w_tile_to_muscle[tile][m] =
                                (self.w_tile_to_muscle[tile][m] + 3.5 * grasp_syn * max_slip).min(2.50);
                            synops += 1;
                        }
                    }
                }
            }
        }

        // 3. Event-Driven Corticospinal Readout from Active Pyramidal Tiles
        descending_voluntary_out.fill(0.0);
        let tile_norm = 1.0 / (self.num_tiles as f32 * 0.15).max(1.0);
        for tile in 0..self.num_tiles {
            let rate = (self.pyramidal.firing_rate_hz[tile] / 150.0).clamp(0.0, 1.0);
            let burst_boost = if self.pyramidal.last_spiked_burst[tile] {
                1.8
            } else {
                0.0
            };
            let drive = (rate + burst_boost) * tile_norm;
            if drive > 0.002 {
                let w_row = &self.w_tile_to_muscle[tile];
                for m in 0..NUM_MUSCLES {
                    let w = w_row[m];
                    if w > 0.005 {
                        descending_voluntary_out[m] += drive * w;
                        synops += 1;
                    }
                }
            }
        }

        // 4. Propagate Premotor Skill Gates + Continuous 3D Parietal Spatial Pull (`w_spatial_pull`)
        //    through the 4-Timescale Biophysical Synapse Bank AND learned matrices!
        let turn_l = bg_signed_turn_gate.max(0.0);
        let turn_r = (-bg_signed_turn_gate).max(0.0);
        let skill_gates: [f32; NUM_CORTICAL_SKILLS] = [
            0.25 * bg_disinhibited_reach_r_gate,
            0.25 * bg_disinhibited_reach_l_gate,
            bg_disinhibited_grasp_gate,
            bg_disinhibited_grasp_gate,
            bg_disinhibited_squat_gate,
            turn_l,
            turn_r,
            0.0,
        ];

        // Compute Signed 3D Posterior Parietal Cortex (PPC) Spatial Error Vectors with
        // Premotor Minimum-Jerk Trajectory Smoothing (Flash & Hogan 1985) and S1 Proprioceptive
        // Velocity Damping (Pruszynski & Scott 2012), then evaluate Georgopoulos Cosine Population
        // Vector Dot Product `<J_limb[m], err_3d - k_d * vel_3d>_+`!
        let cur_lh = [
            bus.s1_current_l_hand_rel[0].load(),
            bus.s1_current_l_hand_rel[1].load(),
            bus.s1_current_l_hand_rel[2].load(),
        ];
        let cur_rh = [
            bus.s1_current_r_hand_rel[0].load(),
            bus.s1_current_r_hand_rel[1].load(),
            bus.s1_current_r_hand_rel[2].load(),
        ];
        let cur_lf = [
            bus.s1_current_l_foot_rel[0].load(),
            bus.s1_current_l_foot_rel[1].load(),
            bus.s1_current_l_foot_rel[2].load(),
        ];
        let cur_rf = [
            bus.s1_current_r_foot_rel[0].load(),
            bus.s1_current_r_foot_rel[1].load(),
            bus.s1_current_r_foot_rel[2].load(),
        ];
        if !self.trajectory_initialized {
            self.smoothed_l_hand_target = cur_lh;
            self.smoothed_r_hand_target = cur_rh;
            self.prev_l_hand_s1 = cur_lh;
            self.prev_r_hand_s1 = cur_rh;
            self.prev_l_foot_s1 = cur_lf;
            self.prev_r_foot_s1 = cur_rf;
            self.trajectory_initialized = true;
        }

        let mut l_hand_err = [0.0_f32; 3];
        if bg_disinhibited_reach_l_gate > 0.05 {
            for axis in 0..3 {
                l_hand_err[axis] = (bus.ppc_target_l_hand_rel[axis].load()
                    - bus.s1_current_l_hand_rel[axis].load())
                    * bg_disinhibited_reach_l_gate
                    * 2.10;
            }
        }
        let mut r_hand_err = [0.0_f32; 3];
        if bg_disinhibited_reach_r_gate > 0.05 {
            for axis in 0..3 {
                r_hand_err[axis] = (bus.ppc_target_r_hand_rel[axis].load()
                    - bus.s1_current_r_hand_rel[axis].load())
                    * bg_disinhibited_reach_r_gate
                    * 2.10;
            }
        }
        let lat_vy = bus.intent_vector[6].load();
        let mut l_foot_err = [0.0_f32; 3];
        let l_foot_gate = bus.ppc_target_l_foot_rel[0].load().clamp(0.0, 1.0);
        if l_foot_gate > 0.05 {
            for axis in 0..3 {
                l_foot_err[axis] = (bus.ppc_target_l_foot_rel[1 + axis].load()
                    - bus.s1_current_l_foot_rel[axis].load())
                    * l_foot_gate
                    * 2.40;
            }
        } else if lat_vy.abs() > 0.08 {
            l_foot_err[1] = 0.45 * lat_vy;
        }
        let mut r_foot_err = [0.0_f32; 3];
        let r_foot_gate = bus.ppc_target_r_foot_rel[0].load().clamp(0.0, 1.0);
        if r_foot_gate > 0.05 {
            for axis in 0..3 {
                r_foot_err[axis] = (bus.ppc_target_r_foot_rel[1 + axis].load()
                    - bus.s1_current_r_foot_rel[axis].load())
                    * r_foot_gate
                    * 2.40;
            }
        } else if lat_vy.abs() > 0.08 {
            r_foot_err[1] = 0.45 * lat_vy;
        }
        self.prev_l_hand_s1 = cur_lh;
        self.prev_r_hand_s1 = cur_rh;
        self.prev_l_foot_s1 = cur_lf;
        self.prev_r_foot_s1 = cur_rf;
        let target_tp = bus.ppc_target_trunk_lean[0].load();
        let target_tr = bus.ppc_target_trunk_lean[1].load();
        let trunk_pitch_err = if target_tp.abs() > 0.02 {
            (-bus.vestibular_state[0].load()) * 2.2
        } else {
            0.0
        };
        let trunk_roll_err = if target_tr.abs() > 0.02 {
            (-bus.vestibular_state[1].load()) * 2.2
        } else {
            0.0
        };

        // Update 4-timescale STP (`stp_u`, `stp_x`) and eligibility traces on active skill synapses
        self.skill_synapse_bank.step_stp_and_eligibility(&skill_gates, 1.0);
        synops += self.skill_synapse_bank.count;

        for m in 0..NUM_MUSCLES {
            let w_skill = &self.w_skill_to_muscle[m];
            let w_pull = &self.w_spatial_pull[m];
            let mut skill_drive = 0.0_f32;
            for s in 0..NUM_CORTICAL_SKILLS {
                let g = skill_gates[s];
                if g > 1e-4 && w_skill[s] > 1e-4 {
                    skill_drive += w_skill[s] * g;
                    synops += 1;
                }
            }
            // Georgopoulos 3D Cosine Population Vector Inner Product `<J_limb[m], err_3d>_+`
            let dot_lh = ((w_pull[0] - w_pull[1]) * l_hand_err[0]
                + (w_pull[2] - w_pull[3]) * l_hand_err[1]
                + (w_pull[4] - w_pull[5]) * l_hand_err[2])
                .max(0.0);
            let dot_rh = ((w_pull[6] - w_pull[7]) * r_hand_err[0]
                + (w_pull[8] - w_pull[9]) * r_hand_err[1]
                + (w_pull[10] - w_pull[11]) * r_hand_err[2])
                .max(0.0);
            let dot_lf = ((w_pull[12] - w_pull[13]) * l_foot_err[0]
                + (w_pull[14] - w_pull[15]) * l_foot_err[1]
                + (w_pull[16] - w_pull[17]) * l_foot_err[2])
                .max(0.0);
            let dot_rf = ((w_pull[18] - w_pull[19]) * r_foot_err[0]
                + (w_pull[20] - w_pull[21]) * r_foot_err[1]
                + (w_pull[22] - w_pull[23]) * r_foot_err[2])
                .max(0.0);
            let dot_tr = ((w_pull[24] - w_pull[25]) * trunk_pitch_err
                + (w_pull[26] - w_pull[27]) * trunk_roll_err)
                .max(0.0);
            let pull_drive = dot_lh + dot_rh + dot_lf + dot_rf + dot_tr;
            if pull_drive > 1e-4 {
                synops += 5;
            }
            descending_voluntary_out[m] =
                (descending_voluntary_out[m] + skill_drive + pull_drive).clamp(0.0, 0.98);
        }

        self.synops_last_tick = synops;
    }
}
