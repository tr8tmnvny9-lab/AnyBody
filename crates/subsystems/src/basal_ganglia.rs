//! `subsystems::basal_ganglia`: Striatal D1 (Go) / D2 (No-Go) / Striosome Value Compartments,
//! Subthalamic Nucleus (STN) Hyperdirect Emergency Brake, GPi/SNr Tonic Inhibition,
//! and Endogenous Dopamine Reward Prediction Error (SNc / Lateral Habenula).
//!
//! Supports 7 Whole-Body Action Channels:
//! - `CH_STAND_BALANCE (0)`: Upright postural equilibrium
//! - `CH_WALK_LOCOMOTE (1)`: Bipedal 3D walking & running
//! - `CH_REACH_ARM (2)`: Right arm 3D spatial target reaching
//! - `CH_GRASP_HAND (3)`: Bimanual 20-DoF hand grasping & slip reflex
//! - `CH_TURN_STEER (4)`: 3D axial & pelvic yaw steering (`[-1, +1]` signed gate)
//! - `CH_SQUAT_POSTURE (5)`: Deep bipedal knee/hip/ankle squat-and-rise
//! - `CH_REACH_LEFT_ARM (6)`: Left arm 3D spatial target reaching

use body_physics::NeuralBodyBus;

pub const NUM_BG_CHANNELS: usize = 7;
pub const CH_STAND_BALANCE: usize = 0;
pub const CH_WALK_LOCOMOTE: usize = 1;
pub const CH_REACH_ARM: usize = 2;
pub const CH_GRASP_HAND: usize = 3;
pub const CH_TURN_STEER: usize = 4;
pub const CH_SQUAT_POSTURE: usize = 5;
pub const CH_REACH_LEFT_ARM: usize = 6;

#[derive(Clone, Debug)]
pub struct BasalGangliaSubsystem {
    /// Striatal Direct Pathway ("Go") D1 MSN activity per action channel
    pub d1_msn_go: [f32; NUM_BG_CHANNELS],
    /// Striatal Indirect Pathway ("No-Go") D2 MSN activity per action channel
    pub d2_msn_nogo: [f32; NUM_BG_CHANNELS],
    /// Corticostriatal synaptic weights onto D1 and D2 MSNs (modulated by Dopamine RPE)
    pub w_cortex_to_d1: [f32; NUM_BG_CHANNELS],
    pub w_cortex_to_d2: [f32; NUM_BG_CHANNELS],

    /// Subthalamic Nucleus (STN) Hyperdirect global brake (driven by cortical conflict or vestibular trip alarm)
    pub stn_hyperdirect_activity: f32,
    /// Globus Pallidus Externus (GPe) activity
    pub gpe_activity: [f32; NUM_BG_CHANNELS],
    /// Globus Pallidus Internus (GPi / SNr) tonic inhibitory output (`0.85` at rest = brake ON!)
    pub gpi_tonic_inhibition: [f32; NUM_BG_CHANNELS],
    /// Ventrolateral Motor Thalamus (`VL`) disinhibited output gate (`[0, 1]`)
    pub thalamic_motor_gate: [f32; NUM_BG_CHANNELS],
    /// Signed steering direction from `bus.intent_vector[2]` (`[-1, +1]`), gated by `thalamic_motor_gate[CH_TURN_STEER]`
    pub signed_turn_gate: f32,

    /// Striosomal expected stability/success value and SNc Dopamine concentration (`1.0` = baseline tonic DA)
    pub striosome_value_pred: f32,
    pub snc_dopamine_level: f32,
}

impl Default for BasalGangliaSubsystem {
    fn default() -> Self {
        Self::new()
    }
}

impl BasalGangliaSubsystem {
    pub const fn new() -> Self {
        Self {
            d1_msn_go: [0.0; NUM_BG_CHANNELS],
            d2_msn_nogo: [0.0; NUM_BG_CHANNELS],
            w_cortex_to_d1: [0.85; NUM_BG_CHANNELS],
            w_cortex_to_d2: [0.35; NUM_BG_CHANNELS],
            stn_hyperdirect_activity: 0.05,
            gpe_activity: [0.60; NUM_BG_CHANNELS],
            gpi_tonic_inhibition: [0.85; NUM_BG_CHANNELS],
            thalamic_motor_gate: [0.0; NUM_BG_CHANNELS],
            signed_turn_gate: 0.0,
            striosome_value_pred: 0.5,
            snc_dopamine_level: 1.0,
        }
    }

    /// Steps the Basal Ganglia - Thalamocortical gating loop (`dt = 1 ms`).
    /// Reads upstream intent from `bus.intent_vector` and postural/slip stability from `bus`.
    pub fn step_1ms(&mut self, bus: &NeuralBodyBus) {
        let walk_cmd = bus.intent_vector[0]
            .load()
            .abs()
            .max(bus.intent_vector[6].load().abs())
            .clamp(0.0, 1.0);
        let squat_cmd = bus.intent_vector[1].load().clamp(0.0, 1.0);
        let raw_turn = bus.intent_vector[2].load().clamp(-1.0, 1.0);
        let turn_mag = raw_turn.abs();
        let stand_cmd = bus.intent_vector[3].load().clamp(0.0, 1.0);
        let reach_r_cmd = bus.intent_vector[4].load().clamp(0.0, 1.0);
        let reach_l_cmd = bus.intent_vector[5].load().clamp(0.0, 1.0);
        let grasp_cmd = bus.intent_vector[7].load().clamp(0.0, 1.0);

        let raw_intents = [
            stand_cmd,
            walk_cmd,
            reach_r_cmd,
            grasp_cmd,
            turn_mag,
            squat_cmd,
            reach_l_cmd,
        ];

        // 1. Compute Endogenous Dopamine RPE via Lateral Habenula (LHb) & PPN:
        let pitch_err = bus.vestibular_state[0].load().abs();
        let slip_err = bus.tactile_hand_state[0].load().max(bus.tactile_hand_state[2].load());
        let grip_confirm = bus.tactile_hand_state[1].load().max(bus.tactile_hand_state[3].load());

        let instant_biological_outcome =
            (0.6 * (1.0 - 2.0 * pitch_err).max(0.0) + 0.4 * grip_confirm - 0.8 * slip_err).clamp(0.0, 1.5);
        let rpe = instant_biological_outcome - self.striosome_value_pred;
        self.striosome_value_pred = 0.98 * self.striosome_value_pred + 0.02 * instant_biological_outcome;
        self.snc_dopamine_level = (1.0 + 0.8 * rpe).clamp(0.2, 2.0);

        // 2. Subthalamic Nucleus (STN) Hyperdirect Emergency Brake:
        let intent_conflict = (walk_cmd * stand_cmd).max(0.0);
        let fall_alarm = if pitch_err > 0.45 { (pitch_err - 0.45) * 3.0 } else { 0.0 };
        let stn_target = (0.05 + 0.4 * intent_conflict + fall_alarm).clamp(0.05, 1.5);
        self.stn_hyperdirect_activity = 0.85 * self.stn_hyperdirect_activity + 0.15 * stn_target;

        // 3. Striatal D1 (Go) vs D2 (No-Go) Competition & GPi Disinhibition
        let da = self.snc_dopamine_level;
        for ch in 0..NUM_BG_CHANNELS {
            let c_in = raw_intents[ch];

            let d1_drive = c_in * self.w_cortex_to_d1[ch] * (0.6 + 0.4 * da);
            let d2_drive = c_in * self.w_cortex_to_d2[ch] / (0.6 + 0.4 * da);

            let mut lateral_inh = 0.0_f32;
            for other in 0..NUM_BG_CHANNELS {
                if other != ch {
                    // Stand-Still (0) and Deep Squat (5) suppress Walk (1) so squatting happens cleanly in place;
                    // Turning, Bimanual Reaching, and Grasping co-execute seamlessly with Walking or Squatting!
                    let conflict_weight = match (ch, other) {
                        (CH_STAND_BALANCE, CH_WALK_LOCOMOTE) | (CH_WALK_LOCOMOTE, CH_STAND_BALANCE) => 0.75,
                        (CH_WALK_LOCOMOTE, CH_SQUAT_POSTURE) => 1.35,
                        _ => 0.0,
                    };
                    lateral_inh += conflict_weight * self.d1_msn_go[other];
                }
            }

            self.d1_msn_go[ch] = 0.85 * self.d1_msn_go[ch] + 0.15 * (d1_drive - lateral_inh).max(0.0);
            self.d2_msn_nogo[ch] = 0.85 * self.d2_msn_nogo[ch] + 0.15 * d2_drive;

            self.gpe_activity[ch] = (0.75 - 0.80 * self.d2_msn_nogo[ch]).clamp(0.0, 1.0);

            let gpi_target = (0.85 + 0.65 * self.stn_hyperdirect_activity
                + 0.45 * (0.75 - self.gpe_activity[ch])
                - 1.35 * self.d1_msn_go[ch])
                .clamp(0.0, 1.5);
            self.gpi_tonic_inhibition[ch] = 0.85 * self.gpi_tonic_inhibition[ch] + 0.15 * gpi_target;

            let thal_drive = ((0.75 - self.gpi_tonic_inhibition[ch]) * 1.6).clamp(0.0, 1.0);
            self.thalamic_motor_gate[ch] = 0.85 * self.thalamic_motor_gate[ch] + 0.15 * thal_drive;
        }
        self.signed_turn_gate = raw_turn.signum() * self.thalamic_motor_gate[CH_TURN_STEER];
    }
}
