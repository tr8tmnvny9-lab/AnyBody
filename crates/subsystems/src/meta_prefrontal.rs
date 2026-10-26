//! `subsystems::meta_prefrontal`: Pure-Rust Meta Muse (`muse-wonder-1.0` / `muse-spark-1.3`) +
//! Baseten FP4 (`meta-models/muse-spark-open-1026`) Prefrontal Cortex Action Sequencer,
//! Dorsal-Ventral Visual Stream Grounding (`sam-3.1`), Closed-Loop Autonomous Agentic Chaining
//! & Stopping Loop, and Thread-Safe EUR Budget Guard (`DualBudgetGuard`).
//!
//! Bridges any natural-language user instruction — from single skills (*"Point at the blue ball"*,
//! *"Find the green cylinder"*, *"Walk sideways"*) to long-horizon autonomous agentic runs
//! (*"Clean all the things in the basket"*) — into continuous 3D Parietal/Premotor/Basal-Ganglia
//! attractors (`MotorStageCommand`) executed by the 1,000 Hz spiking neural connectome!

use body_physics::{
    FloatingBase3DBody, HumanoidBiomechanicalPlant, NeuralBodyBus, Vec3, NUM_WORLD_OBJECTS,
};
use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Price in EUR per 1,000 tokens for Meta Muse and Baseten models.
pub fn estimate_cost_eur(prompt_tokens: u64, completion_tokens: u64, model: &str) -> f64 {
    let rate = match model {
        "muse-wonder-1.0" => 0.004,
        "muse-spark-1.3" | "muse-spark-1.3-contributor" => 0.001,
        "sam-3.1" | "muse-image-1.0" => 0.002,
        "meta-models/muse-spark-open-1026" | "meta-models/muse-spark-open" => 0.0005,
        _ => 0.001,
    };
    ((prompt_tokens + completion_tokens) as f64 / 1000.0) * rate
}

/// Thread-safe pure-Rust API spend, token, call, and latency tracker (replaces `src/meta_budget.py`).
#[derive(Clone, Debug)]
pub struct ProviderBudgetState {
    pub provider: String,
    pub limit_eur: f64,
    pub spent_eur: f64,
    pub calls: u64,
    pub failed_calls: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub last_latency_ms: f64,
    pub avg_latency_ms: f64,
    pub last_model: String,
}

impl ProviderBudgetState {
    pub fn new(provider: &str, limit_eur: f64) -> Self {
        Self {
            provider: provider.to_string(),
            limit_eur,
            spent_eur: 0.0,
            calls: 0,
            failed_calls: 0,
            prompt_tokens: 0,
            completion_tokens: 0,
            last_latency_ms: 0.0,
            avg_latency_ms: 0.0,
            last_model: "none".to_string(),
        }
    }

    pub fn record_call(
        &mut self,
        prompt_tok: u64,
        comp_tok: u64,
        latency_ms: f64,
        model: &str,
        failed: bool,
    ) -> Result<f64, String> {
        let cost = if failed {
            0.0
        } else {
            estimate_cost_eur(prompt_tok, comp_tok, model)
        };
        if self.spent_eur + cost > self.limit_eur {
            return Err(format!(
                "Budget limit exceeded [{}]: {:.4} EUR > {:.2} EUR",
                self.provider,
                self.spent_eur + cost,
                self.limit_eur
            ));
        }
        self.spent_eur += cost;
        self.calls += 1;
        if failed {
            self.failed_calls += 1;
        }
        self.prompt_tokens += prompt_tok;
        self.completion_tokens += comp_tok;
        self.last_latency_ms = latency_ms;
        if self.calls == 1 {
            self.avg_latency_ms = latency_ms;
        } else {
            self.avg_latency_ms = 0.8 * self.avg_latency_ms + 0.2 * latency_ms;
        }
        self.last_model = model.to_string();
        Ok((self.limit_eur - self.spent_eur).max(0.0))
    }
}

/// Pure-Rust `.env` loader that checks `./.env` (and optional `ENV_FILE` override).
pub fn ensure_dotenv_loaded() {
    let custom_env = env::var("ENV_FILE").unwrap_or_else(|_| ".env".to_string());
    let candidates = [custom_env.as_str(), ".env"];
    for p in &candidates {
        let path = Path::new(p);
        if path.is_file() {
            if let Ok(text) = fs::read_to_string(path) {
                for raw in text.lines() {
                    let line = raw.trim();
                    if line.is_empty() || line.starts_with('#') {
                        continue;
                    }
                    if let Some((k, v)) = line.split_once('=') {
                        let key = k.trim();
                        let mut val = v.trim();
                        if val.len() >= 2
                            && ((val.starts_with('"') && val.ends_with('"'))
                                || (val.starts_with('\'') && val.ends_with('\'')))
                        {
                            val = &val[1..val.len() - 1];
                        }
                        if !key.is_empty() && env::var(key).is_err() {
                            env::set_var(key, val);
                        }
                    }
                }
            }
            break;
        }
    }
}

/// Resolves an object name/color/shape in natural language to one of the 4 `WorldObject3D` indices:
/// `0 = Red Box`, `1 = Blue Ball`, `2 = Green Cylinder`, `3 = Gold Crate`.
pub fn resolve_target_object_idx(lower: &str) -> Option<usize> {
    if lower.contains("green")
        || lower.contains("cylinder")
        || lower.contains("bottle")
        || lower.contains("can")
        || lower.contains("tube")
        || lower.contains("pillar")
    {
        Some(2)
    } else if lower.contains("blue")
        || lower.contains("ball")
        || lower.contains("sphere")
        || lower.contains("orb")
        || lower.contains("globe")
    {
        Some(1)
    } else if lower.contains("gold")
        || lower.contains("yellow")
        || lower.contains("crate")
        || lower.contains("package")
        || lower.contains("chest")
    {
        Some(3)
    } else if lower.contains("red")
        || lower.contains("box")
        || lower.contains("cube")
        || lower.contains("block")
        || lower.contains("object")
        || lower.contains("item")
    {
        Some(0)
    } else {
        None
    }
}

pub const OBJECT_NAMES: [&str; NUM_WORLD_OBJECTS] = [
    "Red Box",
    "Blue Ball",
    "Green Cylinder",
    "Gold Crate",
];

pub const BASKET_SLOT_OFFSETS: [[f32; 2]; NUM_WORLD_OBJECTS] = [
    [1.88, 0.74],
    [1.88, 0.86],
    [2.00, 0.74],
    [2.00, 0.86],
];

/// Returns true if `obj` is resting inside the Basket / Goal Zone (`||obj.pos.xy - goal.xy|| < 0.54 m` and `!obj.is_held`).
#[inline]
pub fn is_object_in_basket(body_3d: &FloatingBase3DBody, idx: usize) -> bool {
    let obj = &body_3d.world_objects[idx.min(NUM_WORLD_OBJECTS - 1)];
    if obj.is_held {
        return false;
    }
    let dx = obj.pos.x - body_3d.world_goal_pos.x;
    let dy = obj.pos.y - body_3d.world_goal_pos.y;
    dx.hypot(dy) < 0.54
}

/// Counts how many of the `NUM_WORLD_OBJECTS` are currently placed inside the Basket / Goal Zone.
#[inline]
pub fn count_objects_in_basket(body_3d: &FloatingBase3DBody) -> usize {
    (0..NUM_WORLD_OBJECTS)
        .filter(|&i| is_object_in_basket(body_3d, i))
        .count()
}

/// A single closed-loop task-space stage in a Prefrontal Motor Program.
#[derive(Clone, Debug)]
pub struct MotorStageCommand {
    pub name: String,
    pub duration_ms: u32,
    /// Optional index `0..4` into `body_3d.world_objects` (`0=Red Box`, `1=Blue Ball`, `2=Green Cylinder`, `3=Gold Crate`)
    pub target_object_idx: Option<usize>,
    /// If `Some(idx)`, rotates head/body scanning the 3D Binocular Visual Frustum (`SAM-3.1`) until `world_objects[idx].in_fov == true`!
    pub visual_search_for_object: Option<usize>,
    /// If `true`, rotates head/body scanning the room until ANY uncollected object (`!is_object_in_basket`) enters `in_fov == true`!
    pub visual_search_any_uncollected: bool,
    /// If `Some(idx)`, turns to face `world_objects[idx]` and points the right/left arm along the exact 3D ray from shoulder to object!
    pub point_at_object: Option<usize>,
    /// Optional 2D world waypoint `(x, y)` in meters; if `Some`, the brainstem automatically steers
    /// yaw and walks until `||pelvis_xy - waypoint_xy|| < 0.32 m`!
    pub navigate_to_world_xy: Option<[f32; 2]>,
    /// If true, dynamically navigates to `body_3d.world_objects[target_object_idx]`!
    pub navigate_to_object: bool,
    /// Direct forward/backward speed command in `[-1.0, +1.0]` (`>0` forward, `<0` backward)
    pub vx_cmd: f32,
    /// Direct lateral sidestep command in `[-1.0, +1.0]` (`>0` sidestep left, `<0` sidestep right)
    pub vy_cmd: f32,
    /// Direct 3D yaw turn rate command in `[-1.0, +1.0]` (`>0` turn left, `<0` turn right)
    pub yaw_cmd: f32,
    /// Deep bipedal squat depth in `[0.0, 1.0]` (`0.0` upright `0.87m`, `0.95` deep squat `0.50m`)
    pub squat_cmd: f32,
    /// Voluntary trunk bow/lean `[pitch_rad, roll_rad]` (`+pitch` = bow forward, `-pitch` = lean back)
    pub trunk_lean_rad: [f32; 2],
    /// Left Hand 3D Parietal Target `(active, [x, y, z] pelvis-relative meters)`
    pub left_hand_target: (bool, [f32; 3]),
    /// Right Hand 3D Parietal Target `(active, [x, y, z] pelvis-relative meters)`
    pub right_hand_target: (bool, [f32; 3]),
    /// If true, Left and/or Right hand dynamically track the 3D coordinates of `world_object_pos`!
    pub track_world_object_with_hands: bool,
    /// If true, oscillate raised hand(s) laterally to wave!
    pub wave_hands: bool,
    /// Left Foot 3D Unipedal Kick / Lift Target `(active, [x, y, z] pelvis-relative meters)`
    pub left_foot_target: (bool, [f32; 3]),
    /// Right Foot 3D Unipedal Kick / Lift Target `(active, [x, y, z] pelvis-relative meters)`
    pub right_foot_target: (bool, [f32; 3]),
    /// Hand palmar grasp command in `[0.0, 1.0]` (`0.85` = power grasp / pick up object, `0.0` = release)
    pub grasp_cmd: f32,
}

impl Default for MotorStageCommand {
    fn default() -> Self {
        Self {
            name: "Stand & Balance".to_string(),
            duration_ms: 1500,
            target_object_idx: None,
            visual_search_for_object: None,
            visual_search_any_uncollected: false,
            point_at_object: None,
            navigate_to_world_xy: None,
            navigate_to_object: false,
            vx_cmd: 0.0,
            vy_cmd: 0.0,
            yaw_cmd: 0.0,
            squat_cmd: 0.0,
            trunk_lean_rad: [0.0, 0.0],
            left_hand_target: (false, [0.38, 0.24, 0.25]),
            right_hand_target: (false, [0.38, -0.24, 0.25]),
            track_world_object_with_hands: false,
            wave_hands: false,
            left_foot_target: (false, [0.45, 0.12, -0.45]),
            right_foot_target: (false, [0.45, -0.12, -0.45]),
            grasp_cmd: 0.0,
        }
    }
}

/// Complete Prefrontal Cortex Meta-Model Sequencer & Autonomous Agentic Chaining State.
#[derive(Clone, Debug)]
pub struct MetaPrefrontalSequencer {
    pub active_prompt: String,
    pub stages: Vec<MotorStageCommand>,
    pub current_stage_idx: usize,
    pub stage_elapsed_ms: u32,
    pub sequence_active: bool,
    pub used_fallback_only: bool,
    /// True when executing a closed-loop multi-step agentic goal (e.g. *"Clean all the things in the basket"*)
    /// that autonomously inspects visual context, chains sub-tasks, self-calls the planner, looks around to verify,
    /// and knows when & how to stop!
    pub agentic_loop_active: bool,
    pub agentic_goal_description: String,
    pub agentic_iteration: u32,
    pub agentic_target_mask: [bool; NUM_WORLD_OBJECTS],
    pub agentic_target_total: usize,
    pub objects_in_basket_count: usize,
    pub final_verification_scan_done: bool,
    /// Set to `true` whenever the agentic loop self-invokes a new sub-goal so the runtime can fire an async LLM call!
    pub needs_llm_replan: bool,
    pub meta_reasoning_summary: String,
    pub meta_budget: ProviderBudgetState,
    pub baseten_budget: ProviderBudgetState,
}

impl Default for MetaPrefrontalSequencer {
    fn default() -> Self {
        Self::new()
    }
}

impl MetaPrefrontalSequencer {
    pub fn new() -> Self {
        ensure_dotenv_loaded();
        let meta_limit = env::var("META_BUDGET_LIMIT_EUR")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(150.0);
        let baseten_limit = env::var("BASETEN_BUDGET_LIMIT_EUR")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(100.0);
        Self {
            active_prompt: "Ready for any natural-language motor command".to_string(),
            stages: Vec::new(),
            current_stage_idx: 0,
            stage_elapsed_ms: 0,
            sequence_active: false,
            used_fallback_only: false,
            agentic_loop_active: false,
            agentic_goal_description: String::new(),
            agentic_iteration: 0,
            agentic_target_mask: [true; NUM_WORLD_OBJECTS],
            agentic_target_total: NUM_WORLD_OBJECTS,
            objects_in_basket_count: 0,
            final_verification_scan_done: false,
            needs_llm_replan: false,
            meta_reasoning_summary:
                "Prefrontal Cortex online (Meta Muse + SAM-3.1 Dorsal Visual Stream + Agentic Chaining ready)."
                    .to_string(),
            meta_budget: ProviderBudgetState::new("meta", meta_limit),
            baseten_budget: ProviderBudgetState::new("baseten", baseten_limit),
        }
    }

    /// Generates the 4-stage Pick-and-Place sub-program for a single discovered object `idx` into the Basket.
    fn build_single_object_basket_subprogram(&self, idx: usize) -> Vec<MotorStageCommand> {
        let obj_name = OBJECT_NAMES[idx.min(NUM_WORLD_OBJECTS - 1)];
        let slot_xy = BASKET_SLOT_OFFSETS[idx.min(NUM_WORLD_OBJECTS - 1)];
        vec![
            MotorStageCommand {
                name: format!(
                    "[Agentic #{}] Walk to {}",
                    self.agentic_iteration, obj_name
                ),
                duration_ms: 5800,
                target_object_idx: Some(idx),
                navigate_to_object: true,
                ..Default::default()
            },
            MotorStageCommand {
                name: format!(
                    "[Agentic #{}] Squat, Reach & Grasp {}",
                    self.agentic_iteration, obj_name
                ),
                duration_ms: 1450,
                target_object_idx: Some(idx),
                squat_cmd: 0.72,
                trunk_lean_rad: [0.18, 0.0],
                track_world_object_with_hands: true,
                left_hand_target: (true, [0.38, 0.16, 0.22]),
                right_hand_target: (true, [0.38, -0.16, 0.22]),
                grasp_cmd: 0.92,
                ..Default::default()
            },
            MotorStageCommand {
                name: format!(
                    "[Agentic #{}] Carry {} to Basket",
                    self.agentic_iteration, obj_name
                ),
                duration_ms: 6200,
                target_object_idx: Some(idx),
                navigate_to_world_xy: Some(slot_xy),
                left_hand_target: (true, [0.38, 0.16, 0.28]),
                right_hand_target: (true, [0.38, -0.16, 0.28]),
                grasp_cmd: 0.92,
                ..Default::default()
            },
            MotorStageCommand {
                name: format!(
                    "[Agentic #{}] Place {} in Basket",
                    self.agentic_iteration, obj_name
                ),
                duration_ms: 950,
                target_object_idx: Some(idx),
                squat_cmd: 0.22,
                left_hand_target: (true, [0.44, 0.18, 0.08]),
                right_hand_target: (true, [0.44, -0.18, 0.08]),
                grasp_cmd: 0.0,
                ..Default::default()
            },
        ]
    }

    /// Closed-Loop Autonomous Agentic Chaining, Self-Calling & Stopping Evaluator!
    /// Called at the start of an agentic run and automatically whenever a sub-task stage list finishes.
    /// 1. Counts how many objects are currently inside the Basket (`objects_in_basket_count`).
    /// 2. If `< NUM_WORLD_OBJECTS` are in the basket:
    ///    - Selects the nearest uncollected object that is currently visible (`in_fov || ever_seen`).
    ///    - If found: increments `agentic_iteration`, sets `needs_llm_replan = true` (self-calling the planner),
    ///      and dispatches the 4-stage Pick-and-Place sub-program for that object.
    ///    - If NO uncollected object has been seen yet (`!in_fov && !ever_seen`): dispatches an active
    ///      `"Agentic Visual Search: Look Around Room (SAM-3.1)"` stage (`visual_search_any_uncollected = true`)
    ///      that rotates the humanoid's 116° retinal camera until a new uncollected object enters `in_fov`!
    /// 3. If all `4/4` objects are in the basket (`objects_in_basket_count == NUM_WORLD_OBJECTS`):
    ///    - If `!final_verification_scan_done`: dispatches one `"Final 360° Visual Verification Scan"` stage
    ///      to look around the room and confirm zero objects remain outside the basket.
    ///    - Once that verification scan finishes: **STOPS THE AGENTIC RUN** (`agentic_loop_active = false`,
    ///      `sequence_active = false`)!
    pub fn advance_autonomous_agentic_loop(
        &mut self,
        bus: &NeuralBodyBus,
        body_3d: &mut FloatingBase3DBody,
    ) {
        self.objects_in_basket_count = count_objects_in_basket(body_3d);

        // Find uncollected targeted objects (`self.agentic_target_mask[idx] && !is_object_in_basket`)
        let mut nearest_visible_uncollected: Option<(usize, f32)> = None;
        let mut any_uncollected_remaining = false;

        for idx in 0..NUM_WORLD_OBJECTS {
            if self.agentic_target_mask[idx] && !is_object_in_basket(body_3d, idx) {
                any_uncollected_remaining = true;
                let obj = &body_3d.world_objects[idx];
                if obj.in_fov || obj.ever_seen {
                    let dist = (obj.pos.x - body_3d.com_pos.x).hypot(obj.pos.y - body_3d.com_pos.y);
                    // Prioritize currently in-FOV objects slightly so the humanoid picks what is right in front of it
                    let score = if obj.in_fov { dist * 0.85 } else { dist };
                    if nearest_visible_uncollected.map_or(true, |(_, best_d)| score < best_d) {
                        nearest_visible_uncollected = Some((idx, score));
                    }
                }
            }
        }

        if any_uncollected_remaining {
            self.agentic_iteration += 1;
            self.needs_llm_replan = true;
            let _ = self
                .meta_budget
                .record_call(95, 36, 31.0, "sam-3.1", false);

            if let Some((next_idx, _)) = nearest_visible_uncollected {
                body_3d.active_object_idx = next_idx;
                let obj_name = OBJECT_NAMES[next_idx];
                self.stages = self.build_single_object_basket_subprogram(next_idx);
                self.current_stage_idx = 0;
                self.stage_elapsed_ms = 0;
                self.sequence_active = true;
                self.meta_reasoning_summary = format!(
                    "[Agentic Self-Call #{} | Basket: {}/{}] Visual target locked on {} -> Executing 4-stage Pick & Place into Basket",
                    self.agentic_iteration,
                    self.objects_in_basket_count,
                    self.agentic_target_total,
                    obj_name
                );
            } else {
                // Uncollected objects remain in the room, but none have entered the 116° retinal FOV yet!
                // Dispatch an active visual room scan to look around and discover them!
                self.stages = vec![MotorStageCommand {
                    name: format!(
                        "[Agentic #{}] Visual Search: Look Around Room (SAM-3.1)",
                        self.agentic_iteration
                    ),
                    duration_ms: 2600,
                    visual_search_any_uncollected: true,
                    ..Default::default()
                }];
                self.current_stage_idx = 0;
                self.stage_elapsed_ms = 0;
                self.sequence_active = true;
                self.meta_reasoning_summary = format!(
                    "[Agentic Self-Call #{} | Basket: {}/{}] No uncollected targets in current FOV -> Rotating head/body to scan room with SAM-3.1",
                    self.agentic_iteration,
                    self.objects_in_basket_count,
                    self.agentic_target_total
                );
            }
        } else if !self.final_verification_scan_done {
            // All targeted objects are in the basket! Perform one final 360° visual scan to verify room is clean before stopping!
            self.final_verification_scan_done = true;
            self.agentic_iteration += 1;
            self.needs_llm_replan = true;
            let _ = self
                .meta_budget
                .record_call(80, 28, 27.0, "sam-3.1", false);
            self.stages = vec![MotorStageCommand {
                name: format!(
                    "[Agentic #{}] Final 360° Visual Verification Scan (Confirm Room Clean)",
                    self.agentic_iteration
                ),
                duration_ms: 1450,
                yaw_cmd: 0.88,
                vx_cmd: 0.32,
                ..Default::default()
            }];
            self.current_stage_idx = 0;
            self.stage_elapsed_ms = 0;
            self.sequence_active = true;
            self.meta_reasoning_summary = format!(
                "[Agentic Self-Call #{} | Basket: {}/{}] All targeted objects placed in Basket -> Running Final 360° Visual Verification Scan before stopping",
                self.agentic_iteration,
                self.objects_in_basket_count,
                self.agentic_target_total
            );
        } else {
            // Final verification scan completed and all targeted objects are confirmed in the Basket!
            // Autonomously STOP the agentic run!
            self.agentic_loop_active = false;
            self.sequence_active = false;
            self.stages.clear();
            self.current_stage_idx = 0;
            self.stage_elapsed_ms = 0;
            bus.intent_vector[0].store(0.0);
            bus.intent_vector[1].store(0.0);
            bus.intent_vector[2].store(0.0);
            bus.intent_vector[4].store(0.0);
            bus.intent_vector[5].store(0.0);
            bus.intent_vector[6].store(0.0);
            bus.intent_vector[7].store(0.0);
            bus.ppc_target_l_foot_rel[0].store(0.0);
            bus.ppc_target_r_foot_rel[0].store(0.0);
            bus.ppc_target_trunk_lean[0].store(0.0);
            bus.ppc_target_trunk_lean[1].store(0.0);
            self.meta_reasoning_summary = format!(
                "[Agentic Run Complete & Halted | {} iterations] {}/{} targeted objects placed in Basket & verified via 360° visual scan. Autonomous stop triggered.",
                self.agentic_iteration,
                self.objects_in_basket_count,
                self.agentic_target_total
            );
        }
    }

    /// Compiles any natural-language user prompt into a multi-stage `Vec<MotorStageCommand>` or activates
    /// the **Closed-Loop Autonomous Agentic Chaining & Stopping Loop** for multi-object collection goals.
    pub fn compile_natural_language_prompt(&mut self, prompt: &str) -> Vec<MotorStageCommand> {
        let cleaned = prompt.trim();
        if cleaned.is_empty() {
            return Vec::new();
        }
        self.active_prompt = cleaned.to_string();
        self.used_fallback_only = false;
        self.agentic_loop_active = false;
        self.final_verification_scan_done = false;
        self.agentic_iteration = 0;
        self.agentic_target_mask = [true; NUM_WORLD_OBJECTS];
        self.agentic_target_total = NUM_WORLD_OBJECTS;

        let full_lower = cleaned.to_lowercase();

        // Check for Autonomous Agentic Multi-Object Basket Cleaning / Selective Collection Goal:
        // Supports:
        // 1) Full-room collection: "clean all the things in the basket", "collect everything"
        // 2) Selective subset collection: "only put the blue ball and green cylinder in the basket"
        // 3) Negation-filtered collection: "clean everything except the red box into the basket"
        let has_collect_verb = full_lower.contains("pick up")
            || full_lower.contains("pickup")
            || full_lower.contains("collect")
            || full_lower.contains("gather")
            || full_lower.contains("clean")
            || full_lower.contains("put")
            || full_lower.contains("place")
            || full_lower.contains("bring");
        let has_basket_or_plural = full_lower.contains("things")
            || full_lower.contains("objects")
            || full_lower.contains("items")
            || full_lower.contains("everything")
            || full_lower.contains("all")
            || full_lower.contains("basket")
            || full_lower.contains("room");

        let mentions_obj = [
            full_lower.contains("red") || full_lower.contains("box"),
            full_lower.contains("blue") || full_lower.contains("ball") || full_lower.contains("sphere"),
            full_lower.contains("green") || full_lower.contains("cylinder") || full_lower.contains("can"),
            full_lower.contains("gold") || full_lower.contains("yellow") || full_lower.contains("crate"),
        ];
        let mentioned_count = mentions_obj.iter().filter(|&&b| b).count();
        let has_negation = full_lower.contains("except")
            || full_lower.contains("without")
            || full_lower.contains("not the")
            || full_lower.contains("leave the");
        let is_chained_sequence = full_lower.contains("then ")
            || full_lower.contains("; ")
            || full_lower.contains("after that");

        let is_multi_object_agentic = !is_chained_sequence
            && has_collect_verb
            && has_basket_or_plural
            && (mentioned_count == 0 || mentioned_count >= 2 || has_negation);

        if is_multi_object_agentic {
            let mask = if has_negation && mentioned_count > 0 {
                [
                    !mentions_obj[0],
                    !mentions_obj[1],
                    !mentions_obj[2],
                    !mentions_obj[3],
                ]
            } else if mentioned_count >= 2 {
                mentions_obj
            } else {
                [true; NUM_WORLD_OBJECTS]
            };
            let target_total = mask.iter().filter(|&&b| b).count().max(1);
            self.agentic_target_mask = mask;
            self.agentic_target_total = target_total;
            self.agentic_loop_active = true;
            self.agentic_goal_description = cleaned.to_string();
            self.agentic_iteration = 1;
            self.needs_llm_replan = true;
            let _ = self
                .meta_budget
                .record_call(110, 42, 34.0, "sam-3.1", false);

            // Bootstrap with the first targeted object in frontal FOV (`0, 1, 3`), or Visual Search if only `2` is targeted
            let first_visible = [0_usize, 1, 3]
                .into_iter()
                .find(|&i| self.agentic_target_mask[i]);
            let stages = if let Some(first_idx) = first_visible {
                self.build_single_object_basket_subprogram(first_idx)
            } else {
                vec![MotorStageCommand {
                    name: "[Agentic #1] Visual Search: Look Around Room (SAM-3.1)".to_string(),
                    duration_ms: 2600,
                    visual_search_any_uncollected: true,
                    ..Default::default()
                }]
            };
            self.stages = stages.clone();
            self.current_stage_idx = 0;
            self.stage_elapsed_ms = 0;
            self.sequence_active = true;
            self.meta_reasoning_summary = format!(
                "[Agentic Loop Armed | {}/{} Targets] Autonomous visual-motor loop active for \"{}\": will self-call, scan room, and auto-stop when {} targeted objects are in Basket",
                0,
                self.agentic_target_total,
                self.active_prompt,
                self.agentic_target_total
            );
            return stages;
        }

        // Normalize sequential connectors into `|` stage delimiters
        let normalized = cleaned
            .replace(" and then ", " | ")
            .replace(", then ", " | ")
            .replace(" then ", " | ")
            .replace(" after that ", " | ")
            .replace(" finally ", " | ")
            .replace(" and point ", " | point ")
            .replace(" and pick ", " | pick ")
            .replace(" and grab ", " | grab ")
            .replace(" and put ", " | put ")
            .replace(" and place ", " | place ")
            .replace(" and drop ", " | drop ")
            .replace(" and wave", " | wave")
            .replace(" and bow", " | bow")
            .replace(';', " | ")
            .replace(',', " | ");

        let mut stages = Vec::new();
        let mut carrying_object = false;
        let mut active_obj_idx: usize = resolve_target_object_idx(&full_lower).unwrap_or(0);
        let mut any_recognized = false;

        for raw_clause in normalized.split('|') {
            let clause = raw_clause.trim();
            if clause.is_empty() {
                continue;
            }
            let lower = clause.to_lowercase();
            if let Some(idx) = resolve_target_object_idx(&lower) {
                active_obj_idx = idx;
            }
            let obj_name = OBJECT_NAMES[active_obj_idx];

            let mut st = MotorStageCommand {
                name: clause.chars().take(42).collect(),
                duration_ms: 1800,
                target_object_idx: Some(active_obj_idx),
                grasp_cmd: if carrying_object { 0.88 } else { 0.0 },
                left_hand_target: if carrying_object {
                    (true, [0.38, 0.16, 0.22])
                } else {
                    (false, [0.38, 0.24, 0.25])
                },
                right_hand_target: if carrying_object {
                    (true, [0.38, -0.16, 0.22])
                } else {
                    (false, [0.38, -0.24, 0.25])
                },
                ..Default::default()
            };

            let coords = extract_numbers(&lower);

            // 0. Check for Visual Search / "Go and Find" / "Search for" / "Look for" / "Locate" / "Where is"
            if lower.contains("find")
                || lower.contains("search")
                || lower.contains("look for")
                || lower.contains("locate")
                || lower.contains("where is")
                || lower.contains("spot")
            {
                any_recognized = true;
                let _ = self.meta_budget.record_call(64, 24, 29.0, "sam-3.1", false);
                let mut scan_st = st.clone();
                scan_st.name = format!("SAM-3.1 Visual Search for {}", obj_name);
                scan_st.visual_search_for_object = Some(active_obj_idx);
                scan_st.duration_ms = 2600;
                stages.push(scan_st);

                if lower.contains("go")
                    || lower.contains("walk")
                    || lower.contains("find")
                    || lower.contains("approach")
                {
                    let mut nav_st = st.clone();
                    nav_st.name = format!("Walk to Discovered {}", obj_name);
                    nav_st.navigate_to_object = true;
                    nav_st.duration_ms = 3000;
                    stages.push(nav_st);
                }

                // Point at the discovered object to confirm visual lock!
                let mut pt_st = st.clone();
                pt_st.name = format!("Point 3D Ray at {}", obj_name);
                pt_st.point_at_object = Some(active_obj_idx);
                pt_st.duration_ms = 1800;
                stages.push(pt_st);
                continue;
            }

            // 0B. Check for Pointing at an Object ("point at the box", "point to the blue ball", "show me the cylinder")
            if (lower.contains("point")
                || lower.contains("show me")
                || lower.contains("aim at")
                || lower.contains("indicate"))
                && (resolve_target_object_idx(&lower).is_some()
                    || lower.contains("it")
                    || lower.contains("that")
                    || coords.is_empty())
            {
                any_recognized = true;
                let already_pointed = stages.last().map_or(false, |prev| {
                    prev.point_at_object == Some(active_obj_idx)
                });
                if already_pointed {
                    continue;
                }
                let _ = self.meta_budget.record_call(48, 18, 24.0, "sam-3.1", false);
                let mut scan_st = st.clone();
                scan_st.name = format!("SAM-3.1 Gaze Lock on {}", obj_name);
                scan_st.visual_search_for_object = Some(active_obj_idx);
                scan_st.duration_ms = 2000;
                stages.push(scan_st);

                st.name = format!("Point 3D Ray at {}", obj_name);
                st.point_at_object = Some(active_obj_idx);
                st.duration_ms = 1800;
                stages.push(st);
                continue;
            }

            // 1. Check for Pick Up / Grab / Grasp Object
            if lower.contains("pick up")
                || lower.contains("pickup")
                || lower.contains("grab")
                || lower.contains("grasp")
                || lower.contains("take the")
            {
                any_recognized = true;
                let _ = self.meta_budget.record_call(56, 22, 26.0, "sam-3.1", false);
                let already_navigated = stages.last().map_or(false, |prev| {
                    prev.navigate_to_object && prev.target_object_idx == Some(active_obj_idx)
                });
                if !already_navigated {
                    if active_obj_idx >= 2 {
                        let mut scan = st.clone();
                        scan.name = format!("SAM-3.1 Locate {}", obj_name);
                        scan.visual_search_for_object = Some(active_obj_idx);
                        scan.duration_ms = 2000;
                        stages.push(scan);
                    }
                    let mut approach = st.clone();
                    approach.name = format!("Walk to {}", obj_name);
                    approach.navigate_to_object = true;
                    approach.duration_ms = 5800;
                    stages.push(approach);
                }

                st.name = format!("Squat, Reach & Grasp {}", obj_name);
                st.squat_cmd = 0.72;
                st.trunk_lean_rad = [0.18, 0.0];
                st.track_world_object_with_hands = true;
                st.left_hand_target.0 = true;
                st.right_hand_target.0 = true;
                st.grasp_cmd = 0.92;
                st.duration_ms = 1500;
                carrying_object = true;
                stages.push(st);
                continue;
            }

            // 2. Check for Drop / Release / Place / Put Down Object (including into the Basket / Goal Table!)
            if lower.contains("drop")
                || lower.contains("release")
                || lower.contains("put down")
                || lower.contains("put it")
                || lower.contains("place")
                || lower.contains("set down")
                || lower.contains("in the basket")
                || lower.contains("on the table")
            {
                any_recognized = true;
                if lower.contains("basket")
                    || lower.contains("table")
                    || lower.contains("goal")
                    || lower.contains("zone")
                {
                    let slot_xy = BASKET_SLOT_OFFSETS[active_obj_idx.min(NUM_WORLD_OBJECTS - 1)];
                    let mut carry_to_basket = st.clone();
                    carry_to_basket.name = format!("Carry {} to Basket", obj_name);
                    carry_to_basket.navigate_to_world_xy = Some(slot_xy);
                    carry_to_basket.grasp_cmd = 0.92;
                    carry_to_basket.left_hand_target = (true, [0.38, 0.16, 0.28]);
                    carry_to_basket.right_hand_target = (true, [0.38, -0.16, 0.28]);
                    carry_to_basket.duration_ms = 6200;
                    stages.push(carry_to_basket);
                }
                st.name = format!("Lower & Release {}", obj_name);
                st.squat_cmd = if lower.contains("floor") || lower.contains("down") {
                    0.65
                } else {
                    0.22
                };
                st.left_hand_target = (true, [0.44, 0.18, 0.08]);
                st.right_hand_target = (true, [0.44, -0.18, 0.08]);
                st.grasp_cmd = 0.0;
                st.duration_ms = 1100;
                carrying_object = false;
                stages.push(st);
                continue;
            }

            // 3. Check for Unipedal Leg Kick / High Knee Lift
            if lower.contains("kick")
                || lower.contains("lift left leg")
                || lower.contains("lift right leg")
                || lower.contains("one leg")
                || lower.contains("high knee")
            {
                any_recognized = true;
                let use_left = lower.contains("left");
                let target_xyz = if coords.len() >= 3 {
                    [coords[0], coords[1], coords[2]]
                } else if lower.contains("high") {
                    [0.52, if use_left { 0.14 } else { -0.14 }, -0.25]
                } else {
                    [0.48, if use_left { 0.14 } else { -0.14 }, -0.38]
                };
                if use_left {
                    st.name = "Unipedal Left-Leg Kick & APA Balance".to_string();
                    st.left_foot_target = (true, target_xyz);
                    st.trunk_lean_rad = [-0.10, -0.08];
                } else {
                    st.name = "Unipedal Right-Leg Kick & APA Balance".to_string();
                    st.right_foot_target = (true, target_xyz);
                    st.trunk_lean_rad = [-0.10, 0.08];
                }
                st.left_hand_target = (true, [0.22, 0.48, 0.25]);
                st.right_hand_target = (true, [0.22, -0.48, 0.25]);
                st.duration_ms = 1600;
                stages.push(st);
                continue;
            }

            // 3B. Check for Jump / Hop / Leap / Dance
            if lower.contains("jump") || lower.contains("hop") || lower.contains("leap") {
                any_recognized = true;
                let mut crouch = st.clone();
                crouch.name = "Countermovement Squat Prep".to_string();
                crouch.squat_cmd = 0.80;
                crouch.left_hand_target = (true, [0.10, 0.28, -0.12]);
                crouch.right_hand_target = (true, [0.10, -0.28, -0.12]);
                crouch.duration_ms = 750;
                stages.push(crouch);

                st.name = "Explosive Extension & Overhead Reach".to_string();
                st.squat_cmd = 0.0;
                st.left_hand_target = (true, [0.42, 0.24, 0.86]);
                st.right_hand_target = (true, [0.42, -0.24, 0.86]);
                st.duration_ms = 1250;
                stages.push(st);
                continue;
            }
            if lower.contains("dance") || lower.contains("boogie") || lower.contains("groove") {
                any_recognized = true;
                let mut d1 = st.clone();
                d1.name = "Dance Step Left + Dual Wave".to_string();
                d1.vy_cmd = 0.88;
                d1.wave_hands = true;
                d1.left_hand_target = (true, [0.28, 0.46, 0.72]);
                d1.right_hand_target = (true, [0.28, -0.46, 0.72]);
                d1.duration_ms = 1600;
                stages.push(d1);

                let mut d2 = st.clone();
                d2.name = "Dance Squat Bounce + T-Pose".to_string();
                d2.squat_cmd = 0.65;
                d2.left_hand_target = (true, [0.10, 0.66, 0.40]);
                d2.right_hand_target = (true, [0.10, -0.66, 0.40]);
                d2.duration_ms = 1100;
                stages.push(d2);

                st.name = "Dance Step Right + Overhead Celebrate".to_string();
                st.vy_cmd = -0.88;
                st.squat_cmd = 0.0;
                st.left_hand_target = (true, [0.40, 0.26, 0.85]);
                st.right_hand_target = (true, [0.40, -0.26, 0.85]);
                st.duration_ms = 1600;
                stages.push(st);
                continue;
            }

            // 4. Locomotion: Sideways / Lateral / Walk / Run / Step / Go to (x,y) or Object / Backward
            let is_sideways = lower.contains("sideways")
                || lower.contains("lateral")
                || lower.contains("sidestep")
                || lower.contains("strafe")
                || lower.contains("crab")
                || lower.contains("step left")
                || lower.contains("step right")
                || lower.contains("walk left")
                || lower.contains("walk right")
                || lower.contains("move left")
                || lower.contains("move right")
                || lower.contains("slide left")
                || lower.contains("slide right");

            if is_sideways {
                any_recognized = true;
                let has_left = lower.contains("left");
                let has_right = lower.contains("right");
                if !has_left && !has_right {
                    let mut left_st = st.clone();
                    left_st.name = "Walk Sideways Left (+Y)".to_string();
                    left_st.vy_cmd = 0.90;
                    left_st.duration_ms = 1900;
                    stages.push(left_st);

                    st.name = "Walk Sideways Right (-Y)".to_string();
                    st.vy_cmd = -0.90;
                    st.duration_ms = 1900;
                } else {
                    let dir = if has_right && !has_left { -0.90 } else { 0.90 };
                    st.name = if dir > 0.0 {
                        "Walk Sideways Left (+Y)".to_string()
                    } else {
                        "Walk Sideways Right (-Y)".to_string()
                    };
                    st.vy_cmd = dir;
                    st.duration_ms = 2100;
                }
            } else if lower.contains("backward")
                || lower.contains("back up")
                || lower.contains("walk back")
                || lower.contains("move back")
                || lower.contains("step back")
                || lower.contains("retreat")
            {
                any_recognized = true;
                st.name = "Walk Backward (-X)".to_string();
                st.vx_cmd = -0.85;
                st.duration_ms = 2000;
            } else if lower.contains("basket") || lower.contains("table") || lower.contains("goal") || lower.contains("drop zone") {
                any_recognized = true;
                st.name = "Navigate to Basket / Goal Zone (1.95, 0.80)".to_string();
                st.navigate_to_world_xy = Some([1.95, 0.80]);
                st.duration_ms = 3200;
            } else if resolve_target_object_idx(&lower).is_some()
                && (lower.contains("walk")
                    || lower.contains("go")
                    || lower.contains("approach")
                    || lower.contains("move")
                    || lower.contains("to the"))
            {
                any_recognized = true;
                if active_obj_idx >= 2 {
                    let mut scan_st = st.clone();
                    scan_st.name = format!("SAM-3.1 Visual Lock on {}", obj_name);
                    scan_st.visual_search_for_object = Some(active_obj_idx);
                    scan_st.duration_ms = 1800;
                    stages.push(scan_st);
                }

                st.name = format!("Navigate to {}", obj_name);
                st.navigate_to_object = true;
                st.duration_ms = 3200;
            } else if (lower.contains("walk")
                || lower.contains("go")
                || lower.contains("move")
                || lower.contains("carry")
                || lower.contains("run"))
                && coords.len() >= 2
            {
                any_recognized = true;
                st.name = format!("Navigate to ({:.1}, {:.1})", coords[0], coords[1]);
                st.navigate_to_world_xy = Some([coords[0], coords[1]]);
                st.duration_ms = 3200;
            } else if lower.contains("run") || lower.contains("sprint") || lower.contains("fast") || lower.contains("jog") {
                any_recognized = true;
                st.name = "Fast Run Forward".to_string();
                st.vx_cmd = 1.0;
                st.duration_ms = 2200;
            } else if lower.contains("walk")
                || lower.contains("forward")
                || lower.contains("advance")
                || lower.contains("carry")
                || lower.contains("step forward")
            {
                any_recognized = true;
                st.name = "Walk Forward".to_string();
                st.vx_cmd = 0.85;
                st.duration_ms = 2200;
            }

            // 5. Turning / Spinning / Visual Room Scan
            if lower.contains("turn left") || lower.contains("steer left") || lower.contains("rotate left") || lower.contains("look left") {
                any_recognized = true;
                st.yaw_cmd = 0.80;
                if st.vx_cmd.abs() < 0.1 && st.vy_cmd.abs() < 0.1 {
                    st.vx_cmd = 0.45;
                }
            } else if lower.contains("turn right") || lower.contains("steer right") || lower.contains("rotate right") || lower.contains("look right") {
                any_recognized = true;
                st.yaw_cmd = -0.80;
                if st.vx_cmd.abs() < 0.1 && st.vy_cmd.abs() < 0.1 {
                    st.vx_cmd = 0.45;
                }
            } else if lower.contains("spin")
                || lower.contains("turn around")
                || lower.contains("circle")
                || lower.contains("look around")
                || lower.contains("scan")
            {
                any_recognized = true;
                st.name = "360° Visual Room Scan / Turn".to_string();
                st.yaw_cmd = 0.90;
                st.vx_cmd = 0.45;
                st.duration_ms = 2600;
            }

            // 6. Squat / Crouch / Stand Up / Rise / Sit
            if lower.contains("squat") || lower.contains("crouch") || lower.contains("duck") || lower.contains("kneel") || lower.contains("sit") {
                any_recognized = true;
                st.squat_cmd = if lower.contains("half") || lower.contains("slight") {
                    0.50
                } else {
                    0.92
                };
                st.vx_cmd = 0.0;
                st.duration_ms = 1600;
            } else if lower.contains("stand") || lower.contains("rise") || lower.contains("get up") || lower.contains("upright") {
                any_recognized = true;
                st.squat_cmd = 0.0;
                st.trunk_lean_rad = [0.0, 0.0];
                st.duration_ms = 1200;
            }

            // 7. Bow / Lean
            if lower.contains("bow") || lower.contains("lean forward") || lower.contains("bend forward") || lower.contains("nod") {
                any_recognized = true;
                st.trunk_lean_rad[0] = 0.42;
                st.duration_ms = 1500;
            } else if lower.contains("lean back") {
                any_recognized = true;
                st.trunk_lean_rad[0] = -0.25;
                st.duration_ms = 1500;
            } else if lower.contains("lean left") {
                any_recognized = true;
                st.trunk_lean_rad[1] = 0.28;
            } else if lower.contains("lean right") {
                any_recognized = true;
                st.trunk_lean_rad[1] = -0.28;
            }

            // 8. 3D Arm Reaching / Waving / T-Pose / Overhead Lift / Salute / Clap / Push
            if lower.contains("wave") || lower.contains("greet") || lower.contains("hello") || lower.contains("hi ") {
                any_recognized = true;
                let use_left = lower.contains("left") && !lower.contains("right");
                let use_both = lower.contains("both") || (!lower.contains("left") && !lower.contains("right"));
                st.wave_hands = true;
                if use_left || use_both {
                    st.left_hand_target = (true, [0.25, 0.45, 0.68]);
                }
                if !use_left || use_both {
                    st.right_hand_target = (true, [0.25, -0.45, 0.68]);
                }
                st.duration_ms = 2000;
            } else if lower.contains("salute") {
                any_recognized = true;
                st.name = "Right-Hand Salute".to_string();
                st.right_hand_target = (true, [0.24, -0.18, 0.62]);
                st.duration_ms = 1600;
            } else if lower.contains("clap") || lower.contains("applaud") {
                any_recognized = true;
                st.name = "Bimanual Midline Clap".to_string();
                st.wave_hands = true;
                st.left_hand_target = (true, [0.42, 0.06, 0.36]);
                st.right_hand_target = (true, [0.42, -0.06, 0.36]);
                st.duration_ms = 1800;
            } else if lower.contains("t-pose") || lower.contains("spread") || lower.contains("out to the side") || lower.contains("wings") {
                any_recognized = true;
                st.left_hand_target = (true, [0.05, 0.68, 0.38]);
                st.right_hand_target = (true, [0.05, -0.68, 0.38]);
                st.duration_ms = 1600;
            } else if lower.contains("overhead") || lower.contains("high") || lower.contains("hands up") || lower.contains("raise") || lower.contains("lift") || lower.contains("celebrate") || lower.contains("sky") {
                any_recognized = true;
                let use_left = lower.contains("left") && !lower.contains("right");
                let use_right = lower.contains("right") && !lower.contains("left");
                if use_left {
                    st.left_hand_target = (true, [0.42, 0.24, 0.85]);
                } else if use_right {
                    st.right_hand_target = (true, [0.42, -0.24, 0.85]);
                } else {
                    st.left_hand_target = (true, [0.42, 0.24, 0.85]);
                    st.right_hand_target = (true, [0.42, -0.24, 0.85]);
                }
                st.duration_ms = 1600;
            } else if lower.contains("reach") || lower.contains("point") || lower.contains("stretch") || lower.contains("arm") || lower.contains("hand") || lower.contains("push") || lower.contains("punch") {
                any_recognized = true;
                let target = if coords.len() >= 3 {
                    [
                        coords[0].clamp(-0.2, 0.65),
                        coords[1].clamp(-0.7, 0.7),
                        coords[2].clamp(-0.3, 0.85),
                    ]
                } else if lower.contains("down") || lower.contains("low") {
                    [0.42, 0.20, -0.15]
                } else if lower.contains("side") {
                    [0.15, 0.62, 0.35]
                } else {
                    [0.54, 0.20, 0.40]
                };
                if lower.contains("left") && !lower.contains("right") {
                    st.left_hand_target = (true, [target[0], target[1].abs(), target[2]]);
                } else if lower.contains("right") && !lower.contains("left") {
                    st.right_hand_target = (true, [target[0], -target[1].abs(), target[2]]);
                } else {
                    st.left_hand_target = (true, [target[0], target[1].abs(), target[2]]);
                    st.right_hand_target = (true, [target[0], -target[1].abs(), target[2]]);
                }
                st.duration_ms = 1600;
            }

            if lower.contains("second") || lower.contains("sec") {
                if let Some(&sec) = coords.last() {
                    if (0.3..=15.0).contains(&sec) {
                        st.duration_ms = (sec * 1000.0) as u32;
                    }
                }
            }

            stages.push(st);
        }

        if !any_recognized {
            self.used_fallback_only = true;
        }
        self.stages = stages.clone();
        self.current_stage_idx = 0;
        self.stage_elapsed_ms = 0;
        self.sequence_active = !self.stages.is_empty();
        self.meta_reasoning_summary = format!(
            "Compiled {} cortical task-space stages (Active Target: {}) from prompt: \"{}\"",
            self.stages.len(),
            OBJECT_NAMES[active_obj_idx],
            self.active_prompt
        );
        stages
    }

    /// Steps the active Prefrontal Motor Stage at 1,000 Hz, writes continuous 3D task-space
    /// targets onto `NeuralBodyBus`, and runs the **Autonomous Agentic Chaining & Stopping Loop**!
    pub fn step_sequencer_1ms(
        &mut self,
        bus: &NeuralBodyBus,
        body_3d: &mut FloatingBase3DBody,
        _plant: &HumanoidBiomechanicalPlant,
    ) {
        self.objects_in_basket_count = count_objects_in_basket(body_3d);

        if !self.sequence_active || self.stages.is_empty() {
            return;
        }
        if self.current_stage_idx >= self.stages.len() {
            // If an Autonomous Agentic Loop is active, self-call `advance_autonomous_agentic_loop`
            // to inspect the visual room & basket state, chain the next sub-goal or verification scan,
            // or stop the agentic run once all objects are verified in the basket!
            if self.agentic_loop_active {
                self.advance_autonomous_agentic_loop(bus, body_3d);
                return;
            }

            self.sequence_active = false;
            bus.intent_vector[0].store(0.0);
            bus.intent_vector[1].store(0.0);
            bus.intent_vector[2].store(0.0);
            bus.intent_vector[6].store(0.0);
            bus.ppc_target_l_foot_rel[0].store(0.0);
            bus.ppc_target_r_foot_rel[0].store(0.0);
            bus.ppc_target_trunk_lean[0].store(0.0);
            bus.ppc_target_trunk_lean[1].store(0.0);
            return;
        }

        self.stage_elapsed_ms += 1;
        let st = &self.stages[self.current_stage_idx];

        if let Some(obj_idx) = st.target_object_idx {
            body_3d.active_object_idx = obj_idx.min(NUM_WORLD_OBJECTS - 1);
        }
        let active_idx = body_3d.active_object_idx.min(NUM_WORLD_OBJECTS - 1);
        let target_obj = body_3d.world_objects[active_idx];

        // 0A. Check Agentic Open-Ended Visual Room Scan (`visual_search_any_uncollected`)
        if st.visual_search_any_uncollected {
            let mut newly_spotted: Option<usize> = None;
            for i in 0..NUM_WORLD_OBJECTS {
                if self.agentic_target_mask[i]
                    && !is_object_in_basket(body_3d, i)
                    && body_3d.world_objects[i].in_fov
                {
                    newly_spotted = Some(i);
                    break;
                }
            }
            if let Some(found_idx) = newly_spotted {
                body_3d.active_object_idx = found_idx;
                body_3d.world_objects[found_idx].ever_seen = true;
                bus.intent_vector[0].store(0.0);
                bus.intent_vector[2].store(0.0);
                self.current_stage_idx += 1;
                self.stage_elapsed_ms = 0;
                return;
            } else {
                bus.intent_vector[2].store(0.92);
                bus.intent_vector[0].store(0.0);
                bus.intent_vector[6].store(0.0);
                bus.intent_vector[1].store(0.0);
                if self.stage_elapsed_ms >= st.duration_ms {
                    // Full 360° room sweep completed; mark remaining targeted uncollected objects as observed
                    for i in 0..NUM_WORLD_OBJECTS {
                        if self.agentic_target_mask[i] && !is_object_in_basket(body_3d, i) {
                            body_3d.world_objects[i].ever_seen = true;
                        }
                    }
                    self.current_stage_idx += 1;
                    self.stage_elapsed_ms = 0;
                }
                return;
            }
        }

        // 0B. Check Active 3D Visual Search for a specific object (`SAM-3.1` + Dorsal Stream Binocular Frustum Scan)
        if let Some(search_idx) = st.visual_search_for_object {
            let idx = search_idx.min(NUM_WORLD_OBJECTS - 1);
            body_3d.active_object_idx = idx;
            let obj = body_3d.world_objects[idx];
            let dx = obj.pos.x - body_3d.com_pos.x;
            let dy = obj.pos.y - body_3d.com_pos.y;
            let desired_yaw = dy.atan2(dx);
            let mut yaw_err = desired_yaw - body_3d.yaw_rad;
            while yaw_err > std::f32::consts::PI {
                yaw_err -= 2.0 * std::f32::consts::PI;
            }
            while yaw_err < -std::f32::consts::PI {
                yaw_err += 2.0 * std::f32::consts::PI;
            }
            if (obj.in_fov && yaw_err.abs() < 0.45 && self.stage_elapsed_ms >= 180)
                || dx.hypot(dy) <= 0.48
            {
                body_3d.world_objects[idx].ever_seen = true;
                bus.intent_vector[0].store(0.0);
                bus.intent_vector[2].store(0.0);
                self.current_stage_idx += 1;
                self.stage_elapsed_ms = 0;
                return;
            } else {
                let scan_dir = if yaw_err >= 0.0 { 0.92 } else { -0.92 };
                bus.intent_vector[2].store(scan_dir);
                bus.intent_vector[0].store(0.0);
                bus.intent_vector[6].store(0.0);
                bus.intent_vector[1].store(0.0);
                if self.stage_elapsed_ms >= st.duration_ms {
                    body_3d.world_objects[idx].ever_seen = true;
                    self.current_stage_idx += 1;
                    self.stage_elapsed_ms = 0;
                }
                return;
            }
        }

        // 1. Check 2D Waypoint / 3D Object Navigation vs Direct Velocity Command
        let mut waypoint_reached = false;
        let nav_target = if st.navigate_to_object {
            Some([target_obj.pos.x, target_obj.pos.y])
        } else {
            st.navigate_to_world_xy
        };

        if let Some([tx, ty]) = nav_target {
            let dx = tx - body_3d.com_pos.x;
            let dy = ty - body_3d.com_pos.y;
            let dist = dx.hypot(dy);
            let stop_dist = if st.navigate_to_object { 0.38 } else { 0.30 };
            if dist <= stop_dist {
                waypoint_reached = true;
                bus.intent_vector[0].store(0.0);
                bus.intent_vector[2].store(0.0);
            } else {
                let desired_yaw = dy.atan2(dx);
                let mut yaw_err = desired_yaw - body_3d.yaw_rad;
                while yaw_err > std::f32::consts::PI {
                    yaw_err -= 2.0 * std::f32::consts::PI;
                }
                while yaw_err < -std::f32::consts::PI {
                    yaw_err += 2.0 * std::f32::consts::PI;
                }
                // Critically Damped PD Heading Control + Pivot-Then-Stride Law:
                // - Pivots crisply in place (`2.4x` rotational authority) when `|yaw_err| > 0.42 rad` (`~24°`)
                // - Smoothly strides forward along the target vector once aligned within `24°`!
                let pd_yaw = (yaw_err * 1.85 - 0.12 * body_3d.yaw_vel_rad_s).clamp(-0.98, 0.98);
                let vx_nav = if yaw_err.abs() > 0.42 {
                    0.0
                } else {
                    (0.92 * yaw_err.cos()).clamp(0.55, 0.92)
                };
                bus.intent_vector[2].store(pd_yaw);
                bus.intent_vector[0].store(vx_nav);
            }
        } else if let Some(pt_idx) = st.point_at_object {
            let obj = body_3d.world_objects[pt_idx.min(NUM_WORLD_OBJECTS - 1)];
            let dx = obj.pos.x - body_3d.com_pos.x;
            let dy = obj.pos.y - body_3d.com_pos.y;
            let desired_yaw = dy.atan2(dx);
            let mut yaw_err = desired_yaw - body_3d.yaw_rad;
            while yaw_err > std::f32::consts::PI {
                yaw_err -= 2.0 * std::f32::consts::PI;
            }
            while yaw_err < -std::f32::consts::PI {
                yaw_err += 2.0 * std::f32::consts::PI;
            }
            if yaw_err.abs() > 0.16 {
                bus.intent_vector[2].store((yaw_err * 1.35).clamp(-0.85, 0.85));
                bus.intent_vector[0].store(0.32);
            } else {
                bus.intent_vector[2].store((yaw_err * 0.65).clamp(-0.35, 0.35));
                bus.intent_vector[0].store(0.0);
            }
        } else {
            bus.intent_vector[0].store(st.vx_cmd);
            bus.intent_vector[2].store(st.yaw_cmd);
        }
        bus.intent_vector[6].store(st.vy_cmd);
        bus.intent_vector[1].store(st.squat_cmd);
        bus.intent_vector[7].store(st.grasp_cmd);
        bus.ppc_target_trunk_lean[0].store(st.trunk_lean_rad[0]);
        bus.ppc_target_trunk_lean[1].store(st.trunk_lean_rad[1]);

        // 2. Compute 3D Parietal Hand Targets (including 3D Object Pointing Ray, World-Object Tracking & Waving!)
        let wave_offset = if st.wave_hands {
            0.22 * ((self.stage_elapsed_ms as f32) * 0.016).sin()
        } else {
            0.0
        };

        if let Some(pt_idx) = st.point_at_object {
            let obj = body_3d.world_objects[pt_idx.min(NUM_WORLD_OBJECTS - 1)];
            let dx_w = obj.pos.x - body_3d.com_pos.x;
            let dy_w = obj.pos.y - body_3d.com_pos.y;
            let dz_w = obj.pos.z - (body_3d.com_pos.z + 0.34);
            let cos_y = body_3d.head_yaw_rad.cos();
            let sin_y = body_3d.head_yaw_rad.sin();
            let rx = dx_w * cos_y + dy_w * sin_y;
            let ry = -dx_w * sin_y + dy_w * cos_y;
            let rz = dz_w;
            let norm = (rx * rx + ry * ry + rz * rz).sqrt().max(0.05);
            let pt_x = (0.56 * rx / norm).clamp(0.24, 0.60);
            let pt_y = (0.56 * ry / norm).clamp(-0.55, 0.55);
            let pt_z = (0.34 + 0.56 * rz / norm).clamp(-0.10, 0.78);

            if ry > 0.12 {
                bus.intent_vector[5].store(0.95);
                bus.intent_vector[4].store(0.0);
                bus.ppc_target_l_hand_rel[0].store(pt_x);
                bus.ppc_target_l_hand_rel[1].store(pt_y.max(0.06));
                bus.ppc_target_l_hand_rel[2].store(pt_z);
            } else {
                bus.intent_vector[4].store(0.95);
                bus.intent_vector[5].store(0.0);
                bus.ppc_target_r_hand_rel[0].store(pt_x);
                bus.ppc_target_r_hand_rel[1].store(pt_y.min(-0.06));
                bus.ppc_target_r_hand_rel[2].store(pt_z);
            }
        } else if st.track_world_object_with_hands {
            let dx_w = target_obj.pos.x - body_3d.com_pos.x;
            let dy_w = target_obj.pos.y - body_3d.com_pos.y;
            let dz_w = target_obj.pos.z - body_3d.com_pos.z;
            // Maintain active visual foveation alignment toward the object while squatting & reaching
            if nav_target.is_none() && st.yaw_cmd.abs() < 0.01 {
                let desired_yaw = dy_w.atan2(dx_w);
                let mut yaw_err = desired_yaw - body_3d.head_yaw_rad;
                while yaw_err > std::f32::consts::PI {
                    yaw_err -= 2.0 * std::f32::consts::PI;
                }
                while yaw_err < -std::f32::consts::PI {
                    yaw_err += 2.0 * std::f32::consts::PI;
                }
                if yaw_err.abs() > 0.05 {
                    bus.intent_vector[2].store((yaw_err * 1.15).clamp(-0.65, 0.65));
                }
            }
            let cos_y = body_3d.head_yaw_rad.cos();
            let sin_y = body_3d.head_yaw_rad.sin();
            let rel_x = (dx_w * cos_y + dy_w * sin_y).clamp(0.18, 0.58);
            let rel_y = (-dx_w * sin_y + dy_w * cos_y).clamp(-0.45, 0.45);
            let rel_z = dz_w.clamp(-0.28, 0.65);

            bus.intent_vector[5].store(0.95);
            bus.intent_vector[4].store(0.95);
            bus.ppc_target_l_hand_rel[0].store(rel_x);
            bus.ppc_target_l_hand_rel[1].store((rel_y + 0.10).max(0.05));
            bus.ppc_target_l_hand_rel[2].store(rel_z);
            bus.ppc_target_r_hand_rel[0].store(rel_x);
            bus.ppc_target_r_hand_rel[1].store((rel_y - 0.10).min(-0.05));
            bus.ppc_target_r_hand_rel[2].store(rel_z);
        } else {
            if st.left_hand_target.0 {
                bus.intent_vector[5].store(0.92);
                bus.ppc_target_l_hand_rel[0].store(st.left_hand_target.1[0]);
                bus.ppc_target_l_hand_rel[1].store(st.left_hand_target.1[1] + wave_offset);
                bus.ppc_target_l_hand_rel[2].store(st.left_hand_target.1[2]);
            } else {
                bus.intent_vector[5].store(0.0);
            }
            if st.right_hand_target.0 {
                bus.intent_vector[4].store(0.92);
                bus.ppc_target_r_hand_rel[0].store(st.right_hand_target.1[0]);
                bus.ppc_target_r_hand_rel[1].store(st.right_hand_target.1[1] - wave_offset);
                bus.ppc_target_r_hand_rel[2].store(st.right_hand_target.1[2]);
            } else {
                bus.intent_vector[4].store(0.0);
            }
        }

        // 3. Unipedal Foot Kick / High-Step Targets
        if st.left_foot_target.0 {
            bus.ppc_target_l_foot_rel[0].store(0.95);
            bus.ppc_target_l_foot_rel[1].store(st.left_foot_target.1[0]);
            bus.ppc_target_l_foot_rel[2].store(st.left_foot_target.1[1]);
            bus.ppc_target_l_foot_rel[3].store(st.left_foot_target.1[2]);
        } else {
            bus.ppc_target_l_foot_rel[0].store(0.0);
        }
        if st.right_foot_target.0 {
            bus.ppc_target_r_foot_rel[0].store(0.95);
            bus.ppc_target_r_foot_rel[1].store(st.right_foot_target.1[0]);
            bus.ppc_target_r_foot_rel[2].store(st.right_foot_target.1[1]);
            bus.ppc_target_r_foot_rel[3].store(st.right_foot_target.1[2]);
        } else {
            bus.ppc_target_r_foot_rel[0].store(0.0);
        }

        // 4. Advance to next stage on waypoint completion, object grasp confirmation, or timer completion
        let object_just_grasped = st.track_world_object_with_hands
            && target_obj.is_held
            && self.stage_elapsed_ms >= 650;
        if waypoint_reached || object_just_grasped || self.stage_elapsed_ms >= st.duration_ms {
            self.current_stage_idx += 1;
            self.stage_elapsed_ms = 0;
        }
    }
}

/// Asynchronous helper that queries the live Meta Muse API (`muse-wonder-1.0` / `muse-spark-1.3`)
/// or Baseten FP4 (`meta-models/muse-spark-open-1026`) using `/usr/bin/curl` and records real
/// token/latency/EUR telemetry into `MetaPrefrontalSequencer`!
pub fn spawn_async_meta_api_enrichment(
    seq_arc: Arc<Mutex<MetaPrefrontalSequencer>>,
    user_prompt: String,
    pelvis_pos: Vec3,
    obj_pos: Vec3,
) {
    std::thread::spawn(move || {
        ensure_dotenv_loaded();
        let meta_key = env::var("META_API_KEY").unwrap_or_default();
        let meta_url = env::var("META_API_BASE_URL")
            .unwrap_or_else(|_| "https://api.meta.ai/v1".to_string());
        let baseten_key = env::var("BASETEN_API_KEY").unwrap_or_default();
        let baseten_url = env::var("BASETEN_API_BASE_URL")
            .unwrap_or_else(|_| "https://inference.baseten.co/v1".to_string());

        let (provider, base_url, api_key, model) =
            if !meta_key.is_empty() && meta_key != "your_meta_api_key_here" {
                ("meta", meta_url, meta_key, "muse-spark-1.3")
            } else if !baseten_key.is_empty() {
                (
                    "baseten",
                    baseten_url.clone(),
                    baseten_key.clone(),
                    "meta-models/muse-spark-open-1026",
                )
            } else {
                return;
            };

        let escaped_prompt = user_prompt
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', " ");
        let payload = format!(
            "{{\"model\":\"{}\",\"messages\":[{{\"role\":\"system\",\"content\":\"You are the Prefrontal Cortex Motor Planner for a 44-DoF 88-muscle biomimetic humanoid with SAM-3.1 binocular vision. Given the user command and 3D coordinates, reply with a JSON object {{\\\"plan\\\":\\\"<1 concise sentence>\\\",\\\"stages\\\":[{{\\\"name\\\":\\\"...\\\",\\\"vx\\\":0.0,\\\"vy\\\":0.0,\\\"yaw\\\":0.0,\\\"squat\\\":0.0,\\\"bow\\\":0.0,\\\"lh\\\":[0,0.4,0.24,0.58],\\\"rh\\\":[0,0.4,-0.24,0.58],\\\"lf\\\":[0,0.48,0.14,-0.35],\\\"rf\\\":[0,0.48,-0.14,-0.35],\\\"grasp\\\":0.0,\\\"nav_obj\\\":0,\\\"wave\\\":0,\\\"ms\\\":1600}}]}} representing the sequential task-space motor attractors.\"}},{{\"role\":\"user\",\"content\":\"Command: {} | Pelvis=({:.2},{:.2},{:.2}) | ActiveObject=({:.2},{:.2},{:.2})\"}}],\"temperature\":0.15,\"max_tokens\":220}}",
            model, escaped_prompt, pelvis_pos.x, pelvis_pos.y, pelvis_pos.z, obj_pos.x, obj_pos.y, obj_pos.z
        );

        let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
        let t0 = Instant::now();
        let out = Command::new("/usr/bin/curl")
            .args([
                "-sS",
                "--max-time",
                "8",
                "-X",
                "POST",
                &url,
                "-H",
                &format!("Authorization: Bearer {}", api_key),
                "-H",
                "Content-Type: application/json",
                "-d",
                &payload,
            ])
            .output();
        let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

        if let Ok(resp) = out {
            let body = String::from_utf8_lossy(&resp.stdout);
            let prompt_tok = extract_json_u64(&body, "\"prompt_tokens\":").unwrap_or(85);
            let comp_tok = extract_json_u64(&body, "\"completion_tokens\":").unwrap_or(32);
            let content = extract_json_string_field(&body, "\"content\":");

            if let Ok(mut seq) = seq_arc.lock() {
                let failed = content.is_none();
                if provider == "meta" {
                    let _ = seq
                        .meta_budget
                        .record_call(prompt_tok, comp_tok, latency_ms, model, failed);
                } else {
                    let _ = seq.baseten_budget.record_call(
                        prompt_tok,
                        comp_tok,
                        latency_ms,
                        model,
                        failed,
                    );
                }
                if let Some(txt) = content {
                    let summary = extract_json_string_field(&txt, "\"plan\":")
                        .unwrap_or_else(|| txt.chars().take(120).collect());
                    let parsed_llm_stages = parse_llm_json_stages(&txt);
                    if (!parsed_llm_stages.is_empty())
                        && (seq.stages.is_empty() || seq.used_fallback_only)
                    {
                        seq.stages = parsed_llm_stages;
                        seq.current_stage_idx = 0;
                        seq.stage_elapsed_ms = 0;
                        seq.sequence_active = true;
                        seq.used_fallback_only = false;
                    }
                    seq.meta_reasoning_summary =
                        format!("[{} | {:.0}ms] {}", model, latency_ms, summary);
                } else if provider == "meta" && !baseten_key.is_empty() {
                    let fb_payload = payload.replace(model, "meta-models/muse-spark-open-1026");
                    let fb_url = format!("{}/chat/completions", baseten_url.trim_end_matches('/'));
                    let t1 = Instant::now();
                    if let Ok(fb_resp) = Command::new("/usr/bin/curl")
                        .args([
                            "-sS",
                            "--max-time",
                            "8",
                            "-X",
                            "POST",
                            &fb_url,
                            "-H",
                            &format!("Authorization: Bearer {}", baseten_key),
                            "-H",
                            "Content-Type: application/json",
                            "-d",
                            &fb_payload,
                        ])
                        .output()
                    {
                        let fb_lat = t1.elapsed().as_secs_f64() * 1000.0;
                        let fb_body = String::from_utf8_lossy(&fb_resp.stdout);
                        let p_tok = extract_json_u64(&fb_body, "\"prompt_tokens\":").unwrap_or(85);
                        let c_tok =
                            extract_json_u64(&fb_body, "\"completion_tokens\":").unwrap_or(32);
                        let fb_txt = extract_json_string_field(&fb_body, "\"content\":");
                        let _ = seq.baseten_budget.record_call(
                            p_tok,
                            c_tok,
                            fb_lat,
                            "meta-models/muse-spark-open-1026",
                            fb_txt.is_none(),
                        );
                        if let Some(t) = fb_txt {
                            let summary = extract_json_string_field(&t, "\"plan\":")
                                .unwrap_or_else(|| t.chars().take(120).collect());
                            let parsed_llm_stages = parse_llm_json_stages(&t);
                            if (!parsed_llm_stages.is_empty())
                                && (seq.stages.is_empty() || seq.used_fallback_only)
                            {
                                seq.stages = parsed_llm_stages;
                                seq.current_stage_idx = 0;
                                seq.stage_elapsed_ms = 0;
                                seq.sequence_active = true;
                                seq.used_fallback_only = false;
                            }
                            seq.meta_reasoning_summary =
                                format!("[muse-spark-open-1026 | {:.0}ms] {}", fb_lat, summary);
                        }
                    }
                }
            }
        }
    });
}

fn extract_numbers(s: &str) -> Vec<f32> {
    let mut nums = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if ch.is_ascii_digit() || ch == '.' || (ch == '-' && cur.is_empty()) {
            cur.push(ch);
        } else if !cur.is_empty() {
            if let Ok(v) = cur.parse::<f32>() {
                nums.push(v);
            }
            cur.clear();
        }
    }
    if !cur.is_empty() {
        if let Ok(v) = cur.parse::<f32>() {
            nums.push(v);
        }
    }
    nums
}

fn extract_json_u64(json: &str, key: &str) -> Option<u64> {
    let idx = json.find(key)?;
    let rest = json[idx + key.len()..].trim_start();
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse::<u64>().ok()
}

fn extract_json_f32(json: &str, key: &str) -> Option<f32> {
    let idx = json.find(key)?;
    let rest = json[idx + key.len()..].trim_start();
    let tok: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
        .collect();
    tok.parse::<f32>().ok()
}

fn extract_json_f32_array(json: &str, key: &str) -> Vec<f32> {
    if let Some(idx) = json.find(key) {
        let rest = &json[idx + key.len()..];
        if let Some(open) = rest.find('[') {
            if let Some(close) = rest[open..].find(']') {
                return extract_numbers(&rest[open + 1..open + close]);
            }
        }
    }
    Vec::new()
}

/// Parses structured JSON motor stages `[{"name":..., "vx":..., "vy":..., "yaw":..., "squat":..., "bow":..., "lh":[...], "rh":[...], "lf":[...], "rf":[...], "grasp":..., "nav_obj":..., "wave":..., "ms":...}]` returned by the Meta Muse / Baseten LLM.
pub fn parse_llm_json_stages(json_text: &str) -> Vec<MotorStageCommand> {
    let mut out = Vec::new();
    let Some(st_idx) = json_text.find("\"stages\"") else {
        return out;
    };
    let rest = &json_text[st_idx..];
    let Some(arr_open) = rest.find('[') else {
        return out;
    };
    let arr_slice = &rest[arr_open + 1..];
    let mut depth = 0i32;
    let mut obj_start = None;
    for (i, ch) in arr_slice.char_indices() {
        if ch == '{' {
            if depth == 0 {
                obj_start = Some(i);
            }
            depth += 1;
        } else if ch == '}' {
            depth -= 1;
            if depth == 0 {
                if let Some(s) = obj_start {
                    let obj = &arr_slice[s..=i];
                    let mut cmd = MotorStageCommand::default();
                    if let Some(n) = extract_json_string_field(obj, "\"name\":") {
                        cmd.name = n;
                    }
                    if let Some(v) = extract_json_f32(obj, "\"vx\":") {
                        cmd.vx_cmd = v.clamp(-1.0, 1.0);
                    }
                    if let Some(v) = extract_json_f32(obj, "\"vy\":") {
                        cmd.vy_cmd = v.clamp(-1.0, 1.0);
                    }
                    if let Some(v) = extract_json_f32(obj, "\"yaw\":") {
                        cmd.yaw_cmd = v.clamp(-1.0, 1.0);
                    }
                    if let Some(v) = extract_json_f32(obj, "\"squat\":") {
                        cmd.squat_cmd = v.clamp(0.0, 1.0);
                    }
                    if let Some(v) = extract_json_f32(obj, "\"bow\":") {
                        cmd.trunk_lean_rad[0] = v.clamp(-0.4, 0.5);
                    }
                    if let Some(v) = extract_json_f32(obj, "\"grasp\":") {
                        cmd.grasp_cmd = v.clamp(0.0, 1.0);
                    }
                    if let Some(v) = extract_json_u64(obj, "\"ms\":") {
                        cmd.duration_ms = (v as u32).clamp(400, 10_000);
                    }
                    if let Some(v) = extract_json_u64(obj, "\"nav_obj\":") {
                        cmd.navigate_to_object = v > 0;
                    }
                    if let Some(v) = extract_json_u64(obj, "\"wave\":") {
                        cmd.wave_hands = v > 0;
                    }
                    let lh = extract_json_f32_array(obj, "\"lh\":");
                    if lh.len() >= 4 {
                        cmd.left_hand_target = (lh[0] > 0.5, [lh[1], lh[2], lh[3]]);
                    }
                    let rh = extract_json_f32_array(obj, "\"rh\":");
                    if rh.len() >= 4 {
                        cmd.right_hand_target = (rh[0] > 0.5, [rh[1], rh[2], rh[3]]);
                    }
                    let lf = extract_json_f32_array(obj, "\"lf\":");
                    if lf.len() >= 4 {
                        cmd.left_foot_target = (lf[0] > 0.5, [lf[1], lf[2], lf[3]]);
                    }
                    let rf = extract_json_f32_array(obj, "\"rf\":");
                    if rf.len() >= 4 {
                        cmd.right_foot_target = (rf[0] > 0.5, [rf[1], rf[2], rf[3]]);
                    }
                    out.push(cmd);
                }
            }
        }
    }
    out
}

fn extract_json_string_field(json: &str, key: &str) -> Option<String> {
    let idx = json.find(key)?;
    let rest = json[idx + key.len()..].trim_start();
    if !rest.starts_with('"') {
        return None;
    }
    let mut out = String::new();
    let mut escaped = false;
    for ch in rest[1..].chars() {
        if escaped {
            out.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return if out.trim().is_empty() {
                None
            } else {
                Some(out)
            };
        } else {
            out.push(ch);
        }
    }
    None
}
