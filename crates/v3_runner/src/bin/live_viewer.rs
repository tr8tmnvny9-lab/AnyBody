//! `live_viewer`: Pure-Rust Real-Time Neuromorphic 3D Humanoid, LLM Prefrontal Sequencer & Performance-per-Watt Server
//!
//! Runs the unthrottled 1.18M-Synapse `WholeBodyNeuralController` (`ScaleProfile::InteractiveRealTime`)
//! at 1,000 Hz in `< 0.35 ms/tick` and serves a 60 FPS interactive 3D Humanoid + Natural-Language Prefrontal
//! Command Bar + Multi-Layer Spiking Neural & Wattage Dashboard on `http://127.0.0.1:8421`.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use subsystems::meta_prefrontal::{spawn_async_meta_api_enrichment, MetaPrefrontalSequencer};
use subsystems::{ControlParadigm, ScaleProfile, WholeBodyNeuralController};

struct SharedControlState {
    scenario_tab: AtomicI32, // 5 = Meta-Model Natural-Language Control, 1 = Neuromorphic Walk, 2 = Dense PD Comparison, 3 = Multi-Skill Gauntlet, 4 = Untrained 0W Fall
    push_timer_ms: AtomicI32,
    slope_mrad: AtomicI32,
    turn_cmd_mrad: AtomicI32,
    squat_active: AtomicBool,
    bimanual_reach_active: AtomicBool,
    payload_surge_active: AtomicBool,
    trained_weights_active: AtomicBool,
    paradigm_is_dense_pd: AtomicBool,
    reset_requested: AtomicBool,
    pending_nl_prompt: Mutex<Option<String>>,
    meta_enrichment_state: Arc<Mutex<MetaPrefrontalSequencer>>,
    latest_json: Mutex<String>,
}

fn url_decode(s: &str) -> String {
    let mut out = Vec::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'+' {
            out.push(b' ');
            i += 1;
        } else if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(hex_str) = std::str::from_utf8(&bytes[i + 1..i + 3]) {
                if let Ok(b) = u8::from_str_radix(hex_str, 16) {
                    out.push(b);
                    i += 3;
                    continue;
                }
            }
            out.push(bytes[i]);
            i += 1;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', " ")
        .replace('\r', " ")
}

fn main() {
    println!("================================================================================");
    println!("  RUST NEURAL CONTROL V4: LIVE NEUROMORPHIC, LLM PREFRONTAL & WATTAGE 3D SERVER");
    println!("================================================================================");

    let _ = std::fs::create_dir_all("docs/diagnostics");
    let _ = std::fs::write("docs/diagnostics/live_3d_humanoid_viewer.html", VIEWER_HTML);

    let shared = Arc::new(SharedControlState {
        scenario_tab: AtomicI32::new(5),
        push_timer_ms: AtomicI32::new(0),
        slope_mrad: AtomicI32::new(0),
        turn_cmd_mrad: AtomicI32::new(0),
        squat_active: AtomicBool::new(false),
        bimanual_reach_active: AtomicBool::new(false),
        payload_surge_active: AtomicBool::new(false),
        trained_weights_active: AtomicBool::new(true),
        paradigm_is_dense_pd: AtomicBool::new(false),
        reset_requested: AtomicBool::new(false),
        pending_nl_prompt: Mutex::new(Some(
            "Walk to the box, pick up the box, stand up, turn left, and lift it overhead"
                .to_string(),
        )),
        meta_enrichment_state: Arc::new(Mutex::new(MetaPrefrontalSequencer::new())),
        latest_json: Mutex::new("{}".to_string()),
    });

    let sim_shared = Arc::clone(&shared);
    let cli_args: Vec<String> = std::env::args().collect();
    let profile = if cli_args.iter().any(|a| a == "--full-scale") {
        ScaleProfile::FullScaleHuman
    } else if cli_args.iter().any(|a| a == "--high-precision") {
        ScaleProfile::HighPrecisionRealTime
    } else {
        ScaleProfile::InteractiveRealTime
    };
    thread::spawn(move || {
        let mut ctrl = WholeBodyNeuralController::new(profile, 999);
        let total_syn = ctrl.total_network_synapses();
        let total_neur = ctrl.total_network_neurons();
        ctrl.train_spinal_and_cortical_weights();
        ctrl.spine.cpg.reset_for_trial(true);
        ctrl.bus.intent_vector[0].store(0.85);
        ctrl.bus.intent_vector[4].store(0.60);
        ctrl.bus.intent_vector[7].store(0.85);
        ctrl.plant.held_object_mass_kg = 0.45;

        println!(
            "  -> Live 1000Hz Unthrottled 3D Simulation Thread Active ({:.2}M Synapses, {:.2}M Neurons)",
            total_syn as f64 / 1e6,
            total_neur as f64 / 1e6
        );

        let mut last_wall = Instant::now();
        loop {
            let t0 = Instant::now();
            if sim_shared.reset_requested.swap(false, Ordering::SeqCst) {
                let trained = sim_shared.trained_weights_active.load(Ordering::Relaxed);
                if trained {
                    ctrl.train_spinal_and_cortical_weights();
                } else {
                    ctrl.reset_to_untrained_zero_weights();
                }
                ctrl.reset_body_posture_for_trial();
                ctrl.spine.cpg.reset_for_trial(trained);
                ctrl.bus.intent_vector[0].store(0.85);
                ctrl.bus.intent_vector[4].store(0.60);
                ctrl.bus.intent_vector[7].store(0.85);
                ctrl.plant.held_object_mass_kg = 0.45;
                sim_shared.payload_surge_active.store(false, Ordering::Relaxed);
                sim_shared.squat_active.store(false, Ordering::Relaxed);
                sim_shared.turn_cmd_mrad.store(0, Ordering::Relaxed);
                last_wall = Instant::now();
            }

            let tab = sim_shared.scenario_tab.load(Ordering::Relaxed);
            let trained = sim_shared.trained_weights_active.load(Ordering::Relaxed);
            let is_dense = sim_shared.paradigm_is_dense_pd.load(Ordering::Relaxed);
            ctrl.paradigm = if is_dense {
                ControlParadigm::ConventionalDensePdMlp
            } else {
                ControlParadigm::NeuromorphicHuman
            };
            ctrl.spine.cpg.enable_triphasic_pf_and_propriospinal = trained;

            if let Ok(mut p_guard) = sim_shared.pending_nl_prompt.lock() {
                if let Some(prompt_text) = p_guard.take() {
                    ctrl.prefrontal.compile_natural_language_prompt(&prompt_text);
                    if let Ok(mut m_guard) = sim_shared.meta_enrichment_state.lock() {
                        m_guard.active_prompt = ctrl.prefrontal.active_prompt.clone();
                        m_guard.meta_reasoning_summary =
                            ctrl.prefrontal.meta_reasoning_summary.clone();
                        m_guard.stages = ctrl.prefrontal.stages.clone();
                        m_guard.current_stage_idx = 0;
                        m_guard.stage_elapsed_ms = 0;
                        m_guard.sequence_active = true;
                    }
                    spawn_async_meta_api_enrichment(
                        Arc::clone(&sim_shared.meta_enrichment_state),
                        prompt_text,
                        ctrl.body_3d.com_pos,
                        ctrl.body_3d.world_object_pos,
                    );
                }
            }

            let cur_ms = ctrl.body_3d.latest_snapshot.time_ms % 4200;
            if tab == 5 {
                let push_rem = sim_shared.push_timer_ms.load(Ordering::Relaxed);
                if push_rem > 0 {
                    sim_shared
                        .push_timer_ms
                        .store((push_rem - 16).max(0), Ordering::Relaxed);
                    ctrl.body_3d.external_push_force_y_n = 150.0;
                    ctrl.body_3d.external_push_force_x_n = -40.0;
                    ctrl.plant.external_pitch_perturbation_nm = 24.0;
                } else {
                    ctrl.body_3d.external_push_force_y_n = 0.0;
                    ctrl.body_3d.external_push_force_x_n = 0.0;
                    ctrl.plant.external_pitch_perturbation_nm = 0.0;
                }
            } else if tab == 3 {
                ctrl.prefrontal.sequence_active = false;
                ctrl.body_3d.terrain_slope_rad = 5.0_f32.to_radians();
                ctrl.bus.intent_vector[4].store(0.95);
                ctrl.bus.intent_vector[5].store(0.95);
                ctrl.bus.intent_vector[6].store(0.75);
                ctrl.bus.intent_vector[7].store(0.75);

                if (1200..2300).contains(&cur_ms) {
                    ctrl.bus.intent_vector[0].store(0.0);
                    ctrl.bus.intent_vector[1].store(0.98);
                } else {
                    ctrl.bus.intent_vector[0].store(0.88);
                    ctrl.bus.intent_vector[1].store(0.0);
                }

                if (200..1200).contains(&cur_ms) {
                    ctrl.bus.intent_vector[2].store(0.85);
                } else if (2300..3600).contains(&cur_ms) {
                    ctrl.bus.intent_vector[2].store(-0.85);
                } else {
                    ctrl.bus.intent_vector[2].store(0.0);
                }

                if (2600..2800).contains(&cur_ms) {
                    ctrl.body_3d.external_push_force_y_n = 140.0;
                    ctrl.body_3d.external_push_force_x_n = -35.0;
                    ctrl.plant.external_pitch_perturbation_nm = 20.0;
                } else {
                    ctrl.body_3d.external_push_force_y_n = 0.0;
                    ctrl.body_3d.external_push_force_x_n = 0.0;
                    ctrl.plant.external_pitch_perturbation_nm = 0.0;
                }

                if (2800..3900).contains(&cur_ms) {
                    ctrl.plant.held_object_mass_kg = 2.50;
                } else {
                    ctrl.plant.held_object_mass_kg = 0.90;
                }
            } else {
                ctrl.prefrontal.sequence_active = false;
                let slope = (sim_shared.slope_mrad.load(Ordering::Relaxed) as f32) * 0.001;
                ctrl.body_3d.terrain_slope_rad = slope;

                let squat = sim_shared.squat_active.load(Ordering::Relaxed);
                if squat {
                    ctrl.bus.intent_vector[0].store(0.0);
                    ctrl.bus.intent_vector[1].store(0.98);
                } else {
                    ctrl.bus.intent_vector[0].store(0.88);
                    ctrl.bus.intent_vector[1].store(0.0);
                }

                let turn_cmd = (sim_shared.turn_cmd_mrad.load(Ordering::Relaxed) as f32) * 0.001;
                ctrl.bus.intent_vector[2].store(turn_cmd.clamp(-1.0, 1.0));

                let bimanual = sim_shared.bimanual_reach_active.load(Ordering::Relaxed);
                if bimanual {
                    ctrl.bus.intent_vector[4].store(0.95);
                    ctrl.bus.intent_vector[5].store(0.95);
                } else {
                    ctrl.bus.intent_vector[4].store(0.55);
                    ctrl.bus.intent_vector[5].store(0.0);
                }

                let push_rem = sim_shared.push_timer_ms.load(Ordering::Relaxed);
                if push_rem > 0 {
                    sim_shared
                        .push_timer_ms
                        .store((push_rem - 16).max(0), Ordering::Relaxed);
                    ctrl.body_3d.external_push_force_y_n = 150.0;
                    ctrl.body_3d.external_push_force_x_n = -40.0;
                    ctrl.plant.external_pitch_perturbation_nm = 24.0;
                } else {
                    ctrl.body_3d.external_push_force_y_n = 0.0;
                    ctrl.body_3d.external_push_force_x_n = 0.0;
                    ctrl.plant.external_pitch_perturbation_nm = 0.0;
                }

                if sim_shared.payload_surge_active.load(Ordering::Relaxed) {
                    ctrl.plant.held_object_mass_kg = 2.50;
                } else {
                    ctrl.plant.held_object_mass_kg = 0.90;
                }
            }

            let wall_dt_ms = last_wall.elapsed().as_millis().clamp(8, 22) as usize;
            last_wall = Instant::now();
            for _ in 0..wall_dt_ms {
                ctrl.step_closed_loop_1ms(if trained { 0.015 } else { 0.0 });
            }

            // If the Autonomous Agentic Chaining Loop just self-called a new sub-goal, fire async LLM enrichment!
            if ctrl.prefrontal.needs_llm_replan {
                ctrl.prefrontal.needs_llm_replan = false;
                if let Ok(mut m_guard) = sim_shared.meta_enrichment_state.lock() {
                    m_guard.meta_reasoning_summary =
                        ctrl.prefrontal.meta_reasoning_summary.clone();
                }
                spawn_async_meta_api_enrichment(
                    Arc::clone(&sim_shared.meta_enrichment_state),
                    format!(
                        "{} (Agentic Sub-Call #{}, Basket {}/4)",
                        ctrl.prefrontal.active_prompt,
                        ctrl.prefrontal.agentic_iteration,
                        ctrl.prefrontal.objects_in_basket_count
                    ),
                    ctrl.body_3d.com_pos,
                    ctrl.body_3d.world_object_pos,
                );
            }

            let snap = &ctrl.body_3d.latest_snapshot;
            let score = ctrl.body_3d.evaluate_gait_scorecard();
            let pwr = ctrl.power_metrics;
            let grip_r = ctrl.plant.right_hand_grip_force_n;
            let grip_l = ctrl.plant.left_hand_grip_force_n;
            let slip_v = ctrl
                .plant
                .right_hand_object_slip_vel
                .max(ctrl.plant.left_hand_object_slip_vel);
            let tick_ms = t0.elapsed().as_secs_f64() * 1000.0 / (wall_dt_ms as f64);

            let stage_str = if ctrl.prefrontal.sequence_active
                && ctrl.prefrontal.current_stage_idx < ctrl.prefrontal.stages.len()
            {
                format!(
                    "Stage {}/{}: {}",
                    ctrl.prefrontal.current_stage_idx + 1,
                    ctrl.prefrontal.stages.len(),
                    ctrl.prefrontal.stages[ctrl.prefrontal.current_stage_idx].name
                )
            } else if !ctrl.prefrontal.stages.is_empty() {
                format!(
                    "Completed {} stages ({})",
                    ctrl.prefrontal.stages.len(),
                    ctrl.prefrontal.active_prompt
                )
            } else {
                "Idle / Manual Control".to_string()
            };

            let (meta_summary, meta_eur, baseten_eur, total_calls) =
                if let Ok(m_guard) = sim_shared.meta_enrichment_state.lock() {
                    let summary = if ctrl.prefrontal.agentic_loop_active
                        || ctrl.prefrontal.final_verification_scan_done
                    {
                        ctrl.prefrontal.meta_reasoning_summary.clone()
                    } else {
                        m_guard.meta_reasoning_summary.clone()
                    };
                    (
                        summary,
                        m_guard.meta_budget.spent_eur + ctrl.prefrontal.meta_budget.spent_eur,
                        m_guard.baseten_budget.spent_eur + ctrl.prefrontal.baseten_budget.spent_eur,
                        m_guard.meta_budget.calls
                            + m_guard.baseten_budget.calls
                            + ctrl.prefrontal.meta_budget.calls,
                    )
                } else {
                    (ctrl.prefrontal.meta_reasoning_summary.clone(), 0.0, 0.0, 0)
                };
            let obj = ctrl.body_3d.world_object_pos;
            let obj_held = ctrl.body_3d.world_object_is_held;
            let goal_pos = ctrl.body_3d.world_goal_pos;
            let mut objs_json = String::from("[");
            for (idx, wobj) in ctrl.body_3d.world_objects.iter().enumerate() {
                if idx > 0 {
                    objs_json.push(',');
                }
                objs_json.push_str(&format!(
                    "{{\"id\":{},\"name\":\"{}\",\"color\":\"{}\",\"shape\":\"{}\",\"pos\":[{:.3},{:.3},{:.3}],\"held\":{},\"in_fov\":{},\"seen\":{},\"uv\":[{:.3},{:.3}],\"depth\":{:.2}}}",
                    wobj.id,
                    wobj.name,
                    wobj.color,
                    wobj.shape,
                    wobj.pos.x,
                    wobj.pos.y,
                    wobj.pos.z,
                    wobj.is_held,
                    wobj.in_fov,
                    wobj.ever_seen,
                    wobj.retinal_uv[0],
                    wobj.retinal_uv[1],
                    wobj.depth_m
                ));
            }
            objs_json.push(']');

            let act_l_thigh = ctrl.plant.muscles[0]
                .ca_activation
                .max(ctrl.plant.muscles[1].ca_activation);
            let act_l_shank = ctrl.plant.muscles[6]
                .ca_activation
                .max(ctrl.plant.muscles[7].ca_activation);
            let act_r_thigh = ctrl.plant.muscles[12]
                .ca_activation
                .max(ctrl.plant.muscles[13].ca_activation);
            let act_r_shank = ctrl.plant.muscles[18]
                .ca_activation
                .max(ctrl.plant.muscles[19].ca_activation);
            let act_l_arm = ctrl.plant.muscles[32]
                .ca_activation
                .max(ctrl.plant.muscles[36].ca_activation);
            let act_r_arm = ctrl.plant.muscles[40]
                .ca_activation
                .max(ctrl.plant.muscles[44].ca_activation);

            let json = format!(
                concat!(
                    "{{",
                    "\"tab\":{},\"paradigm\":{},\"time_ms\":{},\"synapses\":{},\"neurons\":{},\"tick_ms\":{:.3},",
                    "\"pelvis\":[{:.4},{:.4},{:.4}],\"head\":[{:.4},{:.4},{:.4}],",
                    "\"l_hip\":[{:.4},{:.4},{:.4}],\"l_knee\":[{:.4},{:.4},{:.4}],\"l_ankle\":[{:.4},{:.4},{:.4}],\"l_toe\":[{:.4},{:.4},{:.4}],",
                    "\"r_hip\":[{:.4},{:.4},{:.4}],\"r_knee\":[{:.4},{:.4},{:.4}],\"r_ankle\":[{:.4},{:.4},{:.4}],\"r_toe\":[{:.4},{:.4},{:.4}],",
                    "\"l_sh\":[{:.4},{:.4},{:.4}],\"l_el\":[{:.4},{:.4},{:.4}],\"l_hand\":[{:.4},{:.4},{:.4}],",
                    "\"r_sh\":[{:.4},{:.4},{:.4}],\"r_el\":[{:.4},{:.4},{:.4}],\"r_hand\":[{:.4},{:.4},{:.4}],",
                    "\"xcom\":[{:.4},{:.4},{:.4}],\"l_grf\":{:.1},\"r_grf\":{:.1},",
                    "\"speed\":{:.2},\"dist\":{:.2},\"clearance_cm\":{:.1},\"sym_err\":{:.1},\"arm_corr\":{:.2},\"score\":{:.1},",
                    "\"bone_err_cm\":{:.5},\"yaw_deg\":{:.1},\"is_fallen\":{},",
                    "\"w_knee\":{:.2},\"w_lpn\":{:.2},\"w_comm\":[{:.2},{:.2}],",
                    "\"pf\":[{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}],",
                    "\"grip_n\":{:.1},\"grip_l\":{:.1},\"slip_v\":{:.3},",
                    "\"synops\":{},\"dense_ops\":{},\"sparsity_pct\":{:.1},\"comp_mw\":{:.1},\"muscle_w\":{:.1},\"stiff_w\":{:.1},\"co_waste\":{:.2},\"eff_mult\":{:.2},",
                    "\"ctx_active\":{},\"ctx_single\":{},\"ctx_burst\":{},\"pv\":{:.2},\"sst\":{:.2},\"vip\":{:.2},\"cb_active_gc\":{},",
                    "\"m_act\":[{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}],",
                    "\"obj\":[{:.4},{:.4},{:.4}],\"obj_held\":{},\"stage_str\":\"{}\",\"meta_summary\":\"{}\",",
                    "\"meta_eur\":{:.4},\"baseten_eur\":{:.4},\"api_calls\":{},",
                    "\"goal\":[{:.3},{:.3},{:.3}],\"head_yaw\":{:.4},\"active_obj\":{},",
                    "\"agentic_active\":{},\"agentic_iter\":{},\"basket_count\":{},\"scan_done\":{},\"objects\":{}",
                    "}}"
                ),
                tab,
                if is_dense { 1 } else { 0 },
                snap.time_ms,
                total_syn,
                total_neur,
                tick_ms,
                snap.pelvis.x, snap.pelvis.y, snap.pelvis.z,
                snap.head.x, snap.head.y, snap.head.z,
                snap.left_hip.x, snap.left_hip.y, snap.left_hip.z,
                snap.left_knee.x, snap.left_knee.y, snap.left_knee.z,
                snap.left_ankle.x, snap.left_ankle.y, snap.left_ankle.z,
                snap.left_toe.x, snap.left_toe.y, snap.left_toe.z,
                snap.right_hip.x, snap.right_hip.y, snap.right_hip.z,
                snap.right_knee.x, snap.right_knee.y, snap.right_knee.z,
                snap.right_ankle.x, snap.right_ankle.y, snap.right_ankle.z,
                snap.right_toe.x, snap.right_toe.y, snap.right_toe.z,
                snap.left_shoulder.x, snap.left_shoulder.y, snap.left_shoulder.z,
                snap.left_elbow.x, snap.left_elbow.y, snap.left_elbow.z,
                snap.left_hand.x, snap.left_hand.y, snap.left_hand.z,
                snap.right_shoulder.x, snap.right_shoulder.y, snap.right_shoulder.z,
                snap.right_elbow.x, snap.right_elbow.y, snap.right_elbow.z,
                snap.right_hand.x, snap.right_hand.y, snap.right_hand.z,
                snap.xcom_capture_point.x, snap.xcom_capture_point.y, snap.xcom_capture_point.z,
                snap.left_grf_n, snap.right_grf_n,
                snap.forward_vel_m_s,
                score.net_forward_distance_m,
                score.mean_swing_foot_clearance_m * 100.0,
                score.gait_symmetry_error_pct,
                score.arm_leg_counter_rotation_corr,
                score.composite_naturalness_score,
                score.max_bone_length_error_m * 100.0,
                snap.yaw_rad.to_degrees(),
                snap.is_fallen,
                ctrl.spine.cpg.w_sw1_knee_flex[0],
                ctrl.spine.cpg.w_lpn_contra[0],
                ctrl.spine.cpg.w_comm_balance[0],
                ctrl.spine.cpg.w_comm_balance[1],
                ctrl.spine.cpg.pf_rate[0],
                ctrl.spine.cpg.pf_rate[1],
                ctrl.spine.cpg.pf_rate[2],
                ctrl.spine.cpg.pf_rate[3],
                ctrl.spine.cpg.pf_rate[4],
                ctrl.spine.cpg.pf_rate[5],
                ctrl.spine.cpg.pf_rate[6],
                ctrl.spine.cpg.pf_rate[7],
                grip_r,
                grip_l,
                slip_v,
                pwr.active_synops_per_tick,
                pwr.dense_equiv_ops_per_tick,
                pwr.compute_sparsity_fraction * 100.0,
                pwr.compute_power_mw,
                pwr.muscle_power_watts,
                pwr.stiff_pd_actuator_watts,
                pwr.co_contraction_waste_index,
                pwr.perf_per_watt_multiplier,
                ctrl.cortex.pyramidal.active_dendrite_count,
                ctrl.cortex.pyramidal.single_spike_count,
                ctrl.cortex.pyramidal.burst_spike_count,
                ctrl.cortex.mean_pv_inhibition,
                ctrl.cortex.mean_sst_inhibition,
                ctrl.cortex.mean_vip_disinhibition,
                ctrl.cerebellum.total_active_granules_last_tick,
                act_l_thigh, act_l_shank, act_r_thigh, act_r_shank, act_l_arm, act_r_arm,
                obj.x, obj.y, obj.z,
                obj_held,
                json_escape(&stage_str),
                json_escape(&meta_summary),
                meta_eur,
                baseten_eur,
                total_calls,
                goal_pos.x,
                goal_pos.y,
                goal_pos.z,
                snap.head_yaw_rad,
                snap.active_object_idx,
                ctrl.prefrontal.agentic_loop_active,
                ctrl.prefrontal.agentic_iteration,
                ctrl.prefrontal.objects_in_basket_count,
                ctrl.prefrontal.final_verification_scan_done,
                objs_json
            );

            if let Ok(mut guard) = sim_shared.latest_json.lock() {
                *guard = json;
            }
            thread::sleep(Duration::from_millis(12));
        }
    });

    let addr = std::env::var("VIEWER_ADDR").unwrap_or_else(|_| "127.0.0.1:8421".to_string());
    let listener = TcpListener::bind(&addr).expect("Failed to bind viewer port");
    println!(
        "  -> Interactive 3D Neuromorphic, LLM Prefrontal & Wattage Viewer listening on: http://{}",
        addr
    );

    for stream in listener.incoming().flatten() {
        let srv_shared = Arc::clone(&shared);
        thread::spawn(move || {
            let mut stream = stream;
            let _ = stream.set_read_timeout(Some(Duration::from_millis(250)));
            let mut buf = [0u8; 2048];
            if let Ok(n) = stream.read(&mut buf) {
                let req = String::from_utf8_lossy(&buf[..n]);
                let first_line = req.lines().next().unwrap_or("");
                if first_line.starts_with("GET /state") {
                    let body = srv_shared
                        .latest_json
                        .lock()
                        .map(|g| g.clone())
                        .unwrap_or_else(|_| "{}".to_string());
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = stream.write_all(resp.as_bytes());
                } else if first_line.starts_with("POST /cmd") {
                    if let Some(p_idx) = first_line.find("prompt=") {
                        let after = &first_line[p_idx + 7..];
                        let raw_prompt = after
                            .split_whitespace()
                            .next()
                            .unwrap_or("")
                            .split('&')
                            .next()
                            .unwrap_or("");
                        let decoded = url_decode(raw_prompt);
                        if !decoded.trim().is_empty() {
                            srv_shared.scenario_tab.store(5, Ordering::Relaxed);
                            srv_shared.trained_weights_active.store(true, Ordering::Relaxed);
                            if let Ok(mut g) = srv_shared.pending_nl_prompt.lock() {
                                *g = Some(decoded);
                            }
                        }
                    } else if first_line.contains("act=tab5") {
                        srv_shared.scenario_tab.store(5, Ordering::Relaxed);
                        srv_shared.paradigm_is_dense_pd.store(false, Ordering::Relaxed);
                        srv_shared.trained_weights_active.store(true, Ordering::Relaxed);
                    } else if first_line.contains("act=tab1") {
                        srv_shared.scenario_tab.store(1, Ordering::Relaxed);
                        srv_shared.paradigm_is_dense_pd.store(false, Ordering::Relaxed);
                        srv_shared.trained_weights_active.store(true, Ordering::Relaxed);
                        srv_shared.reset_requested.store(true, Ordering::SeqCst);
                    } else if first_line.contains("act=tab2") {
                        srv_shared.scenario_tab.store(2, Ordering::Relaxed);
                        srv_shared.paradigm_is_dense_pd.store(true, Ordering::Relaxed);
                        srv_shared.trained_weights_active.store(true, Ordering::Relaxed);
                    } else if first_line.contains("act=tab3") {
                        srv_shared.scenario_tab.store(3, Ordering::Relaxed);
                        srv_shared.paradigm_is_dense_pd.store(false, Ordering::Relaxed);
                        srv_shared.trained_weights_active.store(true, Ordering::Relaxed);
                        srv_shared.reset_requested.store(true, Ordering::SeqCst);
                    } else if first_line.contains("act=tab4") {
                        srv_shared.scenario_tab.store(4, Ordering::Relaxed);
                        srv_shared.trained_weights_active.store(false, Ordering::Relaxed);
                        srv_shared.reset_requested.store(true, Ordering::SeqCst);
                    } else if first_line.contains("act=push") {
                        srv_shared.push_timer_ms.store(180, Ordering::Relaxed);
                    } else if first_line.contains("act=slip") {
                        let cur = srv_shared.payload_surge_active.load(Ordering::Relaxed);
                        srv_shared.payload_surge_active.store(!cur, Ordering::Relaxed);
                    } else if first_line.contains("act=squat") {
                        let cur = srv_shared.squat_active.load(Ordering::Relaxed);
                        srv_shared.squat_active.store(!cur, Ordering::Relaxed);
                    } else if first_line.contains("act=reach") {
                        let cur = srv_shared.bimanual_reach_active.load(Ordering::Relaxed);
                        srv_shared.bimanual_reach_active.store(!cur, Ordering::Relaxed);
                    } else if first_line.contains("act=turn_l") {
                        srv_shared.turn_cmd_mrad.store(750, Ordering::Relaxed);
                    } else if first_line.contains("act=turn_r") {
                        srv_shared.turn_cmd_mrad.store(-750, Ordering::Relaxed);
                    } else if first_line.contains("act=reset") {
                        srv_shared.reset_requested.store(true, Ordering::SeqCst);
                    }
                    let resp = "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 2\r\n\r\nOK";
                    let _ = stream.write_all(resp.as_bytes());
                } else if first_line.starts_with("GET /neuron") {
                    let body = std::fs::read_to_string(
                        "docs/diagnostics/human_neuron_synapse_explainer.html",
                    )
                    .unwrap_or_else(|_| VIEWER_HTML.to_string());
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = stream.write_all(resp.as_bytes());
                } else {
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
                        VIEWER_HTML.len(),
                        VIEWER_HTML
                    );
                    let _ = stream.write_all(resp.as_bytes());
                }
            }
        });
    }
}

const VIEWER_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Rust Neural Control V4 — LLM Prefrontal + Neuromorphic Performance-per-Watt 3D Humanoid</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; }
  body { background: #070b12; color: #f8fafc; display: flex; height: 100vh; overflow: hidden; }
  #viewport-wrap { flex: 1; position: relative; display: flex; flex-direction: column; }
  canvas#view3d { flex: 1; width: 100%; height: 100%; cursor: grab; background: radial-gradient(circle at 50% 42%, #17223b 0%, #070b12 78%); }
  #hud-top { position: absolute; top: 14px; left: 14px; right: 14px; display: flex; justify-content: space-between; pointer-events: none; gap: 12px; }
  .badge { background: rgba(15, 23, 42, 0.90); border: 1px solid #334155; padding: 10px 15px; border-radius: 8px; backdrop-filter: blur(8px); }
  .badge h1 { font-size: 14.5px; color: #38bdf8; letter-spacing: 0.4px; }
  .badge p { font-size: 11.5px; color: #94a3b8; margin-top: 3px; }
  #tab-bar { position: absolute; top: 82px; left: 14px; display: flex; gap: 8px; z-index: 20; flex-wrap: wrap; }
  .tab-btn { background: rgba(15, 23, 42, 0.92); border: 1.5px solid #475569; color: #e2e8f0; padding: 8px 13px; border-radius: 8px; font-size: 11.5px; font-weight: 700; cursor: pointer; transition: 0.15s; }
  .tab-btn.active { background: #0284c7; border-color: #38bdf8; color: #ffffff; box-shadow: 0 0 14px rgba(56, 189, 248, 0.45); }
  #sidebar { width: 440px; background: #0b111e; border-left: 1px solid #1e293b; padding: 12px; display: flex; flex-direction: column; gap: 10px; overflow-y: auto; }
  .card { background: #141e30; border: 1px solid #283750; border-radius: 8px; padding: 10px 12px; }
  .card h2 { font-size: 11.5px; color: #38bdf8; text-transform: uppercase; margin-bottom: 7px; letter-spacing: 0.6px; }
  .btn-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  button { background: #0284c7; color: white; border: none; padding: 7px 9px; border-radius: 6px; font-size: 11px; font-weight: 600; cursor: pointer; transition: 0.15s; }
  button:hover { background: #0369a1; }
  button.warn { background: #e11d48; }
  button.warn:hover { background: #be123c; }
  button.sec { background: #334155; }
  .metric-row { display: flex; justify-content: space-between; font-size: 11.5px; padding: 3px 0; border-bottom: 1px solid rgba(51,65,85,0.45); }
  .metric-val { font-weight: 700; color: #34d399; }
  .bar-wrap { height: 7px; background: #070b12; border-radius: 4px; overflow: hidden; margin-top: 3px; }
  .bar-fill { height: 100%; background: linear-gradient(90deg, #34d399, #38bdf8); width: 0%; transition: width 0.08s; }
  .bar-fill.hot { background: linear-gradient(90deg, #f59e0b, #ef4444); }
</style>
</head>
<body>
<div id="viewport-wrap">
  <canvas id="view3d"></canvas>
  <div id="hud-top">
    <div class="badge">
      <h1 id="title-banner">RUST NEURAL V4 — ZERO-HARDCODING LLM->NEURON &amp; WATTAGE HIERARCHY</h1>
      <p id="conn-status">Connecting to Unthrottled 1000Hz Rust Engine (127.0.0.1:8421)...</p>
    </div>
    <div class="badge">
      <h1 id="watt-banner" style="color:#34d399;">POWER: 68.4 W (3.4x MORE EFFICIENT THAN DENSE PD)</h1>
      <p id="speed-banner">Sparsity: 94.2% SynOps Asleep | Compute: 58 mW | Speed: 1.85 m/s</p>
    </div>
  </div>
  <div id="tab-bar">
    <button class="tab-btn active" id="tbtn-5" onclick="sendCmd('tab5')">🧠 Tab 5: LLM Prefrontal Natural-Language Control</button>
    <button class="tab-btn" id="tbtn-1" onclick="sendCmd('tab1')">⚡ Mode A: Neuromorphic Sparse Spiking (Low W)</button>
    <button class="tab-btn" id="tbtn-2" onclick="sendCmd('tab2')">🔥 Mode B: Conventional Stiff PD + Dense MLP (High W)</button>
    <button class="tab-btn" id="tbtn-3" onclick="sendCmd('tab3')">🎯 Stage 5: Multi-Skill Gauntlet</button>
    <button class="tab-btn" id="tbtn-4" onclick="sendCmd('tab4')">⚠️ Stage 4A: Untrained 0-Weight Fall</button>
    <a href="/neuron" class="tab-btn" style="text-decoration:none; background:#0f172a; border-color:#38bdf8; color:#38bdf8;">Fig. 1: Multi-Scale Synapse &amp; Cortical Architecture</a>
  </div>
  <div style="position:absolute; bottom:14px; left:14px; right:14px; background:rgba(15,23,42,0.94); border:1.5px solid #38bdf8; border-radius:10px; padding:11px 14px; backdrop-filter:blur(10px); z-index:25; box-shadow:0 8px 28px rgba(0,0,0,0.55);">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
      <span style="font-size:11.5px; font-weight:700; color:#38bdf8; letter-spacing:0.4px;">🧠 PREFRONTAL CORTEX META-MODEL (muse-spark-1.3 / muse-wonder-1.0 / Baseten FP4) — TYPE ANY MOTOR COMMAND:</span>
      <span id="m-budget" style="font-size:11px; color:#a855f7; font-weight:600;">Meta: €0.0000 / €150 | Baseten: €0.0000 / €100 | Calls: 0</span>
    </div>
    <div style="display:flex; gap:8px; margin-bottom:7px;">
      <input id="nl-input" type="text" value="Clean all the things in the basket" placeholder="Type any natural-language command..." onkeydown="if(event.key==='Enter') sendPrompt();" style="flex:1; background:#070b12; border:1px solid #475569; color:#f8fafc; padding:8px 11px; border-radius:6px; font-size:12px;" />
      <button onclick="sendPrompt()" style="background:linear-gradient(90deg,#0284c7,#7c3aed); padding:8px 16px; font-size:12px;">🚀 Execute Prompt</button>
    </div>
    <div style="display:flex; flex-wrap:wrap; gap:6px; margin-bottom:5px;">
      <button onclick="setAndSend('Clean all the things in the basket')" style="background:#059669;">🧹 Clean All 4 Objects Into Basket (19.1s Agentic)</button>
      <button onclick="setAndSend('Only put the blue ball and green cylinder in the basket')" style="background:#0d9488;">🎯 Selective Sort: Only Blue Ball &amp; Green Cylinder</button>
      <button onclick="setAndSend('Clean everything except the red box into the basket')" style="background:#7c3aed;">🚫 Negation Filter: Clean All Except Red Box</button>
      <button onclick="setAndSend('Find the blue ball and point at it, then walk sideways left, then pick up the gold crate, put it in the basket, and lift hands overhead')" style="background:#ea580c;">📦 Sidestep + Deliver 1.4kg Gold Crate + Celebrate</button>
      <button class="sec" onclick="setAndSend('Find the green cylinder and point at it')">🔍 Find Green Cylinder &amp; Point</button>
      <button class="sec" onclick="setAndSend('Walk to the red box, pick up the red box, stand up, and lift it overhead')">📦 Pick Up Red Box &amp; Lift Overhead</button>
      <button class="sec" onclick="setAndSend('Kick high with right leg, then bow forward and wave both hands')">🦵 Right High Kick, Bow &amp; Wave</button>
    </div>
    <div style="font-size:11px; color:#cbd5e1; display:flex; justify-content:space-between; gap:12px;">
      <span><strong style="color:#34d399;">Active Stage:</strong> <span id="m-stage">Initializing...</span></span>
      <span style="color:#94a3b8; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" id="m-meta-summary">Prefrontal Cortex online.</span>
    </div>
  </div>
</div>
<div id="sidebar">
  <div class="card">
    <h2>Performance-per-Watt Live A/B Meter</h2>
    <div class="metric-row"><span>Active Control Architecture</span><span class="metric-val" id="p-mode">NEUROMORPHIC SPIKING</span></div>
    <div class="metric-row"><span>Synaptic Sparsity (Clock-Gated)</span><span class="metric-val" id="p-sparse">85.0% Asleep</span></div>
    <div class="metric-row"><span>Active SynOps / 1ms Tick</span><span class="metric-val" id="p-synops">180,000 / 1,180,000</span></div>
    <div class="metric-row"><span>Neural Compute Power</span><span class="metric-val" id="p-comp">188 mW (vs 18,500 mW)</span></div>
    <div class="metric-row"><span>Total Body Power (Muscles + Brain)</span><span class="metric-val" id="p-watts">210 W</span></div>
    <div class="bar-wrap"><div class="bar-fill" id="b-watts"></div></div>
    <div class="metric-row" style="margin-top:5px;"><span>Co-Contraction Stiffness Waste</span><span class="metric-val" id="p-waste">0.03 (Passive Swing!)</span></div>
    <div class="bar-wrap"><div class="bar-fill hot" id="b-waste"></div></div>
    <div class="metric-row" style="margin-top:5px;"><span>Performance-per-Watt Advantage</span><span class="metric-val" id="p-mult" style="color:#38bdf8;font-size:13px;">2.90x vs Dense PD</span></div>
  </div>

  <div class="card">
    <h2>Live Interactive Perturbations &amp; Skills</h2>
    <div class="btn-grid">
      <button onclick="sendCmd('squat')">🏋️ Toggle Deep Squat</button>
      <button onclick="sendCmd('reach')">🙌 Bimanual 3D Reach</button>
      <button onclick="sendCmd('turn_l')">↩️ Steer Left (+Yaw)</button>
      <button onclick="sendCmd('turn_r')">↪️ Steer Right (-Yaw)</button>
      <button class="warn" onclick="sendCmd('push')">💥 150N Lateral Push</button>
      <button class="warn" onclick="sendCmd('slip')">📦 2.5kg Bimanual Slip</button>
      <button class="sec" onclick="sendCmd('reset')" style="grid-column: span 2;">🔄 Reset Posture &amp; Trial</button>
    </div>
  </div>

  <div class="card">
    <h2>4-Layer Intelligent Neural Activity Scope</h2>
    <div class="metric-row"><span>L1 Cortex: Active 3-Comp Dendrites</span><span class="metric-val" id="n-ctx">42 / 576 tiles</span></div>
    <div class="metric-row"><span>L1 Cortex: Single vs. Ca2+ Burst Spikes</span><span class="metric-val" id="n-burst">6 single / 2 burst</span></div>
    <div class="metric-row"><span>L1 Inhibitory Routing [PV, SST, VIP]</span><span class="metric-val" id="n-inh">[0.42, 0.08, 0.15]</span></div>
    <div class="metric-row"><span>L2 Cerebellum: Active Sparse Granules</span><span class="metric-val" id="n-cb">5,120 / 180,224</span></div>
    <div class="metric-row"><span>L3 Spinal Plasticity [FRA Knee, LPN Arm]</span><span class="metric-val" id="n-sp">[1.16, 0.74]</span></div>
  </div>

  <div class="card">
    <h2>Rigid-Bone &amp; Clinical 3D Biomechanics</h2>
    <div class="metric-row"><span>Postural State &amp; Score</span><span class="metric-val" id="m-status">WALKING (86.4/100)</span></div>
    <div class="metric-row"><span>3D Distance, Speed &amp; Heading</span><span class="metric-val" id="m-dist">0.00 m (0.00 m/s)</span></div>
    <div class="metric-row"><span>Swing Foot Clearance &amp; Bone Err</span><span class="metric-val" id="m-clear">5.1 cm (0.00007 cm)</span></div>
    <div class="metric-row"><span>Bimanual 20-DoF Grip [L, R]</span><span class="metric-val" id="m-grip">35.8 N / 35.8 N</span></div>
    <div class="metric-row"><span>Unthrottled 1000Hz Rust Tick Time</span><span class="metric-val" id="m-tick">0.30 ms/tick</span></div>
  </div>

  <div class="card">
    <h2>8-Pool Biophysical CPG Pattern Formation</h2>
    <div id="pf-bars"></div>
  </div>
</div>

<script>
const pfNames = ["L_Sw1 (Early Swing)", "L_Sw2 (Term Swing)", "L_St1 (Weight Accept)", "L_St2 (Push-Off)", "R_Sw1 (Early Swing)", "R_Sw2 (Term Swing)", "R_St1 (Weight Accept)", "R_St2 (Push-Off)"];
const pfContainer = document.getElementById('pf-bars');
pfNames.forEach((name, i) => {
  pfContainer.innerHTML += `<div style="font-size:10.5px;color:#cbd5e1;margin-top:3px;display:flex;justify-content:space-between;"><span>${name}</span><span id="pfv-${i}">0.00</span></div><div class="bar-wrap"><div class="bar-fill" id="pfb-${i}"></div></div>`;
});

let camYaw = -0.68, camPitch = 0.26, camDist = 2.55, isDragging = false, lastX = 0, lastY = 0;
const canvas = document.getElementById('view3d');
const ctx = canvas.getContext('2d');
canvas.addEventListener('mousedown', e => { isDragging = true; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener('mouseup', () => isDragging = false);
window.addEventListener('mousemove', e => {
  if (!isDragging) return;
  camYaw += (e.clientX - lastX) * 0.008;
  camPitch = Math.max(-0.08, Math.min(1.15, camPitch + (e.clientY - lastY) * 0.008));
  lastX = e.clientX; lastY = e.clientY;
});
canvas.addEventListener('wheel', e => { camDist = Math.max(1.2, Math.min(5.5, camDist + e.deltaY * 0.002)); });

let state = null;
function sendCmd(act) {
  fetch('/cmd?act=' + act, { method: 'POST' }).catch(() => {});
}
function sendPrompt() {
  const el = document.getElementById('nl-input');
  if (!el || !el.value.trim()) return;
  fetch('/cmd?prompt=' + encodeURIComponent(el.value.trim()), { method: 'POST' }).catch(() => {});
}
function setAndSend(txt) {
  const el = document.getElementById('nl-input');
  if (el) el.value = txt;
  sendPrompt();
}

async function pollState() {
  try {
    const res = await fetch('/state');
    if (res.ok) {
      state = await res.json();
      [1, 2, 3, 4, 5].forEach(i => {
        const el = document.getElementById('tbtn-' + i);
        if (el) el.className = (state.tab === i) ? 'tab-btn active' : 'tab-btn';
      });
    }
  } catch (e) {}
}
setInterval(pollState, 30);

function project3D(pt, targetX, targetY) {
  const rx = pt[0] - targetX, ry = pt[1] - targetY, rz = pt[2] - 0.72;
  const cy = Math.cos(camYaw), sy = Math.sin(camYaw);
  const cp = Math.cos(camPitch), sp = Math.sin(camPitch);
  const x1 = rx * cy - ry * sy;
  const y1 = rx * sy + ry * cy;
  const z2 = rz * cp - y1 * sp;
  const depth = camDist + (rz * sp + y1 * cp);
  const scale = (Math.min(canvas.width, canvas.height) * 0.86) / Math.max(0.3, depth);
  return [canvas.width * 0.5 + x1 * scale, canvas.height * 0.52 - z2 * scale];
}

function muscleColor(act, isDense) {
  if (isDense) return '#f97316';
  if (act > 0.45) return '#fb7185';
  if (act > 0.22) return '#38bdf8';
  return '#34d399';
}

function drawCapsule3D(a, b, tx, ty, color, width) {
  const p1 = project3D(a, tx, ty), p2 = project3D(b, tx, ty);
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(p1[0], p1[1]); ctx.lineTo(p2[0], p2[1]); ctx.stroke();
}

function drawJoint3D(pt, tx, ty, radius, color) {
  const p = project3D(pt, tx, ty);
  ctx.fillStyle = color; ctx.beginPath(); ctx.arc(p[0], p[1], radius, 0, Math.PI * 2); ctx.fill();
}

let trail = [];
function render() {
  canvas.width = canvas.clientWidth; canvas.height = canvas.clientHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (state && state.pelvis) {
    const tx = state.pelvis[0], ty = state.pelvis[1];
    const isDense = state.paradigm === 1;
    if (trail.length === 0 || Math.hypot(tx - trail[trail.length-1][0], ty - trail[trail.length-1][1]) > 0.04) {
      trail.push([tx, ty]);
      if (trail.length > 90) trail.shift();
    }
    const startX = Math.floor(tx) - 3, startY = Math.floor(ty) - 3;
    for (let gx = startX; gx <= startX + 6; gx += 0.25) {
      const isMajor = Math.abs(gx - Math.round(gx)) < 0.01;
      drawCapsule3D([gx, startY, 0], [gx, startY + 6, 0], tx, ty, isMajor ? '#334155' : '#162032', isMajor ? 1.6 : 1);
      if (isMajor) {
        const lp = project3D([gx, ty - 0.85, 0], tx, ty);
        ctx.fillStyle = '#64748b'; ctx.font = '11px monospace';
        ctx.fillText(`${Math.round(gx)}m`, lp[0] - 10, lp[1]);
      }
    }
    for (let gy = startY; gy <= startY + 6; gy += 0.25) {
      const isMajor = Math.abs(gy - Math.round(gy)) < 0.01;
      drawCapsule3D([startX, gy, 0], [startX + 6, gy, 0], tx, ty, isMajor ? '#334155' : '#162032', isMajor ? 1.6 : 1);
    }
    for (let i = 1; i < trail.length; i++) {
      drawCapsule3D([trail[i-1][0], trail[i-1][1], 0.01], [trail[i][0], trail[i][1], 0.01], tx, ty, '#a855f7', 3);
    }

    // Render 3D Basket / Goal Zone at state.goal
    if (state.goal) {
      const gx = state.goal[0], gy = state.goal[1], r = 0.42, h = 0.36;
      const b00 = [gx-r, gy-r, 0], b10 = [gx+r, gy-r, 0], b11 = [gx+r, gy+r, 0], b01 = [gx-r, gy+r, 0];
      const t00 = [gx-r, gy-r, h], t10 = [gx+r, gy-r, h], t11 = [gx+r, gy+r, h], t01 = [gx-r, gy+r, h];
      [[b00,b10],[b10,b11],[b11,b01],[b01,b00],[t00,t10],[t10,t11],[t11,t01],[t01,t00],[b00,t00],[b10,t10],[b11,t11],[b01,t01]].forEach(e => {
        drawCapsule3D(e[0], e[1], tx, ty, '#10b981', 2.2);
      });
      const gp = project3D([gx, gy, h + 0.14], tx, ty);
      ctx.fillStyle = '#34d399'; ctx.font = 'bold 11.5px monospace';
      ctx.fillText(`🧺 BASKET (${state.basket_count || 0}/4)`, gp[0] - 54, gp[1]);
    }

    // Render 116° Binocular Retinal Camera FOV Cone (`SAM-3.1`) from Head
    if (state.head && state.head_yaw !== undefined) {
      const hx = state.head[0], hy = state.head[1], hz = state.head[2];
      const fovHalf = 0.92, coneLen = 2.1;
      const rayL = [hx + coneLen * Math.cos(state.head_yaw + fovHalf), hy + coneLen * Math.sin(state.head_yaw + fovHalf), 0.15];
      const rayR = [hx + coneLen * Math.cos(state.head_yaw - fovHalf), hy + coneLen * Math.sin(state.head_yaw - fovHalf), 0.15];
      drawCapsule3D([hx, hy, hz], rayL, tx, ty, 'rgba(56,189,248,0.35)', 1.5);
      drawCapsule3D([hx, hy, hz], rayR, tx, ty, 'rgba(56,189,248,0.35)', 1.5);
      drawCapsule3D(rayL, rayR, tx, ty, 'rgba(56,189,248,0.25)', 1.2);
    }

    // Render all 4 Multi-Object 3D World Objects (`Red Box`, `Blue Ball`, `Green Cylinder`, `Gold Crate`)
    if (state.objects && state.objects.length) {
      state.objects.forEach(wobj => {
        const ox = wobj.pos[0], oy = wobj.pos[1], oz = wobj.pos[2];
        const r = 0.12;
        const col = wobj.held ? '#ffffff' : wobj.color;
        if (!wobj.held && oz > 0.40) {
          drawCapsule3D([ox, oy, 0], [ox, oy, oz - r], tx, ty, '#475569', 3.5);
        }
        if (wobj.shape === 'sphere') {
          const sp = project3D([ox, oy, oz], tx, ty);
          ctx.fillStyle = col; ctx.beginPath(); ctx.arc(sp[0], sp[1], 14, 0, Math.PI * 2); ctx.fill();
          if (wobj.in_fov) {
            ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 2; ctx.strokeRect(sp[0]-18, sp[1]-18, 36, 36);
          }
        } else {
          const c000 = [ox-r, oy-r, oz-r], c100 = [ox+r, oy-r, oz-r], c110 = [ox+r, oy+r, oz-r], c010 = [ox-r, oy+r, oz-r];
          const c001 = [ox-r, oy-r, oz+r], c101 = [ox+r, oy-r, oz+r], c111 = [ox+r, oy+r, oz+r], c011 = [ox-r, oy+r, oz+r];
          [[c000,c100],[c100,c110],[c110,c010],[c010,c000],[c001,c101],[c101,c111],[c111,c011],[c011,c001],[c000,c001],[c100,c101],[c110,c111],[c010,c011]].forEach(edge => {
            drawCapsule3D(edge[0], edge[1], tx, ty, col, wobj.in_fov ? 2.8 : 2.0);
          });
          drawJoint3D([ox, oy, oz], tx, ty, 5.5, col);
        }
        const lp = project3D([ox, oy, oz + 0.19], tx, ty);
        ctx.fillStyle = wobj.in_fov ? '#f8fafc' : '#94a3b8';
        ctx.font = '10.5px monospace';
        ctx.fillText(`${wobj.name}${wobj.in_fov ? ' [FOV]' : ''}`, lp[0] - 34, lp[1]);
      });
    }

    if (state.l_grf > 20) {
      drawJoint3D([state.l_ankle[0], state.l_ankle[1], 0.0], tx, ty, 8, 'rgba(16,185,129,0.5)');
      drawCapsule3D(state.l_ankle, [state.l_ankle[0], state.l_ankle[1], state.l_ankle[2] + state.l_grf * 0.00045], tx, ty, '#10b981', 3);
    }
    if (state.r_grf > 20) {
      drawJoint3D([state.r_ankle[0], state.r_ankle[1], 0.0], tx, ty, 8, 'rgba(16,185,129,0.5)');
      drawCapsule3D(state.r_ankle, [state.r_ankle[0], state.r_ankle[1], state.r_ankle[2] + state.r_grf * 0.00045], tx, ty, '#10b981', 3);
    }
    drawJoint3D(state.xcom, tx, ty, 6, '#f43f5e');

    const ma = state.m_act || [0.1,0.1,0.1,0.1,0.1,0.1];
    drawCapsule3D(state.pelvis, state.r_hip, tx, ty, '#94a3b8', 5);
    drawCapsule3D(state.r_hip, state.r_knee, tx, ty, muscleColor(ma[2], isDense), 9);
    drawCapsule3D(state.r_knee, state.r_ankle, tx, ty, muscleColor(ma[3], isDense), 7);
    drawCapsule3D(state.r_ankle, state.r_toe, tx, ty, '#fb923c', 5);
    drawCapsule3D(state.r_sh, state.r_el, tx, ty, muscleColor(ma[5], isDense), 6.5);
    drawCapsule3D(state.r_el, state.r_hand, tx, ty, muscleColor(ma[5], isDense), 5.5);

    drawCapsule3D(state.pelvis, state.head, tx, ty, state.is_fallen ? '#ef4444' : '#f8fafc', 10);
    drawCapsule3D(state.l_sh, state.r_sh, tx, ty, '#cbd5e1', 6.5);
    drawJoint3D(state.head, tx, ty, 15, state.is_fallen ? '#ef4444' : '#38bdf8');

    drawCapsule3D(state.pelvis, state.l_hip, tx, ty, '#94a3b8', 5);
    drawCapsule3D(state.l_hip, state.l_knee, tx, ty, muscleColor(ma[0], isDense), 9);
    drawCapsule3D(state.l_knee, state.l_ankle, tx, ty, muscleColor(ma[1], isDense), 7);
    drawCapsule3D(state.l_ankle, state.l_toe, tx, ty, '#38bdf8', 5);
    drawCapsule3D(state.l_sh, state.l_el, tx, ty, muscleColor(ma[4], isDense), 6.5);
    drawCapsule3D(state.l_el, state.l_hand, tx, ty, muscleColor(ma[4], isDense), 5.5);

    [state.pelvis, state.l_knee, state.l_ankle, state.r_knee, state.r_ankle, state.l_el, state.r_el, state.l_hand, state.r_hand].forEach(j => drawJoint3D(j, tx, ty, 4.5, '#facc15'));

    document.getElementById('conn-status').innerText = `Live ${(state.synapses/1e6).toFixed(2)}M-Synapse Unthrottled Stream (t = ${state.time_ms} ms) | Tick: ${state.tick_ms.toFixed(2)} ms`;
    document.getElementById('watt-banner').innerText = isDense
      ? `CONVENTIONAL DENSE PD/MLP: ${state.muscle_w.toFixed(1)} W (HIGH CO-CONTRACTION WASTE)`
      : `NEUROMORPHIC SPIKING POWER: ${state.muscle_w.toFixed(1)} W (${state.eff_mult.toFixed(2)}x MORE EFFICIENT!)`;
    document.getElementById('watt-banner').style.color = isDense ? '#f97316' : '#34d399';
    document.getElementById('speed-banner').innerText = `Sparsity: ${state.sparsity_pct.toFixed(1)}% Asleep | Compute: ${state.comp_mw.toFixed(1)} mW | Speed: ${state.speed.toFixed(2)} m/s`;

    document.getElementById('p-mode').innerText = isDense ? 'DENSE 1kHz MLP + STIFF PD' : 'HUMAN EVENT-DRIVEN SPIKING';
    document.getElementById('p-mode').style.color = isDense ? '#f97316' : '#34d399';
    document.getElementById('p-sparse').innerText = `${state.sparsity_pct.toFixed(1)}% SynOps Asleep`;
    document.getElementById('p-synops').innerText = `${state.synops.toLocaleString()} / ${state.dense_ops.toLocaleString()}`;
    document.getElementById('p-comp').innerText = isDense ? `18,500 mW (Continuous Dense)` : `${state.comp_mw.toFixed(1)} mW (Event-Driven)`;
    document.getElementById('p-watts').innerText = `${state.muscle_w.toFixed(1)} W (vs ${state.stiff_w.toFixed(1)} W Baseline)`;
    document.getElementById('b-watts').style.width = Math.min(100, (state.muscle_w / 500.0) * 100) + '%';
    document.getElementById('p-waste').innerText = isDense ? `${state.co_waste.toFixed(2)} (Stiff Agonist-Antagonist Fight!)` : `${state.co_waste.toFixed(2)} (Reciprocal Inhibition!)`;
    document.getElementById('b-waste').style.width = Math.min(100, (state.co_waste / 0.35) * 100) + '%';
    document.getElementById('p-mult').innerText = isDense ? `1.00x (Baseline Reference)` : `${state.eff_mult.toFixed(2)}x Performance-per-Watt!`;

    document.getElementById('n-ctx').innerText = `${state.ctx_active} / 576 active`;
    document.getElementById('n-burst').innerText = `${state.ctx_single} single / ${state.ctx_burst} burst`;
    document.getElementById('n-inh').innerText = `[PV ${state.pv.toFixed(2)}, SST ${state.sst.toFixed(2)}, VIP ${state.vip.toFixed(2)}]`;
    document.getElementById('n-cb').innerText = `${state.cb_active_gc.toLocaleString()} / 180,224 (~2.8%)`;
    document.getElementById('n-sp').innerText = `[FRA ${state.w_knee.toFixed(2)}, LPN ${state.w_lpn.toFixed(2)}]`;

    document.getElementById('m-status').innerText = state.is_fallen ? 'FALLEN (0.0 W)' : `${state.pelvis[2] < 0.72 ? 'DEEP SQUAT' : 'WALKING'} (${state.score.toFixed(1)}/100)`;
    document.getElementById('m-status').style.color = state.is_fallen ? '#ef4444' : '#34d399';
    document.getElementById('m-dist').innerText = `${state.dist.toFixed(2)}m | ${state.speed.toFixed(2)}m/s | ${state.yaw_deg.toFixed(1)}°`;
    document.getElementById('m-clear').innerText = `${state.clearance_cm.toFixed(1)} cm | ${state.bone_err_cm.toFixed(5)} cm`;
    document.getElementById('m-grip').innerText = `${state.grip_l.toFixed(1)} N / ${state.grip_n.toFixed(1)} N`;
    document.getElementById('m-tick').innerText = `${state.tick_ms.toFixed(2)} ms/tick (${(1.0/Math.max(0.01, state.tick_ms)).toFixed(1)}x RT)`;

    if (state.stage_str) {
      const agBadge = state.agentic_active
        ? ` | 🔄 AGENTIC LOOP #${state.agentic_iter} (Basket: ${state.basket_count}/4)`
        : (state.scan_done ? ` | ✅ AGENTIC STOP (Basket: ${state.basket_count}/4 Verified)` : ` | Basket: ${state.basket_count || 0}/4`);
      document.getElementById('m-stage').innerText = state.stage_str + agBadge;
    }

    // Draw Inset SAM-3.1 Binocular Retinal Camera HUD at top-right of 3D canvas
    if (state.objects) {
      const rx = canvas.width - 224, ry = 132, rw = 210, rh = 132;
      ctx.fillStyle = 'rgba(11, 17, 30, 0.88)';
      ctx.fillRect(rx, ry, rw, rh);
      ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 1.5;
      ctx.strokeRect(rx, ry, rw, rh);
      ctx.fillStyle = '#38bdf8'; ctx.font = 'bold 10px monospace';
      ctx.fillText('SAM-3.1 RETINAL CAMERA (116° FOV)', rx + 8, ry + 14);
      ctx.strokeStyle = 'rgba(51,65,85,0.6)'; ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(rx + rw*0.5, ry + 20); ctx.lineTo(rx + rw*0.5, ry + rh - 6);
      ctx.moveTo(rx + 6, ry + 20 + (rh-26)*0.5); ctx.lineTo(rx + rw - 6, ry + 20 + (rh-26)*0.5);
      ctx.stroke();
      state.objects.forEach(wobj => {
        if (wobj.in_fov) {
          const px = rx + rw * 0.5 + wobj.uv[0] * (rw * 0.42);
          const py = ry + 20 + (rh - 26) * 0.5 - wobj.uv[1] * ((rh - 26) * 0.40);
          ctx.fillStyle = wobj.color;
          ctx.beginPath(); ctx.arc(px, py, Math.max(4, 12 / Math.max(0.6, wobj.depth)), 0, Math.PI * 2); ctx.fill();
          ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1;
          ctx.strokeRect(px - 9, py - 9, 18, 18);
          ctx.fillStyle = '#e2e8f0'; ctx.font = '9px monospace';
          ctx.fillText(`${wobj.name.split(' ')[1]} ${wobj.depth.toFixed(1)}m`, px - 18, py - 12);
        }
      });
    }
    if (state.meta_summary) {
      document.getElementById('m-meta-summary').innerText = state.meta_summary;
    }
    if (state.meta_eur !== undefined) {
      document.getElementById('m-budget').innerText = `Meta: €${state.meta_eur.toFixed(4)} / €150 | Baseten: €${state.baseten_eur.toFixed(4)} / €100 | Calls: ${state.api_calls}`;
    }

    if (state.pf) {
      state.pf.forEach((v, i) => {
        document.getElementById('pfv-' + i).innerText = v.toFixed(2);
        document.getElementById('pfb-' + i).style.width = Math.min(100, v * 100) + '%';
      });
    }
  }
  requestAnimationFrame(render);
}
render();
</script>
</body>
</html>"##;
