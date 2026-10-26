use std::fs::File;
use std::io::Write;
use subsystems::{ScaleProfile, WholeBodyNeuralController};

fn dump_run(prompt: &str, out_path: &str, max_ticks: usize) {
    let mut ctrl = WholeBodyNeuralController::new(ScaleProfile::FastVerification, 999);
    ctrl.train_spinal_and_cortical_weights();
    ctrl.spine.cpg.reset_for_trial(true);
    ctrl.prefrontal.compile_natural_language_prompt(prompt);

    let mut out_file = File::create(out_path).expect("Failed to create trajectory file");
    println!("Dumping trajectory for prompt: {} -> {}", prompt, out_path);

    for step in 1..=max_ticks {
        ctrl.step_closed_loop_1ms(0.010);
        if step % 33 == 0 {
            let snap = &ctrl.body_3d.latest_snapshot;
            let pre = &ctrl.prefrontal;
            let stage_desc = if pre.current_stage_idx < pre.stages.len() {
                pre.stages[pre.current_stage_idx].name.clone()
            } else {
                "Idle / Goal Achieved".to_string()
            };
            let line = format!(
                concat!(
                    "{{\"time_ms\":{},\"stage\":\"{}\",",
                    "\"pelvis\":[{:.3},{:.3},{:.3}],\"yaw\":{:.3},",
                    "\"l_ankle\":[{:.3},{:.3},{:.3}],\"r_ankle\":[{:.3},{:.3},{:.3}],",
                    "\"l_hand\":[{:.3},{:.3},{:.3}],\"r_hand\":[{:.3},{:.3},{:.3}],",
                    "\"head\":[{:.3},{:.3},{:.3}],",
                    "\"l_knee\":[{:.3},{:.3},{:.3}],\"r_knee\":[{:.3},{:.3},{:.3}],",
                    "\"l_elbow\":[{:.3},{:.3},{:.3}],\"r_elbow\":[{:.3},{:.3},{:.3}],",
                    "\"l_shoulder\":[{:.3},{:.3},{:.3}],\"r_shoulder\":[{:.3},{:.3},{:.3}],",
                    "\"l_hip\":[{:.3},{:.3},{:.3}],\"r_hip\":[{:.3},{:.3},{:.3}],",
                    "\"l_grf\":{:.1},\"r_grf\":{:.1},",
                    "\"objects\":[",
                    "{{\"name\":\"Red Box\",\"color\":\"#ef4444\",\"pos\":[{:.3},{:.3},{:.3}],\"held\":{},\"fov\":{}}},",
                    "{{\"name\":\"Blue Ball\",\"color\":\"#0082fb\",\"pos\":[{:.3},{:.3},{:.3}],\"held\":{},\"fov\":{}}},",
                    "{{\"name\":\"Green Cylinder\",\"color\":\"#10b981\",\"pos\":[{:.3},{:.3},{:.3}],\"held\":{},\"fov\":{}}},",
                    "{{\"name\":\"Gold Crate\",\"color\":\"#f59e0b\",\"pos\":[{:.3},{:.3},{:.3}],\"held\":{},\"fov\":{}}}",
                    "],",
                    "\"basket\":[{:.3},{:.3},{:.3}],",
                    "\"agentic_cycle\":{},\"in_basket\":{}",
                    "}}\n"
                ),
                snap.time_ms,
                stage_desc,
                snap.pelvis.x, snap.pelvis.y, snap.pelvis.z,
                snap.yaw_rad,
                snap.left_ankle.x, snap.left_ankle.y, snap.left_ankle.z,
                snap.right_ankle.x, snap.right_ankle.y, snap.right_ankle.z,
                snap.left_hand.x, snap.left_hand.y, snap.left_hand.z,
                snap.right_hand.x, snap.right_hand.y, snap.right_hand.z,
                snap.head.x, snap.head.y, snap.head.z,
                snap.left_knee.x, snap.left_knee.y, snap.left_knee.z,
                snap.right_knee.x, snap.right_knee.y, snap.right_knee.z,
                snap.left_elbow.x, snap.left_elbow.y, snap.left_elbow.z,
                snap.right_elbow.x, snap.right_elbow.y, snap.right_elbow.z,
                snap.left_shoulder.x, snap.left_shoulder.y, snap.left_shoulder.z,
                snap.right_shoulder.x, snap.right_shoulder.y, snap.right_shoulder.z,
                snap.left_hip.x, snap.left_hip.y, snap.left_hip.z,
                snap.right_hip.x, snap.right_hip.y, snap.right_hip.z,
                snap.left_grf_n, snap.right_grf_n,
                snap.world_objects[0].pos.x, snap.world_objects[0].pos.y, snap.world_objects[0].pos.z, snap.world_objects[0].is_held, snap.world_objects[0].in_fov,
                snap.world_objects[1].pos.x, snap.world_objects[1].pos.y, snap.world_objects[1].pos.z, snap.world_objects[1].is_held, snap.world_objects[1].in_fov,
                snap.world_objects[2].pos.x, snap.world_objects[2].pos.y, snap.world_objects[2].pos.z, snap.world_objects[2].is_held, snap.world_objects[2].in_fov,
                snap.world_objects[3].pos.x, snap.world_objects[3].pos.y, snap.world_objects[3].pos.z, snap.world_objects[3].is_held, snap.world_objects[3].in_fov,
                snap.world_goal_pos.x, snap.world_goal_pos.y, snap.world_goal_pos.z,
                pre.agentic_iteration,
                pre.objects_in_basket_count,
            );
            out_file.write_all(line.as_bytes()).unwrap();
        }
        if !ctrl.prefrontal.sequence_active && !ctrl.prefrontal.agentic_loop_active {
            break;
        }
    }
}

fn main() {
    // Demo 1: Locomotion & Dexterity (Walk to box, pick up, overhead lift)
    dump_run(
        "Walk to the box, pick up the box, stand up, turn left, and lift it overhead",
        "target/demo1_lift.jsonl",
        9000
    );

    // Demo 2: Agility & Balance (Unipedal Kick, Bow & Wave)
    dump_run(
        "Kick high with right leg, then bow forward and wave both hands",
        "target/demo2_kick_wave.jsonl",
        6000
    );

    // Demo 3: Climax Autonomous Agentic Sorting ("Clean all the things in the basket")
    dump_run(
        "Clean all the things in the basket",
        "target/demo3_agentic_clean.jsonl",
        22000
    );
}
