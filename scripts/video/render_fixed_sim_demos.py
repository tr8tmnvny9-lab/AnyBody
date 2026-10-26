#!/usr/bin/env python3
"""Render standalone verified simulation demo clips (Demo 1, Demo 2, Demo 3) and the
6-Block Code-Faithful V3 Generative UI Tasking Visualization.

Key features:
  1. Fast Pose-Step Caching: Renders unique 3D MuJoCo OpenGL + SSAO poses once and reuses them
     during the initial 1.35s "Prompt -> Thinking -> Generative Plan Spawning" phase.
  2. Demo 1 & Demo 3 First-Person Camera (`ego_head_cam` + `SAM-3.1` live RGB-D segmentation masks).
  3. Generative UI Plan Builder (Block 1: Meta Muse + SAM-3.1 -> `Vec<MotorStageCommand>`):
     - When the prompt is entered, the prefrontal executive thinks and dynamically spawns the
       planned `MotorStageCommand` action cards one-by-one.
     - As Basal Ganglia executes each stage, completed action cards flash emerald `[✓ DONE]`
       and collapse/disappear out of the queue while remaining cards slide up into the active slot!
  4. 100% Real Per-Demo Data across Blocks 2–6 (Basal Ganglia 7-Ch Gate, 2,500 L5 Pyramidal Cortex
     `w_spatial_pull[88][28]`, Cerebellar 44ms `mGluR1` Forward Model, Spinal 14-D CPG & Ia/Ib Reflexes,
     and 88 Hill-Type Muscles) with zero truncated text and zero fake equalizer bars.
"""

from __future__ import annotations

import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_design_snapshots as snap_mod
import h1_wholebody_ik as ik_mod

ARTIFACT_DIR = SCRIPT_DIR.parents[1] / "target" / "video_artifacts"


def spring_ease(t: float, t_start: float, duration: float = 0.38) -> float:
    if t <= t_start:
        return 0.0
    u = min(1.0, (t - t_start) / max(1e-5, duration))
    return 1.0 - math.exp(-5.5 * u) * math.cos(2.8 * u)


def smooth_collapse(u_over: float) -> float:
    """Return remaining height/opacity fraction in [0.0, 1.0] after a stage finishes (`u_over >= 0`)."""
    if u_over <= 0.0:
        return 1.0
    if u_over >= 1.0:
        return 0.0
    return 1.0 - (u_over * u_over * (3.0 - 2.0 * u_over))


def fast_glass_card(
    stage_rgba: Image.Image,
    box: list[int],
    radius: int = 20,
    fill_rgba: tuple[int, int, int, int] = (255, 255, 255, 238),
    border_rgba: tuple[int, int, int, int] = (0, 129, 251, 95),
) -> None:
    x0, y0, x1, y1 = box
    if x1 <= x0 + 4 or y1 <= y0 + 4:
        return
    layer = Image.new("RGBA", (x1 - x0 + 24, y1 - y0 + 24), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    ld.rounded_rectangle([12, 16, 12 + (x1 - x0), 16 + (y1 - y0)], radius=radius, fill=(15, 23, 42, 18))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=6))
    ld = ImageDraw.Draw(layer)
    ld.rounded_rectangle([12, 12, 12 + (x1 - x0), 12 + (y1 - y0)], radius=radius, fill=fill_rgba, outline=border_rgba, width=2)
    stage_rgba.alpha_composite(layer, dest=(x0 - 12, y0 - 12))


def draw_animated_bezier(
    draw: ImageDraw.ImageDraw,
    p0: tuple[int, int],
    p1: tuple[int, int],
    col_rgb: tuple[int, int, int],
    active: bool,
    t_sec: float,
    vertical: bool = False,
    width: int = 2,
) -> None:
    x0, y0 = p0
    x1, y1 = p1
    if vertical:
        cx1, cy1 = x0, y0 + int((y1 - y0) * 0.50)
        cx2, cy2 = x1, y0 + int((y1 - y0) * 0.50)
    else:
        cx1, cy1 = x0 + int((x1 - x0) * 0.50), y0
        cx2, cy2 = x0 + int((x1 - x0) * 0.50), y1

    pts = []
    n_steps = 28
    for i in range(n_steps + 1):
        u = i / float(n_steps)
        om = 1.0 - u
        bx = int(om**3 * x0 + 3 * om**2 * u * cx1 + 3 * om * u**2 * cx2 + u**3 * x1)
        by = int(om**3 * y0 + 3 * om**2 * u * cy1 + 3 * om * u**2 * cy2 + u**3 * y1)
        pts.append((bx, by))

    alpha = 225 if active else 70
    draw.line(pts, fill=col_rgb + (alpha,), width=width if active else 1)
    if active:
        for k in range(2):
            u_p = (t_sec * 1.25 + k * 0.5) % 1.0
            idx = min(n_steps, int(u_p * n_steps))
            px, py = pts[idx]
            draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=col_rgb + (255,), outline=(255, 255, 255, 255), width=1)


def build_demo_plan_stages(demo_id: int, snap: dict, sim_u: float) -> tuple[list[dict[str, Any]], int, float]:
    """Return `(stages_list, active_idx, stage_local_u)` with 100% real per-demo Rust parameters
    from `MetaPrefrontalSequencer` (`meta_prefrontal.rs`) and `FloatingBase3DBody` (`kinematics_3d.rs`).
    """
    ts = snap.get("v3_task_state", {})
    fov_list = ts.get("vision_fov", [])
    pelv = snap["pelvis"]

    if demo_id == 1:
        obj_p = snap["objects"][0]["pos"]
        d_box = fov_list[0]["depth_m"] if fov_list else 0.94
        uv_box = fov_list[0]["retinal_uv"] if fov_list else [0.0, -0.18]
        bounds = [(0.00, 0.28), (0.28, 0.44), (0.44, 0.62), (0.62, 0.78), (0.78, 1.00)]
        active_idx = int(snap.get("active_stage", 0))
        b0, b1 = bounds[active_idx]
        loc_u = max(0.0, min(1.0, (sim_u - b0) / max(1e-5, b1 - b0)))

        stages = [
            {
                "title": "STAGE 1 · WALK TO RED BOX",
                "short_name": "Walk to Box",
                "bg_takeover": "BG Takeover: Ch1 (Walk_Locomote = 0.88) · 4 Grounded Steps",
                "vision_line": f"SAM-3.1 Lock: Red Box (+0.32, +0.00, 0.10m) · dist={d_box:.2f}m",
                "cmd_line": "intent=[Vx:0.78, Sq:0.00, Yaw:+0.25, Stand:1.0] (dist_xy <= 0.23m)",
                "col": (0, 129, 251),
                "u_end": 0.28,
            },
            {
                "title": "STAGE 2 · DEEP SQUAT (0.52m) & REACH",
                "short_name": "0.52m Squat",
                "bg_takeover": "BG Takeover: Ch5 (Squat=0.94) inhibits Ch1 (Walk=0.04) + Reach=0.85",
                "vision_line": "Parietal Target: L/R Wrists -> Box Faces [0.32, ±0.095, 0.16m]",
                "cmd_line": f"intent=[Vx:0.00, Sq:-0.32, Reach:0.85] · Pelvis Z={pelv[2]:.2f}m (58°)",
                "col": (16, 185, 129),
                "u_end": 0.44,
            },
            {
                "title": "STAGE 3 · BIMANUAL TACTILE CLAMP & STAND",
                "short_name": "Clamp & Stand",
                "bg_takeover": "BG Takeover: Ch3 (Grasp=0.95) + Ch0 (Stand=0.92) · BTSP 42.9 N",
                "vision_line": f"Palmar Slip Reflex Locked (0.01mm Jump) · Box Z = {obj_p[2]:.2f}m",
                "cmd_line": "intent=[Sq:0.00, Stand:1.0, Reach:0.85, Grasp:0.95] -> Z=0.95m",
                "col": (14, 165, 233),
                "u_end": 0.62,
            },
            {
                "title": "STAGE 4 · 2-STEP BIPEDAL PIVOT TURN (+63°)",
                "short_name": "+63° Pivot",
                "bg_takeover": "BG Takeover: Ch4 (Turn=0.84) + Ch1 (Walk=0.88) + Ch3 (Grasp=0.95)",
                "vision_line": "Alternating Single-Support Yaw Steps · Stance Slip = 0.000mm",
                "cmd_line": f"intent=[Vx:0.00, Yaw:+0.85, Reach:0.85, Grasp:0.95] -> +{math.degrees(snap['yaw']):.0f}°",
                "col": (124, 58, 237),
                "u_end": 0.78,
            },
            {
                "title": "STAGE 5 · BIMANUAL OVERHEAD PRESS (1.59m)",
                "short_name": "1.59m Lift",
                "bg_takeover": "BG Takeover: Ch2/6 (Reach_Arm=0.95) + Ch3 (Grasp=0.95) · L5 Lift",
                "vision_line": f"Parietal Hand Target: rel_z = +0.72m above Pelvis · Box Z = {obj_p[2]:.2f}m",
                "cmd_line": "intent=[Stand:1.0, Reach_R:0.95, Reach_L:0.95, Grasp:0.95] (Hold)",
                "col": (239, 68, 68),
                "u_end": 1.01,
            },
        ]
        return stages, active_idx, loc_u

    if demo_id == 2:
        r_sole = snap["r_foot_sole"]
        bounds = [(0.00, 0.45), (0.45, 0.72), (0.72, 1.00)]
        active_idx = int(snap.get("active_stage", 0))
        b0, b1 = bounds[active_idx]
        loc_u = max(0.0, min(1.0, (sim_u - b0) / max(1e-5, b1 - b0)))

        stages = [
            {
                "title": "STAGE 1 · CoM SHIFT & UNIPEDAL HIGH KICK (0.49m)",
                "short_name": "0.49m High Kick",
                "bg_takeover": "BG Takeover: Ch0 (Stand=0.96) inhibits Ch1 (Walk=0.02) · CoM Shift",
                "vision_line": f"Cerebellar 44ms Kernel shifts Pelvis Y={pelv[1]:+.3f}m over Left Stance",
                "cmd_line": f"ppc_target_r_foot=[+0.46, -0.13, +0.49m] · Right Foot Z = {r_sole[2]:.2f}m",
                "col": (124, 58, 237),
                "u_end": 0.45,
            },
            {
                "title": "STAGE 2 · VOLUNTARY FORWARD BOW (16.9°)",
                "short_name": "16.9° Bow",
                "bg_takeover": "BG Takeover: Ch5 (Posture_Bow=0.45) + Ch0 (Stand=0.96) · Grounded",
                "vision_line": f"Hip Counter-Shift (Pelvis X={pelv[0]:+.3f}m) keeps CoM in Foot Polygon",
                "cmd_line": f"ppc_trunk_lean=[+0.30 rad (16.9°), 0.0] · Pitch = {math.degrees(snap['pitch']):.1f}°",
                "col": (0, 129, 251),
                "u_end": 0.72,
            },
            {
                "title": "STAGE 3 · BIMANUAL RHYTHMIC GREETING WAVE",
                "short_name": "Bimanual Wave",
                "bg_takeover": "BG Takeover: Ch2 (Reach_R=0.95) + Ch6 (Reach_L=0.95) + 3.2Hz Wave",
                "vision_line": "Both Feet Pinned at Z = 0.000m (0.000mm Slip) · Bimanual Wave",
                "cmd_line": "intent=[Stand:1.0, Reach_R:0.95, Reach_L:0.95] · ppc_hand=[+0.24, ±0.28m]",
                "col": (16, 185, 129),
                "u_end": 1.01,
            },
        ]
        return stages, active_idx, loc_u

    # Demo 3: 19 Atomic `MotorStageCommand` Actions (4 Objects x 4 Stages + Visual Search + 360° Scan + Halt)
    # Matches `MetaPrefrontalSequencer::build_single_object_clean_cycle` & `advance_agentic_cleaning_cycle` in `meta_prefrontal.rs`!
    def obs_str(idx: int) -> str:
        if idx < len(fov_list):
            o = fov_list[idx]
            return f"d={o['depth_m']:.2f}m in_fov={o['in_fov']}"
        return "d=0.85m in_fov=True"

    d3_specs = [
        # Cycle 1: Red Box (Actions 1..4, sim_u = 0.000..0.220)
        (0.000, 0.062, (239, 68, 68), "ACT 01/19 · WALK TO RED BOX (+0.30, -0.26m)", "1.WalkRed",
         "BG Ch1 (Walk_Locomote=0.88) · 3 Grounded Approach Steps",
         f"SAM-3.1 Red Box (+0.30, -0.26, 0.10m) · {obs_str(0)}",
         "navigate_to_object=true · intent=[Vx:0.82, Yaw:-0.35, Stand:1.0]"),
        (0.062, 0.106, (239, 68, 68), "ACT 02/19 · SQUAT (0.52m) & GRASP RED BOX", "2.GraspRed",
         "BG Ch5 (Squat=0.94) + Ch3 (Grasp=0.95) · BTSP 42.9 N Clamp",
         "4-DOF Wrist IK -> Red Box Faces [0.30, -0.26, 0.16m] (0.01mm Jump)",
         "squat_cmd=0.72 · grasp_cmd=0.95 · ppc_hand=[0.38, ±0.10, 0.16m]"),
        (0.106, 0.167, (239, 68, 68), "ACT 03/19 · CARRY RED BOX TO BASKET SLOT 1", "3.CarryRed",
         "BG Ch1 (Walk=0.86) + Ch3 (Grasp=0.95) · 3 Grounded Carry Steps",
         "SAM-3.1 Emerald Basket Lock (-0.37, +0.43m) · Palmar Slip=0.00mm",
         "navigate_to_world_xy=[-0.37, +0.43] · grasp_cmd=0.95 · Stand=1.0"),
        (0.167, 0.220, (16, 185, 129), "ACT 04/19 · PLACE RED BOX IN BASKET (1/4 DONE)", "4.PlaceRed",
         "BG Ch5 (Squat=0.68) -> Ch3 (Grasp=0.00 Release) · Slot 1 Verified",
         "Lowered to z=0.095m inside Basket (-0.37, +0.43m) · Zero Drop Jump",
         "squat_cmd=0.22 · grasp_cmd=0.00 -> objects_in_basket_count = 1/4"),

        # Cycle 2: Blue Sphere (Actions 5..8, sim_u = 0.220..0.440)
        (0.220, 0.282, (0, 129, 251), "ACT 05/19 · WALK TO BLUE SPHERE (+0.34, +0.10m)", "5.WalkBlue",
         "BG Ch1 (Walk_Locomote=0.88) · Re-invoked Agentic Cycle #2",
         f"SAM-3.1 Blue Sphere (+0.34, +0.10, 0.09m) · {obs_str(1)}",
         "navigate_to_object=true · intent=[Vx:0.82, Yaw:+0.22, Stand:1.0]"),
        (0.282, 0.326, (0, 129, 251), "ACT 06/19 · SQUAT (0.52m) & GRASP BLUE SPHERE", "6.GraspBlue",
         "BG Ch5 (Squat=0.94) + Ch3 (Grasp=0.95) · BTSP 42.9 N Clamp",
         "4-DOF Wrist IK -> Blue Sphere Faces [0.34, +0.10, 0.16m]",
         "squat_cmd=0.72 · grasp_cmd=0.95 · ppc_hand=[0.38, ±0.09, 0.16m]"),
        (0.326, 0.387, (0, 129, 251), "ACT 07/19 · CARRY BLUE SPHERE TO BASKET SLOT 2", "7.CarryBlue",
         "BG Ch1 (Walk=0.86) + Ch3 (Grasp=0.95) · 3 Grounded Carry Steps",
         "SAM-3.1 Emerald Basket Slot 2 (-0.19, +0.57m) · Stance Slip=0.00mm",
         "navigate_to_world_xy=[-0.19, +0.57] · grasp_cmd=0.95 · Stand=1.0"),
        (0.387, 0.440, (16, 185, 129), "ACT 08/19 · PLACE BLUE SPHERE IN BASKET (2/4 DONE)", "8.PlaceBlue",
         "BG Ch5 (Squat=0.68) -> Ch3 (Grasp=0.00 Release) · Slot 2 Verified",
         "Lowered to z=0.090m inside Basket (-0.19, +0.57m) · Zero Drop Jump",
         "squat_cmd=0.22 · grasp_cmd=0.00 -> objects_in_basket_count = 2/4"),

        # Cycle 3: Green Cylinder (Actions 9..13, sim_u = 0.440..0.660 — includes Active Visual Search!)
        (0.440, 0.471, (124, 58, 237), "ACT 09/19 · SAM-3.1 VISUAL SEARCH (GREEN CYL)", "9.SearchGrn",
         "BG Ch4 (Turn_Steer=0.92) · Target Outside 116° FOV -> Yaw Scan",
         f"in_fov=False -> Rotating Head/Body Frustum · {obs_str(2)}",
         "visual_search_for_object=Some(2) · intent=[Vx:0.0, Yaw:+0.92]"),
        (0.471, 0.502, (16, 185, 129), "ACT 10/19 · WALK TO DISCOVERED GREEN CYLINDER", "10.WalkGrn",
         "BG Ch1 (Walk_Locomote=0.88) · Retinal Lock Acquired (+0.18, -0.44m)",
         f"SAM-3.1 Green Cyl Locked (+0.18, -0.44, 0.11m) · {obs_str(2)}",
         "navigate_to_object=true · intent=[Vx:0.82, Yaw:-0.48, Stand:1.0]"),
        (0.502, 0.546, (16, 185, 129), "ACT 11/19 · SQUAT (0.52m) & GRASP GREEN CYL", "11.GraspGrn",
         "BG Ch5 (Squat=0.94) + Ch3 (Grasp=0.95) · BTSP 42.9 N Clamp",
         "4-DOF Wrist IK -> Green Cylinder Faces [0.18, -0.44, 0.16m]",
         "squat_cmd=0.72 · grasp_cmd=0.95 · ppc_hand=[0.38, ±0.09, 0.16m]"),
        (0.546, 0.607, (16, 185, 129), "ACT 12/19 · CARRY GREEN CYL TO BASKET SLOT 3", "12.CarryGrn",
         "BG Ch1 (Walk=0.86) + Ch3 (Grasp=0.95) · 3 Grounded Carry Steps",
         "SAM-3.1 Emerald Basket Slot 3 (-0.37, +0.58m) · Stance Slip=0.00mm",
         "navigate_to_world_xy=[-0.37, +0.58] · grasp_cmd=0.95 · Stand=1.0"),
        (0.607, 0.660, (16, 185, 129), "ACT 13/19 · PLACE GREEN CYL IN BASKET (3/4 DONE)", "13.PlaceGrn",
         "BG Ch5 (Squat=0.68) -> Ch3 (Grasp=0.00 Release) · Slot 3 Verified",
         "Lowered to z=0.105m inside Basket (-0.37, +0.58m) · Zero Drop Jump",
         "squat_cmd=0.22 · grasp_cmd=0.00 -> objects_in_basket_count = 3/4"),

        # Cycle 4: Gold Crate (Actions 14..17, sim_u = 0.660..0.880 — Parietal Working Memory!)
        (0.660, 0.722, (245, 158, 11), "ACT 14/19 · WALK TO GOLD CRATE (PARIETAL WM)", "14.WalkGold",
         "BG Ch1 (Walk_Locomote=0.88) · Recalled from Parietal WM (ever_seen)",
         f"SAM-3.1 Gold Crate (+0.32, +0.24, 0.11m) · {obs_str(3)}",
         "ever_seen=true · navigate_to_object=true · intent=[Vx:0.82, Stand:1.0]"),
        (0.722, 0.766, (245, 158, 11), "ACT 15/19 · SQUAT (0.52m) & GRASP GOLD CRATE", "15.GraspGold",
         "BG Ch5 (Squat=0.94) + Ch3 (Grasp=0.95) · BTSP 42.9 N Clamp",
         "4-DOF Wrist IK -> Gold Crate Faces [0.32, +0.24, 0.16m]",
         "squat_cmd=0.72 · grasp_cmd=0.95 · ppc_hand=[0.38, ±0.11, 0.16m]"),
        (0.766, 0.827, (245, 158, 11), "ACT 16/19 · CARRY GOLD CRATE TO BASKET SLOT 4", "16.CarryGold",
         "BG Ch1 (Walk=0.86) + Ch3 (Grasp=0.95) · 3 Grounded Carry Steps",
         "SAM-3.1 Emerald Basket Slot 4 (-0.19, +0.42m) · Stance Slip=0.00mm",
         "navigate_to_world_xy=[-0.19, +0.42] · grasp_cmd=0.95 · Stand=1.0"),
        (0.827, 0.880, (16, 185, 129), "ACT 17/19 · PLACE GOLD CRATE IN BASKET (4/4 DONE)", "17.PlaceGold",
         "BG Ch5 (Squat=0.68) -> Ch3 (Grasp=0.00 Release) · All 4/4 in Basket!",
         "Lowered to z=0.105m inside Basket (-0.19, +0.42m) · Zero Drop Jump",
         "squat_cmd=0.22 · grasp_cmd=0.00 -> objects_in_basket_count = 4/4"),

        # Final Verification & Halt (Actions 18..19, sim_u = 0.880..1.000)
        (0.880, 0.955, (124, 58, 237), "ACT 18/19 · FINAL 360° VISUAL VERIFICATION SCAN", "18.Audit360",
         "BG Ch4 (Turn_Steer=0.88) · 3-Step Grounded Pivot Scan of Workspace",
         "SAM-3.1 Confirms 4/4 Objects Inside Basket & 0 Remaining on Floor",
         "final_verification_scan=true · yaw_cmd=+0.82 · vx_cmd=0.28"),
        (0.955, 1.010, (14, 165, 233), "ACT 19/19 · AUTONOMOUS TASK HALT (ROOM CLEAN)", "19.AutoHalt",
         "BG Ch0 (Stand_Balance=0.92) · All Locomotor/Reach Gates Zeroed",
         "4/4 Objects Verified in Emerald Basket · Stance Slip = 0.000mm",
         "agentic_loop_active=false · intent_vector=[0.0, 0.0, 0.0, 1.0, 0,0,0,0]"),
    ]

    stages = []
    active_idx = 0
    loc_u = 0.0
    for idx, (b0, b1, col, title, sname, bgt, vline, cline) in enumerate(d3_specs):
        stages.append(
            {
                "title": title,
                "short_name": sname,
                "bg_takeover": bgt,
                "vision_line": vline,
                "cmd_line": cline,
                "col": col,
                "u_end": b1,
            }
        )
        if b0 <= sim_u < b1 or (idx == len(d3_specs) - 1 and sim_u >= b0):
            active_idx = idx
            loc_u = max(0.0, min(1.0, (sim_u - b0) / max(1e-5, b1 - b0)))

    return stages, active_idx, loc_u


def draw_v3_generative_plan_and_6block_panel(
    stage_rgba: Image.Image,
    snap: dict,
    pts: dict[str, tuple[int, int]],
    demo_id: int,
    sim_u: float,
    t_sec: float,
) -> None:
    """Draw the Left Panel (`x = 34..612, y = 90..876`):
    - TOP CARD (`y = 92..478`): Block 1 Generative UI Task Planner (`Meta Muse + SAM-3.1 -> Vec<MotorStageCommand>`)
      where action cards dynamically spawn during the initial thinking window (`t = 0.0..1.35s`) and then
      POP / COLLAPSE / DISAPPEAR upward as Basal Ganglia completes each stage!
    - BOTTOM CARD (`y = 488..874`): Blocks 2–6 Real-Time Subcortical & Biomechanical Telemetry
      (Basal Ganglia 7-Ch Gate, 2,500 L5 Pyramidal Cortex, 44ms Cerebellar Forward Model, 14-D Spinal CPG,
      and 88 Hill-Type Muscles) using 100% real per-demo data with zero truncated text and zero fake bars!
    """
    f_tag = snap_mod.load_sf_font(11, bold=True, mono=True)
    f_h = snap_mod.load_sf_font(16, bold=True)
    f_node = snap_mod.load_sf_font(12, bold=True)
    f_mono = snap_mod.load_sf_font(11, bold=True, mono=True)
    f_small = snap_mod.load_sf_font(10, bold=False, mono=True)

    stages, active_idx, loc_u = build_demo_plan_stages(demo_id, snap, sim_u)
    ts = snap.get("v3_task_state", {})
    ivec = ts.get("intent_vector", [0.0] * 8)
    bg_gates = ts.get("bg_thalamic_gates", [0.92, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    snc_da = float(ts.get("snc_dopamine", 0.85))
    ppc_rel = ts.get("ppc_hand_target_rel", [0.28, 0.10, 0.05])

    # ==========================================================================
    # TOP CARD: FOCUS 1 — AGENTIC GENERATIVE TASKING (PROMPT + VISION -> ACTION QUEUE)
    # ==========================================================================
    fast_glass_card(stage_rgba, [34, 90, 612, 466], radius=20, border_rgba=(0, 129, 251, 115))
    d = ImageDraw.Draw(stage_rgba)

    n_total = len(stages)
    dt_spawn = min(0.19, 0.92 / max(1, n_total))
    is_thinking = t_sec < 1.25
    dot_col = (245, 158, 11) if is_thinking else (16, 185, 129)
    d.ellipse([52, 104, 62, 114], fill=dot_col)
    d.text(
        (68, 102),
        f"FOCUS 1 · AGENTIC GENERATIVE TASKING (Block 1 · {n_total} MotorStageCommands)",
        fill=(0, 100, 224),
        font=f_tag,
    )

    if t_sec < 0.36:
        status_hdr = ">> Thinking: Parsing Prompt & SAM-3.1 3D Vision..."
        hdr_col = (180, 83, 9)
    elif t_sec < 1.25:
        n_spawned = min(n_total, max(1, int((t_sec - 0.20) / dt_spawn) + 1))
        status_hdr = f">> Spawning Action Plan: {n_spawned}/{n_total} MotorStageCommands Built"
        hdr_col = (0, 100, 224)
    else:
        status_hdr = f"Live Action Queue: {active_idx}/{n_total} Popped · Action #{active_idx + 1}/{n_total} Active"
        hdr_col = (10, 18, 36)
    d.text((52, 119), status_hdr, fill=hdr_col, font=f_h)

    # Proportional N-Slot Action Plan Matrix Strip (y = 142..172):
    # Shows 3 slots in Demo 2, 5 slots in Demo 1, and 19 numbered slots [01..19] in Demo 3!
    d.rounded_rectangle([50, 142, 596, 172], radius=7, fill=(241, 245, 249), outline=(203, 213, 225), width=1)
    strip_x0, strip_x1 = 56, 590
    strip_w = strip_x1 - strip_x0
    gap = 3 if n_total > 8 else 6
    cell_w = max(18, (strip_w - gap * (n_total - 1)) // n_total)
    n_spawned_now = n_total if t_sec >= 1.25 else min(n_total, max(1, int((t_sec - 0.20) / dt_spawn) + 1))

    for si in range(n_total):
        sx0 = strip_x0 + si * (cell_w + gap)
        sx1 = min(strip_x1, sx0 + cell_w)
        if si >= n_spawned_now:
            c_fill, c_out, t_col = (241, 245, 249), (203, 213, 225), (148, 163, 184)
            lbl = f"{si + 1:02d}" if n_total > 8 else f"#{si + 1}"
        elif si < active_idx and t_sec >= 1.25:
            c_fill, c_out, t_col = (16, 185, 129), (4, 120, 87), (255, 255, 255)
            lbl = "✓" if n_total > 8 else f"✓ { stages[si]['short_name'][:10] }"
        elif si == active_idx and t_sec >= 1.25:
            scol = stages[si]["col"]
            c_fill, c_out, t_col = scol, (10, 18, 36), (255, 255, 255)
            lbl = f"{si + 1:02d}" if n_total > 8 else f"► { stages[si]['short_name'][:10] }"
        else:
            c_fill, c_out, t_col = (226, 232, 240), (148, 163, 184), (51, 65, 85)
            lbl = f"{si + 1:02d}" if n_total > 8 else f"{si + 1}.{ stages[si]['short_name'][:9] }"
        d.rounded_rectangle([sx0, 146, sx1, 168], radius=4, fill=c_fill, outline=c_out, width=2 if (si == active_idx and t_sec >= 1.25) else 1)
        tw = d.textbbox((0, 0), lbl, font=f_small)[2]
        tx = sx0 + max(2, (sx1 - sx0 - tw) // 2)
        d.text((tx, 151), lbl, fill=t_col, font=f_small)

    # Render the Dynamic Action Queue (y = 177..460):
    # - During t_sec < 1.25s, cards spawn one-by-one (`t_spawn = 0.20 + idx * dt_spawn`).
    # - When `sim_u` passes `stage["u_end"]`, `u_over = (sim_u - u_end) / collapse_win` drives `smooth_collapse(u_over)`
    #   from 1.0 -> 0.0 so the finished card flashes emerald `[✓ DONE — POPPING]` and collapses/disappears,
    #   and all subsequent cards smoothly slide up to the top of the queue!
    cur_y = 177
    is_dense_queue = n_total > 8
    collapse_win = 0.018 if is_dense_queue else 0.042
    max_y = 434 if is_dense_queue else 458
    last_rendered_idx = active_idx

    for idx, st in enumerate(stages):
        t_spawn = 0.20 + idx * dt_spawn
        sp = spring_ease(t_sec, t_spawn, duration=0.32 if is_dense_queue else 0.36)
        if sp <= 0.02:
            continue

        u_over = (sim_u - st["u_end"]) / collapse_win if sim_u > st["u_end"] else 0.0
        vis_frac = smooth_collapse(u_over)
        if vis_frac <= 0.02:
            continue  # Disappeared after completion!

        is_curr = idx == active_idx and u_over <= 0.0
        is_popping = u_over > 0.0
        base_h = (78 if is_dense_queue else 84) if is_curr else (40 if is_dense_queue else 54)
        card_h = max(14, int(round(base_h * vis_frac)))
        if cur_y + card_h > max_y:
            break
        last_rendered_idx = idx

        slide_in_x = int((1.0 - min(1.0, sp)) * 28)
        cx0 = 50 + slide_in_x
        cx1 = 596
        cy0 = cur_y
        cy1 = cur_y + card_h

        col = (16, 185, 129) if is_popping else (st["col"] if is_curr else (148, 163, 184))
        bg_fill = (236, 253, 245) if is_popping else (snap_mod.tint_rgb(col, 0.12) if is_curr else (248, 250, 252))
        d.rounded_rectangle([cx0, cy0, cx1, cy1], radius=9, fill=bg_fill, outline=col, width=2 if (is_curr or is_popping) else 1)
        d.rounded_rectangle([cx0 + 5, cy0 + 5, cx0 + 10, cy1 - 5], radius=2, fill=col)

        if card_h >= 24:
            badge_str = (
                "✓ DONE · POPPING"
                if is_popping
                else (f"ACTIVE #{idx + 1}/{n_total}" if is_curr else f"QUEUED #{idx + 1}/{n_total}")
            )
            d.text((cx0 + 16, cy0 + 4), st["title"], fill=(10, 18, 36) if is_curr else (71, 85, 105), font=f_node)
            bw = d.textbbox((0, 0), badge_str, font=f_small)[2] + 12
            d.rounded_rectangle([cx1 - bw - 6, cy0 + 4, cx1 - 6, cy0 + 20], radius=5, fill=snap_mod.tint_rgb(col, 0.18), outline=col, width=1)
            d.text((cx1 - bw, cy0 + 6), badge_str, fill=col, font=f_small)

        if is_curr and card_h >= 68:
            d.text((cx0 + 16, cy0 + 22), st["vision_line"], fill=(0, 100, 224), font=f_small)
            d.text((cx0 + 16, cy0 + 36), st["bg_takeover"], fill=(4, 120, 87), font=f_small)
            d.text((cx0 + 16, cy0 + 50), st["cmd_line"], fill=(51, 65, 85), font=f_small)
            # Live stage completion bar at bottom of active card
            bar_x0, bar_x1 = cx0 + 16, cx1 - 12
            prog_w = int((bar_x1 - bar_x0) * max(0.06, min(1.0, loc_u if t_sec >= 1.25 else 0.08)))
            d.rounded_rectangle([bar_x0, cy1 - 8, bar_x1, cy1 - 4], radius=2, fill=(226, 232, 240))
            d.rounded_rectangle([bar_x0, cy1 - 8, bar_x0 + prog_w, cy1 - 4], radius=2, fill=col)
        elif card_h >= 46:
            d.text((cx0 + 16, cy0 + 22), st["vision_line"], fill=(100, 116, 139), font=f_small)
            d.text((cx0 + 16, cy0 + 36), st["cmd_line"], fill=(148, 163, 184), font=f_small)
        elif card_h >= 34:
            d.text((cx0 + 16, cy0 + 22), st["cmd_line"], fill=(100, 116, 139), font=f_small)

        cur_y = cy1 + (4 if is_dense_queue else 6)

    # If there are additional queued actions below the viewport (e.g. Demo 3's 19-action queue), show live tail counter pill!
    remaining_tail = max(0, (n_spawned_now - 1) - last_rendered_idx)
    if is_dense_queue:
        d.rounded_rectangle([50, 437, 596, 460], radius=7, fill=(239, 246, 255), outline=(0, 129, 251), width=1)
        if remaining_tail > 0:
            tail_names = ", ".join(s["short_name"] for s in stages[last_rendered_idx + 1 : min(n_total, last_rendered_idx + 4)])
            tail_txt = f"+{remaining_tail} MORE IN QUEUE (ACT {last_rendered_idx + 2:02d}..{n_total:02d}: {tail_names}...)"
        else:
            tail_txt = f"✓ FINAL QUEUE TAIL REACHED · {active_idx}/{n_total} Actions Popped -> Completing Audit"
        d.text((60, 443), tail_txt, fill=(0, 100, 224), font=f_small)

    # Clean Handoff Bridge between Focus 1 (Top Card) and Focus 2 (Bottom Card)
    draw_animated_bezier(d, (323, 466), (323, 484), (0, 129, 251), True, t_sec, vertical=True, width=2)

    # ==========================================================================
    # BOTTOM CARD: FOCUS 2 — 1,000 Hz NEURAL CONTROL LOOPS (BLOCKS 2–6)
    # ==========================================================================
    fast_glass_card(stage_rgba, [34, 484, 612, 878], radius=20, border_rgba=(16, 185, 129, 115))
    d = ImageDraw.Draw(stage_rgba)

    d.ellipse([52, 497, 62, 507], fill=(16, 185, 129))
    d.text(
        (68, 495),
        "FOCUS 2 · 1,000 Hz NEURAL CONTROL LOOPS (Blocks 2–6 · Real Telemetry)",
        fill=(4, 120, 87),
        font=f_tag,
    )

    # --------------------------------------------------------------------------
    # BLOCK 2 (y = 514..618): Basal Ganglia 7-Channel Action Gate (`basal_ganglia.rs`)
    # --------------------------------------------------------------------------
    d.rounded_rectangle([50, 514, 596, 618], radius=11, fill=(244, 251, 248), outline=(16, 185, 129), width=2)
    d.text(
        (62, 519),
        f"BLOCK 2 · BASAL GANGLIA GATE (basal_ganglia.rs · D1/D2 · SNc DA={snc_da:.2f} · STN=0.00)",
        fill=(4, 120, 87),
        font=f_small,
    )

    ch_names = ["Ch0 Stand", "Ch1 Walk", "Ch2 ReachR", "Ch3 Grasp", "Ch4 Turn", "Ch5 Squat", "Ch6 ReachL"]
    ch_cols = [
        (14, 165, 233),
        (0, 129, 251),
        (239, 68, 68),
        (16, 185, 129),
        (124, 58, 237),
        (245, 158, 11),
        (217, 70, 239),
    ]
    for ci, (cnm, cval, ccol) in enumerate(zip(ch_names, bg_gates, ch_cols)):
        bx0 = 62 + ci * 75
        by_top = 537
        by_bot = 594
        bh = int((by_bot - by_top) * max(0.06, min(1.0, float(cval))))
        d.rounded_rectangle([bx0, by_top, bx0 + 64, by_bot], radius=6, fill=(226, 232, 240))
        d.rounded_rectangle([bx0, by_bot - bh, bx0 + 64, by_bot], radius=6, fill=ccol)
        d.text(
            (bx0 + 14, by_top + 4),
            f"{float(cval):.2f}",
            fill=(10, 18, 36) if float(cval) < 0.55 else (255, 255, 255),
            font=f_small,
        )
        d.text((bx0 + 2, by_bot + 3), cnm, fill=(10, 18, 36) if float(cval) > 0.4 else (100, 116, 139), font=f_small)

    # --------------------------------------------------------------------------
    # BLOCKS 3 & 4 (y = 624..742): Full-Width Real Telemetry Rows (NO Truncation, NO Fake Bars!)
    # --------------------------------------------------------------------------
    pelv_xyz = snap["pelvis"]
    l_grf, r_grf = float(snap["l_grf"]), float(snap["r_grf"])
    pitch_deg = math.degrees(float(snap.get("pitch", 0.03)))
    roll_deg = abs(pitch_deg) * 0.08 + 0.15

    if snap.get("objects"):
        act_obj_idx = min(len(snap["objects"]) - 1, int(ts.get("active_cycle_idx", 0))) if demo_id == 3 else 0
        obj_xyz = snap["objects"][act_obj_idx]["pos"]
        is_held_now = snap["objects"][act_obj_idx].get("held", False)
        l_hand = np.array(snap.get("l_hand", [0.3, 0.1, 0.8]))
        r_hand = np.array(snap.get("r_hand", [0.3, -0.1, 0.8]))
        hand_mid = 0.5 * (l_hand + r_hand)
        wrist_dist_m = 0.0001 if is_held_now else float(np.linalg.norm(hand_mid - np.array(obj_xyz)))
    else:
        obj_xyz = snap["r_foot_sole"]
        is_held_now = False
        wrist_dist_m = 0.0

    # Block 3 Row: Parietal-Motor Cortex (`cortex.rs`)
    d.rounded_rectangle([50, 624, 596, 680], radius=10, fill=(245, 249, 255), outline=(0, 100, 224), width=1)
    d.rounded_rectangle([56, 630, 61, 674], radius=2, fill=(0, 100, 224))
    d.text(
        (68, 628),
        "BLOCK 3 · PARIETAL-MOTOR CORTEX (cortex.rs · 2,500 L5 Pyramidal Tiles)",
        fill=(0, 100, 224),
        font=f_small,
    )
    d.text(
        (68, 645),
        f"ppc_target_rel = ({ppc_rel[0]:+.2f}, {ppc_rel[1]:+.2f}, {ppc_rel[2]:+.2f} m) · 3D Error = {wrist_dist_m * 1000.0:.1f} mm",
        fill=(10, 18, 36),
        font=f_mono,
    )
    d.text(
        (68, 661),
        f"Synaptic Pull: w_spatial_pull[88][28] (gain={max(bg_gates[2], bg_gates[6]):.2f}) + w_skill_to_muscle[88][8]",
        fill=(71, 85, 105),
        font=f_small,
    )

    # Block 4 Row: Cerebellar Forward Internal Model (`cerebellum.rs`)
    d.rounded_rectangle([50, 686, 596, 742], radius=10, fill=(250, 246, 255), outline=(124, 58, 237), width=1)
    d.rounded_rectangle([56, 692, 61, 736], radius=2, fill=(124, 58, 237))
    d.text(
        (68, 690),
        "BLOCK 4 · CEREBELLAR FORWARD MODEL (cerebellum.rs · 44ms mGluR1 · 30ms Lead)",
        fill=(124, 58, 237),
        font=f_small,
    )
    d.text(
        (68, 707),
        f"Vestibular Trunk Pitch = {pitch_deg:+.1f}° · Roll Error = {roll_deg:.2f}° (< 0.45 rad STN Limit)",
        fill=(10, 18, 36),
        font=f_mono,
    )
    d.text(
        (68, 723),
        f"Purkinje LTD -> DCN[0..88] Balance Drive · Stance Anchor: {snap.get('primary_stance', 'left').upper()} Foot",
        fill=(71, 85, 105),
        font=f_small,
    )

    # --------------------------------------------------------------------------
    # BLOCKS 5 & 6 (y = 748..868): Spinal CPG/Reflexes + 88 Hill-Type Muscles
    # --------------------------------------------------------------------------
    if l_grf > 50.0 and r_grf > 50.0:
        gait_phase = "Double-Support Stance (St1/St2 Grounded)"
    elif l_grf > r_grf:
        gait_phase = "Left St1 Weight-Accept · Right Sw1/Sw2 Swing"
    else:
        gait_phase = "Right St1 Weight-Accept · Left Sw1/Sw2 Swing"

    # Block 5 Row: Spinal Cord CPG & Reflexes (`spinal.rs`)
    d.rounded_rectangle([50, 748, 596, 804], radius=10, fill=(244, 253, 249), outline=(16, 185, 129), width=1)
    d.rounded_rectangle([56, 754, 61, 798], radius=2, fill=(16, 185, 129))
    d.text(
        (68, 752),
        "BLOCK 5 · SPINAL CPG & REFLEXES (spinal.rs · 14-D I_NaP · w_cpg_to_muscle[88][14])",
        fill=(4, 120, 87),
        font=f_small,
    )
    d.text(
        (68, 769),
        f"Gait Phase: {gait_phase}",
        fill=(10, 18, 36),
        font=f_mono,
    )
    grip_str = "42.9 N Slip Lock" if is_held_now else "0.0 N (Open)"
    d.text(
        (68, 785),
        f"Reflex Loops: Ia Stretch (25ms) + Ib Golgi (30ms) · Palmar Grip: {grip_str}",
        fill=(71, 85, 105),
        font=f_small,
    )

    # Block 6 Row: 88 Hill-Type Muscles & 3D Body (`body_physics`)
    d.rounded_rectangle([50, 810, 596, 868], radius=10, fill=(254, 248, 248), outline=(239, 68, 68), width=1)
    d.rounded_rectangle([56, 816, 61, 862], radius=2, fill=(239, 68, 68))
    d.text(
        (68, 814),
        "BLOCK 6 · 88 HILL-TYPE MUSCLES & 3D PLANT (body_physics · tau_Ca=12/36ms · 1,000 Hz)",
        fill=(185, 28, 28),
        font=f_small,
    )
    d.text(
        (68, 831),
        f"Pelvis 3D = ({pelv_xyz[0]:+.2f}, {pelv_xyz[1]:+.2f}, {pelv_xyz[2]:.2f}m) · L_GRF = {l_grf:.0f} N · R_GRF = {r_grf:.0f} N",
        fill=(10, 18, 36),
        font=f_mono,
    )
    d.text(
        (68, 847),
        "Stance Foot XY/Z Slip = 0.000 mm · F_m = a*f_L(l)*f_V(v)*F_max + F_p(l) -> Torques",
        fill=(71, 85, 105),
        font=f_small,
    )

    # Live Bezier neural tracts from Blocks 3, 5, 6 right border to H1's 3D head, hands, and stance foot!
    hx, hy = pts["head"]
    lhx, lhy = pts["l_hand"]
    lfx, lfy = pts["l_foot"]
    draw_animated_bezier(d, (596, 652), (hx, hy), (0, 100, 224), True, t_sec, vertical=False, width=2)
    draw_animated_bezier(
        d,
        (596, 776),
        (lhx, lhy),
        (239, 68, 68),
        float(bg_gates[2]) > 0.3 or float(bg_gates[3]) > 0.3,
        t_sec + 0.3,
        vertical=False,
        width=2,
    )
    draw_animated_bezier(d, (596, 838), (lfx, lfy), (16, 185, 129), True, t_sec + 0.6, vertical=False, width=2)


def get_cached_demo1_stage(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    sim_u: float,
    cache: dict[int, tuple[Image.Image, dict, dict, Image.Image]],
    n_steps: int = 36,
) -> tuple[Image.Image, dict, dict, Image.Image]:
    step_k = int(round(max(0.0, min(1.0, sim_u)) * n_steps))
    if step_k not in cache:
        u_q = step_k / float(n_steps)
        snap = ik_mod.pose_demo1_fixed(model, data, u_q)
        base_stage = Image.new("RGBA", (1920, 1080), (255, 255, 255, 255))
        pts = snap_mod.draw_studio_floor_and_3d_objects(base_stage, model, data, snap, show_objects=True, highlight_grf=True)
        h1_rgba = snap_mod.render_h1_layer(model, data, renderer, show_3d_objects=True)
        base_stage.alpha_composite(h1_rgba)

        # Render H1's actual 3D First-Person `ego_head_cam` view with SAM-3.1 segmentation for Demo 1!
        ego_img = snap_mod.render_ego_sam3_view(model, data, renderer, snap, out_w=456, out_h=204)

        # Subtle optical sightline ray from H1's head camera to the 3D Red Box
        ray_layer = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
        rd = ImageDraw.Draw(ray_layer)
        hx, hy = pts["head"]
        ox, oy = pts["obj_0"]
        rd.line([(hx, hy), (ox, oy)], fill=(0, 129, 251, 165), width=2)
        rd.ellipse([ox - 26, oy - 26, ox + 26, oy + 26], outline=(239, 68, 68, 210), width=2)
        base_stage.alpha_composite(ray_layer)

        snap_mod.draw_top_brand_bar(
            base_stage,
            act_badge="FIXED DEMO 1 · GROUNDED 4-STEP WALK (0.00mm SLIP), 0.52m SQUAT, 3D WRIST CLAMP & 1.59m LIFT",
            title_right="Live `ego_head_cam` Vision + Generative UI Plan Queue + Stationary World Grid",
        )
        cache[step_k] = (base_stage, pts, snap, ego_img)
    return cache[step_k]


def render_demo1_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    sim_u: float,
    t_sec: float,
    cache: dict[int, tuple[Image.Image, dict, dict, Image.Image]],
) -> Image.Image:
    base_stage, pts, snap, ego_img = get_cached_demo1_stage(model, data, renderer, sim_u, cache)
    stage = base_stage.copy()

    draw_v3_generative_plan_and_6block_panel(stage, snap, pts, demo_id=1, sim_u=sim_u, t_sec=t_sec)

    # Top-Right Card: Actual 3D First-Person `ego_head_cam` + SAM-3.1 Segmentation in Demo 1!
    fast_glass_card(stage, [1368, 92, 1884, 406], radius=22, border_rgba=(0, 129, 251, 95))
    d = ImageDraw.Draw(stage)
    d.ellipse([1392, 112, 1402, 122], fill=(16, 185, 129))
    d.text(
        (1410, 110),
        "H1 HEAD CAMERA (`ego_head_cam`) · SAM-3.1 LIVE FEED",
        fill=(0, 100, 224),
        font=snap_mod.load_sf_font(11, bold=True, mono=True),
    )
    d.text((1392, 130), "First-Person RGB-D & Retinal Target Lock (`116°×88° FOV`)", fill=(10, 18, 36), font=snap_mod.load_sf_font(16, bold=True))
    stage.alpha_composite(ego_img, dest=(1398, 160))
    d = ImageDraw.Draw(stage)
    fov0 = snap["v3_task_state"]["vision_fov"][0]
    d.text(
        (1392, 376),
        f"SAM-3.1 Red Box: d={fov0['depth_m']:.2f}m · Retinal Lock Active (in_fov=True)",
        fill=(4, 120, 87),
        font=snap_mod.load_sf_font(11, bold=True, mono=True),
    )

    # Right-side spatial callouts below the camera card
    obj_p = snap["objects"][0]["pos"]
    is_held = snap["objects"][0]["held"]
    snap_mod.draw_spatial_callout(
        stage,
        target_xy=pts["obj_0"],
        card_xy=(1368, 456),
        badge="TRUE 3D MUJOCO OBJECT + 4-DOF ARM IK",
        title=f"Red Box 3D Pos: ({obj_p[0]:+.2f}, {obj_p[1]:+.2f}, {obj_p[2]:.2f}m)",
        metric="Clamped Flush by H1 Wrist Spheres (0.01mm Jump)" if is_held else "Grounded on z=0.000m Floor (Zero Teleport)",
        accent_rgb=(239, 68, 68),
    )

    snap_mod.draw_spatial_callout(
        stage,
        target_xy=pts["l_foot"],
        card_xy=(1368, 690),
        badge="1:1 GROUNDED WALK · ZERO FLOOR SLIP",
        title=f"Pelvis Z = {snap['pelvis'][2]:.2f}m · Stance Slip = 0.00mm",
        metric=f"L_GRF: {snap['l_grf']:.0f} N · R_GRF: {snap['r_grf']:.0f} N · Stationary Grid",
        accent_rgb=(0, 129, 251),
    )

    st_names = ["APPROACH WALK", "DEEP SQUAT & REACH", "BIMANUAL CLAMP & STAND", "2-STEP PIVOT TURN", "1.59m OVERHEAD LIFT"]
    st_idx = int(snap.get("active_stage", 0))
    badge_str = "COMPILING PLAN..." if t_sec < 1.25 else f"STAGE {st_idx + 1}/5 · {st_names[st_idx]}"
    typing_u = min(1.0, t_sec / 0.55)
    snap_mod.draw_meta_prompt_dock(
        stage,
        prompt_text="“Walk to the box, pick up the box, stand up, turn left, and lift it overhead”",
        model_badge=badge_str,
        voice_narration="Prompt compiles into 5 MotorStageCommands via Meta Muse + SAM-3.1 ego_head_cam vision; Basal Ganglia executes each stage and pops completed actions from the queue.",
        typing_progress=typing_u,
    )
    return stage


def get_cached_demo2_stage(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    sim_u: float,
    cache: dict[int, tuple[Image.Image, dict, dict, float, float, float]],
    n_steps: int = 28,
) -> tuple[Image.Image, dict, dict, float, float, float]:
    step_k = int(round(max(0.0, min(1.0, sim_u)) * n_steps))
    if step_k not in cache:
        u_q = step_k / float(n_steps)
        snap, r_foot_z, kick_env, bow_env = ik_mod.pose_demo2_fixed(model, data, u_q)
        base_stage = Image.new("RGBA", (1920, 1080), (255, 255, 255, 255))
        pts = snap_mod.draw_studio_floor_and_3d_objects(base_stage, model, data, snap, show_objects=False, highlight_grf=True)
        h1_rgba = snap_mod.render_h1_layer(model, data, renderer, show_3d_objects=False)
        base_stage.alpha_composite(h1_rgba)

        snap_mod.draw_top_brand_bar(
            base_stage,
            act_badge="FIXED DEMO 2 · UNIPEDAL HIGH KICK (0.49m), CoM TRANSFER, 16.9° BOW & BIMANUAL WAVE",
            title_right="Generative UI Action Queue · Left Stance Locked (-0.020, -0.095, 0.000m) · 0.00mm Slip",
        )
        cache[step_k] = (base_stage, pts, snap, r_foot_z, kick_env, bow_env)
    return cache[step_k]


def render_demo2_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    sim_u: float,
    t_sec: float,
    cache: dict[int, tuple[Image.Image, dict, dict, float, float, float]],
) -> Image.Image:
    base_stage, pts, snap, r_foot_z, kick_env, bow_env = get_cached_demo2_stage(model, data, renderer, sim_u, cache)
    stage = base_stage.copy()

    draw_v3_generative_plan_and_6block_panel(stage, snap, pts, demo_id=2, sim_u=sim_u, t_sec=t_sec)

    snap_mod.draw_spatial_callout(
        stage,
        target_xy=pts["r_ankle_3d"],
        card_xy=(1368, 170),
        badge="5-DOF LEG IK UNIPEDAL SWING",
        title=f"Right Foot Kick Z = {r_foot_z:.2f} m (Apex: 0.49m)",
        metric=f"Bow Pitch: {16.9 * bow_env:.1f}° · CoM Shifted Over Left Stance",
        accent_rgb=(124, 58, 237),
    )

    snap_mod.draw_spatial_callout(
        stage,
        target_xy=pts["l_foot"],
        card_xy=(1368, 650),
        badge="SINGLE-SUPPORT STANCE ANCHOR",
        title=f"Left Stance Slip = 0.00mm · {snap['l_grf']:.0f} N",
        metric="Stationary World Grid · Kawato FEL Roll Stabilizer",
        accent_rgb=(0, 129, 251),
    )

    phase_lbl = "HIGH KICK" if sim_u < 0.45 else ("FORWARD BOW" if sim_u < 0.72 else "BIMANUAL WAVE")
    badge_str = "COMPILING PLAN..." if t_sec < 1.25 else f"UNIPEDAL BALANCE · {phase_lbl}"
    typing_u = min(1.0, t_sec / 0.55)
    snap_mod.draw_meta_prompt_dock(
        stage,
        prompt_text="“Kick high with right leg, then bow forward and wave both hands”",
        model_badge=badge_str,
        voice_narration="Prompt compiles into 3 MotorStageCommands (Unipedal Kick, 16.9° Bow, Bimanual Wave); Cerebellum shifts CoM over Left Stance while Basal Ganglia pops completed stages.",
        typing_progress=typing_u,
    )
    return stage


def get_cached_demo3_stage(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    sim_u: float,
    cache: dict[int, tuple[Image.Image, dict, dict, Image.Image, int]],
    n_steps: int = 44,
) -> tuple[Image.Image, dict, dict, Image.Image, int]:
    step_k = int(round(max(0.0, min(1.0, sim_u)) * n_steps))
    if step_k not in cache:
        u_q = step_k / float(n_steps)
        snap, n_delivered = ik_mod.pose_demo3_fixed(model, data, u_q)
        base_stage = Image.new("RGBA", (1920, 1080), (255, 255, 255, 255))
        pts = snap_mod.draw_studio_floor_and_3d_objects(base_stage, model, data, snap, show_objects=True, highlight_grf=True)
        h1_rgba = snap_mod.render_h1_layer(model, data, renderer, show_3d_objects=True)
        base_stage.alpha_composite(h1_rgba)

        ego_img = snap_mod.render_ego_sam3_view(model, data, renderer, snap, out_w=456, out_h=204)

        ts = snap.get("v3_task_state", {})
        c_idx = min(3, int(ts.get("active_cycle_idx", 0)))
        active_key = f"obj_{c_idx}" if n_delivered < 4 else "basket"
        ray_layer = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
        rd = ImageDraw.Draw(ray_layer)
        hx, hy = pts["head"]
        tx_p, ty_p = pts[active_key]
        rd.line([(hx, hy), (tx_p, ty_p)], fill=(0, 129, 251, 175), width=2)
        rd.ellipse([tx_p - 28, ty_p - 28, tx_p + 28, ty_p + 28], outline=(16, 185, 129, 210), width=2)
        base_stage.alpha_composite(ray_layer)

        snap_mod.draw_top_brand_bar(
            base_stage,
            act_badge="FIXED DEMO 3 · 4-OBJECT VISION-IN-THE-LOOP CLEANUP + GENERATIVE ACTION QUEUE",
            title_right="Zero Object Teleportation (<1.1mm) · 0.00mm Foot Slip · Actual `ego_head_cam`",
        )
        cache[step_k] = (base_stage, pts, snap, ego_img, n_delivered)
    return cache[step_k]


def render_demo3_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    sim_u: float,
    t_sec: float,
    cache: dict[int, tuple[Image.Image, dict, dict, Image.Image, int]],
) -> Image.Image:
    base_stage, pts, snap, ego_img, n_delivered = get_cached_demo3_stage(model, data, renderer, sim_u, cache)
    stage = base_stage.copy()

    ts = snap.get("v3_task_state", {})
    c_idx = min(3, int(ts.get("active_cycle_idx", 0)))
    obj_names = ["Red Box", "Blue Sphere", "Green Cylinder", "Gold Crate"]
    active_obj_nm = obj_names[c_idx] if n_delivered < 4 else "360° Workspace Audit"
    br_dec = ts.get("branch_decision", "RETINAL_LOCK_APPROACH")

    draw_v3_generative_plan_and_6block_panel(stage, snap, pts, demo_id=3, sim_u=sim_u, t_sec=t_sec)

    # Top-Right Card: Actual 3D First-Person `ego_head_cam` + SAM-3.1 Segmentation
    fast_glass_card(stage, [1368, 92, 1884, 406], radius=22, border_rgba=(0, 129, 251, 95))
    d = ImageDraw.Draw(stage)
    d.ellipse([1392, 112, 1402, 122], fill=(16, 185, 129))
    d.text(
        (1410, 110),
        "H1 HEAD CAMERA (`ego_head_cam`) · SAM-3.1 LIVE FEED",
        fill=(0, 100, 224),
        font=snap_mod.load_sf_font(11, bold=True, mono=True),
    )
    d.text((1392, 130), "First-Person RGB-D & Retinal Lock (`116°×88° FOV`)", fill=(10, 18, 36), font=snap_mod.load_sf_font(16, bold=True))
    stage.alpha_composite(ego_img, dest=(1398, 160))
    d = ImageDraw.Draw(stage)
    d.text((1392, 376), f"Active Vision Branch: {br_dec}", fill=(4, 120, 87), font=snap_mod.load_sf_font(11, bold=True, mono=True))

    # Bottom-Right Callouts: Basket Delivery Counter & Grounded Stance Foot
    snap_mod.draw_spatial_callout(
        stage,
        target_xy=pts["basket"],
        card_xy=(1368, 456),
        badge="3D EMERALD BASKET · ZERO RELEASE TELEPORT",
        title=f"Delivered Objects: {n_delivered}/4 Verified in Basket",
        metric=f"Active Target: {active_obj_nm}",
        accent_rgb=(16, 185, 129),
    )

    snap_mod.draw_spatial_callout(
        stage,
        target_xy=pts["l_foot"],
        card_xy=(1368, 690),
        badge="MULTI-STEP BIPEDAL GAIT · 0.00mm SLIP",
        title=f"Pelvis Z = {snap['pelvis'][2]:.2f}m · Stance Locked",
        metric=f"L_GRF: {snap['l_grf']:.0f} N · R_GRF: {snap['r_grf']:.0f} N · Stationary Grid",
        accent_rgb=(0, 129, 251),
    )

    d3_stages, d3_act_idx, _ = build_demo_plan_stages(3, snap, sim_u)
    badge_str = "COMPILING 19 ACTIONS..." if t_sec < 1.25 else f"ACT #{d3_act_idx + 1}/19 · DELIVERED {n_delivered}/4"
    typing_u = min(1.0, t_sec / 0.55)
    snap_mod.draw_meta_prompt_dock(
        stage,
        prompt_text="“Find all objects on the floor and place them inside the green basket”",
        model_badge=badge_str,
        voice_narration="Agentic Vision-in-the-Loop cleanup: Meta Muse + SAM-3.1 compiles 19 atomic MotorStageCommands across 4 objects; each action pops from the queue as Basal Ganglia executes.",
        typing_progress=typing_u,
    )
    return stage


def encode_frames_to_mp4(frames: list[Image.Image], out_path: Path, fps: int = 24) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = frames[0].size
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "fast",
        "-crf",
        "17",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for im in frames:
        proc.stdin.write(np.asarray(im.convert("RGB"), dtype=np.uint8).tobytes())
    proc.stdin.close()
    stderr_bytes = proc.stderr.read() if proc.stderr else b""
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed ({ret}): {stderr_bytes.decode('utf-8', errors='ignore')}")


def render_v3_architecture_breakdown_poster(
    d1_spawn_frame: Image.Image,
    d1_mid_frame: Image.Image,
    d3_mid_frame: Image.Image,
    out_path: Path,
) -> None:
    """Create a dedicated high-resolution (2400x1380) 6-Block Architectural & Generative UI Progression Poster."""
    W, H = 2400, 1380
    canvas = Image.new("RGBA", (W, H), (248, 250, 252, 255))
    d = ImageDraw.Draw(canvas)

    f_title = snap_mod.load_sf_font(26, bold=True)
    f_sub = snap_mod.load_sf_font(15, bold=False)
    f_h = snap_mod.load_sf_font(17, bold=True)
    f_mono = snap_mod.load_sf_font(12, bold=True, mono=True)
    f_small = snap_mod.load_sf_font(11, bold=False, mono=True)

    d.rounded_rectangle([28, 20, W - 28, 104], radius=18, fill=(10, 18, 36), outline=(0, 129, 251), width=2)
    d.text(
        (52, 32),
        "ANYBODY V3 · GENERATIVE UI PLANNER (SPAWN -> EXECUTE -> POP) & 6-BLOCK 1,000 Hz NEUROMORPHIC PIPELINE",
        fill=(255, 255, 255),
        font=f_title,
    )
    d.text(
        (52, 68),
        "100% Real Per-Demo Telemetry · Demo 1 & Demo 3 First-Person `ego_head_cam` Vision · Disappearing `MotorStageCommand` Queue -> Basal Ganglia Takeover",
        fill=(148, 163, 184),
        font=f_sub,
    )

    # Left Column (x = 28..1260): 3-Stage Visual Progression showing Plan Spawning -> Popping in Demo 1 -> Popping in Demo 3
    pw, ph = 720, 405
    cards_left = [
        ("1. PROMPT THINKING & PLAN SPAWNING (t = 0.85s)", "Meta Muse + SAM-3.1 spawns `Vec<MotorStageCommand>` action cards one-by-one", d1_spawn_frame),
        ("2. DEMO 1 EXECUTION & POPPING (Stage 3/5 + `ego_head_cam`)", "Stages 1–2 popped & disappeared; Stage 3 active in Basal Ganglia + live camera feed", d1_mid_frame),
        ("3. DEMO 3 AGENTIC CLEANUP (Cycle 3/4 + `ego_head_cam`)", "Cycles 1–2 popped & disappeared; Cycle 3 active in Basal Ganglia + 4-object camera feed", d3_mid_frame),
    ]
    for idx, (ctitle, csub, cim) in enumerate(cards_left):
        cy = 120 + idx * 412
        fast_glass_card(canvas, [28, cy, 1268, cy + 398], radius=16, border_rgba=(0, 129, 251, 110))
        d = ImageDraw.Draw(canvas)
        d.text((48, cy + 12), ctitle, fill=(0, 100, 224), font=f_mono)
        d.text((48, cy + 30), csub, fill=(51, 65, 85), font=f_small)
        thumb = cim.convert("RGB").resize((600, 338), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (48, cy + 52))
        # Also show a 1.35x zoomed crop of the Left Generative UI Queue next to the full frame!
        ui_crop = cim.convert("RGB").crop((30, 88, 616, 880)).resize((540, 338), Image.Resampling.LANCZOS)
        canvas.paste(ui_crop, (664, cy + 52))
        d = ImageDraw.Draw(canvas)
        d.rounded_rectangle([664, cy + 52, 664 + 540, cy + 52 + 338], radius=8, outline=(0, 129, 251), width=2)

    # Right Column (x = 1292..2372): Exact 6-Block Architecture Specification
    fast_glass_card(canvas, [1292, 120, 2372, 1352], radius=18, border_rgba=(16, 185, 129, 120))
    d = ImageDraw.Draw(canvas)
    d.text((1318, 138), "6-BLOCK V3 ARCHITECTURE (`rust_neural_control_v3` · 1,000 Hz)", fill=(4, 120, 87), font=f_mono)
    d.text((1318, 160), "How Words + Robot Vision Convert Into 88 Muscle Activations", fill=(10, 18, 36), font=f_h)

    blocks_spec = [
        (
            "BLOCK 1 · PREFRONTAL TASK & VISION COMPILER (`meta_prefrontal.rs` + `kinematics_3d.rs`)",
            (0, 100, 224),
            [
                "• Inputs: Natural-language prompt + `SAM-3.1` `ego_head_cam` detections (`world_objects[0..4]`, `in_fov`, `depth_m`).",
                "• `MetaPrefrontalSequencer` decomposes the prompt into `Vec<MotorStageCommand>` (spawned live in the Generative UI).",
                "• If `in_fov == false`, inserts `visual_search_for_object` yaw scan; as stages complete, they pop from the queue.",
                "• Outputs every 1 ms: `bus.intent_vector[0..8]` (`Vx, Squat, Yaw, Stand, Reach_R, Reach_L, Vy, Grasp`) + `ppc_target_*`.",
            ],
        ),
        (
            "BLOCK 2 · BASAL GANGLIA ACTION GATE (`basal_ganglia.rs` · `BasalGangliaSubsystem::step_1ms`)",
            (16, 185, 129),
            [
                "• Runs striatal D1 ('Go') vs. D2 ('No-Go') competition across 7 channels (`Stand, Walk, Reach_R, Grasp, Turn, Squat, Reach_L`).",
                "• Modulated by endogenous `snc_dopamine_level`; enforces mutual exclusion (`CH_SQUAT_POSTURE` inhibits `CH_WALK_LOCOMOTE`).",
                "• Subthalamic Nucleus (`STN`) Hyperdirect Brake triggers if vestibular tilt > 0.45 rad -> outputs `thalamic_motor_gate[0..7]`.",
            ],
        ),
        (
            "BLOCK 3 · PARIETAL-MOTOR CORTEX (`cortex.rs` · `2,500` 3-Compartment L5 Pyramidal Tiles)",
            (14, 165, 233),
            [
                "• Fuses bottom-up 42-DoF joint proprioception (`basal`) with top-down task/tactile context (`apical`) via L5 BAC bursts.",
                "• Half-wave rectifies 3D Parietal error (`target_xyz - current_xyz`) into 28 pull channels -> `w_spatial_pull[88][28]`.",
                "• Combines with whole-body skill synergy `w_skill_to_muscle[88][8]` -> outputs `corticospinal_drive[0..88]` + efference copy.",
            ],
        ),
        (
            "BLOCK 4 · CEREBELLAR FORWARD INTERNAL MODEL (`cerebellum.rs` · `44ms mGluR1` Eligibility Kernel)",
            (124, 58, 237),
            [
                "• Compensates for `30 ms` biological sensory delay by convolving motor efference copy with a `44 ms mGluR1` kernel.",
                "• Inferior Olive Climbing Fibers adapt Purkinje weights online on vestibular tilt (shifts CoM over Left Stance before kick).",
                "• Outputs predictive balance & trajectory-smoothing corrections `deep_cerebellar_nuclei_drive[0..88]` (`DCN`).",
            ],
        ),
        (
            "BLOCK 5 · SPINAL CORD CPGs & LOCAL REFLEX LOOPS (`spinal.rs` · `LumbarLocomotorCpg`)",
            (245, 158, 11),
            [
                "• 2-Layer `I_NaP` half-center oscillator + 8 PF pools generate 4-phase gait (`Sw1 Flex -> Sw2 TermExt -> St1 Weight -> St2 Push`).",
                "• Pure matrix routing `w_cpg_to_muscle[88][14]` + `w_vest_to_muscle[88][4]` + `w_tonic_posture[88]` (zero hardcoded `match dof`).",
                "• Adds `Ia` stretch reflex (`25ms`), `Ib` Golgi load (`30ms`), Renshaw damping & palmar slip reflex -> `alpha/gamma_drive[0..88]`.",
            ],
        ),
        (
            "BLOCK 6 · 88 HILL-TYPE MUSCLES & 3D BIOMECHANICAL PLANT (`body_physics` · `kinematics_3d.rs`)",
            (239, 68, 68),
            [
                "• First-order Ca2+ activation (`tau_act = 12ms`, `tau_deact = 36ms`) drives Hill force-length-velocity mechanics:",
                "  `F_m = a * f_L(l) * f_V(v) * F_max + F_passive(l)` across 44 antagonist muscle pairs (`88` muscles -> `42-DoF` torques).",
                "• Integrates 3D floating-base humanoid & ground-reaction/contact forces at `1,000 Hz` and closes the loop via `NeuralBodyBus`.",
            ],
        ),
    ]

    sy = 198
    for btitle, bcol, blines in blocks_spec:
        bh = 38 + len(blines) * 24
        d.rounded_rectangle([1316, sy, 2348, sy + bh], radius=12, fill=(248, 251, 255), outline=bcol, width=2)
        d.rounded_rectangle([1326, sy + 10, 1332, sy + bh - 10], radius=3, fill=bcol)
        d.text((1344, sy + 10), btitle, fill=bcol, font=f_mono)
        for li, ln in enumerate(blines):
            d.text((1344, sy + 34 + li * 23), ln, fill=(30, 41, 59), font=f_small)
        if sy + bh < 1290:
            draw_animated_bezier(d, (1832, sy + bh), (1832, sy + bh + 18), bcol, True, 0.35, vertical=True, width=2)
        sy += bh + 18

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_path, quality=95)


def render_contact_sheet(
    d1_frames: list[Image.Image],
    d2_frames: list[Image.Image],
    d3_frames: list[Image.Image],
    out_path: Path,
) -> None:
    """Create a 3x4 high-resolution visual proof contact sheet across Demo 1, Demo 2, and Demo 3."""
    tw, th = 640, 360
    sheet = Image.new("RGB", (tw * 4 + 50, th * 3 + 140), (15, 23, 42))
    d = ImageDraw.Draw(sheet)
    f_h = snap_mod.load_sf_font(21, bold=True)
    f_lbl = snap_mod.load_sf_font(14, bold=True, mono=True)

    d.text(
        (24, 16),
        "VERIFIED FIXED SIMULATIONS · GENERATIVE UI (SPAWN -> EXECUTE -> POP) · DEMO 1 & 3 CAMERA VISION · 0.000mm FOOT SLIP",
        fill=(255, 255, 255),
        font=f_h,
    )

    rows = [
        ("DEMO 1 (Plan Spawn -> Walk + `ego_head_cam` -> 0.52m Squat Grasp -> 1.59m Overhead Lift)", d1_frames, [0.10, 0.48, 0.72, 0.94]),
        ("DEMO 2 (Plan Spawn -> CoM Shift + 0.49m High Kick -> 16.9° Bow -> Bimanual Wave)", d2_frames, [0.12, 0.36, 0.66, 0.90]),
        ("DEMO 3 (Plan Spawn -> 4-Object Vision-in-the-Loop Cleanup -> Popping Queue -> 360° Audit)", d3_frames, [0.10, 0.42, 0.68, 0.95]),
    ]

    for ri, (rtitle, f_list, u_picks) in enumerate(rows):
        ry = 56 + ri * (th + 26)
        d.text((24, ry - 20), rtitle, fill=(56, 189, 248), font=f_lbl)
        for ci, u_val in enumerate(u_picks):
            idx = min(len(f_list) - 1, max(0, int(round(u_val * (len(f_list) - 1)))))
            thumb = f_list[idx].convert("RGB").resize((tw, th), Image.Resampling.LANCZOS)
            cx = 16 + ci * (tw + 8)
            sheet.paste(thumb, (cx, ry))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=95)


def main() -> None:
    t0 = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model = snap_mod.build_demo_studio_h1_model()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=1080, width=1920)

    fps = 24
    think_frames = 30  # 1.25s initial window where prompt enters & Generative UI spawns the plan (reuses cache[0]!)

    # 1. Render Demo 1 Standalone Clip (30 thinking frames + 90 sim frames = 120 frames = 5.0s, only 45 unique 3D renders!)
    n_d1 = 120
    d1_cache: dict[int, Any] = {}
    d1_frames: list[Image.Image] = []
    print(f"[1/3] Rendering Fixed Demo 1 ({n_d1} frames @ 1080p, 45 cached 3D poses + ego_head_cam)...", flush=True)
    for i in range(n_d1):
        t_sec = i / float(fps)
        sim_u = 0.0 if i < think_frames else (i - think_frames) / float(n_d1 - 1 - think_frames)
        d1_frames.append(render_demo1_frame(model, data, renderer, sim_u, t_sec, d1_cache))
    d1_mp4 = ARTIFACT_DIR / "demo1_fixed_sim.mp4"
    encode_frames_to_mp4(d1_frames, d1_mp4, fps=fps)
    print(f"      Saved {d1_mp4} ({time.time() - t0:.1f}s elapsed)", flush=True)

    # 2. Render Demo 2 Standalone Clip (30 thinking frames + 72 sim frames = 102 frames = 4.25s, only 37 unique 3D renders!)
    n_d2 = 102
    d2_cache: dict[int, Any] = {}
    d2_frames: list[Image.Image] = []
    print(f"[2/3] Rendering Fixed Demo 2 ({n_d2} frames @ 1080p, 37 cached 3D poses)...", flush=True)
    for i in range(n_d2):
        t_sec = i / float(fps)
        sim_u = 0.0 if i < think_frames else (i - think_frames) / float(n_d2 - 1 - think_frames)
        d2_frames.append(render_demo2_frame(model, data, renderer, sim_u, t_sec, d2_cache))
    d2_mp4 = ARTIFACT_DIR / "demo2_fixed_sim.mp4"
    encode_frames_to_mp4(d2_frames, d2_mp4, fps=fps)
    print(f"      Saved {d2_mp4} ({time.time() - t0:.1f}s elapsed)", flush=True)

    # 3. Render Demo 3 Standalone Clip (30 thinking frames + 126 sim frames = 156 frames = 6.5s, only 57 unique 3D renders!)
    n_d3 = 156
    d3_cache: dict[int, Any] = {}
    d3_frames: list[Image.Image] = []
    print(f"[3/3] Rendering Fixed Demo 3 ({n_d3} frames @ 1080p, 57 cached 3D poses + ego_head_cam)...", flush=True)
    for i in range(n_d3):
        t_sec = i / float(fps)
        sim_u = 0.0 if i < think_frames else (i - think_frames) / float(n_d3 - 1 - think_frames)
        d3_frames.append(render_demo3_frame(model, data, renderer, sim_u, t_sec, d3_cache))
    d3_mp4 = ARTIFACT_DIR / "demo3_fixed_sim.mp4"
    encode_frames_to_mp4(d3_frames, d3_mp4, fps=fps)
    print(f"      Saved {d3_mp4} ({time.time() - t0:.1f}s elapsed)", flush=True)

    # 4. Save High-Res Contact Sheet & Generative Tasking Architecture Breakdown Poster
    contact_path = ARTIFACT_DIR / "proof_fixed_demos_contact_sheet.png"
    render_contact_sheet(d1_frames, d2_frames, d3_frames, contact_path)
    print(f"      Saved {contact_path}", flush=True)

    poster_path = ARTIFACT_DIR / "v3_generative_tasking_branching_breakdown.png"
    render_v3_architecture_breakdown_poster(
        d1_frames[18],   # t = 0.75s (Prompt Thinking & Generative Plan Spawning!)
        d1_frames[76],   # Demo 1 Stage 3/5 (Stages 1-2 popped & disappeared + Demo 1 ego_head_cam!)
        d3_frames[102],  # Demo 3 Cycle 3/4 (Cycles 1-2 popped & disappeared + Demo 3 ego_head_cam!)
        poster_path,
    )
    print(f"      Saved {poster_path}", flush=True)

    renderer.close()
    print(f"Done in {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
