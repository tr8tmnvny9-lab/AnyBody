"""Generate the AnyBody Silicon Valley Series-A Launch Intro Video (Photorealistic 4K + Studio Audio & Voiceover).

Supports two tiers:
  1. `--tier low` (960x540 fast iteration preview with full audio & photorealistic screen/boom compositing)
  2. `--tier 4k` (3840x2160 4K UHD master with 8192-shadow Unitree H1, 3D Depth SSAO, Studio HDRI metallic
     reflections, photorealistic American black walnut lower boom + circular titanium pull-ring, true physical
     top-down canvas unrolling, and 48kHz stereo keynote voiceover + tactile sound effects).
"""
from __future__ import annotations

import argparse
import asyncio
import math
from pathlib import Path
import subprocess
import time
import wave
import xml.etree.ElementTree as ET

import edge_tts
import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

MENAGERIE_H1_DIR = Path.home() / ".cache" / "robot_descriptions" / "mujoco_menagerie" / "unitree_h1"
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "target" / "video_artifacts"
WALNUT_PHOTO_PATH = REPO_ROOT / "assets" / "figures" / "walnut_boom_photo.jpg"


def _load_font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        candidates = [
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/SFNSMono.ttf",
            "/System/Library/Fonts/Supplemental/Courier New Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Courier New.ttf",
        ]
    elif bold:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/System/Library/Fonts/SFNS.ttf",
        ]
    else:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/System/Library/Fonts/SFNS.ttf",
        ]
    for c in candidates:
        try:
            if c.endswith("Menlo.ttc") and bold:
                return ImageFont.truetype(c, size=size, index=1)
            return ImageFont.truetype(c, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def prepare_photoreal_walnut_boom_assets(target_bar_w: int) -> dict:
    """Extract the photorealistic walnut boom & circular titanium pull-ring from the studio macro photograph."""
    im = Image.open(WALNUT_PHOTO_PATH).convert("RGB")
    arr = np.asarray(im, dtype=np.float32)

    # Compute alpha against pure #FFFFFF background while keeping interior metallic speculars 100% opaque
    dist = np.max(255.0 - arr, axis=2)
    raw_alpha = np.clip((dist - 7.0) / 18.0, 0.0, 1.0)

    # Fill only small bright specular highlight holes (< 450 px) so the circular pull-ring's center hole stays open!
    bg_holes = dist <= 14.0
    labeled, n_feat = ndimage.label(bg_holes)
    sizes = ndimage.sum(bg_holes, labeled, range(1, n_feat + 1))
    small_hole_mask = np.zeros_like(bg_holes)
    for idx, sz in enumerate(sizes, start=1):
        if sz < 450:
            small_hole_mask |= labeled == idx

    alpha = np.where(small_hole_mask, 1.0, raw_alpha)
    alpha_u8 = (alpha * 255.0).astype(np.uint8)
    rgba_full = np.dstack([im, alpha_u8])
    pil_rgba = Image.fromarray(rgba_full, mode="RGBA")

    # In the 1376x768 source photo:
    # - Horizontal walnut bar + titanium end-caps: x in [100..1275], y in [314..453] (w=1175, h=139)
    # - Center gunmetal clasp + braided black cord + circular titanium pull-ring: x in [410..558], y in [445..684]
    #   with pull-ring center at (484, 612) in source coordinates.
    bar_crop = pil_rgba.crop((100, 314, 1275, 453))
    ring_crop = pil_rgba.crop((410, 445, 558, 684))

    # Scale the walnut bar to `target_bar_w` with a sleek architectural height (bar_h = target_bar_w / 36)
    bar_h = max(14, int(round(target_bar_w / 36.0)))
    bar_scaled = bar_crop.resize((target_bar_w, bar_h), Image.Resampling.LANCZOS)

    # Scale the clasp + braided cord + circular titanium pull-ring isotropically (1:1 aspect ratio so ring is circular!)
    ring_scale = (target_bar_w / 1175.0) * 0.52
    ring_w = max(16, int(round(ring_crop.width * ring_scale)))
    ring_h = max(24, int(round(ring_crop.height * ring_scale)))
    ring_scaled = ring_crop.resize((ring_w, ring_h), Image.Resampling.LANCZOS)

    # Ring center offset relative to the top-left of `ring_scaled`:
    # In ring_crop (148x239), the circular ring hole center is at (74, 167)
    ring_cx_local = int(round(74 * ring_scale))
    ring_cy_local = int(round(167 * ring_scale))

    # Horizontal attachment point of the clasp along the bar (38.5% from left end of bar, matching H1's reach)
    clasp_bar_x = int(round(target_bar_w * 0.385))

    return {
        "bar_img": bar_scaled,
        "bar_w": target_bar_w,
        "bar_h": bar_h,
        "ring_img": ring_scaled,
        "ring_w": ring_w,
        "ring_h": ring_h,
        "ring_cx_local": ring_cx_local,
        "ring_cy_local": ring_cy_local,
        "clasp_bar_x": clasp_bar_x,
    }


def generate_anybody_canvas_image(width: int, height: int) -> np.ndarray:
    """Create the photorealistic matte woven projection screen canvas with the 'A://' directory destination logo."""
    # Fine woven matte projection screen fabric micro-texture
    rng = np.random.default_rng(7)
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]

    # Subtle gallery-grade off-white canvas tone (249..253) so the canvas has real physical presence against pure 255 white
    weave_h = np.sin(np.arange(height, dtype=np.float32)[:, None] * 1.57) * 0.8
    weave_v = np.sin(np.arange(width, dtype=np.float32)[None, :] * 1.57) * 0.8
    grain = rng.normal(0.0, 0.45, (height, width)).astype(np.float32)
    vignette = 252.0 - 2.5 * ((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
    luma = np.clip(vignette + weave_h + weave_v + grain, 246.0, 254.0)

    rgb = np.stack([luma, luma, np.clip(luma + 0.6, 0, 255)], axis=-1).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img)

    # Matte black fibreglass tension borders on left and right edges of the roll-down projection screen
    border_w = max(4, int(round(width * 0.011)))
    draw.rectangle([0, 0, border_w - 1, height - 1], fill=(22, 25, 32))
    draw.rectangle([width - border_w, 0, width - 1, height - 1], fill=(22, 25, 32))
    # Subtle inner seam line
    seam_x = border_w + max(2, int(round(width * 0.003)))
    draw.line([(seam_x, 0), (seam_x, height)], fill=(228, 232, 240), width=max(1, width // 1200))
    draw.line([(width - seam_x - 1, 0), (width - seam_x - 1, height)], fill=(228, 232, 240), width=max(1, width // 1200))

    # Position the `A://` logo on the canvas at a FIXED distance above the bottom walnut boom!
    # Because the canvas moves down WITH the walnut boom (`Screen[0:y_boom] = Canvas[H - y_boom : H]`),
    # placing the logo center at `cy = int(height * 0.50)` means:
    # - At rest (y_boom near 0), the logo is rolled up above the top bezel (`y < 0`).
    # - As H1 pulls the walnut boom down, the logo unrolls from the top bezel (`y = 0`) and travels DOWNWARD
    #   at the exact speed of the walnut boom until settling at the center of the screen!
    cx = int(width * 0.55)
    cy = int(height * 0.48)

    # 1. Hero Mark: `A://` File-Directory Root Destination
    f_hero = _load_font(int(height * 0.25), bold=True, mono=True)
    a_part = "A:"
    slash_part = "//"
    b_a = draw.textbbox((0, 0), a_part, font=f_hero)
    b_s = draw.textbbox((0, 0), slash_part, font=f_hero)
    w_a = b_a[2] - b_a[0]
    w_s = b_s[2] - b_s[0]
    h_hero = b_a[3] - b_a[1]

    cursor_w = max(8, int(width * 0.024))
    cursor_h = int(h_hero * 0.80)
    cursor_gap = max(6, int(width * 0.012))
    total_w = w_a + w_s + cursor_gap + cursor_w

    start_x = cx - total_w // 2
    start_y = cy - h_hero // 2 - int(height * 0.06)

    draw.text((start_x, start_y), a_part, fill=(14, 18, 26), font=f_hero)
    draw.text((start_x + w_a, start_y), slash_part, fill=(0, 129, 251), font=f_hero)

    cur_x0 = start_x + w_a + w_s + cursor_gap
    cur_y0 = cy - cursor_h // 2 - int(height * 0.008)
    draw.rounded_rectangle(
        [cur_x0, cur_y0, cur_x0 + cursor_w, cur_y0 + cursor_h],
        radius=max(2, int(width * 0.003)),
        fill=(14, 18, 26),
    )

    # 2. Brand Pill below `A://`
    f_dir = _load_font(max(12, int(height * 0.040)), bold=True, mono=True)
    dir_suffix = "AnyBody"
    bs = draw.textbbox((0, 0), dir_suffix, font=f_dir)
    ws = bs[2] - bs[0]
    hd = bs[3] - bs[1]

    pad_x = max(14, int(width * 0.016))
    pad_y = max(8, int(height * 0.012))
    pill_w = ws + pad_x * 2
    pill_h = hd + pad_y * 2
    px0 = cx - pill_w // 2
    py0 = cy + int(height * 0.14)
    px1 = px0 + pill_w
    py1 = py0 + pill_h
    draw.rounded_rectangle(
        [px0, py0, px1, py1],
        radius=max(6, int(pill_h * 0.32)),
        fill=(242, 246, 252),
        outline=(204, 214, 230),
        width=max(1, int(width * 0.0012)),
    )

    tx_dir = px0 + pad_x
    ty_dir = py0 + pad_y - max(1, int(height * 0.003))
    draw.text((tx_dir, ty_dir), dir_suffix, fill=(0, 100, 224), font=f_dir)

    # 3. Meta-gradient accent bar below the directory path pill
    bar_w = int(width * 0.18)
    by = py1 + int(height * 0.048)
    bar_thick = max(2, int(height * 0.0045))
    bx_start = cx - bar_w // 2
    for i in range(bar_w):
        u = i / max(1, bar_w - 1)
        if u < 0.5:
            v = u * 2.0
            col = (0, int(100 * (1 - v) + 129 * v), int(224 * (1 - v) + 251 * v))
        else:
            v = (u - 0.5) * 2.0
            col = (int(124 * v), int(129 * (1 - v) + 58 * v), int(251 * (1 - v) + 237 * v))
        draw.line([(bx_start + i, by), (bx_start + i, by + bar_thick)], fill=col, width=1)

    return np.asarray(img, dtype=np.uint8)


def build_intro_scene_xml(tier: str) -> mujoco.MjModel:
    """Build the MuJoCo scene using the 100% authentic MuJoCo Menagerie Unitree H1 CAD meshes (h1.xml) for all tiers."""
    offw, offh = (3840, 2160) if tier == "4k" else (1280, 720)
    shadow_sz = "8192" if tier == "4k" else "4096"
    samples = "8" if tier == "4k" else "4"

    # Official MuJoCo Menagerie Unitree H1 (zero primitive cylinders/boxes added—100% pure factory CAD STL meshes)
    h1_tree = ET.parse(MENAGERIE_H1_DIR / "h1.xml")
    h1_root = h1_tree.getroot()

    comp = h1_root.find("compiler")
    if comp is not None:
        comp.set("meshdir", str(MENAGERIE_H1_DIR / "assets"))

    vis = ET.SubElement(h1_root, "visual")
    ET.SubElement(vis, "global", {"offwidth": str(offw), "offheight": str(offh), "azimuth": "0", "elevation": "-10"})
    ET.SubElement(vis, "quality", {"shadowsize": shadow_sz, "offsamples": samples})
    ET.SubElement(
        vis,
        "headlight",
        {"diffuse": "0.10 0.11 0.13", "ambient": "0.17 0.18 0.20", "specular": "0.06 0.07 0.08"},
    )

    asset = h1_root.find("asset")
    for mat in asset.findall("material"):
        if mat.get("name") == "black":
            # Authentic Unitree H1 dark matte-anodized black/titanium finish (matching h1.png with studio speculars)
            mat.set("rgba", "0.16 0.17 0.20 1")
            mat.set("specular", "0.92")
            mat.set("shininess", "0.68")
            mat.set("reflectance", "0.22")
        elif mat.get("name") == "white":
            # Crisp factory white/silver 'Unitree H1' chest emblem (`logo_link.stl`)
            mat.set("rgba", "0.94 0.96 0.99 1")
            mat.set("specular", "1.0")
            mat.set("shininess", "0.88")
            mat.set("reflectance", "0.28")

    wb = h1_root.find("worldbody")
    for l_el in list(wb.findall("light")):
        wb.remove(l_el)

    # 5-Light Studio Rig with 8192x8192 Self-Shadowing (`castshadow="true"`) on Key and Rim lights
    ET.SubElement(
        wb,
        "light",
        {
            "name": "key_light",
            "pos": "2.2 -1.6 3.0",
            "dir": "-0.55 0.45 -0.60",
            "diffuse": "0.52 0.54 0.58",
            "specular": "0.82 0.86 0.92",
            "castshadow": "true",
        },
    )
    ET.SubElement(
        wb,
        "light",
        {
            "name": "fill_light",
            "pos": "2.0 1.8 2.2",
            "dir": "-0.60 -0.50 -0.35",
            "diffuse": "0.36 0.38 0.42",
            "specular": "0.62 0.66 0.72",
            "castshadow": "false",
        },
    )
    ET.SubElement(
        wb,
        "light",
        {
            "name": "rim_light",
            "pos": "-1.5 -0.8 2.8",
            "dir": "0.65 0.25 -0.65",
            "diffuse": "0.34 0.36 0.40",
            "specular": "0.72 0.76 0.84",
            "castshadow": "true",
        },
    )
    ET.SubElement(
        wb,
        "light",
        {
            "name": "graze_side",
            "pos": "1.2 -2.2 1.5",
            "dir": "-0.35 0.88 -0.10",
            "diffuse": "0.26 0.28 0.32",
            "specular": "0.68 0.72 0.80",
            "castshadow": "false",
        },
    )
    ET.SubElement(
        wb,
        "light",
        {
            "name": "low_bounce",
            "pos": "2.0 0.0 0.45",
            "dir": "-0.80 0.0 0.35",
            "diffuse": "0.20 0.22 0.25",
            "specular": "0.48 0.50 0.55",
            "castshadow": "false",
        },
    )

    ET.SubElement(
        wb,
        "camera",
        {
            "name": "intro_cam",
            "pos": "2.38 0.04 1.32",
            "xyaxes": "0 1 0 -0.015 0 0.9999",
            "fovy": "25.5",
        },
    )

    sheet = ET.SubElement(wb, "body", {"name": "screen_sheet_body", "pos": "-0.118 0.10 1.76"})
    ET.SubElement(sheet, "joint", {"name": "screen_pull", "type": "slide", "axis": "0 0 -1", "range": "0 1.05"})
    ET.SubElement(sheet, "inertial", {"pos": "0 0 0", "mass": "1.0", "diaginertia": "0.01 0.01 0.01"})

    xml_str = ET.tostring(h1_root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml_str)


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    u = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return u * u * (3.0 - 2.0 * u)


def _smootherstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    u = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def _quat_from_yaw_pitch_roll(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def _solve_right_arm_ik_to_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_xyz: np.ndarray,
    jadr: dict[str, int],
    iters: int = 26,
) -> tuple[float, float, float, float]:
    """Damped least-squares Jacobian IK on H1's 4-DoF right arm to place the right wrist end on target_xyz."""
    arm_joints = ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow"]
    q_ids = [jadr[j] for j in arm_joints]
    dof_ids = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in arm_joints]
    j_ranges = [model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in arm_joints]

    elbow_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_elbow_link")
    local_hand_offset = np.array([0.275, 0.0, -0.015], dtype=np.float64)

    for _ in range(iters):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        rot = data.xmat[elbow_body_id].reshape(3, 3)
        hand_pos = data.xpos[elbow_body_id] + rot @ local_hand_offset
        err = target_xyz - hand_pos
        if float(np.linalg.norm(err)) < 0.003:
            break
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(model, data, jacp, None, hand_pos, elbow_body_id)
        J = jacp[:, dof_ids]
        JJT = J @ J.T + 0.010 * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, err)
        for idx, q_idx in enumerate(q_ids):
            val = float(data.qpos[q_idx] + 0.65 * dq[idx])
            lo, hi = float(j_ranges[idx][0]), float(j_ranges[idx][1])
            if lo < hi:
                val = max(lo, min(hi, val))
            data.qpos[q_idx] = val

    return tuple(float(data.qpos[q_idx]) for q_idx in q_ids)


def set_intro_kinematic_state(model: mujoco.MjModel, data: mujoco.MjData, t: float) -> dict:
    """Compute exact H1 whole-body pose, pull-ring 3D world position, and optical wink state at time t."""
    jadr: dict[str, int] = {}
    root_qadr = 0
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            root_qadr = int(model.jnt_qposadr[i])
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if nm:
            jadr[nm] = int(model.jnt_qposadr[i])

    # 1. Screen Pull progress:
    # At t < 8.2s: pull_m = 0.0m -> walnut lower boom hugs the very top bezel of the frame (zero canvas hanging down!)
    # At 8.2s <= t <= 11.0s: H1 pulls the circular titanium pull-ring down by 0.78m, unrolling the canvas WITH the boom!
    # At 11.0s <= t <= 11.25s: spring-ratchet latch settles upward by 1.6cm (`0.78 -> 0.764m`).
    if t < 8.2:
        pull_m = 0.0
    elif t < 11.0:
        u = _smootherstep(8.2, 11.0, t)
        pull_m = 0.78 * u
    elif t < 11.25:
        u_latch = _smoothstep(11.0, 11.25, t)
        pull_m = 0.78 - 0.016 * u_latch
    else:
        pull_m = 0.764

    # 3D world position of the circular brushed-titanium pull-ring (at z = 1.725m when pull_m=0, right below top bezel)
    handle_target = np.array([-0.055, -0.08, 1.725 - pull_m], dtype=np.float64)

    # 2. H1 Base Trajectory (Strictly IN FRONT of the canvas at x = +0.22m .. +0.32m)
    walk_in_u = _smootherstep(3.85, 5.55, t)
    walk_out_u = _smootherstep(12.65, 14.35, t)

    if t < 3.85:
        base_x = 0.30
        base_y = -1.45
        pelvis_yaw = 1.22
        walk_amp = 0.0
        step_phase = 0.0
    elif t < 5.55:
        base_x = 0.30
        base_y = -1.45 + (-0.42 - (-1.45)) * walk_in_u
        turn_u = _smootherstep(4.8, 5.6, t)
        pelvis_yaw = 1.22 * (1.0 - turn_u) + 0.14 * turn_u
        walk_amp = math.sin(math.pi * walk_in_u)
        step_phase = (t - 3.85) * 2.0 * math.pi * 1.85
    elif t < 6.9:
        base_x = 0.30
        base_y = -0.42
        pelvis_yaw = 0.14
        walk_amp = 0.0
        step_phase = 0.0
    elif t < 11.2:
        prep_u = _smootherstep(6.9, 7.9, t)
        unprep_u = _smootherstep(10.85, 11.3, t)
        active_u = prep_u * (1.0 - unprep_u)
        base_x = 0.30 - 0.08 * active_u
        base_y = -0.42 + 0.14 * active_u
        pelvis_yaw = 0.14 + 0.55 * active_u
        walk_amp = 0.25 * math.sin(math.pi * _smoothstep(6.9, 7.7, t)) if t < 7.7 else 0.0
        step_phase = (t - 6.9) * 2.0 * math.pi * 1.5
    elif t < 12.65:
        turn_back_u = _smootherstep(11.1, 11.6, t)
        base_x = 0.22 + 0.10 * turn_back_u
        base_y = -0.28 - 0.26 * turn_back_u
        pelvis_yaw = 0.69 * (1.0 - turn_back_u) + 0.12 * turn_back_u
        walk_amp = 0.0
        step_phase = 0.0
    else:
        turn_exit_u = _smootherstep(12.55, 13.05, t)
        base_x = 0.32
        base_y = -0.54 + (-1.45 - (-0.54)) * walk_out_u
        pelvis_yaw = 0.12 * (1.0 - turn_exit_u) + (-1.38) * turn_exit_u
        walk_amp = math.sin(math.pi * min(1.0, walk_out_u * 1.15))
        step_phase = (t - 12.65) * 2.0 * math.pi * 1.95

    pull_squat = 0.0
    if 8.2 <= t <= 11.4:
        pull_squat = 0.11 * _smoothstep(8.4, 10.8, t) * (1.0 - _smoothstep(10.9, 11.4, t))

    breath_z = 0.004 * math.sin(t * 2.3)
    walk_bounce = 0.016 * abs(math.sin(step_phase)) * walk_amp
    base_z = 0.98 - pull_squat + breath_z + walk_bounce

    pelvis_roll = 0.032 * math.sin(step_phase) * walk_amp
    pelvis_pitch = 0.04 * walk_amp + 0.32 * pull_squat

    data.qpos[root_qadr + 0] = base_x
    data.qpos[root_qadr + 1] = base_y
    data.qpos[root_qadr + 2] = base_z
    data.qpos[root_qadr + 3 : root_qadr + 7] = _quat_from_yaw_pitch_roll(pelvis_yaw, pelvis_pitch, pelvis_roll)
    data.qpos[jadr["screen_pull"]] = pull_m

    # 3. Leg Joints
    s_l = math.sin(step_phase) * walk_amp
    s_r = -s_l
    lift_l = max(0.0, s_l)
    lift_r = max(0.0, s_r)

    knee_bend_extra = 1.8 * pull_squat
    hip_bend_extra = -0.9 * pull_squat
    ankle_bend_extra = -0.9 * pull_squat

    data.qpos[jadr["left_hip_yaw"]] = 0.0
    data.qpos[jadr["left_hip_roll"]] = 0.03 + 0.04 * s_l
    data.qpos[jadr["left_hip_pitch"]] = -0.38 + hip_bend_extra - 0.34 * s_l - 0.18 * lift_l
    data.qpos[jadr["left_knee"]] = 0.76 + knee_bend_extra + 0.52 * lift_l
    data.qpos[jadr["left_ankle"]] = -0.38 + ankle_bend_extra + 0.22 * s_l - 0.28 * lift_l

    data.qpos[jadr["right_hip_yaw"]] = 0.0
    data.qpos[jadr["right_hip_roll"]] = -0.03 + 0.04 * s_r
    data.qpos[jadr["right_hip_pitch"]] = -0.38 + hip_bend_extra - 0.34 * s_r - 0.18 * lift_r
    data.qpos[jadr["right_knee"]] = 0.76 + knee_bend_extra + 0.52 * lift_r
    data.qpos[jadr["right_ankle"]] = -0.38 + ankle_bend_extra + 0.22 * s_r - 0.28 * lift_r

    # 4. Torso Yaw
    if t < 5.3:
        torso_yaw = -0.10 * _smoothstep(4.5, 5.3, t)
    elif t < 6.9:
        torso_yaw = -0.08 + 0.04 * math.sin((t - 5.3) * 4.0)
    elif t < 11.15:
        u_reach = _smootherstep(6.9, 7.8, t)
        torso_yaw = -0.08 * (1.0 - u_reach) + 0.42 * u_reach
    elif t < 12.65:
        u_look = _smootherstep(11.15, 11.55, t)
        torso_yaw = 0.42 * (1.0 - u_look) + (-0.06) * u_look
    else:
        u_leave = _smootherstep(12.65, 13.1, t)
        torso_yaw = -0.06 * (1.0 - u_leave)

    data.qpos[jadr["torso"]] = torso_yaw

    # 5. Left Arm
    data.qpos[jadr["left_shoulder_pitch"]] = 0.14 + 0.30 * s_r
    data.qpos[jadr["left_shoulder_roll"]] = 0.16
    data.qpos[jadr["left_shoulder_yaw"]] = 0.05
    data.qpos[jadr["left_elbow"]] = 0.58 + 0.14 * abs(s_r)

    # 6. Right Arm Choreography
    rsp_nom = 0.10 + 0.30 * s_l
    rsr_nom = -0.16
    rsy_nom = -0.05
    rel_nom = 0.55

    wave_freq = (t - 5.4) * 2.0 * math.pi * 2.6
    rsp_wave = -1.22
    rsr_wave = -0.78 - 0.20 * math.sin(wave_freq)
    rsy_wave = -0.48 + 0.26 * math.sin(wave_freq)
    rel_wave = 1.15 + 0.22 * math.cos(wave_freq)

    gwave_freq = (t - 11.4) * 2.0 * math.pi * 2.8
    rsp_gwave = -1.12
    rsr_gwave = -0.72 - 0.18 * math.sin(gwave_freq)
    rsy_gwave = -0.42 + 0.24 * math.sin(gwave_freq)
    rel_gwave = 1.10 + 0.20 * math.cos(gwave_freq)

    if t < 5.15:
        rsp, rsr, rsy, rel = rsp_nom, rsr_nom, rsy_nom, rel_nom
    elif t < 6.95:
        w = _smootherstep(5.15, 5.55, t) * (1.0 - _smootherstep(6.55, 6.95, t))
        rsp = rsp_nom * (1.0 - w) + rsp_wave * w
        rsr = rsr_nom * (1.0 - w) + rsr_wave * w
        rsy = rsy_nom * (1.0 - w) + rsy_wave * w
        rel = rel_nom * (1.0 - w) + rel_wave * w
    elif t < 11.35:
        data.qpos[jadr["right_shoulder_pitch"]] = -1.55 + 1.05 * (pull_m / 0.78)
        data.qpos[jadr["right_shoulder_roll"]] = -0.12 # abducted outward to prevent clipping torso
        data.qpos[jadr["right_shoulder_yaw"]] = -0.10
        data.qpos[jadr["right_elbow"]] = 0.42 + 0.45 * (pull_m / 0.78)
        ik_rsp, ik_rsr, ik_rsy, ik_rel = _solve_right_arm_ik_to_target(model, data, handle_target, jadr, iters=26)

        w_ik = _smootherstep(6.95, 8.05, t) * (1.0 - _smootherstep(10.95, 11.35, t))
        if t < 8.05:
            rsp = rsp_nom * (1.0 - w_ik) + ik_rsp * w_ik
            rsr = rsr_nom * (1.0 - w_ik) + ik_rsr * w_ik
            rsy = rsy_nom * (1.0 - w_ik) + ik_rsy * w_ik
            rel = rel_nom * (1.0 - w_ik) + ik_rel * w_ik
        elif t <= 10.95:
            rsp, rsr, rsy, rel = ik_rsp, ik_rsr, ik_rsy, ik_rel
        else:
            u_to_gwave = _smootherstep(10.95, 11.45, t)
            rsp = ik_rsp * (1.0 - u_to_gwave) + rsp_gwave * u_to_gwave
            rsr = ik_rsr * (1.0 - u_to_gwave) + rsr_gwave * u_to_gwave
            rsy = ik_rsy * (1.0 - u_to_gwave) + rsy_gwave * u_to_gwave
            rel = ik_rel * (1.0 - u_to_gwave) + rel_gwave * u_to_gwave
    elif t < 12.75:
        w_out = 1.0 - _smootherstep(12.25, 12.75, t)
        rsp = rsp_nom * (1.0 - w_out) + rsp_gwave * w_out
        rsr = rsr_nom * (1.0 - w_out) + rsr_gwave * w_out
        rsy = rsy_nom * (1.0 - w_out) + rsy_gwave * w_out
        rel = rel_nom * (1.0 - w_out) + rel_gwave * w_out
    else:
        rsp, rsr, rsy, rel = rsp_nom, rsr_nom, rsy_nom, rel_nom

    data.qpos[jadr["right_shoulder_pitch"]] = rsp
    data.qpos[jadr["right_shoulder_roll"]] = rsr
    data.qpos[jadr["right_shoulder_yaw"]] = rsy
    data.qpos[jadr["right_elbow"]] = rel

    wink_active = 0.0
    if 11.50 <= t <= 12.10:
        wink_active = math.sin(math.pi * (t - 11.50) / 0.60) ** 2

    mujoco.mj_forward(model, data)

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if torso_id >= 0:
        eye_xpos = data.xpos[torso_id] + data.xmat[torso_id].reshape(3, 3) @ np.array([0.088, -0.025, 0.650])
    else:
        eye_xpos = np.array([0.32, -0.42, 1.62])
    return {
        "pull_m": pull_m,
        "handle_target": handle_target,
        "wink_active": wink_active,
        "eye_xpos": eye_xpos,
    }


def _project_world_to_pixel(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cam_name: str,
    world_xyz: np.ndarray,
    width: int,
    height: int,
) -> tuple[int, int]:
    """Project a 3D world coordinate into exact 2D image pixel coordinates (px, py)."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
    rel = cam_mat.T @ (world_xyz - cam_pos)
    if rel[2] >= -0.05:
        return width // 2, height // 2
    fovy_rad = math.radians(float(model.cam_fovy[cam_id]))
    f = 0.5 * height / math.tan(0.5 * fovy_rad)
    px = int(round(width * 0.5 + f * (rel[0] / -rel[2])))
    py = int(round(height * 0.5 - f * (rel[1] / -rel[2])))
    return px, py


def _shade_photoreal_h1(
    raw_rgb: np.ndarray,
    depth: np.ndarray,
    seg_geom: np.ndarray,
    scale: float,
) -> Image.Image:
    """Extract Unitree H1 via exact segmentation mask and apply 3D Depth SSAO + Studio Softbox HDRI Metallic PBR."""
    h_full, w_full = raw_rgb.shape[:2]
    bot_mask_full = seg_geom >= 0
    if not np.any(bot_mask_full):
        return Image.new("RGBA", (w_full, h_full), (0, 0, 0, 0))

    # Tight bounding-box ROI around Unitree H1 for 6x faster 4K SSAO + Normal-Mapped PBR
    ys, xs = np.where(bot_mask_full)
    pad = max(16, int(44 * scale))
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h_full, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w_full, int(xs.max()) + pad + 1)

    bot_mask = bot_mask_full[y0:y1, x0:x1]
    arr = raw_rgb[y0:y1, x0:x1].astype(np.float32)
    depth_roi = depth[y0:y1, x0:x1]

    # 1. Compute 3-Scale 3D Screen-Space Ambient Occlusion (SSAO) from metric depth buffer `depth_roi`
    bot_depths = depth_roi[bot_mask]
    mean_d = float(np.median(bot_depths)) if len(bot_depths) else 2.1
    d_clean = np.where(bot_mask, depth_roi, mean_d).astype(np.float32)

    s1 = max(1.2, 3.0 * scale)
    s2 = max(3.0, 10.0 * scale)
    s3 = max(8.0, 26.0 * scale)
    b1 = ndimage.gaussian_filter(d_clean, sigma=s1)
    b2 = ndimage.gaussian_filter(d_clean, sigma=s2)
    b3 = ndimage.gaussian_filter(d_clean, sigma=s3)

    occ1 = np.clip((d_clean - b1) / 0.008, 0.0, 1.0)
    occ2 = np.clip((d_clean - b2) / 0.022, 0.0, 1.0)
    occ3 = np.clip((d_clean - b3) / 0.055, 0.0, 1.0)
    ao = 1.0 - np.clip(0.35 * occ1 + 0.40 * occ2 + 0.25 * occ3, 0.0, 0.72)

    # 2. Compute 3D Camera-Space Surface Normals N = normalize(-dz/dx, -dz/dy, pixel_step) for Studio HDRI Reflections
    d_smooth = ndimage.gaussian_filter(d_clean, sigma=max(0.8, 1.4 * scale))
    dz_dy, dz_dx = np.gradient(d_smooth)
    px_step = 0.0012 / max(0.25, scale)
    nx = -dz_dx
    ny = -dz_dy
    nz = np.full_like(nx, px_step)
    norm_len = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    nx /= norm_len
    ny /= norm_len
    nz /= norm_len

    fresnel_rim = np.clip((1.0 - nz) ** 3.2, 0.0, 1.0)
    softbox_key = np.clip(-0.55 * nx - 0.65 * ny + 0.52 * nz, 0.0, 1.0) ** 9.0
    softbox_fill = np.clip(0.65 * nx - 0.35 * ny + 0.65 * nz, 0.0, 1.0) ** 12.0

    yy = np.arange(y0, y1, dtype=np.float32)[:, None]
    brushed_grain = np.sin(yy * (2.4 / max(0.4, scale))) * 1.8

    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    is_logo = lum > 215.0

    norm_l = np.clip((lum - 14.0) / 215.0, 0.0, 1.0)
    s_curve = (norm_l**1.42) * (2.55 - 1.55 * norm_l) * 198.0 + 12.0
    shaded_l = np.where(is_logo, lum, s_curve * ao + brushed_grain * (norm_l * 0.5))

    spec_add = (softbox_key * 44.0 + softbox_fill * 26.0 + fresnel_rim * 32.0) * ao
    ratio = (shaded_l + 1e-3) / (lum + 1e-3)

    r_out = np.clip(arr[:, :, 0] * ratio * 0.96 + spec_add * 0.94, 0, 255)
    g_out = np.clip(arr[:, :, 1] * ratio * 1.00 + spec_add * 0.99, 0, 255)
    b_out = np.clip(arr[:, :, 2] * ratio * 1.06 + spec_add * 1.08, 0, 255)

    rgb_roi = np.stack([r_out, g_out, b_out], axis=-1).astype(np.uint8)
    alpha_f = ndimage.gaussian_filter(bot_mask.astype(np.float32), sigma=max(0.5, 0.75 * scale))
    alpha_u8 = (np.clip(alpha_f * 1.15, 0.0, 1.0) * 255.0).astype(np.uint8)

    out_rgba = np.zeros((h_full, w_full, 4), dtype=np.uint8)
    out_rgba[y0:y1, x0:x1, :3] = rgb_roi
    out_rgba[y0:y1, x0:x1, 3] = alpha_u8
    return Image.fromarray(out_rgba, mode="RGBA")


def _meta_color_at(u: float, alpha: int = 255) -> tuple[int, int, int, int]:
    """Interpolate Meta's signature brand gradient (#0064E0 Royal -> #0081FB Azure -> #7C3AED Violet -> #00D4FF Cyan)."""
    u = max(0.0, min(1.0, u))
    if u < 0.45:
        v = u / 0.45
        r = int(0 * (1 - v) + 0 * v)
        g = int(100 * (1 - v) + 129 * v)
        b = int(224 * (1 - v) + 251 * v)
    elif u < 0.78:
        v = (u - 0.45) / 0.33
        r = int(0 * (1 - v) + 124 * v)
        g = int(129 * (1 - v) + 58 * v)
        b = int(251 * (1 - v) + 237 * v)
    else:
        v = (u - 0.78) / 0.22
        r = int(124 * (1 - v) + 0 * v)
        g = int(58 * (1 - v) + 212 * v)
        b = int(237 * (1 - v) + 255 * v)
    return (r, g, b, alpha)


def composite_minimalist_ui(
    raw_rgb: np.ndarray,
    depth: np.ndarray,
    seg_geom: np.ndarray,
    t: float,
    info: dict,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    out_w: int,
    out_h: int,
    canvas_full_rgb: np.ndarray,
    boom_assets: dict,
) -> np.ndarray:
    """4-Layer Photorealistic Compositor:
    Layer 0: Pure [255, 255, 255] infinite studio cyclorama.
    Layer 1: Physically unrolling matte projection canvas (`Screen[0:y_boom] = Canvas[H - y_boom : H]`) +
             photorealistic American black walnut lower boom & circular brushed-titanium pull-ring locked to `handle_target`.
    Layer 2: 8192-shadowed, SSAO + Studio HDRI PBR-shaded Unitree H1 in the foreground.
    Layer 3: Meta-gradient floating prompt bar + optical sensor wink flare.
    """
    scale = out_w / 1920.0

    # Layer 0: Pure [255, 255, 255] white infinite studio backdrop
    stage_rgba = Image.new("RGBA", (out_w, out_h), (255, 255, 255, 255))

    # Layer 1: True Physical Top-Down Unrolling Canvas + Photorealistic Walnut Boom & Titanium Pull-Ring
    # Project the 3D `handle_target` into exact 2D screen pixel coordinates `(ring_px, ring_py)`!
    ring_px, ring_py = _project_world_to_pixel(model, data, "intro_cam", info["handle_target"], out_w, out_h)

    bar_img: Image.Image = boom_assets["bar_img"]
    bar_w: int = boom_assets["bar_w"]
    bar_h: int = boom_assets["bar_h"]
    ring_img: Image.Image = boom_assets["ring_img"]
    ring_cx_local: int = boom_assets["ring_cx_local"]
    ring_cy_local: int = boom_assets["ring_cy_local"]
    clasp_bar_x: int = boom_assets["clasp_bar_x"]

    # Position the clasp + braided cord + circular titanium pull-ring so the center of the pull-ring is at (ring_px, ring_py)
    ring_x0 = ring_px - ring_cx_local
    ring_y0 = ring_py - ring_cy_local

    # The bottom of the horizontal walnut bar sits right where the clasp attaches (`ring_y0 + int(6 * scale)`)
    bar_bottom_y = ring_y0 + int(round(5 * scale))
    bar_y0 = bar_bottom_y - bar_h
    bar_center_y = bar_y0 + bar_h // 2
    bar_x0 = ring_px - clasp_bar_x

    # The canvas sits horizontally aligned with the walnut bar (inset slightly from the titanium end-caps)
    canv_h, canv_w, _ = canvas_full_rgb.shape
    canv_x0 = bar_x0 + (bar_w - canv_w) // 2

    # TRUE PHYSICAL UNROLLING LOGIC:
    # The bottom edge of the canvas is stapled to the walnut bar at `y = bar_center_y`.
    # Therefore, when `bar_center_y > 0`, the visible canvas from `y = 0` (top bezel) down to `y = bar_center_y`
    # is the bottom `vis_h = bar_center_y` rows of `canvas_full_rgb` (`canvas_full_rgb[canv_h - vis_h : canv_h, :]`)!
    # As `bar_center_y` increases, the `A://` logo moves DOWNWARD at the exact speed of the walnut boom!
    vis_h = max(0, min(canv_h, bar_center_y))
    if vis_h > 4:
        sub_canvas = canvas_full_rgb[canv_h - vis_h : canv_h, :, :].copy().astype(np.float32)

        # Add realistic top-bezel cylindrical roll shading (top 28px) + bottom walnut-boom contact shadow (bottom 14px)
        yy_c = np.arange(vis_h, dtype=np.float32)[:, None]
        top_roll_shade = 1.0 - 0.065 * np.exp(-yy_c / max(4.0, 16.0 * scale))
        bot_boom_ao = 1.0 - 0.10 * np.exp(-(vis_h - 1.0 - yy_c) / max(2.5, 6.0 * scale))
        # Subtle dynamic horizontal fabric wave during movement (8.2s < t < 11.2s)
        if 8.2 < t < 11.25:
            wave_amp = 0.012 * math.sin(math.pi * min(1.0, (t - 8.2) / 2.8))
            fabric_wave = 1.0 + wave_amp * np.sin((vis_h - yy_c) / max(8.0, 28.0 * scale) - t * 9.0)
        else:
            fabric_wave = 1.0

        shade_mul = (top_roll_shade * bot_boom_ao * fabric_wave)[:, :, None]
        sub_canvas = np.clip(sub_canvas * shade_mul, 0, 255).astype(np.uint8)

        # Soft drop-shadow behind the canvas edges so the matte fabric has physical thickness against the infinite white
        canv_sh = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
        ImageDraw.Draw(canv_sh).rectangle(
            [canv_x0 + int(3 * scale), 0, canv_x0 + canv_w + int(4 * scale), vis_h + int(6 * scale)],
            fill=(15, 23, 42, 22),
        )
        canv_sh = canv_sh.filter(ImageFilter.GaussianBlur(radius=max(2, int(8 * scale))))
        stage_rgba = Image.alpha_composite(stage_rgba, canv_sh)

        canv_pil = Image.fromarray(sub_canvas, mode="RGB").convert("RGBA")
        stage_rgba.alpha_composite(canv_pil, dest=(canv_x0, 0))

    # Composite the photorealistic clasp + braided black cord + circular titanium pull-ring AND the walnut bar
    screen_hw_layer = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
    screen_hw_layer.paste(ring_img, (ring_x0, ring_y0), ring_img)
    screen_hw_layer.paste(bar_img, (bar_x0, bar_y0), bar_img)
    stage_rgba = Image.alpha_composite(stage_rgba, screen_hw_layer)

    # Layer 2: Photorealistic PBR + SSAO Unitree H1 in the foreground
    h1_rgba = _shade_photoreal_h1(raw_rgb, depth, seg_geom, scale)
    stage_rgba = Image.alpha_composite(stage_rgba, h1_rgba)

    # Layer 3: Meta-Colored Minimalist Floating Chatbox + Anamorphic Optical Sensor Wink Glint
    overlay = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    chat_appear = _smootherstep(0.35, 1.05, t)
    post_send_dim = 1.0 - 0.15 * _smootherstep(7.5, 9.0, t)
    alpha_mul = chat_appear * post_send_dim

    if alpha_mul > 0.01:
        box_w = int(900 * scale)
        box_h = int(80 * scale)
        bx0 = (out_w - box_w) // 2
        slide_y = int((1.0 - chat_appear) * 28 * scale)
        by0 = out_h - int(112 * scale) + slide_y
        bx1 = bx0 + box_w
        by1 = by0 + box_h
        radius = int(40 * scale)

        sh_layer = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
        sh_draw = ImageDraw.Draw(sh_layer)
        sh_draw.rounded_rectangle(
            [bx0 - int(4 * scale), by0 + int(6 * scale), bx0 + box_w // 2, by1 + int(14 * scale)],
            radius=radius,
            fill=(0, 129, 251, int(44 * alpha_mul)),
        )
        sh_draw.rounded_rectangle(
            [bx0 + box_w // 2, by0 + int(6 * scale), bx1 + int(4 * scale), by1 + int(14 * scale)],
            radius=radius,
            fill=(124, 58, 237, int(40 * alpha_mul)),
        )
        sh_layer = sh_layer.filter(ImageFilter.GaussianBlur(radius=max(3, int(14 * scale))))
        stage_rgba = Image.alpha_composite(stage_rgba, sh_layer)

        border_px = max(2, int(3.2 * scale))
        grad_layer = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
        g_draw = ImageDraw.Draw(grad_layer)
        grad_strip = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
        gs_draw = ImageDraw.Draw(grad_strip)
        for ix in range(box_w):
            col = _meta_color_at(ix / max(1, box_w - 1), int(255 * alpha_mul))
            gs_draw.line([(ix, 0), (ix, box_h)], fill=col, width=1)
        mask_outer = Image.new("L", (box_w, box_h), 0)
        ImageDraw.Draw(mask_outer).rounded_rectangle([0, 0, box_w - 1, box_h - 1], radius=radius, fill=255)
        grad_layer.paste(grad_strip, (bx0, by0), mask_outer)
        g_draw.rounded_rectangle(
            [bx0 + border_px, by0 + border_px, bx1 - border_px, by1 - border_px],
            radius=max(4, radius - border_px),
            fill=(248, 251, 255, int(252 * alpha_mul)),
        )
        stage_rgba = Image.alpha_composite(stage_rgba, grad_layer)
        draw = ImageDraw.Draw(overlay)

        ring_r = int(15 * scale)
        ring_thick = max(2, int(4.2 * scale))
        dot_x = bx0 + int(40 * scale)
        dot_y = (by0 + by1) // 2
        for deg in range(0, 360, 6):
            rad0 = math.radians(deg)
            rad1 = math.radians(deg + 8)
            u_col = ((deg / 360.0) + t * 0.25) % 1.0
            c_seg = _meta_color_at(u_col, int(250 * alpha_mul))
            x0_r = dot_x + ring_r * math.cos(rad0)
            y0_r = dot_y + ring_r * math.sin(rad0)
            x1_r = dot_x + ring_r * math.cos(rad1)
            y1_r = dot_y + ring_r * math.sin(rad1)
            draw.line([(x0_r, y0_r), (x1_r, y1_r)], fill=c_seg, width=ring_thick)

        f_prompt = _load_font(max(12, int(21 * scale)), bold=False)
        f_btn = _load_font(max(10, int(15 * scale)), bold=True)

        target_prompt = "Ask H1 to start the presentation and say hello."
        tx = dot_x + int(30 * scale)

        if t < 1.35:
            placeholder_txt = "Tell us what to do..."
            p_box = draw.textbbox((0, 0), placeholder_txt, font=f_prompt)
            ph = p_box[3] - p_box[1]
            draw.text(
                (tx, dot_y - ph // 2 - int(2 * scale)),
                placeholder_txt,
                fill=(92, 112, 145, int(245 * alpha_mul)),
                font=f_prompt,
            )
            if int(t * 2.5) % 2 == 0:
                draw.line(
                    [(tx - int(3 * scale), dot_y - int(13 * scale)), (tx - int(3 * scale), dot_y + int(13 * scale))],
                    fill=(0, 129, 251, int(240 * alpha_mul)),
                    width=max(2, int(2.5 * scale)),
                )
        else:
            type_u = max(0.0, min(1.0, (t - 1.35) / 2.15))
            n_chars = int(round(type_u * len(target_prompt)))
            shown_txt = target_prompt[:n_chars]
            t_box = draw.textbbox((0, 0), shown_txt if shown_txt else "A", font=f_prompt)
            tw = t_box[2] - t_box[0] if shown_txt else 0
            th = t_box[3] - t_box[1]
            ty = dot_y - th // 2 - int(2 * scale)
            draw.text(
                (tx, ty),
                shown_txt,
                fill=(10, 22, 46, int(252 * alpha_mul)),
                font=f_prompt,
            )
            if t < 3.75 and (n_chars < len(target_prompt) or int(t * 4) % 2 == 0):
                cx = tx + tw + int(4 * scale)
                draw.line(
                    [(cx, dot_y - int(13 * scale)), (cx, dot_y + int(13 * scale))],
                    fill=(0, 129, 251, int(245 * alpha_mul)),
                    width=max(2, int(2.5 * scale)),
                )

        btn_w = int(138 * scale)
        btn_h = int(48 * scale)
        btn_x1 = bx1 - int(16 * scale)
        btn_x0 = btn_x1 - btn_w
        btn_y0 = dot_y - btn_h // 2
        btn_y1 = btn_y0 + btn_h

        sent = t >= 3.75
        press_pulse = math.exp(-((t - 3.78) / 0.14) ** 2) if 3.55 <= t <= 4.1 else 0.0
        if sent:
            btn_fill = (0, 100, 224, int(252 * alpha_mul))
            btn_label = "META 0-SHOT"
        else:
            b_r = int(0 + 40 * press_pulse)
            b_g = int(129 + 45 * press_pulse)
            b_b = 251
            btn_fill = (b_r, b_g, b_b, int(250 * alpha_mul))
            btn_label = "Send ->"

        draw.rounded_rectangle([btn_x0, btn_y0, btn_x1, btn_y1], radius=int(24 * scale), fill=btn_fill)
        b_box = draw.textbbox((0, 0), btn_label, font=f_btn)
        bw, bh = b_box[2] - b_box[0], b_box[3] - b_box[1]
        draw.text(
            (btn_x0 + (btn_w - bw) // 2, btn_y0 + (btn_h - bh) // 2 - int(1 * scale)),
            btn_label,
            fill=(255, 255, 255, int(255 * alpha_mul)),
            font=f_btn,
        )

    # Anamorphic Optical Sensor Lens Flare on H1's right stereo depth camera lens during wink (t in 11.50..12.10)
    wink = float(info.get("wink_active", 0.0))
    if wink > 0.08:
        gx, gy = _project_world_to_pixel(model, data, "intro_cam", info["eye_xpos"], out_w, out_h)
        horiz_arm = int(52 * scale * wink)
        vert_arm = int(22 * scale * wink)
        glint_col = (56, 189, 248, int(245 * wink))
        core_col = (255, 255, 255, int(255 * wink))
        draw.line([(gx - horiz_arm, gy), (gx + horiz_arm, gy)], fill=glint_col, width=max(2, int(3.0 * scale)))
        draw.line([(gx, gy - vert_arm), (gx, gy + vert_arm)], fill=glint_col, width=max(2, int(2.5 * scale)))
        r_c = max(2, int(5.0 * scale * wink))
        draw.ellipse([gx - r_c, gy - r_c, gx + r_c, gy + r_c], fill=core_col)

    out_rgba = Image.alpha_composite(stage_rgba, overlay)
    return np.asarray(out_rgba.convert("RGB"), dtype=np.uint8)


def generate_series_a_soundtrack(duration_s: float = 14.5, sr: int = 48000) -> Path:
    """Synthesize a broadcast-grade 48kHz stereo soundtrack with neural keynote voiceover, tactile mechanical
    typing clicks, UI send chime, precision servo footsteps, canvas unrolling whoosh/ratchet, and ambient pad.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vo1_mp3 = OUT_DIR / "vo_line1.mp3"
    vo2_mp3 = OUT_DIR / "vo_line2.mp3"

    async def _synth_vo() -> None:
        if not vo1_mp3.exists() or vo1_mp3.stat().st_size < 1000:
            c1 = edge_tts.Communicate(
                "What if any robot body could master any physical task, from a single prompt?",
                voice="en-US-AndrewNeural",
            )
            await c1.save(str(vo1_mp3))
        if not vo2_mp3.exists() or vo2_mp3.stat().st_size < 1000:
            c2 = edge_tts.Communicate(
                "Meet AnyBody. Body-agnostic, task-agnostic, zero-to-one-shot embodied intelligence.",
                voice="en-US-AndrewNeural",
            )
            await c2.save(str(vo2_mp3))

    asyncio.run(_synth_vo())

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    vo1_wav = OUT_DIR / "vo_line1.wav"
    vo2_wav = OUT_DIR / "vo_line2.wav"
    for src, dst in [(vo1_mp3, vo1_wav), (vo2_mp3, vo2_wav)]:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(src), "-ac", "1", "-ar", str(sr), "-f", "wav", str(dst)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _read_wav_f32(p: Path) -> np.ndarray:
        with wave.open(str(p), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    vo1 = _read_wav_f32(vo1_wav)
    vo2 = _read_wav_f32(vo2_wav)

    n_samples = int(round(duration_s * sr))
    t = np.arange(n_samples, dtype=np.float32) / float(sr)
    left = np.zeros(n_samples, dtype=np.float32)
    right = np.zeros(n_samples, dtype=np.float32)

    # 1. Warm Silicon Valley Keynote Ambient Pad (Open fifths F2 + C3 + G3 + A3 with slow filter swell)
    env_pad = np.clip(t / 1.2, 0.0, 1.0) * np.clip((duration_s - t) / 1.0, 0.0, 1.0)
    pad_l = (
        0.032 * np.sin(2 * np.pi * 87.31 * t)
        + 0.024 * np.sin(2 * np.pi * 130.81 * t + 0.3)
        + 0.016 * np.sin(2 * np.pi * 196.00 * t + 0.7)
        + 0.012 * np.sin(2 * np.pi * 220.00 * t + 1.1)
    ) * env_pad
    pad_r = (
        0.032 * np.sin(2 * np.pi * 87.31 * t)
        + 0.024 * np.sin(2 * np.pi * 130.95 * t - 0.3)
        + 0.016 * np.sin(2 * np.pi * 196.15 * t - 0.5)
        + 0.012 * np.sin(2 * np.pi * 220.12 * t + 0.4)
    ) * env_pad
    left += pad_l
    right += pad_r

    # 2. Tactile Mechanical / Capacitive Keystroke Clicks (1.35s .. 3.50s, 47 characters)
    rng = np.random.default_rng(108)
    prompt_len = 47
    for c_idx in range(prompt_len):
        tc = 1.35 + (c_idx / float(prompt_len)) * 2.15 + rng.uniform(-0.008, 0.008)
        idx0 = int(tc * sr)
        dur = int(0.022 * sr)
        if 0 <= idx0 < n_samples - dur:
            tt = np.arange(dur, dtype=np.float32) / float(sr)
            # Crisp mechanical switch transient (2.4kHz + 4.8kHz burst + low keycap thock at 280Hz)
            f_click = rng.uniform(2200.0, 2750.0)
            click = (
                0.14 * np.sin(2 * np.pi * f_click * tt) * np.exp(-tt * 420.0)
                + 0.09 * np.sin(2 * np.pi * (f_click * 1.9) * tt) * np.exp(-tt * 680.0)
                + 0.11 * np.sin(2 * np.pi * 260.0 * tt) * np.exp(-tt * 140.0)
            )
            pan = 0.45 + 0.15 * (c_idx / float(prompt_len))
            left[idx0 : idx0 + dur] += click * (1.0 - pan * 0.3)
            right[idx0 : idx0 + dur] += click * (0.7 + pan * 0.3)

    # 3. Clean Meta UI Send Confirmation Chime at t = 3.78s (Two-note harmonic fifth C5 -> G5)
    for note_t, freq, amp in [(3.76, 523.25, 0.095), (3.83, 783.99, 0.115)]:
        idx0 = int(note_t * sr)
        dur = int(0.42 * sr)
        if 0 <= idx0 < n_samples - dur:
            tt = np.arange(dur, dtype=np.float32) / float(sr)
            chime = amp * (np.sin(2 * np.pi * freq * tt) + 0.25 * np.sin(2 * np.pi * freq * 2.0 * tt)) * np.exp(-tt * 9.5)
            left[idx0 : idx0 + dur] += chime * 0.85
            right[idx0 : idx0 + dur] += chime * 1.0

    # 4. Subtle Robotic Footsteps & Precision Harmonic-Drive Servo Whir during walk-in (3.9s..5.5s) and walk-out (12.7s..14.3s)
    for step_t in [4.05, 4.35, 4.65, 4.95, 5.25, 12.80, 13.10, 13.40, 13.70, 14.00]:
        idx0 = int(step_t * sr)
        dur = int(0.14 * sr)
        if 0 <= idx0 < n_samples - dur:
            tt = np.arange(dur, dtype=np.float32) / float(sr)
            thud = 0.09 * np.sin(2 * np.pi * 92.0 * np.exp(-tt * 18.0) * tt) * np.exp(-tt * 28.0)
            servo = 0.025 * np.sin(2 * np.pi * (380.0 + 140.0 * np.sin(tt * 22.0)) * tt) * np.exp(-tt * 14.0)
            left[idx0 : idx0 + dur] += (thud + servo) * 0.95
            right[idx0 : idx0 + dur] += (thud + servo) * 0.75

    # 5. Tactile Woven-Canvas Unrolling Friction & Spring-Roller Ratchet Ticks (t = 8.20s .. 11.00s) + Latch Click (11.05s)
    idx_pull0 = int(8.20 * sr)
    idx_pull1 = int(11.00 * sr)
    pull_len = idx_pull1 - idx_pull0
    tt_p = np.arange(pull_len, dtype=np.float32) / float(sr)
    pull_env = np.maximum(0.0, np.sin(np.pi * tt_p / 2.85)) ** 1.2
    # Filtered pink-like fabric friction whoosh
    raw_noise = rng.normal(0.0, 1.0, pull_len).astype(np.float32)
    fabric_whoosh = ndimage.gaussian_filter1d(raw_noise, sigma=18.0) * 0.065 * pull_env
    left[idx_pull0:idx_pull1] += fabric_whoosh
    right[idx_pull0:idx_pull1] += fabric_whoosh

    # Spring-roller ratchet clicks as the roller unspools
    for r_t in np.linspace(8.35, 10.85, 16):
        idx0 = int(r_t * sr)
        dur = int(0.015 * sr)
        if 0 <= idx0 < n_samples - dur:
            tt = np.arange(dur, dtype=np.float32) / float(sr)
            tick = 0.055 * np.sin(2 * np.pi * 1450.0 * tt) * np.exp(-tt * 380.0)
            left[idx0 : idx0 + dur] += tick
            right[idx0 : idx0 + dur] += tick

    # Crisp metallic spring-latch lock click at t = 11.06s
    for c_t, f_l, a_l in [(11.04, 1180.0, 0.12), (11.11, 1680.0, 0.15)]:
        idx0 = int(c_t * sr)
        dur = int(0.035 * sr)
        if 0 <= idx0 < n_samples - dur:
            tt = np.arange(dur, dtype=np.float32) / float(sr)
            latch = a_l * np.sin(2 * np.pi * f_l * tt) * np.exp(-tt * 240.0)
            left[idx0 : idx0 + dur] += latch
            right[idx0 : idx0 + dur] += latch

    # 6. Optical Sensor Wink Crystalline Ping at t = 11.68s
    idx_w = int(11.68 * sr)
    dur_w = int(0.35 * sr)
    if 0 <= idx_w < n_samples - dur_w:
        tt = np.arange(dur_w, dtype=np.float32) / float(sr)
        wink_ping = 0.08 * np.sin(2 * np.pi * 1567.98 * tt) * np.exp(-tt * 14.0)
        left[idx_w : idx_w + dur_w] += wink_ping * 0.9
        right[idx_w : idx_w + dur_w] += wink_ping * 0.7

    # 7. Mix Neural Keynote Voiceover Lines (`vo1` at t = 4.05s, `vo2` at t = 8.35s)
    for vo_arr, start_t in [(vo1, 4.05), (vo2, 8.35)]:
        idx0 = int(start_t * sr)
        n_v = min(len(vo_arr), n_samples - idx0)
        if n_v > 0:
            left[idx0 : idx0 + n_v] += vo_arr[:n_v] * 0.92
            right[idx0 : idx0 + n_v] += vo_arr[:n_v] * 0.92

    # Soft master limiter
    stereo = np.stack([left, right], axis=-1)
    peak = float(np.max(np.abs(stereo))) + 1e-6
    if peak > 0.95:
        stereo = stereo * (0.95 / peak)
    pcm16 = (np.clip(stereo, -1.0, 1.0) * 32767.0).astype(np.int16)

    wav_out = OUT_DIR / "anybody_intro_audio.wav"
    with wave.open(str(wav_out), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())

    return wav_out


def _mux_video_and_audio(video_only_mp4: Path, audio_wav: Path, final_mp4: Path) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    tmp_mp4 = video_only_mp4.with_suffix(".mux_tmp.mp4")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_only_mp4),
            "-i",
            str(audio_wav),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            "-shortest",
            str(tmp_mp4),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    tmp_mp4.replace(final_mp4)


def render_intro_video(
    tier: str = "low",
    fps: int = 30,
    duration_s: float = 14.5,
    save_keyframes: bool = True,
) -> Path:
    """Render the complete AnyBody Intro video with photorealistic compositing and 48kHz stereo audio."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_wav = generate_series_a_soundtrack(duration_s=duration_s)

    model = build_intro_scene_xml(tier=tier)
    data = mujoco.MjData(model)

    if tier == "4k":
        render_w, render_h = 3840, 2160
        out_path = OUT_DIR / "anybody_intro_4k_uhd.mp4"
    else:
        render_w, render_h = 960, 540
        out_path = OUT_DIR / "anybody_intro_low_tier_preview.mp4"

    scale = render_w / 1920.0
    target_bar_w = int(round(1260 * scale))
    boom_assets = prepare_photoreal_walnut_boom_assets(target_bar_w=target_bar_w)
    canv_w = int(round(target_bar_w * 0.945))
    canv_h = int(round(760 * scale))
    canvas_full_rgb = generate_anybody_canvas_image(width=canv_w, height=canv_h)

    renderer = mujoco.Renderer(model, height=render_h, width=render_w)
    scene_opt = mujoco.MjvOption()
    total_frames = int(round(duration_s * fps))
    t0 = time.perf_counter()

    writer = imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        quality=9 if tier == "4k" else 8,
        pixelformat="yuv420p",
        macro_block_size=1,
    )

    keyframe_times = [0.8, 3.6, 5.8, 8.2, 9.4, 10.6, 11.8, 13.5]
    saved_kf: set[int] = set()

    try:
        for f_idx in range(total_frames):
            t = f_idx / float(fps)
            info = set_intro_kinematic_state(model, data, t)

            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "intro_cam")
            if cam_id >= 0:
                zoom_u = _smootherstep(8.2, 12.5, t)
                model.cam_pos[cam_id] = np.array(
                    [
                        2.38 - 0.12 * zoom_u,
                        0.04 + 0.03 * zoom_u,
                        1.32 + 0.01 * zoom_u,
                    ],
                    dtype=np.float64,
                )
                mujoco.mj_forward(model, data)

            renderer.update_scene(data, camera="intro_cam", scene_option=scene_opt)
            raw_rgb = renderer.render()
            renderer.enable_depth_rendering()
            depth = renderer.render().copy()
            renderer.disable_depth_rendering()
            renderer.enable_segmentation_rendering()
            seg = renderer.render()[:, :, 0].copy()
            renderer.disable_segmentation_rendering()

            comp_rgb = composite_minimalist_ui(
                raw_rgb,
                depth,
                seg,
                t,
                info,
                model,
                data,
                render_w,
                render_h,
                canvas_full_rgb,
                boom_assets,
            )
            writer.append_data(comp_rgb)

            if save_keyframes:
                for k_idx, kt in enumerate(keyframe_times):
                    if k_idx not in saved_kf and abs(t - kt) <= (0.55 / fps):
                        saved_kf.add(k_idx)
                        kf_path = OUT_DIR / f"kf_{tier}_{k_idx:02d}_{t:.1f}s.png"
                        Image.fromarray(comp_rgb).save(kf_path)

            if f_idx % max(1, total_frames // 15) == 0 or f_idx == total_frames - 1:
                elapsed = time.perf_counter() - t0
                pct = 100.0 * (f_idx + 1) / total_frames
                print(
                    f"[{tier.upper():3s}] Frame {f_idx + 1:4d}/{total_frames:4d} ({pct:5.1f}%) | "
                    f"t={t:5.2f}s | pull={info['pull_m']:.2f}m | elapsed={elapsed:.1f}s",
                    flush=True,
                )
    finally:
        writer.close()
        renderer.close()

    _mux_video_and_audio(out_path, audio_wav, out_path)
    print(f"Saved {tier} intro video with stereo audio -> {out_path} ({out_path.stat().st_size / (1024 * 1024):.2f} MB)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render AnyBody minimalist intro animation")
    parser.add_argument("--tier", choices=["low", "4k", "both"], default="low")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    if args.tier in ("low", "both"):
        render_intro_video(tier="low", fps=args.fps)
    if args.tier in ("4k", "both"):
        render_intro_video(tier="4k", fps=args.fps)
