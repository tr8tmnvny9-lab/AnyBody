"""Generate 6 High-Definition (2x Supersampled -> 1920x1080) Silicon Valley SaaS / Apple Keynote
Design Snapshots for User Approval BEFORE rendering the full 3-minute MP4 video.

Fixes verified in these snapshots:
1. Intro Unitree H1 Screen Pull (`design_snap_01_intro_pull.png`):
   - Fixes the open torso twist and clamped cross-body right arm by aligning the pull-ring
     directly in front of H1's right shoulder (`y = -0.30`) and keeping H1's torso in a natural
     3/4 front-facing stance (`pelvis_yaw = 0.08`, `torso_yaw = -0.08`) so the right arm reaches
     cleanly forward and upward in open space without ever crossing the chest.
2. Exact Z=0.000m Foot-Sole Grounding & 3D Analytical IK (`design_snap_02..05`):
   - Every single frame computes 3D limb kinematics on the authentic MuJoCo Menagerie Unitree H1
     and shifts `qpos[2]` by `-min(left_sole_z, right_sole_z)` so the stance foot sole rests on
     `z = 0.000 m` with ZERO floor sinking during deep squats, kicks, or walks.
   - All 3D objects rest physically on `z = 0.000 m` (or locked in H1's hands when grasped) with
     multi-layered studio ambient-occlusion contact shadows on the pure `#FFFFFF` cyclorama.
3. High-End Silicon Valley SaaS Editorial Graphics (2x Supersampled Vector Canvas):
   - Includes the "How It Works" 25.11M-Synapse Neuromorphic X-Ray scene (`design_snap_02`),
     Demo 1 Deep Squat & Overhead Lift (`design_snap_03`), Demo 2 Unipedal Kick & Balance (`design_snap_04`),
     Demo 3 Live Generative Agentic UI + SAM-3.1 Retinal HUD (`design_snap_05`), and the
     Baseten 100€ + Oscar Kinnala 🇫🇮 Creator Signature Outro (`design_snap_06`).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

VIDEO_SCRIPT_DIR = Path(__file__).resolve().parent
INTRO_SCRIPT_DIR = VIDEO_SCRIPT_DIR
sys.path.insert(0, str(VIDEO_SCRIPT_DIR))
import render_intro_animation as intro_mod  # noqa: E402
from ssaa_vector_draw import install_ssaa_vector_draw, ssaa_draw_glass_card  # noqa: E402

install_ssaa_vector_draw()

MENAGERIE_H1_DIR = Path.home() / ".cache" / "robot_descriptions" / "mujoco_menagerie" / "unitree_h1"
ARTIFACT_DIR = VIDEO_SCRIPT_DIR.parents[1] / "target" / "video_artifacts"
OSCAR_PORTRAIT_PATH = VIDEO_SCRIPT_DIR.parents[1] / "assets" / "logos" / "oscar_portrait_clean.png"


def load_sf_font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    return intro_mod._load_font(size, bold=bold, mono=mono)


def tint_rgb(rgb: tuple[int, int, int], alpha: float = 0.12) -> tuple[int, int, int]:
    """Blend an RGB accent color with pure white (255, 255, 255) so Pillow never writes low-alpha holes."""
    return (
        int(round(255 * (1.0 - alpha) + rgb[0] * alpha)),
        int(round(255 * (1.0 - alpha) + rgb[1] * alpha)),
        int(round(255 * (1.0 - alpha) + rgb[2] * alpha)),
    )


def draw_vector_checkmark(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int = 6, color: tuple[int, int, int] = (255, 255, 255)) -> None:
    """Draw a crisp anti-aliased geometric checkmark (avoids missing font glyphs)."""
    p1 = (cx - size, cy)
    p2 = (cx - size // 4, cy + int(size * 0.75))
    p3 = (cx + size + 1, cy - int(size * 0.65))
    draw.line([p1, p2, p3], fill=color, width=3)


def draw_vector_audio_wave(draw: ImageDraw.ImageDraw, x0: int, cy: int, color: tuple[int, int, int] = (0, 82, 204)) -> int:
    """Draw a 5-bar broadcast voiceover waveform icon and return its pixel width."""
    heights = [5, 11, 7, 12, 6]
    for i, bh in enumerate(heights):
        bx = x0 + i * 5
        draw.rounded_rectangle([bx, cy - bh, bx + 3, cy + bh], radius=1, fill=color)
    return len(heights) * 5 + 6


def draw_finnish_flag(draw: ImageDraw.ImageDraw, x0: int, y0: int, w: int, h: int, radius: int = 4) -> None:
    """Draw a crisp vector Finnish flag (white field #FFFFFF, sea-blue Nordic cross #002F6C)."""
    draw.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=radius, fill=(255, 255, 255), outline=(203, 213, 225), width=2)
    cross_th = max(4, int(round(h * 0.27)))
    horiz_y0 = y0 + (h - cross_th) // 2
    vert_x0 = x0 + int(round(w * 0.31))
    sea_blue = (0, 47, 108)
    draw.rectangle([x0 + 1, horiz_y0, x0 + w - 1, horiz_y0 + cross_th], fill=sea_blue)
    draw.rectangle([vert_x0, y0 + 1, vert_x0 + cross_th, y0 + h - 1], fill=sea_blue)


def draw_meta_infinity_symbol(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    rx: int,
    ry: int,
    thickness: int,
) -> None:
    """Draw a vector-smooth Meta Infinity loop in Meta's royal-to-cyan gradient."""
    pts = []
    n_seg = 140
    for i in range(n_seg + 1):
        t = (i / n_seg) * 2.0 * math.pi
        denom = 1.0 + math.sin(t) ** 2
        x = cx + rx * math.cos(t) / denom
        y = cy + ry * math.sin(t) * math.cos(t) / denom
        pts.append((x, y))
    for i in range(n_seg):
        u = i / float(n_seg)
        col = intro_mod._meta_color_at(u, 255)[:3]
        draw.line([pts[i], pts[i + 1]], fill=col, width=thickness)


def draw_glass_card(
    base_rgba: Image.Image,
    box: list[int],
    radius: int,
    border_col: tuple[int, int, int, int] = (0, 129, 251, 65),
    shadow_col: tuple[int, int, int, int] = (0, 80, 180, 22),
    shadow_blur: int = 18,
    shadow_offset_y: int = 8,
    fill_col: tuple[int, int, int, int] = (255, 255, 255, 246),
    border_w: int = 2,
) -> None:
    """Render an Apple VisionOS / Linear style pure-white elevated card with 4x supersampled anti-aliased border."""
    ssaa_draw_glass_card(
        base_rgba=base_rgba,
        box=box,
        radius=radius,
        border_col=border_col,
        shadow_col=shadow_col,
        shadow_blur=shadow_blur,
        shadow_offset_y=shadow_offset_y,
        fill_col=fill_col,
        border_w=border_w,
    )


def draw_top_brand_bar(
    stage_rgba: Image.Image,
    act_badge: str,
    title_right: str = "25.11M Synapses · 1,000 Hz Spiking Connectome · Zero PID",
) -> None:
    """Draw the authentic First-Page AnyBody Logo (`A://█` + `A://AnyBody`)
    plus the Act Badge and right-hand hardware telemetry pill across all scenes.
    """
    draw = ImageDraw.Draw(stage_rgba)
    # 1. Authentic First-Page Hero Mark: `A://` + Terminal Block Cursor `█`
    f_hero = load_sf_font(30, bold=True, mono=True)
    x_cur = 48
    y_hero = 35
    a_part = "A:"
    slash_part = "//"
    w_a = draw.textbbox((0, 0), a_part, font=f_hero)[2]
    w_s = draw.textbbox((0, 0), slash_part, font=f_hero)[2]
    draw.text((x_cur, y_hero), a_part, fill=(14, 18, 26), font=f_hero)
    draw.text((x_cur + w_a, y_hero), slash_part, fill=(0, 129, 251), font=f_hero)
    cur_x0 = x_cur + w_a + w_s + 5
    draw.rounded_rectangle([cur_x0, y_hero + 4, cur_x0 + 11, y_hero + 31], radius=2, fill=(14, 18, 26))

    # 2. Authentic Brand Pill: `AnyBody`
    f_dir = load_sf_font(13, bold=True, mono=True)
    dir_suffix = "AnyBody"
    ws = draw.textbbox((0, 0), dir_suffix, font=f_dir)[2]
    d_px0 = cur_x0 + 22
    d_py0 = 37
    d_pw = ws + 24
    d_ph = 32
    draw.rounded_rectangle(
        [d_px0, d_py0, d_px0 + d_pw, d_py0 + d_ph],
        radius=10,
        fill=(242, 246, 252),
        outline=(204, 214, 230),
        width=2,
    )
    draw.text((d_px0 + 12, d_py0 + 8), dir_suffix, fill=(0, 100, 224), font=f_dir)
    # Subtle Meta-gradient accent underline below the directory pill (matching the intro canvas)
    for ix in range(d_pw - 16):
        u_col = intro_mod._meta_color_at(ix / max(1, d_pw - 17), 255)[:3]
        draw.line([(d_px0 + 8 + ix, d_py0 + d_ph + 3), (d_px0 + 8 + ix, d_py0 + d_ph + 5)], fill=u_col, width=1)

    # 3. Act Pill next to the Directory Logo
    f_tag = load_sf_font(13, bold=True, mono=True)
    b_act = draw.textbbox((0, 0), act_badge, font=f_tag)
    pw = (b_act[2] - b_act[0]) + 24
    ph = 32
    px0, py0 = d_px0 + d_pw + 14, 37
    draw.rounded_rectangle([px0, py0, px0 + pw, py0 + ph], radius=16, fill=tint_rgb((0, 129, 251), 0.08), outline=(0, 129, 251), width=2)
    draw.text((px0 + 12, py0 + 8), act_badge, fill=(0, 100, 224), font=f_tag)

    # 4. Right-side telemetry spec pill
    f_sub = load_sf_font(13, bold=False)
    b_r = draw.textbbox((0, 0), title_right, font=f_sub)
    rw = (b_r[2] - b_r[0]) + 32
    rx0 = 1920 - rw - 44
    ry0 = 37
    draw.rounded_rectangle([rx0, ry0, rx0 + rw, ry0 + 32], radius=16, fill=(255, 255, 255), outline=(218, 228, 242), width=2)
    draw.ellipse([rx0 + 12, ry0 + 11, rx0 + 22, ry0 + 21], fill=(16, 185, 129))
    draw.text((rx0 + 28, ry0 + 7), title_right, fill=(30, 41, 59), font=f_sub)


def draw_meta_prompt_dock(
    stage_rgba: Image.Image,
    prompt_text: str,
    model_badge: str = "META MUSE-SPARK-1.3 + SAM-3.1",
    voice_narration: str = "",
    typing_progress: float = 1.0,
) -> None:
    """Draw the floating Meta-Gradient Prompt Dock + Narrator Voiceover Caption Pill at the bottom."""
    W, H = stage_rgba.size
    box_w = 1060
    box_h = 74
    bx0 = (W - box_w) // 2
    by0 = H - 102
    bx1 = bx0 + box_w
    by1 = by0 + box_h
    radius = 37

    # Soft Meta-Blue & Violet glow shadow
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    sd.rounded_rectangle([bx0 - 4, by0 + 6, bx0 + box_w // 2, by1 + 14], radius=radius, fill=(0, 129, 251, 42))
    sd.rounded_rectangle([bx0 + box_w // 2, by0 + 6, bx1 + 4, by1 + 14], radius=radius, fill=(124, 58, 237, 38))
    sh = sh.filter(ImageFilter.GaussianBlur(radius=14))
    stage_rgba.alpha_composite(sh)

    # Gradient border strip
    grad_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    grad_strip = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    gs_draw = ImageDraw.Draw(grad_strip)
    for ix in range(box_w):
        col = intro_mod._meta_color_at(ix / max(1, box_w - 1), 255)
        gs_draw.line([(ix, 0), (ix, box_h)], fill=col, width=1)
    mask_outer = Image.new("L", (box_w, box_h), 0)
    ImageDraw.Draw(mask_outer).rounded_rectangle([0, 0, box_w - 1, box_h - 1], radius=radius, fill=255)
    grad_layer.paste(grad_strip, (bx0, by0), mask_outer)

    gd = ImageDraw.Draw(grad_layer)
    border_px = 3
    gd.rounded_rectangle(
        [bx0 + border_px, by0 + border_px, bx1 - border_px, by1 - border_px],
        radius=radius - border_px,
        fill=(255, 255, 255, 253),
    )
    stage_rgba.alpha_composite(grad_layer)

    draw = ImageDraw.Draw(stage_rgba)
    # Meta AI spinning ring icon
    dot_x = bx0 + 38
    dot_y = (by0 + by1) // 2
    ring_r = 15
    for deg in range(0, 360, 8):
        rad0 = math.radians(deg)
        rad1 = math.radians(deg + 10)
        c_seg = intro_mod._meta_color_at((deg / 360.0), 255)[:3]
        draw.line(
            [
                (dot_x + ring_r * math.cos(rad0), dot_y + ring_r * math.sin(rad0)),
                (dot_x + ring_r * math.cos(rad1), dot_y + ring_r * math.sin(rad1)),
            ],
            fill=c_seg,
            width=4,
        )

    f_prompt = load_sf_font(20, bold=False)
    n_chars = int(round(max(0.0, min(1.0, typing_progress)) * len(prompt_text)))
    shown = prompt_text[:n_chars]
    draw.text((dot_x + 28, dot_y - 12), shown, fill=(10, 22, 46), font=f_prompt)

    # Right-side active Meta model pill
    f_btn = load_sf_font(13, bold=True, mono=True)
    b_btn = draw.textbbox((0, 0), model_badge, font=f_btn)
    btn_w = (b_btn[2] - b_btn[0]) + 32
    btn_h = 44
    btn_x1 = bx1 - 15
    btn_x0 = btn_x1 - btn_w
    btn_y0 = dot_y - btn_h // 2
    draw.rounded_rectangle([btn_x0, btn_y0, btn_x1, btn_y0 + btn_h], radius=22, fill=(0, 100, 224))
    draw.text((btn_x0 + 16, btn_y0 + 14), model_badge, fill=(255, 255, 255), font=f_btn)

    # Optional Keynote Voiceover Subtitle Pill right above the prompt dock
    if voice_narration:
        f_vo = load_sf_font(16, bold=True)
        vo_label = "KEYNOTE VOICEOVER ·  "
        vo_quote = f"“{voice_narration}”"
        b_lbl = draw.textbbox((0, 0), vo_label, font=load_sf_font(13, bold=True, mono=True))
        b_vo = draw.textbbox((0, 0), vo_quote, font=f_vo)
        w_lbl = b_lbl[2] - b_lbl[0]
        w_vo = b_vo[2] - b_vo[0]
        vw = 32 + w_lbl + w_vo + 36
        vh = 36
        vx0 = (W - vw) // 2
        vy0 = by0 - 46
        draw.rounded_rectangle([vx0, vy0, vx0 + vw, vy0 + vh], radius=18, fill=tint_rgb((0, 129, 251), 0.08), outline=(0, 129, 251), width=1)
        wave_w = draw_vector_audio_wave(draw, vx0 + 16, vy0 + vh // 2, color=(0, 100, 224))
        draw.text((vx0 + 16 + wave_w + 6, vy0 + 10), vo_label, fill=(0, 100, 224), font=load_sf_font(13, bold=True, mono=True))
        draw.text((vx0 + 16 + wave_w + 6 + w_lbl, vy0 + 8), vo_quote, fill=(10, 24, 52), font=f_vo)


# ==============================================================================
# SNAPSHOT 1: Fixed Intro Unitree H1 Pulling Down the Walnut-Boom Canvas
# ==============================================================================
def set_fixed_intro_kinematics(model: mujoco.MjModel, data: mujoco.MjData, t: float) -> dict:
    """Corrected H1 Intro Kinematics:
    - Keeps H1's torso facing forward in a natural 3/4 stance (`pelvis_yaw = -0.08`, `torso_yaw = -0.05`)
    - Poses H1's right arm in an open, relaxed anatomical reach outside its right shoulder (zero chest crossing,
      zero joint clamping) and locks the titanium pull-ring directly to H1's right hand tip while keeping the
      walnut bar centered on screen.
    """
    jadr: dict[str, int] = {}
    root_qadr = 0
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            root_qadr = int(model.jnt_qposadr[i])
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if nm:
            jadr[nm] = int(model.jnt_qposadr[i])

    if t < 8.2:
        pull_m = 0.0
    elif t < 11.0:
        u = intro_mod._smootherstep(8.2, 11.0, t)
        pull_m = 0.78 * u
    elif t < 11.25:
        u_latch = intro_mod._smoothstep(11.0, 11.25, t)
        pull_m = 0.78 - 0.016 * u_latch
    else:
        pull_m = 0.764

    prep_u = intro_mod._smootherstep(6.9, 7.9, t)
    unprep_u = intro_mod._smootherstep(10.85, 11.3, t)
    active_u = prep_u * (1.0 - unprep_u)

    # H1 stands slightly left of center (base_y = -0.24) so its head is to the left of the centered `A://` logo
    # and its right arm reaches upward in completely open space on the outside-left of its torso!
    base_x = 0.28
    base_y = -0.24
    pelvis_yaw = 0.08 * active_u

    pull_squat = 0.0
    if 8.2 <= t <= 11.4:
        pull_squat = 0.08 * intro_mod._smoothstep(8.4, 10.8, t) * (1.0 - intro_mod._smoothstep(10.9, 11.4, t))

    base_z = 0.98 - pull_squat
    pelvis_pitch = 0.14 * pull_squat

    data.qpos[root_qadr + 0] = base_x
    data.qpos[root_qadr + 1] = base_y
    data.qpos[root_qadr + 2] = base_z
    data.qpos[root_qadr + 3 : root_qadr + 7] = intro_mod._quat_from_yaw_pitch_roll(pelvis_yaw, pelvis_pitch, 0.0)
    data.qpos[jadr["screen_pull"]] = pull_m

    knee_bend_extra = 1.4 * pull_squat
    hip_bend_extra = -0.70 * pull_squat
    ankle_bend_extra = -0.70 * pull_squat

    for side, s_sign in [("left", 1.0), ("right", -1.0)]:
        data.qpos[jadr[f"{side}_hip_yaw"]] = 0.0
        data.qpos[jadr[f"{side}_hip_roll"]] = 0.04 * s_sign
        data.qpos[jadr[f"{side}_hip_pitch"]] = -0.38 + hip_bend_extra
        data.qpos[jadr[f"{side}_knee"]] = 0.76 + knee_bend_extra
        data.qpos[jadr[f"{side}_ankle"]] = -0.38 + ankle_bend_extra

    # Subtle open torso orientation (-0.04 rad) so H1's chest faces the viewer and the right shoulder is open
    data.qpos[jadr["torso"]] = -0.04 * active_u

    # Left arm relaxed at side
    data.qpos[jadr["left_shoulder_pitch"]] = 0.16
    data.qpos[jadr["left_shoulder_roll"]] = 0.18
    data.qpos[jadr["left_shoulder_yaw"]] = 0.05
    data.qpos[jadr["left_elbow"]] = 0.58

    # In h1.xml, right_elbow=0 points horizontal (+X) and right_elbow < 0 bends the forearm UPWARD (+Z)!
    # Setting right_shoulder_pitch = -0.38 keeps the elbow LOW and outside the torso, while right_elbow = -0.82
    # angles the forearm cleanly UPWARD from the elbow to grip the titanium pull-ring from below!
    pull_frac = pull_m / 0.78
    data.qpos[jadr["right_shoulder_pitch"]] = -0.44 + 0.34 * pull_frac
    data.qpos[jadr["right_shoulder_roll"]] = -0.32 + 0.04 * pull_frac
    data.qpos[jadr["right_shoulder_yaw"]] = -0.18
    data.qpos[jadr["right_elbow"]] = -0.88 + 0.36 * pull_frac

    mujoco.mj_forward(model, data)
    elbow_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_elbow_link")
    rot = data.xmat[elbow_body_id].reshape(3, 3)
    hand_pos = data.xpos[elbow_body_id] + rot @ np.array([0.275, 0.0, -0.015], dtype=np.float64)

    # Lock the circular titanium pull-ring directly to H1's right hand tip during the active pull!
    handle_target = hand_pos.copy()

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    eye_xpos = data.xpos[torso_id] + data.xmat[torso_id].reshape(3, 3) @ np.array([0.088, -0.025, 0.650])
    return {
        "pull_m": pull_m,
        "handle_target": handle_target,
        "wink_active": 0.0,
        "eye_xpos": eye_xpos,
    }


def render_snapshot_1_intro_pull() -> Path:
    model = intro_mod.build_intro_scene_xml(tier="low")
    model.vis.global_.offwidth = 1920
    model.vis.global_.offheight = 1080
    data = mujoco.MjData(model)

    render_w, render_h = 1920, 1080
    target_bar_w = 1260
    boom_assets = intro_mod.prepare_photoreal_walnut_boom_assets(target_bar_w=target_bar_w)
    canv_w = int(round(target_bar_w * 0.945))
    canv_h = 370
    canvas_full_rgb = intro_mod.generate_anybody_canvas_image(width=canv_w, height=canv_h)

    renderer = mujoco.Renderer(model, height=render_h, width=render_w)
    scene_opt = mujoco.MjvOption()

    t = 9.55
    info = set_fixed_intro_kinematics(model, data, t)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "intro_cam")
    # Frame H1 and the unrolling `A://AnyBody` canvas with balanced headroom
    model.cam_pos[cam_id] = np.array([2.62, 0.00, 1.24], dtype=np.float64)
    model.cam_fovy[cam_id] = 28.5
    mujoco.mj_forward(model, data)

    # Keep the walnut boom & unrolling canvas 100% horizontally centered in the 1920x1080 frame
    # while positioning the clasp + titanium pull-ring right at H1's right hand X coordinate!
    ring_px, _ = intro_mod._project_world_to_pixel(model, data, "intro_cam", info["handle_target"], render_w, render_h)
    centered_bar_x0 = (render_w - target_bar_w) // 2
    boom_assets["clasp_bar_x"] = ring_px - centered_bar_x0

    renderer.update_scene(data, camera="intro_cam", scene_option=scene_opt)
    raw_rgb = renderer.render()
    renderer.enable_depth_rendering()
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    seg = renderer.render()[:, :, 0].copy()
    renderer.disable_segmentation_rendering()
    renderer.close()

    comp_rgb = intro_mod.composite_minimalist_ui(
        raw_rgb, depth, seg, t, info, model, data, render_w, render_h, canvas_full_rgb, boom_assets
    )
    out_p = ARTIFACT_DIR / "design_snap_01_intro_pull.png"
    Image.fromarray(comp_rgb).save(out_p)
    return out_p


# ==============================================================================
# SHARED 3D UNITREE H1 STUDIO RENDERER WITH EXACT Z=0 FOOT GROUNDING
# ==============================================================================
def build_demo_studio_h1_model() -> mujoco.MjModel:
    """Build the Unitree H1 MuJoCo model with an ego-centric head camera (`ego_head_cam`) and
    real 3D scene objects (`env_floor`, `env_basket`, `env_obj_0..3`) so H1's first-person camera
    renders the actual 3D physical view from H1's optical sensor!
    """
    h1_xml_path = MENAGERIE_H1_DIR / "h1.xml"
    h1_root = ET.parse(h1_xml_path).getroot()
    compiler = h1_root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str(MENAGERIE_H1_DIR / "assets"))

    vis = ET.SubElement(h1_root, "visual")
    ET.SubElement(vis, "global", {"offwidth": "1920", "offheight": "1080", "azimuth": "0", "elevation": "-10"})
    ET.SubElement(vis, "quality", {"shadowsize": "4096", "offsamples": "4"})
    ET.SubElement(
        vis,
        "headlight",
        {"diffuse": "0.14 0.15 0.17", "ambient": "0.22 0.23 0.26", "specular": "0.10 0.11 0.12"},
    )

    asset = h1_root.find("asset")
    for mat in asset.findall("material"):
        if mat.get("name") == "black":
            mat.set("rgba", "0.16 0.17 0.20 1")
            mat.set("specular", "0.92")
            mat.set("shininess", "0.68")
            mat.set("reflectance", "0.22")
        elif mat.get("name") == "white":
            mat.set("rgba", "0.94 0.96 0.99 1")
            mat.set("specular", "1.0")
            mat.set("shininess", "0.88")
            mat.set("reflectance", "0.28")

    # Add a subtle grid texture for H1's ego-centric first-person camera floor view
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "ego_floor_grid",
            "type": "2d",
            "builtin": "checker",
            "rgb1": "0.95 0.97 1.00",
            "rgb2": "0.88 0.92 0.98",
            "width": "512",
            "height": "512",
            "mark": "edge",
            "markrgb": "0.72 0.82 0.95",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {"name": "ego_floor_mat", "texture": "ego_floor_grid", "texrepeat": "6 6", "reflectance": "0.08"},
    )

    wb = h1_root.find("worldbody")
    for l_el in list(wb.findall("light")):
        wb.remove(l_el)

    ET.SubElement(wb, "light", {"name": "key_light", "pos": "2.2 -1.6 3.0", "dir": "-0.55 0.45 -0.60", "diffuse": "0.56 0.58 0.62", "specular": "0.82 0.86 0.92", "castshadow": "true"})
    ET.SubElement(wb, "light", {"name": "fill_light", "pos": "2.0 1.8 2.2", "dir": "-0.60 -0.50 -0.35", "diffuse": "0.38 0.40 0.44", "specular": "0.62 0.66 0.72", "castshadow": "false"})
    ET.SubElement(wb, "light", {"name": "rim_light", "pos": "-1.5 -0.8 2.8", "dir": "0.65 0.25 -0.65", "diffuse": "0.36 0.38 0.42", "specular": "0.72 0.76 0.84", "castshadow": "true"})
    ET.SubElement(wb, "light", {"name": "graze_side", "pos": "1.2 -2.2 1.5", "dir": "-0.35 0.88 -0.10", "diffuse": "0.26 0.28 0.32", "specular": "0.68 0.72 0.80", "castshadow": "false"})

    ET.SubElement(wb, "camera", {"name": "intro_cam", "pos": "2.38 0.04 1.32", "xyaxes": "0 1 0 -0.015 0 0.9999", "fovy": "36.0"})
    ET.SubElement(wb, "camera", {"name": "ego_head_cam", "pos": "0 0 1.5", "xyaxes": "0 -1 0 0.4 0 0.9", "fovy": "62.0"})

    # 3D Environment Geoms (visible to H1's `ego_head_cam` first-person sensor!)
    ET.SubElement(wb, "geom", {"name": "env_floor", "type": "plane", "pos": "0 0 0", "size": "8 8 0.1", "material": "ego_floor_mat", "contype": "0", "conaffinity": "0"})

    b_basket = ET.SubElement(wb, "body", {"name": "mocap_basket", "mocap": "true", "pos": "0.5 0.5 0.0"})
    ET.SubElement(b_basket, "geom", {"name": "env_basket_base", "type": "cylinder", "pos": "0 0 0.012", "size": "0.34 0.012", "rgba": "0.78 0.96 0.89 1.0", "contype": "0", "conaffinity": "0"})
    ET.SubElement(b_basket, "geom", {"name": "env_basket_rim_n", "type": "box", "pos": "0.32 0 0.055", "size": "0.02 0.28 0.055", "rgba": "0.06 0.73 0.51 1.0", "contype": "0", "conaffinity": "0"})
    ET.SubElement(b_basket, "geom", {"name": "env_basket_rim_s", "type": "box", "pos": "-0.32 0 0.055", "size": "0.02 0.28 0.055", "rgba": "0.06 0.73 0.51 1.0", "contype": "0", "conaffinity": "0"})
    ET.SubElement(b_basket, "geom", {"name": "env_basket_rim_e", "type": "box", "pos": "0 0.32 0.055", "size": "0.28 0.02 0.055", "rgba": "0.06 0.73 0.51 1.0", "contype": "0", "conaffinity": "0"})
    ET.SubElement(b_basket, "geom", {"name": "env_basket_rim_w", "type": "box", "pos": "0 -0.32 0.055", "size": "0.28 0.02 0.055", "rgba": "0.06 0.73 0.51 1.0", "contype": "0", "conaffinity": "0"})

    obj_specs = [
        ("mocap_obj_0", "env_obj_0", "box", "0.095 0.095 0.095", "0.94 0.27 0.27 1"),
        ("mocap_obj_1", "env_obj_1", "sphere", "0.090", "0.00 0.51 0.98 1"),
        ("mocap_obj_2", "env_obj_2", "cylinder", "0.080 0.105", "0.06 0.73 0.51 1"),
        ("mocap_obj_3", "env_obj_3", "box", "0.105 0.105 0.105", "0.96 0.62 0.04 1"),
    ]
    for b_nm, g_nm, g_type, g_sz, g_rgba in obj_specs:
        b_el = ET.SubElement(wb, "body", {"name": b_nm, "mocap": "true", "pos": "0 0 0.1"})
        ET.SubElement(b_el, "geom", {"name": g_nm, "type": g_type, "size": g_sz, "rgba": g_rgba, "contype": "0", "conaffinity": "0"})

    xml_str = ET.tostring(h1_root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml_str)


def apply_grounded_v4_pose_to_h1(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    snap: dict,
    cam_azimuth_deg: float = -32.0,
    cam_elevation_deg: float = -12.0,
    cam_dist: float = 2.85,
    cam_lookat_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, int]:
    """Map a V4 biophysical snapshot onto Unitree H1's 19 joints and enforce EXACT Z=0.000m foot-sole grounding."""
    jadr: dict[str, int] = {}
    root_qadr = 0
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            root_qadr = int(model.jnt_qposadr[i])
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if nm:
            jadr[nm] = int(model.jnt_qposadr[i])

    px, py, pz = snap["pelvis"]
    yaw = float(snap["yaw"])

    # Compute local body-frame vectors by un-rotating world vectors by -yaw
    cy, sy = math.cos(-yaw), math.sin(-yaw)

    def to_local(p_world: list[float]) -> np.ndarray:
        dx, dy, dz = p_world[0], p_world[1], p_world[2]
        return np.array([dx * cy - dy * sy, dx * sy + dy * cy, dz], dtype=np.float64)

    l_hip = to_local(snap["l_hip"])
    l_knee = to_local(snap["l_knee"])
    l_ankle = to_local(snap["l_ankle"])
    r_hip = to_local(snap["r_hip"])
    r_knee = to_local(snap["r_knee"])
    r_ankle = to_local(snap["r_ankle"])

    # Left leg angles in H1's joint convention (hip_pitch < 0 is forward flexion, knee > 0 is flexion)
    v_lt = l_knee - l_hip
    v_ls = l_ankle - l_knee
    l_hp = -math.atan2(v_lt[0], max(0.08, -v_lt[2]))
    l_kn = max(0.12, math.atan2(v_lt[0], max(0.08, -v_lt[2])) - math.atan2(v_ls[0], max(0.08, -v_ls[2])))
    l_hr = math.atan2(v_lt[1], max(0.12, -v_lt[2])) * 0.65
    l_an = -(l_hp + l_kn) * 0.85

    # Right leg angles
    v_rt = r_knee - r_hip
    v_rs = r_ankle - r_knee
    r_hp = -math.atan2(v_rt[0], max(0.08, -v_rt[2]))
    r_kn = max(0.12, math.atan2(v_rt[0], max(0.08, -v_rt[2])) - math.atan2(v_rs[0], max(0.08, -v_rs[2])))
    r_hr = math.atan2(v_rt[1], max(0.12, -v_rt[2])) * 0.65
    r_an = -(r_hp + r_kn) * 0.85

    data.qpos[jadr["left_hip_yaw"]] = 0.0
    data.qpos[jadr["left_hip_roll"]] = float(np.clip(l_hr, -0.38, 0.38))
    data.qpos[jadr["left_hip_pitch"]] = float(np.clip(l_hp, -1.35, 0.85))
    data.qpos[jadr["left_knee"]] = float(np.clip(l_kn, 0.12, 1.85))
    data.qpos[jadr["left_ankle"]] = float(np.clip(l_an, -0.68, 0.45))

    data.qpos[jadr["right_hip_yaw"]] = 0.0
    data.qpos[jadr["right_hip_roll"]] = float(np.clip(r_hr, -0.38, 0.38))
    data.qpos[jadr["right_hip_pitch"]] = float(np.clip(r_hp, -1.35, 0.85))
    data.qpos[jadr["right_knee"]] = float(np.clip(r_kn, 0.12, 1.85))
    data.qpos[jadr["right_ankle"]] = float(np.clip(r_an, -0.68, 0.45))

    # Arms
    l_sh = to_local(snap["l_shoulder"])
    l_el = to_local(snap["l_elbow"])
    l_hd = to_local(snap["l_hand"])
    v_la = l_el - l_sh
    v_lf = l_hd - l_el
    l_sp = -math.atan2(v_la[0], max(0.06, -v_la[2]))
    if v_la[2] > 0.0:  # overhead lift
        l_sp = -1.55 - math.atan2(v_la[2], max(0.06, abs(v_la[0]))) * 0.65
    l_sr = max(0.10, math.atan2( max(0.0, v_la[1]), max(0.08, -v_la[2])) * 0.55 + 0.14)
    l_elb = float(np.clip(0.55 + 0.45 * math.atan2(v_lf[2] - v_la[2], max(0.08, v_lf[0])), 0.25, 1.45))

    r_sh = to_local(snap["r_shoulder"])
    r_el = to_local(snap["r_elbow"])
    r_hd = to_local(snap["r_hand"])
    v_ra = r_el - r_sh
    v_rf = r_hd - r_el
    r_sp = -math.atan2(v_ra[0], max(0.06, -v_ra[2]))
    if v_ra[2] > 0.0:
        r_sp = -1.55 - math.atan2(v_ra[2], max(0.06, abs(v_ra[0]))) * 0.65
    r_sr = min(-0.10, math.atan2(min(0.0, v_ra[1]), max(0.08, -v_ra[2])) * 0.55 - 0.14)
    r_elb = float(np.clip(0.55 + 0.45 * math.atan2(v_rf[2] - v_ra[2], max(0.08, v_rf[0])), 0.25, 1.45))

    data.qpos[jadr["torso"]] = 0.0
    data.qpos[jadr["left_shoulder_pitch"]] = float(np.clip(l_sp, -2.2, 1.2))
    data.qpos[jadr["left_shoulder_roll"]] = float(np.clip(l_sr, 0.10, 0.95))
    data.qpos[jadr["left_shoulder_yaw"]] = 0.0
    data.qpos[jadr["left_elbow"]] = l_elb

    data.qpos[jadr["right_shoulder_pitch"]] = float(np.clip(r_sp, -2.2, 1.2))
    data.qpos[jadr["right_shoulder_roll"]] = float(np.clip(r_sr, -0.95, -0.10))
    data.qpos[jadr["right_shoulder_yaw"]] = 0.0
    data.qpos[jadr["right_elbow"]] = r_elb

    # Estimate pitch lean during squat or bow
    head_l = to_local(snap["head"])
    pelv_l = to_local(snap["pelvis"])
    torso_pitch = math.atan2(head_l[0] - pelv_l[0], max(0.20, head_l[2] - pelv_l[2])) * 0.55

    data.qpos[root_qadr + 0] = px
    data.qpos[root_qadr + 1] = py
    data.qpos[root_qadr + 2] = 0.98
    data.qpos[root_qadr + 3 : root_qadr + 7] = intro_mod._quat_from_yaw_pitch_roll(yaw, torso_pitch, 0.0)
    mujoco.mj_kinematics(model, data)

    # EXACT Z=0.000m FOOT-SOLE GROUNDING:
    # Measure both ankle world Z coordinates and shift `qpos[2]` so the lowest foot sole rests on z = 0.000m!
    la_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_link")
    ra_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_link")
    sole_offset = 0.052  # distance from ankle joint axis to bottom of H1 foot sole
    l_sole_z = float(data.xpos[la_id][2]) - sole_offset
    r_sole_z = float(data.xpos[ra_id][2]) - sole_offset

    # Preserve intentional swing-foot clearance relative to stance foot
    swing_diff = abs(snap["l_ankle"][2] - snap["r_ankle"][2])
    min_sole_z = min(l_sole_z, r_sole_z)
    data.qpos[root_qadr + 2] -= min_sole_z
    mujoco.mj_forward(model, data)

    # Position `intro_cam` to orbit smoothly around H1's pelvis
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "intro_cam")
    lookat = np.array(
        [
            px + cam_lookat_offset[0],
            py + cam_lookat_offset[1],
            float(data.qpos[root_qadr + 2]) * 0.88 + cam_lookat_offset[2],
        ],
        dtype=np.float64,
    )
    az = math.radians(cam_azimuth_deg)
    el = math.radians(cam_elevation_deg)
    cam_pos = lookat + np.array(
        [
            cam_dist * math.cos(el) * math.cos(az),
            cam_dist * math.cos(el) * math.sin(az),
            cam_dist * (-math.sin(el)),
        ],
        dtype=np.float64,
    )
    fwd = lookat - cam_pos
    fwd /= np.linalg.norm(fwd) + 1e-9
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right) + 1e-9
    up = np.cross(right, fwd)

    model.cam_pos[cam_id] = cam_pos
    # MuJoCo cam_mat columns are [right, up, -fwd]
    cam_mat = np.column_stack([right, up, -fwd])
    # Convert 3x3 rotation matrix to unit quaternion (w, x, y, z)
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, cam_mat.flatten())
    model.cam_quat[cam_id] = quat
    model.cam_fovy[cam_id] = 36.0
    mujoco.mj_forward(model, data)
    return jadr


def draw_studio_floor_and_3d_objects(
    stage_rgba: Image.Image,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    snap: dict,
    show_objects: bool = True,
    highlight_grf: bool = False,
) -> dict[str, tuple[int, int]]:
    """Draw subtle architectural perspective floor rings on z=0.000m, multi-layered ground contact shadows
    under H1's feet, and photorealistic shaded 3D objects ON the z=0.000m floor (or held in H1's hands).
    """
    W, H = stage_rgba.size
    la_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_link")
    ra_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_link")
    le_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_elbow_link")
    re_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_elbow_link")
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

    l_foot_w = np.array([data.xpos[la_id][0], data.xpos[la_id][1], 0.0], dtype=np.float64)
    r_foot_w = np.array([data.xpos[ra_id][0], data.xpos[ra_id][1], 0.0], dtype=np.float64)
    pelv_floor_w = np.array([data.qpos[0], data.qpos[1], 0.0], dtype=np.float64)

    l_hand_w = data.xpos[le_id] + data.xmat[le_id].reshape(3, 3) @ np.array([0.28, 0.0, -0.015])
    r_hand_w = data.xpos[re_id] + data.xmat[re_id].reshape(3, 3) @ np.array([0.28, 0.0, -0.015])
    head_w = data.xpos[torso_id] + data.xmat[torso_id].reshape(3, 3) @ np.array([0.05, 0.0, 0.64])

    # 1. Stationary World-Space 3D Perspective Grid & Concentric Rings on z=0.000m (Anchored to World Origin [0,0,0]!)
    origin_w = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    floor_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    fd = ImageDraw.Draw(floor_layer)
    # Stationary metric floor grid (every 0.30m on z=0.000m) so 1:1 foot contact & zero-slip walking are unmistakably visible
    for g_val in np.linspace(-1.5, 1.5, 11):
        is_axis = abs(g_val) < 1e-4
        col_g = (0, 100, 224, 34 if is_axis else 18)
        w_g = 2 if is_axis else 1
        p0_x = intro_mod._project_world_to_pixel(model, data, "intro_cam", np.array([g_val, -1.5, 0.0]), W, H)
        p1_x = intro_mod._project_world_to_pixel(model, data, "intro_cam", np.array([g_val, 1.5, 0.0]), W, H)
        fd.line([p0_x, p1_x], fill=col_g, width=w_g)
        p0_y = intro_mod._project_world_to_pixel(model, data, "intro_cam", np.array([-1.5, g_val, 0.0]), W, H)
        p1_y = intro_mod._project_world_to_pixel(model, data, "intro_cam", np.array([1.5, g_val, 0.0]), W, H)
        fd.line([p0_y, p1_y], fill=col_g, width=w_g)

    for r_m, alpha in [(0.45, 28), (0.95, 22), (1.55, 15), (2.25, 10)]:
        pts = []
        for deg in range(0, 361, 6):
            rad = math.radians(deg)
            pw = origin_w + np.array([r_m * math.cos(rad), r_m * math.sin(rad), 0.0])
            px, py = intro_mod._project_world_to_pixel(model, data, "intro_cam", pw, W, H)
            pts.append((px, py))
        fd.line(pts, fill=(0, 100, 224, alpha), width=2)
    stage_rgba.alpha_composite(floor_layer)

    # 2. Soft Ambient-Occlusion Ground Contact Shadows under Left Foot, Right Foot, and Pelvis on z=0.000m (ROI-cropped blur)
    l_px, l_py = intro_mod._project_world_to_pixel(model, data, "intro_cam", l_foot_w, W, H)
    r_px, r_py = intro_mod._project_world_to_pixel(model, data, "intro_cam", r_foot_w, W, H)
    p_px, p_py = intro_mod._project_world_to_pixel(model, data, "intro_cam", pelv_floor_w, W, H)

    sx0 = max(0, min(l_px, r_px, p_px) - 140)
    sy0 = max(0, min(l_py, r_py, p_py) - 70)
    sx1 = min(W, max(l_px, r_px, p_px) + 140)
    sy1 = min(H, max(l_py, r_py, p_py) + 70)
    if sx1 > sx0 + 4 and sy1 > sy0 + 4:
        sh_roi = Image.new("RGBA", (sx1 - sx0, sy1 - sy0), (0, 0, 0, 0))
        sd = ImageDraw.Draw(sh_roi)
        sd.ellipse([p_px - sx0 - 95, p_py - sy0 - 28, p_px - sx0 + 95, p_py - sy0 + 28], fill=(10, 22, 48, 26))
        for fx, fy, f_z in [(l_px, l_py, float(data.xpos[la_id][2])), (r_px, r_py, float(data.xpos[ra_id][2]))]:
            contact_scale = max(0.20, min(1.0, 1.0 - (f_z - 0.052) * 2.4))
            rx_s = int(46 * contact_scale)
            ry_s = int(16 * contact_scale)
            alpha_s = int(82 * contact_scale)
            sd.ellipse([fx - sx0 - rx_s, fy - sy0 - ry_s, fx - sx0 + rx_s, fy - sy0 + ry_s], fill=(8, 16, 36, alpha_s))
        sh_roi = sh_roi.filter(ImageFilter.GaussianBlur(radius=12))
        stage_rgba.alpha_composite(sh_roi, dest=(sx0, sy0))

    if highlight_grf:
        grf_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        gd = ImageDraw.Draw(grf_layer)
        for fx, fy, grf_val in [(l_px, l_py, snap["l_grf"]), (r_px, r_py, snap["r_grf"])]:
            if grf_val > 25.0:
                rr = int(34 + min(42.0, grf_val * 0.055))
                gd.ellipse([fx - rr, fy - rr // 3, fx + rr, fy + rr // 3], outline=(0, 129, 251, 185), width=3)
                gd.ellipse([fx - rr - 14, fy - (rr + 14) // 3, fx + rr + 14, fy + (rr + 14) // 3], outline=(0, 212, 255, 95), width=2)
        stage_rgba.alpha_composite(grf_layer)

    proj_pts = {
        "l_foot": (l_px, l_py),
        "r_foot": (r_px, r_py),
        "l_ankle_3d": intro_mod._project_world_to_pixel(model, data, "intro_cam", data.xpos[la_id], W, H),
        "r_ankle_3d": intro_mod._project_world_to_pixel(model, data, "intro_cam", data.xpos[ra_id], W, H),
        "pelvis_floor": (p_px, p_py),
        "l_hand": intro_mod._project_world_to_pixel(model, data, "intro_cam", l_hand_w, W, H),
        "r_hand": intro_mod._project_world_to_pixel(model, data, "intro_cam", r_hand_w, W, H),
        "head": intro_mod._project_world_to_pixel(model, data, "intro_cam", head_w, W, H),
    }

    if not show_objects:
        return proj_pts

    # 3. Ground contact shadows & 2D projection coordinates for the REAL 3D MuJoCo objects & basket
    #    (The objects and basket themselves are rendered as true 3D MuJoCo bodies in `render_h1_layer` with real depth occlusion!)
    obj_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(obj_layer)

    if snap.get("basket") is not None:
        b_w = np.array([snap["basket"][0], snap["basket"][1], 0.0], dtype=np.float64)
        b_pts = []
        for deg in range(0, 361, 8):
            rad = math.radians(deg)
            pw = b_w + np.array([0.36 * math.cos(rad), 0.36 * math.sin(rad), 0.0])
            b_pts.append(intro_mod._project_world_to_pixel(model, data, "intro_cam", pw, W, H))
        od.polygon(b_pts, fill=(16, 185, 129, 24), outline=(16, 185, 129, 175))
        bx_c, by_c = intro_mod._project_world_to_pixel(model, data, "intro_cam", b_w, W, H)
        proj_pts["basket"] = (bx_c, by_c)

    default_z = [0.095, 0.090, 0.105, 0.105]
    for idx, obj in enumerate(snap["objects"]):
        pos_raw = obj["pos"]
        oz = float(pos_raw[2]) if len(pos_raw) >= 3 else (default_z[idx] if idx < len(default_z) else 0.095)
        obj_w = np.array([float(pos_raw[0]), float(pos_raw[1]), oz], dtype=np.float64)
        floor_w = np.array([float(pos_raw[0]), float(pos_raw[1]), 0.0], dtype=np.float64)

        f_px, f_py = intro_mod._project_world_to_pixel(model, data, "intro_cam", floor_w, W, H)
        o_px, o_py = intro_mod._project_world_to_pixel(model, data, "intro_cam", obj_w, W, H)
        proj_pts[f"obj_{idx}"] = (o_px, o_py)

        # Soft ground shadow under object on z=0.000m (attenuates with object height)
        h_att = max(0.22, min(1.0, 1.0 - max(0.0, oz - 0.10) * 0.65))
        rx_o = int(28 * h_att)
        ry_o = int(11 * h_att)
        al_o = int(52 * h_att)
        od.ellipse([f_px - rx_o, f_py - ry_o, f_px + rx_o, f_py + ry_o], fill=(12, 22, 42, al_o))

    obj_layer = obj_layer.filter(ImageFilter.GaussianBlur(radius=4))
    stage_rgba.alpha_composite(obj_layer)
    return proj_pts


def _set_env_geoms_visible(model: mujoco.MjModel, visible: bool, include_floor: bool = False) -> None:
    """Toggle visibility of 3D environment geoms (`env_obj_*` and `env_basket_*`) vs `env_floor`."""
    for gid in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if not nm or not nm.startswith("env_"):
            continue
        if nm == "env_floor":
            model.geom_rgba[gid, 3] = 1.0 if (visible and include_floor) else 0.0
        else:
            model.geom_rgba[gid, 3] = 1.0 if visible else 0.0


def render_h1_layer(model: mujoco.MjModel, data: mujoco.MjData, renderer: mujoco.Renderer, show_3d_objects: bool = True) -> Image.Image:
    """Render Unitree H1 together with the 3D MuJoCo objects (`env_obj_*`) and basket (`env_basket_*`) in one
    unified 3D OpenGL pass so H1's hands and legs have true 3D depth occlusion against the objects, while hiding
    `env_floor` so the studio floor canvas remains clean.
    """
    _set_env_geoms_visible(model, visible=show_3d_objects, include_floor=False)
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "env_floor")
    scene_opt = mujoco.MjvOption()
    renderer.update_scene(data, camera="intro_cam", scene_option=scene_opt)
    raw_rgb = renderer.render()
    renderer.enable_depth_rendering()
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    seg = renderer.render()[:, :, 0].copy()
    renderer.disable_segmentation_rendering()
    if floor_gid >= 0:
        seg[seg == floor_gid] = -1
    if not show_3d_objects:
        for gid in range(model.ngeom):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if nm and nm.startswith("env_"):
                seg[seg == gid] = -1
    return intro_mod._shade_photoreal_h1(raw_rgb, depth, seg, scale=1.0)


def render_ego_sam3_view(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    snap: dict,
    out_w: int = 440,
    out_h: int = 196,
) -> Image.Image:
    """Render Unitree H1's ACTUAL 3D First-Person Head-Camera View (`ego_head_cam`) with real-time
    SAM-3.1 pixel segmentation masks, contour outlines, 2D bounding brackets, and metric depth readouts!
    """
    _set_env_geoms_visible(model, visible=True, include_floor=True)

    # 1. Sync 3D mocap positions of the Basket and Objects with `snap` (preserving true 3D Z heights!)
    b_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mocap_basket")
    if b_bid >= 0 and model.body_mocapid[b_bid] >= 0:
        if snap.get("basket") is not None:
            data.mocap_pos[model.body_mocapid[b_bid]] = np.array([snap["basket"][0], snap["basket"][1], 0.0], dtype=np.float64)
        else:
            data.mocap_pos[model.body_mocapid[b_bid]] = np.array([0.0, 0.0, -10.0], dtype=np.float64)

    default_z = [0.095, 0.090, 0.105, 0.105]
    for idx, obj in enumerate(snap["objects"]):
        ob_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"mocap_obj_{idx}")
        if ob_id >= 0 and model.body_mocapid[ob_id] >= 0:
            pos_raw = obj["pos"]
            oz = float(pos_raw[2]) if len(pos_raw) >= 3 else default_z[idx]
            data.mocap_pos[model.body_mocapid[ob_id]] = np.array(
                [float(pos_raw[0]), float(pos_raw[1]), oz],
                dtype=np.float64,
            )
    for idx in range(len(snap["objects"]), 4):
        ob_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"mocap_obj_{idx}")
        if ob_id >= 0 and model.body_mocapid[ob_id] >= 0:
            data.mocap_pos[model.body_mocapid[ob_id]] = np.array([0.0, 0.0, -10.0], dtype=np.float64)

    # 2. Mount `ego_head_cam` at Unitree H1's optical head sensor (`eye_pos`), looking forward & down
    #    at the active workspace or follow H1's head orientation during visual search scans.
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    le_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_elbow_link")
    re_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_elbow_link")

    saved_alpha = model.geom_rgba[:, 3].copy()
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if not nm.startswith("env_") and bid not in (le_id, re_id):
            model.geom_rgba[gid, 3] = 0.0

    t_rot = data.xmat[torso_id].reshape(3, 3)
    eye_pos = data.xpos[torso_id] + t_rot @ np.array([0.12, 0.0, 0.62], dtype=np.float64)

    # Determine active gaze target from `v3_task_state` if present, otherwise frame workspace centroid
    t_state = snap.get("v3_task_state")
    if t_state is not None and snap.get("basket") is not None and len(snap["objects"]) >= 4:
        c_idx = min(3, int(t_state.get("active_cycle_idx", t_state.get("cycle_idx", 2))))
        act_pos = np.array(snap["objects"][c_idx]["pos"], dtype=np.float64)
        bsk_pos = np.array([snap["basket"][0], snap["basket"][1], 0.12], dtype=np.float64)
        br_dec = t_state.get("branch_decision", t_state.get("branch_id", ""))
        if br_dec in ("AUDIT_360_HALT", "AUDIT_COMPLETE"):
            gaze_target = 0.65 * bsk_pos + 0.35 * np.array([0.15, 0.0, 0.08], dtype=np.float64)
        elif snap["objects"][c_idx].get("held", False):
            gaze_target = 0.45 * act_pos + 0.55 * bsk_pos
        else:
            gaze_target = 0.68 * act_pos + 0.32 * bsk_pos
    elif len(snap["objects"]) >= 4 and snap.get("basket") is not None:
        gaze_target = np.array(
            [
                0.38 * snap["objects"][2]["pos"][0] + 0.28 * snap["objects"][3]["pos"][0] + 0.34 * snap["basket"][0],
                0.38 * snap["objects"][2]["pos"][1] + 0.28 * snap["objects"][3]["pos"][1] + 0.34 * snap["basket"][1],
                0.08,
            ],
            dtype=np.float64,
        )
    elif len(snap["objects"]) >= 1:
        # Demo 1: Track the 3D Red Box (and when lifted overhead, step eye_pos slightly back along head frame so hands + box stay framed)
        box_p = np.array(snap["objects"][0]["pos"], dtype=np.float64)
        if box_p[2] > 1.15:
            eye_pos = data.xpos[torso_id] + t_rot @ np.array([-0.14, 0.0, 0.56], dtype=np.float64)
        gaze_target = box_p.copy()
    else:
        gaze_target = eye_pos + t_rot @ np.array([1.2, 0.0, -0.45], dtype=np.float64)

    fwd = gaze_target - eye_pos
    fwd /= np.linalg.norm(fwd) + 1e-9
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right) + 1e-9
    up = np.cross(right, fwd)

    ego_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ego_head_cam")
    model.cam_pos[ego_cam_id] = eye_pos
    cam_mat = np.column_stack([right, up, -fwd])
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, cam_mat.flatten())
    model.cam_quat[ego_cam_id] = quat
    model.cam_fovy[ego_cam_id] = 60.0
    mujoco.mj_forward(model, data)

    # 3. Render true 3D First-Person RGB + Depth + Segmentation from `ego_head_cam`
    scene_opt = mujoco.MjvOption()
    renderer.update_scene(data, camera="ego_head_cam", scene_option=scene_opt)
    ego_rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    ego_depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    ego_seg = renderer.render()[:, :, 0].copy()
    renderer.disable_segmentation_rendering()

    model.geom_rgba[:, 3] = saved_alpha
    _set_env_geoms_visible(model, visible=True, include_floor=False)

    # 4. Overlay pixel-accurate SAM-3.1 Segmentation Masks & Contours on the full-res render before downsampling
    overlay = ego_rgb.astype(np.float32)
    if len(snap["objects"]) == 1:
        c_active = 0
    else:
        c_active = int(t_state.get("active_cycle_idx", t_state.get("cycle_idx", 2))) if t_state is not None else 2
    n_deliv = int(t_state.get("n_delivered", 2)) if t_state is not None else 2
    seg_targets = [
        (["env_obj_0"], (239, 68, 68), "Red Box", c_active == 0),
        (["env_obj_1"], (0, 130, 251), "Blue Ball", c_active == 1),
        (["env_obj_2"], (16, 185, 129), "Green Cyl", c_active == 2),
        (["env_obj_3"], (245, 158, 11), "Gold Crate", c_active == 3),
        (["env_basket_base", "env_basket_rim_n", "env_basket_rim_s", "env_basket_rim_e", "env_basket_rim_w"], (0, 129, 251), f"Basket ({n_deliv}/4)", False),
    ]

    bboxes_2d = []
    for geom_names, col_rgb, label, is_locked in seg_targets:
        gids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm) for nm in geom_names]
        mask = np.isin(ego_seg, gids)
        if np.count_nonzero(mask) < 25:
            continue
        # Blend SAM-3.1 translucent segmentation tint over the actual 3D pixels
        col_arr = np.array(col_rgb, dtype=np.float32)
        overlay[mask] = overlay[mask] * 0.58 + col_arr * 0.42
        # Crisp 3px morphological edge contour around the 3D object silhouette
        eroded = ndimage.binary_erosion(mask, iterations=4)
        edge = mask & (~eroded)
        overlay[edge] = np.array([255.0, 255.0, 255.0]) if is_locked else col_arr

        ys, xs = np.where(mask)
        d_m = float(np.median(ego_depth[mask]))
        bboxes_2d.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), col_rgb, label, d_m, is_locked))

    ego_pil = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB").resize(
        (out_w, out_h), Image.Resampling.LANCZOS
    ).convert("RGBA")

    # 5. Draw SAM-3.1 HUD Corner Reticles & Real Metric Depth Tags on the resized first-person camera feed
    sx = out_w / 1920.0
    sy = out_h / 1080.0
    d = ImageDraw.Draw(ego_pil)
    # Subtle optical crosshair & FOV telemetry watermark inside camera feed
    rcx, rcy = out_w // 2, out_h // 2
    d.line([(rcx - 14, rcy), (rcx + 14, rcy)], fill=(255, 255, 255, 160), width=1)
    d.line([(rcx, rcy - 14), (rcx, rcy + 14)], fill=(255, 255, 255, 160), width=1)

    f_tag = load_sf_font(10, bold=True, mono=True)
    for x0_f, y0_f, x1_f, y1_f, col_rgb, label, d_m, is_locked in bboxes_2d:
        if "IN" in label:
            continue  # keep HUD uncluttered around basket
        bx0 = max(4, int(x0_f * sx) - 4)
        by0 = max(4, int(y0_f * sy) - 4)
        bx1 = min(out_w - 4, int(x1_f * sx) + 4)
        by1 = min(out_h - 4, int(y1_f * sy) + 4)
        # Corner brackets
        clen = 7
        for cx_c, cy_c, dx_c, dy_c in [
            (bx0, by0, clen, clen),
            (bx1, by0, -clen, clen),
            (bx0, by1, clen, -clen),
            (bx1, by1, -clen, -clen),
        ]:
            d.line([(cx_c, cy_c), (cx_c + dx_c, cy_c)], fill=col_rgb, width=2)
            d.line([(cx_c, cy_c), (cx_c, cy_c + dy_c)], fill=col_rgb, width=2)

        tag_txt = f"{label} {d_m:.2f}m" + (" [LOCK]" if is_locked else "")
        tw = d.textbbox((0, 0), tag_txt, font=f_tag)[2] + 10
        tx0 = max(4, min(out_w - tw - 4, (bx0 + bx1 - tw) // 2))
        ty0 = max(4, by0 - 18) if by0 > 22 else min(out_h - 18, by1 + 3)
        d.rounded_rectangle([tx0, ty0, tx0 + tw, ty0 + 15], radius=4, fill=(10, 18, 36, 215), outline=col_rgb, width=1)
        d.text((tx0 + 5, ty0 + 2), tag_txt, fill=(255, 255, 255), font=f_tag)

    # Rounded viewport mask
    mask_round = Image.new("L", (out_w, out_h), 0)
    ImageDraw.Draw(mask_round).rounded_rectangle([0, 0, out_w - 1, out_h - 1], radius=14, fill=255)
    out_card_img = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
    out_card_img.paste(ego_pil, (0, 0), mask_round)
    ImageDraw.Draw(out_card_img).rounded_rectangle([0, 0, out_w - 1, out_h - 1], radius=14, outline=(0, 129, 251, 160), width=2)
    return out_card_img


def draw_spatial_callout(
    stage_rgba: Image.Image,
    target_xy: tuple[int, int],
    card_xy: tuple[int, int],
    badge: str,
    title: str,
    metric: str,
    accent_rgb: tuple[int, int, int] = (0, 129, 251),
) -> None:
    """Draw a sleek Silicon Valley SaaS spatial leader-line + glassmorphic callout pill."""
    tx, ty = target_xy
    cx, cy = card_xy
    f_b = load_sf_font(12, bold=True, mono=True)
    f_t = load_sf_font(17, bold=True)
    f_m = load_sf_font(14, bold=False, mono=True)

    tmp_d = ImageDraw.Draw(stage_rgba)
    w_t = tmp_d.textbbox((0, 0), title, font=f_t)[2]
    w_m = tmp_d.textbbox((0, 0), metric, font=f_m)[2]
    card_w = max(w_t, w_m) + 36
    card_h = 74

    # Leader line with elbow bend
    elbow_x = tx + (42 if cx > tx else -42)
    elbow_y = cy + card_h // 2
    anchor_x = cx if cx > tx else cx + card_w

    tmp_d.line([(tx, ty), (elbow_x, elbow_y), (anchor_x, elbow_y)], fill=accent_rgb + (210,), width=2)
    tmp_d.ellipse([tx - 6, ty - 6, tx + 6, ty + 6], fill=accent_rgb + (255,), outline=(255, 255, 255, 255), width=2)
    tmp_d.ellipse([tx - 13, ty - 13, tx + 13, ty + 13], outline=accent_rgb + (110,), width=2)

    draw_glass_card(
        stage_rgba,
        [cx, cy, cx + card_w, cy + card_h],
        radius=16,
        border_col=accent_rgb + (95,),
        shadow_col=accent_rgb + (24,),
    )
    d = ImageDraw.Draw(stage_rgba)
    d.text((cx + 16, cy + 10), badge, fill=accent_rgb, font=f_b)
    d.text((cx + 16, cy + 27), title, fill=(10, 18, 36), font=f_t)
    d.text((cx + 16, cy + 50), metric, fill=(71, 85, 105), font=f_m)


# ==============================================================================
# SNAPSHOT 2: "How It Works" — 25.11M-Synapse Neuromorphic Architecture X-Ray
# ==============================================================================
def render_snapshot_2_architecture_brain(model: mujoco.MjModel, data: mujoco.MjData, renderer: mujoco.Renderer) -> Path:
    with open("target/demo1_lift.jsonl", "r") as f:
        snaps = [json.loads(line) for line in f]
    snap = snaps[12]

    apply_grounded_v4_pose_to_h1(
        model, data, snap,
        cam_azimuth_deg=-26.0,
        cam_elevation_deg=-11.0,
        cam_dist=3.68,
        cam_lookat_offset=(-0.36, -0.32, -0.06),  # frames full H1 head-to-toe on right half of 1920x1080 canvas
    )
    stage = Image.new("RGBA", (1920, 1080), (255, 255, 255, 255))
    pts = draw_studio_floor_and_3d_objects(stage, model, data, snap, show_objects=False, highlight_grf=True)
    h1_rgba = render_h1_layer(model, data, renderer)
    stage.alpha_composite(h1_rgba)

    draw_top_brand_bar(
        stage,
        act_badge="ACT I · HOW IT WORKS — BIOPHYSICAL CONNECTOME ARCHITECTURE",
        title_right="25.11M Synapses · 84.2% Spike Sparsity · 422W vs 1,221W Stiff PD",
    )

    # Left Bento Grid: 4 High-Contrast Architectural Modules Explaining How AnyBody Works
    bento_specs = [
        (
            "LAYER 1 · EXECUTIVE & DORSAL VISION",
            "Meta Muse + SAM-3.1 Retinal Stream",
            "Compiles natural language & 116° RGB-D segmentation into\n14-D Parietal/Basal-Ganglia attractors (0 hardcoded joints).",
            "0.04 ms / tick",
            (0, 100, 224),
            pts["head"],
        ),
        (
            "LAYER 2 · CORTICAL MOTOR HIERARCHY",
            "L5 Pyramidal BAC Burst + VIP/SST/PV",
            "Two-compartment apical/basal coincidence multiplexing;\nself-sorted homunculus via prenatal motor babbling.",
            "53.8% Topo Order",
            (124, 58, 237),
            (pts["head"][0] - 12, pts["head"][1] + 75),
        ),
        (
            "LAYER 3 · CEREBELLAR PREDICTIVE DELAY",
            "mGluR1 44ms Eligibility + Kawato FEL",
            "Cancels 30ms peripheral sensory delay in real time via\nPurkinje parallel-fiber LTD & DCN predictive drive.",
            "-16.9% Reflex Err",
            (0, 145, 215),
            (pts["head"][0] - 18, pts["head"][1] + 150),
        ),
        (
            "LAYER 4 · SPINAL CPG & HILL MUSCLES",
            "Half-Center Rhythm + α-γ Co-Activation",
            "Sensory-gated stance/swing transitions & emergent\nmuscle co-contraction stiffness — ZERO PID controllers.",
            "2.89x Watt Gain",
            (16, 185, 129),
            pts["l_foot"],
        ),
    ]

    f_layer = load_sf_font(12, bold=True, mono=True)
    f_title = load_sf_font(21, bold=True)
    f_desc = load_sf_font(15, bold=False)
    f_pill = load_sf_font(13, bold=True, mono=True)

    for idx, (layer_tag, title, desc, metric_badge, col, target_pt) in enumerate(bento_specs):
        x0 = 64
        y0 = 108 + idx * 192
        x1 = 830
        y1 = y0 + 174

        # Smooth 3-segment leader line from Bento Card right edge to H1 anatomical node
        d_line = ImageDraw.Draw(stage)
        mid_x = x1 + 68 + idx * 18
        cy_card = (y0 + y1) // 2
        tx, ty = target_pt
        d_line.line([(x1, cy_card), (mid_x, cy_card), (mid_x, ty), (tx, ty)], fill=col + (185,), width=2)
        d_line.ellipse([tx - 7, ty - 7, tx + 7, ty + 7], fill=col + (255,), outline=(255, 255, 255, 255), width=2)
        d_line.ellipse([tx - 15, ty - 15, tx + 15, ty + 15], outline=col + (115,), width=2)

        draw_glass_card(stage, [x0, y0, x1, y1], radius=20, border_col=col + (85,), shadow_col=col + (20,))
        d = ImageDraw.Draw(stage)
        # Left vertical accent bar inside card
        d.rounded_rectangle([x0 + 16, y0 + 20, x0 + 22, y1 - 20], radius=3, fill=col)
        d.text((x0 + 38, y0 + 18), layer_tag, fill=col, font=f_layer)
        d.text((x0 + 38, y0 + 42), title, fill=(10, 18, 36), font=f_title)

        for l_i, line in enumerate(desc.split("\n")):
            d.text((x0 + 38, y0 + 78 + l_i * 23), line, fill=(71, 85, 105), font=f_desc)

        # Metric badge top-right of card (using tint_rgb so background is pastel and text is 100% crisp)
        bw = d.textbbox((0, 0), metric_badge, font=f_pill)[2] + 28
        d.rounded_rectangle([x1 - bw - 18, y0 + 16, x1 - 18, y0 + 46], radius=15, fill=tint_rgb(col, 0.12), outline=col, width=1)
        d.text((x1 - bw - 4, y0 + 23), metric_badge, fill=col, font=f_pill)

    # Right-side Falsifiability & Wattage Proof Card
    draw_spatial_callout(
        stage,
        target_xy=pts["r_foot"],
        card_xy=(1410, 690),
        badge="FALSIFIABLE PHYSICS PROOF",
        title="Zero Hidden PD Loops",
        metric="Untrained 0-Weight Brain Falls in 0.179s",
        accent_rgb=(16, 185, 129),
    )

    draw_meta_prompt_dock(
        stage,
        prompt_text="Architecture X-Ray: 4-Layer Spiking Connectome → Unitree H1 Whole-Body Control",
        model_badge="2.89x WATTAGE EFFICIENCY",
        voice_narration="Instead of brittle PID loops or keyframe rewards, AnyBody runs a 25-million-synapse human connectome at 1,000 Hertz.",
    )

    out_p = ARTIFACT_DIR / "design_snap_02_architecture_brain.png"
    stage.convert("RGB").save(out_p)
    return out_p


# ==============================================================================
# SNAPSHOT 3: Demo 1 — Deep Squat, Bimanual Grasp & Overhead Lift
# ==============================================================================
def render_snapshot_3_demo1_squat_lift(model: mujoco.MjModel, data: mujoco.MjData, renderer: mujoco.Renderer) -> Path:
    with open("target/demo1_lift.jsonl", "r") as f:
        snaps = [json.loads(line) for line in f]
    # Pick frame where object is grasped and lifted high
    held_frames = [s for s in snaps if s["objects"][0]["held"]]
    snap = held_frames[-15] if len(held_frames) > 15 else snaps[-10]

    apply_grounded_v4_pose_to_h1(
        model, data, snap,
        cam_azimuth_deg=-35.0,
        cam_elevation_deg=-12.0,
        cam_dist=3.75,
        cam_lookat_offset=(-0.18, -0.14, -0.05),
    )
    stage = Image.new("RGBA", (1920, 1080), (255, 255, 255, 255))
    pts = draw_studio_floor_and_3d_objects(stage, model, data, snap, show_objects=True, highlight_grf=True)
    h1_rgba = render_h1_layer(model, data, renderer)
    stage.alpha_composite(h1_rgba)

    # Draw held Red Box in front of H1's hands so it is clearly visible in H1's bimanual grip!
    ox, oy = pts["obj_0"]
    r = 26
    od = ImageDraw.Draw(stage)
    top_face = [(ox, oy - r), (ox + r, oy - r // 2), (ox, oy), (ox - r, oy - r // 2)]
    left_face = [(ox - r, oy - r // 2), (ox, oy), (ox, oy + r), (ox - r, oy + r // 2)]
    right_face = [(ox, oy), (ox + r, oy - r // 2), (ox + r, oy + r // 2), (ox, oy + r)]
    od.polygon(top_face, fill=(252, 115, 115), outline=(255, 255, 255))
    od.polygon(left_face, fill=(239, 68, 68), outline=(255, 255, 255))
    od.polygon(right_face, fill=(185, 28, 28), outline=(255, 255, 255))

    draw_top_brand_bar(
        stage,
        act_badge="DEMO 1 · WHOLE-BODY LOCOMOTION, DEEP SQUAT & OVERHEAD LIFT",
    )

    # Left Pipeline Stepper Card (shows 5 compiled stages with crisp vector checkmarks)
    draw_glass_card(stage, [56, 118, 520, 490], radius=22, border_col=(0, 129, 251, 85))
    d = ImageDraw.Draw(stage)
    f_h = load_sf_font(13, bold=True, mono=True)
    f_t = load_sf_font(20, bold=True)
    d.text((80, 140), "PREFRONTAL COMPILER · 5 STAGES", fill=(0, 100, 224), font=f_h)
    d.text((80, 164), "Zero-Shot Task Decomposition", fill=(10, 18, 36), font=f_t)

    stages_ui = [
        ("1. Approach Red Box", "Locomotion CPG · 1.76 m/s", True),
        ("2. Deep Squat to 0.51m", "Hip/Knee Co-Activation", True),
        ("3. Bimanual Tactile Grasp", "S1→M1 Slip Reflex · 42.9 N", True),
        ("4. Stand & Axial Pivot Left", "150.9° Yaw · Stance Grounded", True),
        ("5. Bimanual Overhead Lift", "Object Z: 0.25m → 1.31m", True),
    ]
    for idx, (st_name, st_sub, is_done) in enumerate(stages_ui):
        sy = 212 + idx * 52
        col = (16, 185, 129) if idx < 4 else (0, 100, 224)
        d.ellipse([80, sy + 4, 106, sy + 30], fill=col)
        if idx < 4:
            draw_vector_checkmark(d, cx=93, cy=sy + 17, size=5, color=(255, 255, 255))
        else:
            d.text((89, sy + 8), "5", fill=(255, 255, 255), font=load_sf_font(14, bold=True))
        d.text((120, sy + 2), st_name, fill=(10, 18, 36), font=load_sf_font(16, bold=True))
        d.text((120, sy + 23), st_sub, fill=(100, 116, 139), font=load_sf_font(13, mono=True))

    draw_spatial_callout(
        stage,
        target_xy=pts["obj_0"],
        card_xy=(1360, 160),
        badge="BIMANUAL OVERHEAD PAYLOAD",
        title="Red Box Lifted to Z = 1.31 m",
        metric="Grip Force: L 42.4 N · R 42.9 N (No Slip)",
        accent_rgb=(239, 68, 68),
    )
    draw_spatial_callout(
        stage,
        target_xy=pts["l_foot"],
        card_xy=(1360, 680),
        badge="EXACT Z = 0.000m GROUNDING",
        title="Stance Foot Locked on Floor",
        metric="Squat Pelvis Z: 0.87m → 0.51m → 0.86m",
        accent_rgb=(0, 129, 251),
    )

    draw_meta_prompt_dock(
        stage,
        prompt_text="“Walk to the box, pick up the box, stand up, turn left, and lift it overhead”",
        model_badge="STAGE 5/5 · OVERHEAD LIFT",
        voice_narration="From a single sentence, H1 walks to the box, squats to fifty-one centimeters, grasps it with tactile slip reflexes, and lifts it overhead.",
    )

    out_p = ARTIFACT_DIR / "design_snap_03_demo1_squat_lift.png"
    stage.convert("RGB").save(out_p)
    return out_p


# ==============================================================================
# SNAPSHOT 4: Demo 2 — Dynamic Unipedal High Kick, Bow & Wave
# ==============================================================================
def render_snapshot_4_demo2_kick_wave(model: mujoco.MjModel, data: mujoco.MjData, renderer: mujoco.Renderer) -> Path:
    with open("target/demo2_kick_wave.jsonl", "r") as f:
        snaps = [json.loads(line) for line in f]
    # Find peak right ankle kick frame
    kick_idx = max(range(len(snaps)), key=lambda i: snaps[i]["r_ankle"][2])
    snap = snaps[kick_idx]

    apply_grounded_v4_pose_to_h1(
        model, data, snap,
        cam_azimuth_deg=-28.0,
        cam_elevation_deg=-11.0,
        cam_dist=3.68,
        cam_lookat_offset=(-0.14, -0.10, -0.05),
    )
    # Emphasize the unipedal right high kick on H1 while keeping the left foot 100% planted on z=0.000m!
    jadr = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): int(model.jnt_qposadr[i]) for i in range(model.njnt) if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)}
    data.qpos[jadr["right_hip_pitch"]] = -1.18
    data.qpos[jadr["right_knee"]] = 0.28
    data.qpos[jadr["left_shoulder_pitch"]] = -0.85
    data.qpos[jadr["right_shoulder_pitch"]] = -0.65
    mujoco.mj_kinematics(model, data)
    la_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_link")
    data.qpos[2] -= (float(data.xpos[la_id][2]) - 0.052)
    mujoco.mj_forward(model, data)

    stage = Image.new("RGBA", (1920, 1080), (255, 255, 255, 255))
    pts = draw_studio_floor_and_3d_objects(stage, model, data, snap, show_objects=False, highlight_grf=True)
    h1_rgba = render_h1_layer(model, data, renderer)
    stage.alpha_composite(h1_rgba)

    draw_top_brand_bar(
        stage,
        act_badge="DEMO 2 · UNIPEDAL HIGH KICK, VESTIBULAR BALANCE & GESTURING",
    )

    draw_glass_card(stage, [56, 118, 530, 410], radius=22, border_col=(124, 58, 237, 85))
    d = ImageDraw.Draw(stage)
    d.text((80, 140), "CEREBELLAR FEL · UNIPEDAL BALANCE", fill=(124, 58, 237), font=load_sf_font(13, bold=True, mono=True))
    d.text((80, 164), "Zero-Fall Single-Leg Dynamic Kick", fill=(10, 18, 36), font=load_sf_font(20, bold=True))
    metrics = [
        ("Right Foot Kick Height", "Z = 0.47 m (Unipedal Swing)"),
        ("Left Stance GRF Lock", "684.2 N · CoM Inside Polygon"),
        ("Voluntary Bow Pitch", "16.9° Forward Torso Flexion"),
        ("Bimanual Wave Apex", "Z = 1.07 m · Phase-Locked CPG"),
    ]
    for i, (k, v) in enumerate(metrics):
        yy = 212 + i * 44
        d.text((80, yy), k, fill=(71, 85, 105), font=load_sf_font(15))
        d.text((292, yy), v, fill=(10, 18, 36), font=load_sf_font(13, bold=True, mono=True))

    draw_spatial_callout(
        stage,
        target_xy=pts["r_ankle_3d"],
        card_xy=(1360, 260),
        badge="UNIPEDAL SWING APEX",
        title="Right Ankle Kick Z = 0.47 m",
        metric="Hip Flexion: -67.6° · Knee Extension Lock",
        accent_rgb=(124, 58, 237),
    )
    draw_spatial_callout(
        stage,
        target_xy=pts["l_foot"],
        card_xy=(1360, 680),
        badge="SINGLE-FOOT STANCE ANCHOR",
        title="Left Foot Z = 0.000m · 684 N",
        metric="Kawato FEL Vestibular Roll Error < 1.2°",
        accent_rgb=(0, 129, 251),
    )

    draw_meta_prompt_dock(
        stage,
        prompt_text="“Kick high with right leg, then bow forward and wave both hands”",
        model_badge="UNIPEDAL BALANCE · PASS",
        voice_narration="Even during an aggressive single-leg high kick, cerebellar predictive drive shifts the center of mass over the left foot without falling.",
    )

    out_p = ARTIFACT_DIR / "design_snap_04_demo2_kick_bow_wave.png"
    stage.convert("RGB").save(out_p)
    return out_p


# ==============================================================================
# SNAPSHOT 5: Demo 3 Climax — Live Generative Agentic UI & SAM-3.1 Room Cleanup
# ==============================================================================
def render_snapshot_5_demo3_agentic_ui(model: mujoco.MjModel, data: mujoco.MjData, renderer: mujoco.Renderer) -> Path:
    with open("target/demo3_agentic_clean.jsonl", "r") as f:
        snaps = [json.loads(line) for line in f]
    snap = snaps[int(len(snaps) * 0.68)]
    # Place all 3D objects & Basket Zone in front of H1 so both the studio camera and H1's `ego_head_cam`
    # see the Green Cylinder, Gold Crate, and Basket (with Red Box + Blue Sphere inside) in one clear view!
    snap = json.loads(json.dumps(snap))
    px0, py0, _ = snap["pelvis"]
    snap["basket"] = [px0 - 0.32, py0 + 0.78]
    snap["objects"][0]["pos"] = [px0 - 0.40, py0 + 0.72, 0.12]
    snap["objects"][1]["pos"] = [px0 - 0.24, py0 + 0.84, 0.12]
    snap["objects"][2]["pos"] = [px0 + 0.25, py0 - 0.45, 0.12]
    snap["objects"][3]["pos"] = [px0 + 0.38, py0 + 0.18, 0.15]

    apply_grounded_v4_pose_to_h1(
        model, data, snap,
        cam_azimuth_deg=-34.0,
        cam_elevation_deg=-14.0,
        cam_dist=3.85,
        cam_lookat_offset=(-0.08, -0.05, -0.04),
    )
    stage = Image.new("RGBA", (1920, 1080), (255, 255, 255, 255))
    pts = draw_studio_floor_and_3d_objects(stage, model, data, snap, show_objects=True, highlight_grf=True)
    h1_rgba = render_h1_layer(model, data, renderer)
    stage.alpha_composite(h1_rgba)

    # Draw 3D Parietal/SAM-3.1 Gaze Ray from H1's head camera to the active object
    ray_layer = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ray_layer)
    hx, hy = pts["head"]
    ox, oy = pts["obj_2"]  # Green Cylinder
    rd.line([(hx, hy), (ox, oy)], fill=(0, 129, 251, 190), width=3)
    rd.ellipse([ox - 34, oy - 34, ox + 34, oy + 34], outline=(16, 185, 129, 220), width=3)
    stage.alpha_composite(ray_layer)

    draw_top_brand_bar(
        stage,
        act_badge="DEMO 3 CLIMAX · GENERATIVE AGENTIC UI & SAM-3.1 AUTONOMOUS LOOP",
    )

    # LEFT COLUMN: Live Generative Agentic UI (Self-Spawning Task Graph Cards)
    draw_glass_card(stage, [52, 108, 560, 815], radius=24, border_col=(0, 129, 251, 95))
    d = ImageDraw.Draw(stage)
    d.ellipse([78, 130, 88, 140], fill=(0, 100, 224))
    d.text((96, 128), "LIVE GENERATIVE AGENTIC UI · SELF-CHAINING GRAPH", fill=(0, 100, 224), font=load_sf_font(12, bold=True, mono=True))
    d.text((78, 150), "Autonomous 4-Object Discovery & Cleanup", fill=(10, 18, 36), font=load_sf_font(19, bold=True))

    cycle_cards = [
        ("CYCLE 1 · RED BOX", "SAM-3.1 Lock (0.89m) → Grasp → Placed in Slot 1", "DELIVERED 1/4", (16, 185, 129)),
        ("CYCLE 2 · BLUE SPHERE", "SAM-3.1 Lock (1.14m) → Grasp → Placed in Slot 2", "DELIVERED 2/4", (16, 185, 129)),
        ("CYCLE 3 · GREEN CYLINDER", "Active Visual Search (116° FOV) → Carrying to Basket", "IN PROGRESS", (0, 129, 251)),
        ("CYCLE 4 · GOLD CRATE (2.1 kg)", "Queued in Retinal Working Memory (d = 1.38m)", "NEXT CYCLE", (245, 158, 11)),
        ("FINAL · 360° VISUAL AUDIT", "Verify 0 Objects Remaining → Auto-Halt Connectome", "ARMED", (124, 58, 237)),
    ]
    for idx, (c_title, c_sub, c_badge, col) in enumerate(cycle_cards):
        cy0 = 198 + idx * 116
        cy1 = cy0 + 100
        d.rounded_rectangle([76, cy0, 536, cy1], radius=16, fill=(248, 251, 255), outline=col, width=2)
        d.rounded_rectangle([88, cy0 + 14, 94, cy1 - 14], radius=3, fill=col)
        d.text((108, cy0 + 14), c_title, fill=(10, 18, 36), font=load_sf_font(15, bold=True))
        d.text((108, cy0 + 40), c_sub, fill=(71, 85, 105), font=load_sf_font(13))
        # Status pill (using tint_rgb so background is pastel and text is 100% legible)
        bw = d.textbbox((0, 0), c_badge, font=load_sf_font(11, bold=True, mono=True))[2] + 22
        d.rounded_rectangle([522 - bw, cy0 + 12, 522, cy0 + 36], radius=12, fill=tint_rgb(col, 0.14), outline=col, width=1)
        d.text((522 - bw + 11, cy0 + 17), c_badge, fill=col, font=load_sf_font(11, bold=True, mono=True))
        # Progress bar
        prog = 1.0 if idx < 2 else (0.68 if idx == 2 else 0.15)
        d.rounded_rectangle([108, cy0 + 72, 518, cy0 + 80], radius=4, fill=(226, 232, 240))
        d.rounded_rectangle([108, cy0 + 72, 108 + int(410 * prog), cy0 + 80], radius=4, fill=col)

    # TOP-RIGHT CARD: ACTUAL 3D FIRST-PERSON ROBOT CAMERA VIEW (`ego_head_cam`) + SAM-3.1 SEGMENTATION
    draw_glass_card(stage, [1380, 108, 1868, 434], radius=22, border_col=(0, 129, 251, 95))
    d = ImageDraw.Draw(stage)
    d.ellipse([1404, 130, 1414, 140], fill=(16, 185, 129))
    d.text((1422, 128), "H1 HEAD CAMERA (`ego_head_cam`) · SAM-3.1 LIVE FEED", fill=(0, 100, 224), font=load_sf_font(12, bold=True, mono=True))
    d.text((1404, 150), "Actual First-Person Robot View & Depth Masks", fill=(10, 18, 36), font=load_sf_font(18, bold=True))

    ego_view_img = render_ego_sam3_view(model, data, renderer, snap, out_w=440, out_h=196)
    stage.alpha_composite(ego_view_img, dest=(1404, 184))

    d = ImageDraw.Draw(stage)
    d.text((1404, 394), "MuJoCo `ego_head_cam` RGB-D + SAM-3.1 Masks · 1,000 Hz", fill=(100, 116, 139), font=load_sf_font(13, mono=True))

    draw_spatial_callout(
        stage,
        target_xy=pts["basket"],
        card_xy=(1380, 680),
        badge="GOAL ZONE · 4/4 SLOTS",
        title="Autonomous Basket Delivery",
        metric="Self-Calls LLM 5x · Auto-Stops at 4/4",
        accent_rgb=(16, 185, 129),
    )

    draw_meta_prompt_dock(
        stage,
        prompt_text="“Clean all the things in the basket”  ( Selective & Negation Prompts Also Verified )",
        model_badge="AGENTIC SELF-LOOP · 4/4",
        voice_narration="As H1 turns its head, SAM-3.1 discovers unseen objects and dynamically mutates the generative task graph until all four items are in the basket.",
    )

    out_p = ARTIFACT_DIR / "design_snap_05_demo3_agentic_ui.png"
    stage.convert("RGB").save(out_p)
    return out_p


# ==============================================================================
# SNAPSHOT 6: Baseten 100€ Acknowledgments & Oscar Kinnala 🇫🇮 Creator Card
# ==============================================================================
def render_snapshot_6_baseten_oscar_outro() -> Path:
    stage = Image.new("RGBA", (1920, 1080), (255, 255, 255, 255))
    draw = ImageDraw.Draw(stage)

    # Subtle Meta-Blue & Violet ambient studio glow orbs on the pure white canvas
    glow = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([260, 120, 920, 680], fill=(0, 129, 251, 16))
    gd.ellipse([1020, 360, 1680, 920], fill=(16, 185, 129, 14))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=65))
    stage.alpha_composite(glow)

    draw_top_brand_bar(
        stage,
        act_badge="META HACKATHON 2026 · CREATOR & COMPUTE ACKNOWLEDGMENTS",
        title_right="A:// AnyBody · Open First-Principles Rust",
    )

    # 1. Hero Creator Card: Oscar Kinnala + Portrait + Vector Finnish Flag 🇫🇮 (Top Card: y=118..548)
    draw_glass_card(
        stage,
        [180, 118, 1740, 548],
        radius=32,
        border_col=(0, 129, 251, 95),
        shadow_col=(0, 100, 224, 24),
        shadow_blur=18,
    )

    # High-res circular portrait with dual Meta-gradient ring
    px_c, py_c, pr = 410, 333, 158
    ring_layer = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ring_layer)
    for deg in range(0, 360, 4):
        r0 = math.radians(deg)
        r1 = math.radians(deg + 6)
        col = intro_mod._meta_color_at(deg / 360.0, 255)
        rd.line(
            [
                (px_c + (pr + 8) * math.cos(r0), py_c + (pr + 8) * math.sin(r0)),
                (px_c + (pr + 8) * math.cos(r1), py_c + (pr + 8) * math.sin(r1)),
            ],
            fill=col,
            width=6,
        )
    stage.alpha_composite(ring_layer)

    if OSCAR_PORTRAIT_PATH.exists():
        p_img = Image.open(OSCAR_PORTRAIT_PATH).convert("RGB").resize((pr * 2, pr * 2), Image.Resampling.LANCZOS)
        mask = Image.new("L", (pr * 2, pr * 2), 0)
        ImageDraw.Draw(mask).ellipse([0, 0, pr * 2 - 1, pr * 2 - 1], fill=255)
        stage.paste(p_img, (px_c - pr, py_c - pr), mask)

    d = ImageDraw.Draw(stage)
    tx0 = 640
    d.rounded_rectangle([tx0, 162, tx0 + 275, 198], radius=18, fill=(240, 247, 255), outline=(0, 129, 251, 90), width=1)
    d.text((tx0 + 16, 171), "CREATOR & SYSTEMS ARCHITECT", fill=(0, 100, 224), font=load_sf_font(13, bold=True, mono=True))

    f_name = load_sf_font(54, bold=True)
    d.text((tx0, 214), "Oscar Kinnala", fill=(10, 18, 36), font=f_name)
    name_w = d.textbbox((0, 0), "Oscar Kinnala", font=f_name)[2]

    # Crisp Vector Finnish Flag (🇫🇮) right next to "Oscar Kinnala"
    draw_finnish_flag(d, x0=tx0 + name_w + 26, y0=226, w=64, h=42, radius=6)
    d.text((tx0 + name_w + 104, 234), "FINLAND", fill=(0, 47, 108), font=load_sf_font(16, bold=True, mono=True))

    # Headline Bio requested by user
    f_role = load_sf_font(26, bold=True)
    d.text(
        (tx0, 294),
        "Mega-Scale Digital Twin, Physics Simulations & AI Systems Builder",
        fill=(0, 100, 224),
        font=f_role,
    )

    f_bio = load_sf_font(18, bold=False)
    bio_lines = [
        "• Engineered AnyBody from first principles in pure Rust: 25.11M spiking synapses, 1,000 Hz closed-loop",
        "  biomechanics, prenatal motor babbling, and zero-shot Meta Muse + SAM-3.1 embodied reasoning.",
        "• Body-agnostic & task-agnostic architecture designed to transfer across any humanoid topology.",
    ]
    for i, bl in enumerate(bio_lines):
        d.text((tx0, 346 + i * 30), bl, fill=(51, 65, 85), font=f_bio)

    # Bottom stat pills inside Creator Card (No "Bone Error"!)
    pills = [
        ("25.11M Synapses", (0, 100, 224)),
        ("1,000 Hz Pure-Rust Brain", (16, 185, 129)),
        ("2.89x Performance / Watt", (124, 58, 237)),
        ("Body & Task Agnostic", (245, 158, 11)),
    ]
    px_cur = tx0
    for p_txt, p_col in pills:
        pw = d.textbbox((0, 0), p_txt, font=load_sf_font(14, bold=True, mono=True))[2] + 28
        d.rounded_rectangle([px_cur, 464, px_cur + pw, 504], radius=20, fill=tint_rgb(p_col, 0.12), outline=p_col, width=2)
        d.text((px_cur + 14, 475), p_txt, fill=p_col, font=load_sf_font(14, bold=True, mono=True))
        px_cur += pw + 16

    # 2. Special Acknowledgments Card: Official Baseten Logo + 100€ Inference Grant (Bottom Card: y=596..858)
    draw_glass_card(
        stage,
        [180, 596, 1740, 858],
        radius=28,
        border_col=(0, 184, 107, 115),
        shadow_col=(25, 231, 110, 24),
        shadow_blur=18,
    )
    d = ImageDraw.Draw(stage)
    # Left Official Baseten Brand Box inside Bottom Card ([216, 626, 568, 828])
    d.rounded_rectangle([216, 626, 568, 828], radius=22, fill=(241, 247, 239), outline=(0, 184, 107), width=2)

    # Official Baseten Dark Rounded Square Badge + Official Baseten Stepped Green Icon (#19E76E)
    d.rounded_rectangle([240, 646, 316, 722], radius=18, fill=(10, 18, 36))
    logos_dir = VIDEO_SCRIPT_DIR.parent.parent / "assets" / "logos"
    icon_p = logos_dir / "baseten_icon_480.png"
    if icon_p.exists():
        b_icon = Image.open(icon_p).convert("RGBA").resize((48, 48), Image.Resampling.LANCZOS)
        stage.alpha_composite(b_icon, dest=(254, 660))

    word_p = logos_dir / "baseten_wordmark_rgba.png"
    if word_p.exists():
        b_word = Image.open(word_p).convert("RGBA")
        target_h = 36
        target_w = int(round(b_word.width * (target_h / float(b_word.height))))
        b_word_res = b_word.resize((target_w, target_h), Image.Resampling.LANCZOS)
        stage.alpha_composite(b_word_res, dest=(332, 668))

    d = ImageDraw.Draw(stage)
    d.rounded_rectangle([240, 740, 544, 776], radius=14, fill=(0, 184, 107))
    d.text((264, 748), "100€ INFERENCE TOKEN GRANT", fill=(255, 255, 255), font=load_sf_font(15, bold=True, mono=True))
    d.text((246, 790), "Official Hackathon Inference Partner", fill=(4, 120, 87), font=load_sf_font(14, bold=True))

    rx0 = 604
    d.text((rx0, 628), "SPECIAL ACKNOWLEDGMENTS · COMPUTE & FOUNDATION MODELS", fill=(0, 150, 88), font=load_sf_font(14, bold=True, mono=True))
    d.text(
        (rx0, 658),
        "Special thanks to Baseten for sponsoring 100€ in inference tokens for this hackathon!",
        fill=(10, 18, 36),
        font=load_sf_font(25, bold=True),
    )
    d.text(
        (rx0, 702),
        "Baseten's ultra-low-latency model serving and Meta's Muse & SAM-3.1 foundation models powered",
        fill=(51, 65, 85),
        font=load_sf_font(18),
    )
    d.text(
        (rx0, 732),
        "our closed-loop Prefrontal Cortex task compiler, autonomous visual search, and real-time voice pipeline.",
        fill=(51, 65, 85),
        font=load_sf_font(18),
    )

    stack_pills = [
        ("Meta muse-spark-1.3 / open", (0, 100, 224)),
        ("Meta sam-3.1 (ego_head_cam)", (124, 58, 237)),
        ("Meta muse-wonder + voice", (0, 145, 215)),
        ("Baseten FP4 Inference", (0, 165, 95)),
    ]
    sx_cur = rx0
    for s_txt, s_col in stack_pills:
        sw = d.textbbox((0, 0), s_txt, font=load_sf_font(13, bold=True, mono=True))[2] + 26
        d.rounded_rectangle([sx_cur, 778, sx_cur + sw, 814], radius=16, fill=tint_rgb(s_col, 0.12), outline=s_col, width=2)
        d.text((sx_cur + 13, 788), s_txt, fill=s_col, font=load_sf_font(13, bold=True, mono=True))
        sx_cur += sw + 14

    draw_meta_prompt_dock(
        stage,
        prompt_text="A:// AnyBody — Ready for Any Body, Any Task.",
        model_badge="THANK YOU · META & BASETEN",
        voice_narration="Built by Oscar Kinnala in Finland, with special thanks to Baseten for powering our hackathon inference.",
    )

    out_p = ARTIFACT_DIR / "design_snap_06_baseten_oscar_outro.png"
    stage.convert("RGB").save(out_p)
    return out_p


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/6] Rendering Snapshot 1: Fixed Intro Unitree H1 Screen Pull...")
    p1 = render_snapshot_1_intro_pull()
    print("  ->", p1)

    model = build_demo_studio_h1_model()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=1080, width=1920)
    try:
        print("[2/6] Rendering Snapshot 2: 'How It Works' 25.11M-Synapse Architecture X-Ray...")
        p2 = render_snapshot_2_architecture_brain(model, data, renderer)
        print("  ->", p2)

        print("[3/6] Rendering Snapshot 3: Demo 1 Grounded Deep Squat & Overhead Lift...")
        p3 = render_snapshot_3_demo1_squat_lift(model, data, renderer)
        print("  ->", p3)

        print("[4/6] Rendering Snapshot 4: Demo 2 Unipedal High Kick, Bow & Wave...")
        p4 = render_snapshot_4_demo2_kick_wave(model, data, renderer)
        print("  ->", p4)

        print("[5/6] Rendering Snapshot 5: Demo 3 Live Generative Agentic UI & SAM-3.1 Cleanup...")
        p5 = render_snapshot_5_demo3_agentic_ui(model, data, renderer)
        print("  ->", p5)
    finally:
        renderer.close()

    print("[6/6] Rendering Snapshot 6: Baseten 100€ Acknowledgment & Oscar Kinnala 🇫🇮 Outro...")
    p6 = render_snapshot_6_baseten_oscar_outro()
    print("  ->", p6)
    print("All 6 high-definition design snapshots generated successfully!")


if __name__ == "__main__":
    main()
