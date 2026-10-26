"""Generate a High-CTR 1920x1080 YouTube Thumbnail for AnyBody (`A://`).

Follows the viral tech/robotics YouTube thumbnail playbook:
- Massive, mobile-legible typography (3-second rule):
  - Giant `A://` + `AnyBody` brand mark at top-left
  - Ultra-bold stacked headline: "BODY & TASK-" / "AGNOSTIC" / "HUMANOID CONTROL"
- Hero 3D Unitree H1 Humanoid rendered via MuJoCo on the right (clean floor, zero clutter),
  wrapped in a glowing cortico-spinal spiking neuron connectome aura with live synaptic nodes
  anchored to H1's actual head, torso, waist, hands, and legs.
- High-impact Spiking Neuron Action Potential (BAC Burst) oscilloscope card +
  "25.11M Synapses" & "2.89x PERFORMANCE / WATT" proof badges.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import mujoco
from PIL import Image, ImageDraw, ImageFilter

VIDEO_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VIDEO_SCRIPT_DIR))
import generate_design_snapshots as snap_mod  # noqa: E402


def render_hero_h1_cutout(width: int = 1920, height: int = 1080) -> tuple[Image.Image, dict]:
    """Render a large, dramatic 3/4 hero pose of the Unitree H1 humanoid on a transparent RGBA canvas."""
    with open(VIDEO_SCRIPT_DIR.parents[1] / "target" / "demo1_lift.jsonl", "r") as f:
        snaps = [json.loads(line) for line in f]
    snap = snaps[12]

    model = snap_mod.build_demo_studio_h1_model()
    # Hide all mocap object/basket geoms so H1 stands cleanly on the studio floor
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if gname.startswith("env_basket") or gname.startswith("env_obj"):
            model.geom_rgba[gid, 3] = 0.0

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        snap_mod.apply_grounded_v4_pose_to_h1(
            model,
            data,
            snap,
            cam_azimuth_deg=-26.0,
            cam_elevation_deg=-10.5,
            cam_dist=3.35,
            cam_lookat_offset=(-0.44, -0.36, -0.02),
        )
        if model.nmocap > 0:
            data.mocap_pos[:, :] = [50.0, 50.0, -10.0]
        mujoco.mj_forward(model, data)

        tmp_stage = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        pts = snap_mod.draw_studio_floor_and_3d_objects(
            tmp_stage, model, data, snap, show_objects=False, highlight_grf=True
        )
        h1_rgba = snap_mod.render_h1_layer(model, data, renderer)
        tmp_stage.alpha_composite(h1_rgba)
        return tmp_stage, pts
    finally:
        renderer.close()


def draw_gradient_rounded_rect(
    canvas: Image.Image,
    box: list[int],
    radius: int,
    c_start: tuple[int, int, int],
    c_end: tuple[int, int, int],
) -> None:
    """Draw a horizontal gradient rounded rectangle onto `canvas`."""
    x0, y0, x1, y1 = box
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    grad = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grad)
    for ix in range(w):
        t = ix / max(1, w - 1)
        r = int(round(c_start[0] * (1 - t) + c_end[0] * t))
        g = int(round(c_start[1] * (1 - t) + c_end[1] * t))
        b = int(round(c_start[2] * (1 - t) + c_end[2] * t))
        gd.line([(ix, 0), (ix, h)], fill=(r, g, b, 255), width=1)
    mask = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    canvas.paste(grad, (x0, y0), mask)


def generate_thumbnail() -> Path:
    W, H = 1920, 1080
    base = Image.new("RGBA", (W, H), (248, 250, 255, 255))

    # 1. Subtle radial neuro-electric aura behind the right-side robot & neuron network
    aura = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ad = ImageDraw.Draw(aura)
    ad.ellipse([860, -100, 1960, 960], fill=(0, 122, 255, 42))
    ad.ellipse([980, 140, 1860, 1020], fill=(124, 58, 237, 36))
    ad.ellipse([700, 520, 1420, 1060], fill=(16, 185, 129, 28))
    aura = aura.filter(ImageFilter.GaussianBlur(radius=95))
    base.alpha_composite(aura)

    draw = ImageDraw.Draw(base)

    # Subtle architectural perspective grid lines
    for gx in range(0, W, 96):
        draw.line([(gx, 0), (gx, H)], fill=(226, 234, 246), width=1)
    for gy in range(0, H, 96):
        draw.line([(0, gy), (W, gy)], fill=(226, 234, 246), width=1)

    # 2. Render Hero H1 Cutout and composite on right
    h1_rgba, _ = render_hero_h1_cutout(width=W, height=H)

    # Exact visual body-part anchor points on the rendered H1 robot (never crossing the Unitree H1 chest text)
    # Left-side callouts -> Left side of H1; Right-side callouts -> Right side of H1
    synapse_connections = [
        # (node_x, node_y, color, label, anchor_x, anchor_y)
        (1035, 130, (0, 129, 251), "SAM-3.1 VISION", 1255, 122),      # H1 head camera
        (1010, 275, (124, 58, 237), "L5 CORTEX", 1145, 248),          # H1 right shoulder (left in image)
        (1015, 445, (0, 163, 224), "mGluR1 45ms", 1185, 458),         # H1 pelvis / cerebellar balance
        (1025, 635, (16, 185, 129), "SPINAL CPG", 1192, 685),         # H1 right knee/leg (left in image)
        (1715, 165, (0, 129, 251), "MUSE-SPARK-1.3", 1282, 118),      # H1 executive head bus
        (1735, 365, (124, 58, 237), "25.11M SYNAPSES", 1382, 385),    # H1 left forearm (right in image)
        (1725, 595, (16, 185, 129), "88 HILL MUSCLES", 1345, 675),    # H1 left knee/leg (right in image)
    ]

    glow_net = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gnd = ImageDraw.Draw(glow_net)
    for nx, ny, col, _, rx, ry in synapse_connections:
        mx = (nx + rx) // 2
        my = min(ny, ry) - 22
        c_pts = []
        for s in range(32):
            t = s / 31.0
            px = (1 - t) ** 2 * nx + 2 * (1 - t) * t * mx + t**2 * rx
            py = (1 - t) ** 2 * ny + 2 * (1 - t) * t * my + t**2 * ry
            c_pts.append((px, py))
        gnd.line(c_pts, fill=(*col, 105), width=10)
    glow_net = glow_net.filter(ImageFilter.GaussianBlur(radius=8))
    base.alpha_composite(glow_net)

    base.alpha_composite(h1_rgba)
    draw = ImageDraw.Draw(base)

    # Crisp foreground neural axons + pulsing action-potential spikes onto H1's body
    f_node = snap_mod.load_sf_font(16, bold=True, mono=True)
    for nx, ny, col, label, rx, ry in synapse_connections:
        mx = (nx + rx) // 2
        my = min(ny, ry) - 22
        pts = []
        for s in range(32):
            t = s / 31.0
            px = (1 - t) ** 2 * nx + 2 * (1 - t) * t * mx + t**2 * rx
            py = (1 - t) ** 2 * ny + 2 * (1 - t) * t * my + t**2 * ry
            pts.append((px, py))
        draw.line(pts, fill=col, width=4)
        for pulse_t in (0.38, 0.72):
            idx = int(pulse_t * 31)
            px, py = pts[idx]
            draw.ellipse([px - 7, py - 7, px + 7, py + 7], fill=(255, 255, 255), outline=col, width=3)
        draw.ellipse([rx - 12, ry - 12, rx + 12, ry + 12], fill=(255, 255, 255), outline=col, width=4)
        draw.ellipse([rx - 5, ry - 5, rx + 5, ry + 5], fill=col)

        tw = draw.textbbox((0, 0), label, font=f_node)[2]
        pw = tw + 38
        ph = 40
        px0 = nx - pw // 2
        py0 = ny - ph // 2
        draw.rounded_rectangle([px0, py0, px0 + pw, py0 + ph], radius=20, fill=(255, 255, 255), outline=col, width=3)
        draw.ellipse([px0 + 12, ny - 5, px0 + 22, ny + 5], fill=col)
        draw.text((px0 + 28, py0 + 10), label, fill=(15, 23, 42), font=f_node)

    # =========================================================================
    # 3. LEFT SIDE: MASSIVE VIRAL YOUTUBE THUMBNAIL TYPOGRAPHY & LOGO
    # =========================================================================
    x_left = 72

    # 3A. Giant `A://` Hero Mark + `AnyBody` Brand Pill (Top-Left)
    f_logo = snap_mod.load_sf_font(92, bold=True, mono=True)
    w_a = draw.textbbox((0, 0), "A:", font=f_logo)[2]
    w_s = draw.textbbox((0, 0), "//", font=f_logo)[2]
    y_logo = 54
    draw.text((x_left, y_logo), "A:", fill=(10, 15, 28), font=f_logo)
    draw.text((x_left + w_a, y_logo), "//", fill=(0, 122, 255), font=f_logo)
    cur_x = x_left + w_a + w_s + 12
    draw.rounded_rectangle([cur_x, y_logo + 14, cur_x + 28, y_logo + 90], radius=6, fill=(10, 15, 28))

    # Big `AnyBody` pill badge next to `A://`
    f_brand = snap_mod.load_sf_font(48, bold=True, mono=True)
    bw = draw.textbbox((0, 0), "AnyBody", font=f_brand)[2]
    bx0 = cur_x + 52
    by0 = y_logo + 10
    bpw = bw + 64
    bph = 82
    draw.rounded_rectangle([bx0, by0, bx0 + bpw, by0 + bph], radius=24, fill=(238, 245, 255), outline=(0, 122, 255), width=4)
    draw.text((bx0 + 32, by0 + 14), "AnyBody", fill=(0, 92, 214), font=f_brand)

    # 3B. Ultra-Bold Stacked Headline ("BODY & TASK-" / "AGNOSTIC" / "HUMANOID CONTROL")
    f_h1 = snap_mod.load_sf_font(74, bold=True)
    f_h2 = snap_mod.load_sf_font(80, bold=True)
    f_h3 = snap_mod.load_sf_font(68, bold=True)

    # Line 1: BODY & TASK-
    draw.text((x_left, 186), "BODY & TASK-", fill=(10, 15, 28), font=f_h1)
    # Line 2: AGNOSTIC inside a vivid Meta-Blue -> Violet gradient banner!
    grad_box = [x_left - 4, 278, x_left + 516, 384]
    draw_gradient_rounded_rect(base, grad_box, radius=24, c_start=(0, 114, 255), c_end=(124, 58, 237))
    draw = ImageDraw.Draw(base)
    draw.text((x_left + 24, 285), "AGNOSTIC", fill=(255, 255, 255), font=f_h2)

    # Line 3: HUMANOID CONTROL
    draw.text((x_left, 404), "HUMANOID CONTROL", fill=(10, 15, 28), font=f_h3)

    # =========================================================================
    # 4. BOTTOM-LEFT: HIGH-IMPACT SPIKING NEURON OSCILLOSCOPE + WATTAGE CARDS
    # =========================================================================
    c1_x0, c1_y0, c1_x1, c1_y1 = x_left, 518, x_left + 850, 804
    sh1 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(sh1).rounded_rectangle([c1_x0, c1_y0 + 10, c1_x1, c1_y1 + 14], radius=28, fill=(15, 23, 42, 32))
    sh1 = sh1.filter(ImageFilter.GaussianBlur(radius=16))
    base.alpha_composite(sh1)
    draw = ImageDraw.Draw(base)

    draw.rounded_rectangle([c1_x0, c1_y0, c1_x1, c1_y1], radius=28, fill=(12, 18, 34), outline=(0, 129, 251), width=4)

    # Header inside dark oscilloscope card
    f_card_tag = snap_mod.load_sf_font(20, bold=True, mono=True)
    f_card_big = snap_mod.load_sf_font(42, bold=True)
    draw.ellipse([c1_x0 + 28, c1_y0 + 26, c1_x0 + 46, c1_y0 + 44], fill=(16, 185, 129))
    draw.text((c1_x0 + 58, c1_y0 + 23), "1,000 Hz PURE-RUST SPIKING BRAIN", fill=(96, 165, 250), font=f_card_tag)
    draw.text((c1_x0 + 28, c1_y0 + 60), "25.11M Synapses", fill=(255, 255, 255), font=f_card_big)

    # Sub-badges on left of card with generous horizontal padding
    f_pill = snap_mod.load_sf_font(19, bold=True, mono=True)
    p1_text = "Meta Muse + SAM-3.1 Vision"
    p2_text = "Zero PID · 88 Hill Muscles"
    pw_box = max(draw.textbbox((0, 0), p1_text, font=f_pill)[2], draw.textbbox((0, 0), p2_text, font=f_pill)[2]) + 38

    draw.rounded_rectangle([c1_x0 + 28, c1_y0 + 132, c1_x0 + 28 + pw_box, c1_y0 + 178], radius=14, fill=(22, 34, 62), outline=(59, 130, 246), width=2)
    draw.text((c1_x0 + 46, c1_y0 + 144), p1_text, fill=(191, 219, 254), font=f_pill)

    draw.rounded_rectangle([c1_x0 + 28, c1_y0 + 194, c1_x0 + 28 + pw_box, c1_y0 + 240], radius=14, fill=(22, 48, 44), outline=(16, 185, 129), width=2)
    draw.text((c1_x0 + 46, c1_y0 + 206), p2_text, fill=(167, 243, 208), font=f_pill)

    # Right half of dark card: High-contrast Spiking Neuron BAC Burst Waveform
    wx0, wy0, wx1, wy1 = c1_x0 + 412, c1_y0 + 56, c1_x1 - 26, c1_y1 - 26
    draw.rounded_rectangle([wx0, wy0, wx1, wy1], radius=18, fill=(7, 11, 22), outline=(30, 41, 59), width=2)
    for gy in range(wy0 + 44, wy1, 36):
        draw.line([(wx0 + 10, gy), (wx1 - 10, gy)], fill=(20, 30, 52), width=1)

    ca_pts = []
    soma_pts = []
    n_samples = 180
    for i in range(n_samples):
        u = i / (n_samples - 1)
        px = wx0 + 16 + u * (wx1 - wx0 - 32)
        ca_env = math.exp(-((u - 0.54) ** 2) / 0.045)
        py_ca = (wy1 - 28) - ca_env * 78.0
        ca_pts.append((px, py_ca))

        spike = (
            0.70 * math.exp(-((u - 0.18) ** 2) / 0.00042)
            + 0.96 * math.exp(-((u - 0.44) ** 2) / 0.00036)
            + 0.92 * math.exp(-((u - 0.53) ** 2) / 0.00036)
            + 0.85 * math.exp(-((u - 0.62) ** 2) / 0.00036)
        )
        base_v = 0.18 * ca_env
        py_s = (wy1 - 26) - min(1.0, base_v + spike) * 116.0
        soma_pts.append((px, py_s))

    draw.line(ca_pts, fill=(168, 85, 247), width=4)
    draw.line(soma_pts, fill=(56, 189, 248), width=4)
    f_osc = snap_mod.load_sf_font(15, bold=True, mono=True)
    draw.text((wx0 + 16, wy0 + 12), "L5 NEURON BAC BURST (mV)", fill=(167, 243, 208), font=f_osc)

    # 5. Bottom High-Contrast Efficiency Banner ("2.89x PERFORMANCE / WATT · 422W vs 1,221W STIFF PD")
    b_y0 = 832
    b_y1 = 958
    draw_gradient_rounded_rect(base, [x_left, b_y0, x_left + 850, b_y1], radius=26, c_start=(5, 150, 105), c_end=(16, 185, 129))
    draw = ImageDraw.Draw(base)
    f_eff_big = snap_mod.load_sf_font(44, bold=True)
    f_eff_sub = snap_mod.load_sf_font(22, bold=True, mono=True)
    draw.text((x_left + 30, b_y0 + 16), "2.89x PERFORMANCE / WATT", fill=(255, 255, 255), font=f_eff_big)
    draw.text((x_left + 32, b_y0 + 74), "422W Neuromorphic Muscles  vs.  1,221W Stiff PD", fill=(209, 250, 229), font=f_eff_sub)

    out_repo = VIDEO_SCRIPT_DIR.parents[1] / "assets" / "figures" / "anybody_youtube_thumbnail.png"
    out_desk = Path.home() / "Desktop" / "AnyBody_YouTube_Thumbnail.png"
    out_desk_jpg = Path.home() / "Desktop" / "AnyBody_YouTube_Thumbnail.jpg"

    rgb_final = base.convert("RGB")
    rgb_final.save(out_repo, format="PNG", optimize=True)
    rgb_final.save(out_desk, format="PNG", optimize=True)
    rgb_final.save(out_desk_jpg, format="JPEG", quality=96, subsampling=0)
    print(f"Saved YouTube thumbnail to:\n  - {out_repo}\n  - {out_desk}\n  - {out_desk_jpg}")
    return out_repo


if __name__ == "__main__":
    generate_thumbnail()
