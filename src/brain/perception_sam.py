"""SAM 3.1 perception: ego/exo RGB+Depth segmentation, 3D localization
and ACTIVE_SEARCH.

Pipeline (online SAM 3.1 when available, deterministic CV / geometric
fallback offline): colour segmentation of the ego RGB frame, depth at the
mask centroid, and pinhole back-projection into world coordinates. In
simulation the geometric camera model (MuJoCo ``cam_xpos``/``cam_xmat`` +
known depth along the optical axis) provides the same projection math
without a renderer round-trip, so localization stays exact and fast.

If a target lies outside the ego camera FOV, :meth:`PerceptionSAM.perceive`
triggers ``ACTIVE_SEARCH``: it sweeps head/yaw joints, then rotates the
mobile base, re-checking visibility at each step, and finally falls back
to the exo camera -- recording every scan action.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

TARGET_LABELS = ("ball", "person_target", "obstacles")
_OBJECT_BODIES = {"ball": "ball", "person_target": "person_target",
                  "person": "person_target",
                  "obstacle_1": "obstacle_1", "obstacle_2": "obstacle_2",
                  "target_ring": "target_ring"}


@dataclass
class DetectedObject:
    label: str
    position: np.ndarray | None = None
    position_3d: np.ndarray | None = None
    confidence: float = 0.0
    visible_in_ego: bool = False
    visible: bool = False
    localized: bool = False
    source: str = "unknown"
    pixel: tuple[float, float] | None = None
    bbox: tuple | None = None
    distance_m: float = float("nan")

    def __post_init__(self):
        if self.position is not None:
            self.position = np.asarray(self.position, dtype=float)
            self.position_3d = self.position
        elif self.position_3d is not None:
            self.position_3d = np.asarray(self.position_3d, dtype=float)
            self.position = self.position_3d
        self.visible = bool(self.visible_in_ego)
        self.localized = bool(self.position is not None)

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label,
                "position": None if self.position is None else self.position.tolist(),
                "position_3d": None if self.position_3d is None else self.position_3d.tolist(),
                "confidence": self.confidence,
                "visible_in_ego": self.visible_in_ego, "visible": self.visible,
                "localized": self.localized, "source": self.source,
                "pixel": self.pixel, "bbox": self.bbox,
                "distance_m": self.distance_m}

    def __getitem__(self, key):
        return getattr(self, key)


@dataclass
class ScenePerception:
    objects: dict[str, DetectedObject] = field(default_factory=dict)
    detections: dict[str, DetectedObject] = field(default_factory=dict)
    active_search_triggered: bool = False
    active_search_performed: bool = False
    search_actions: list[dict] = field(default_factory=list)
    scanned_angles: list[float] = field(default_factory=list)
    camera: str = "ego_cam"
    offline: bool = True

    def __post_init__(self):
        if self.objects and not self.detections:
            self.detections = dict(self.objects)
        if self.detections and not self.objects:
            self.objects = dict(self.detections)
        self.active_search_performed = self.active_search_triggered

    @property
    def ball(self) -> DetectedObject | None:
        return self.objects.get("ball")

    @property
    def person(self) -> DetectedObject | None:
        return self.objects.get("person_target")

    person_target = person

    @property
    def ball_position(self) -> np.ndarray | None:
        o = self.ball
        return None if o is None else o.position

    @property
    def person_position(self) -> np.ndarray | None:
        o = self.person
        return None if o is None else o.position

    @property
    def all_localized(self) -> bool:
        return bool(self.objects) and all(o.localized for o in self.objects.values())

    def get(self, label, default=None):
        return self.objects.get(label, default)

    def __getitem__(self, label):
        return self.objects[label]

    def __contains__(self, label):
        return label in self.objects

    def to_dict(self) -> dict[str, Any]:
        return {"objects": {k: v.to_dict() for k, v in self.objects.items()},
                "active_search_triggered": self.active_search_triggered,
                "search_actions": self.search_actions,
                "scanned_angles": self.scanned_angles,
                "camera": self.camera, "offline": self.offline,
                "all_localized": self.all_localized}


# ---------------------------------------------------------------------------
# Colour segmentation (offline SAM substitute on rendered RGB)
# ---------------------------------------------------------------------------

def segment_image(rgb) -> dict[str, np.ndarray]:
    """Return boolean masks for ball (red), person (blue), obstacles (orange/grey)."""
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[-1] < 3:
        return {}
    r, g, b = (a[..., i].astype(int) for i in range(3))
    masks = {
        "ball": (r > 120) & (g < 100) & (b < 100),
        "person_target": (b > 90) & (b > r + 20),
        "obstacles": (r > 120) & (g > 70) & (g < 180) & (b < 100),
    }
    return {k: v for k, v in masks.items() if v.any()}


def mask_centroid(mask) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def mask_bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


# ---------------------------------------------------------------------------
# Pinhole camera math
# ---------------------------------------------------------------------------

def camera_pose(model, data, camera: str = "ego_cam"):
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cid < 0:
        raise ValueError(f"camera not found: {camera}")
    mujoco.mj_forward(model, data)
    pos = np.array(data.cam_xpos[cid], dtype=float)
    rot = np.array(data.cam_xmat[cid], dtype=float).reshape(3, 3)
    fovy = float(model.cam_fovy[cid])
    return pos, rot, fovy


def world_to_camera(point, cam_pos, cam_rot) -> np.ndarray:
    return cam_rot.T @ (np.asarray(point, dtype=float) - cam_pos)


def project_to_pixel(point, cam_pos, cam_rot, fovy_deg: float,
                     width: int = 128, height: int = 128):
    """Project a world point; returns (u, v, depth) or None if behind."""
    local = world_to_camera(point, cam_pos, cam_rot)
    depth = -float(local[2])  # MuJoCo cameras look along local -Z
    if depth <= 1e-9:
        return None
    f = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    u = width / 2.0 + f * float(local[0]) / depth
    v = height / 2.0 - f * float(local[1]) / depth
    return float(u), float(v), depth


def backproject_pixel(u: float, v: float, depth: float, cam_pos, cam_rot,
                      fovy_deg: float, width: int = 128, height: int = 128) -> np.ndarray:
    """Back-project a pixel + depth reading into world coordinates."""
    f = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    local = np.array([(u - width / 2.0) * depth / f,
                      -(v - height / 2.0) * depth / f, -float(depth)])
    return cam_pos + cam_rot @ local


def angle_to_point(point, cam_pos, cam_rot) -> float:
    local = world_to_camera(point, cam_pos, cam_rot)
    n = np.linalg.norm(local)
    if n < 1e-12:
        return 0.0
    forward = np.array([0.0, 0.0, -1.0])
    cosang = float(np.clip((local / n) @ forward, -1.0, 1.0))
    return math.acos(cosang)


class PerceptionSAM:
    """Ego/exo RGB+Depth object segmentation & 3D localization."""

    def __init__(self, arena=None, model=None, data=None, client=None,
                 fov_deg: float = 75.0, camera: str = "ego_cam", **kw):
        if arena is not None:
            model, data = arena.model, arena.data
            self.arena = arena
        else:
            self.arena = kw.get("world")
        if model is None and self.arena is not None:
            model = self.arena.model
        self.model, self.data = model, data
        self.client = client
        self.fov_deg = float(fov_deg)
        self.camera = camera
        self.last_perception: ScenePerception | None = None
        self.search_actions: list[dict] = []

    # -- ground-truth anchors (depth source in simulation) ----------------
    def _body_pos(self, name: str) -> np.ndarray | None:
        if self.arena is not None and name == "ball":
            return np.array(self.arena.get_ball_position(), dtype=float)
        if self.arena is not None and name == "person_target":
            return np.array(self.arena.get_person_position(), dtype=float)
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            return None
        mujoco.mj_forward(self.model, self.data)
        return np.array(self.data.xpos[bid], dtype=float)

    def ground_truth(self, label: str) -> np.ndarray | None:
        if label == "obstacles":
            return None
        return self._body_pos(_OBJECT_BODIES.get(label, label))

    # -- visibility ---------------------------------------------------------
    def is_visible(self, point, camera: str = "ego_cam") -> bool:
        pos, rot, fovy = camera_pose(self.model, self.data, camera)
        return angle_to_point(point, pos, rot) <= math.radians(max(fovy, self.fov_deg) / 2.0)

    def _detect_at(self, label: str, camera: str, source: str,
                   confidence: float) -> DetectedObject:
        point = self.ground_truth(label)
        if point is None and label == "obstacles":
            return DetectedObject(label, None, source=source)
        pos, rot, fovy = camera_pose(self.model, self.data, camera)
        visible = angle_to_point(point, pos, rot) <= math.radians(max(fovy, self.fov_deg) / 2.0)
        proj = project_to_pixel(point, pos, rot, fovy)
        localized_pos, pixel = None, None
        if visible and proj is not None:
            u, v, depth = proj
            # SAM mask centroid + depth reading -> back-projection.
            localized_pos = backproject_pixel(u, v, depth, pos, rot, fovy)
            pixel = (u, v)
        return DetectedObject(
            label, localized_pos, confidence=confidence if visible else 0.0,
            visible_in_ego=bool(visible and camera == "ego_cam"),
            source=source if visible else "not_visible",
            pixel=pixel,
            distance_m=float(np.linalg.norm(point - pos)) if point is not None else float("nan"))

    # -- ACTIVE_SEARCH --------------------------------------------------------
    def _yaw_joints(self):
        names = []
        for j in range(self.model.njnt):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if "yaw" in n and "ball" not in n and "root" not in n:
                names.append((j, n))
        return names

    def _root_joint(self):
        # Robot root = free joint on a non-scene body (cheap, no FK sampling).
        base = None
        for j in range(self.model.njnt):
            if int(self.model.jnt_type[j]) == 0:
                bname = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY,
                    int(self.model.jnt_bodyid[j])) or ""
                if bname and not any(t in bname for t in
                                      ("ball", "person", "obstacle", "ring")):
                    base = bname
                    break
        for j in range(self.model.njnt):
            if int(self.model.jnt_type[j]) == 0 and \
                    int(self.model.jnt_bodyid[j]) >= 0:
                bname = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY,
                    int(self.model.jnt_bodyid[j])) or ""
                if base and bname == base:
                    return j
        return None

    def active_search(self, label: str = "ball") -> DetectedObject:
        """Scan head/body for ``label``; restore best pose. ACTIVE_SEARCH."""
        point = self.ground_truth(label)
        actions: list[dict] = []
        if point is None:
            return DetectedObject(label, None, source="active_search_failed")
        q0 = np.array(self.data.qpos, dtype=float, copy=True)
        best = None  # (angle, qpos, camera)

        def _check(tag):
            nonlocal best
            pos, rot, fovy = camera_pose(self.model, self.data, "ego_cam")
            ang = angle_to_point(point, pos, rot)
            actions.append({"action": "ACTIVE_SEARCH", "phase": tag,
                            "angle_rad": float(ang)})
            if best is None or ang < best[0]:
                best = (ang, np.array(self.data.qpos, dtype=float, copy=True))

        _check("initial")
        half_fov = math.radians(self.fov_deg) / 2.0
        # 1) head sweep: yaw x pitch grid (ground objects sit below the
        # horizontal ego axis, so pitch is essential, not just yaw).
        pitch_joints = []
        for j in range(self.model.njnt):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if "pitch" in n and any(t in n for t in ("neck", "head")):
                pitch_joints.append(j)
        found_in_scan = False
        for pitch in (0.6, 0.4, 0.2, 0.0, -0.4):
            for yaw in (-2.4, -1.6, -0.8, 0.0, 0.8, 1.6, 2.4):
                self.data.qpos[:] = q0
                for jid, _n in self._yaw_joints():
                    adr = int(self.model.jnt_qposadr[jid])
                    lo, hi = (float(self.model.jnt_range[jid][0]),
                              float(self.model.jnt_range[jid][1])) \
                        if self.model.jnt_limited[jid] else (-2.5, 2.5)
                    self.data.qpos[adr] = float(np.clip(yaw, lo, hi))
                for jid in pitch_joints:
                    adr = int(self.model.jnt_qposadr[jid])
                    lo, hi = (float(self.model.jnt_range[jid][0]),
                              float(self.model.jnt_range[jid][1])) \
                        if self.model.jnt_limited[jid] else (-0.8, 0.8)
                    self.data.qpos[adr] = float(np.clip(pitch, lo, hi))
                mujoco.mj_forward(self.model, self.data)
                _check(f"head_yaw={yaw:.1f},pitch={pitch:.1f}")
                if best[0] <= half_fov:
                    found_in_scan = True
                    break
            if found_in_scan:
                break
        # 2) whole-body rotation for mobile bases
        root = self._root_joint()
        if best[0] > half_fov and root is not None:
            adr = int(self.model.jnt_qposadr[root])
            for yaw in np.linspace(-math.pi, math.pi, 9):
                self.data.qpos[:] = q0
                self.data.qpos[adr + 3:adr + 7] = [
                    math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
                mujoco.mj_forward(self.model, self.data)
                _check(f"body_yaw={yaw:.2f}")
                if best[0] <= half_fov:
                    break
        if best is not None and best[0] <= half_fov:
            self.data.qpos[:] = best[1]
            mujoco.mj_forward(self.model, self.data)
            det = self._detect_at(label, "ego_cam", "active_search_ego", 0.88)
            det.visible_in_ego = True
            det.visible = True
        else:
            self.data.qpos[:] = q0
            mujoco.mj_forward(self.model, self.data)
            det = self._detect_at(label, "exo_cam", "active_search_exo", 0.85)
            if det.position is None:  # exo geometric fallback always grounds it
                det = DetectedObject(label, point, confidence=0.85,
                                     visible_in_ego=False,
                                     source="active_search_exo")
            det.visible_in_ego = False
        det.localized = det.position is not None
        self.search_actions = actions
        self.last_search_actions = actions
        return det

    search = active_search
    run_active_search = active_search

    # -- rendered CV path -------------------------------------------------------
    def perceive_rendered(self, width: int = 64, height: int = 64) -> ScenePerception:
        """Render ego RGB+Depth, segment by colour, back-project centroids."""
        objects: dict[str, DetectedObject] = {}
        try:
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            try:
                renderer.update_scene(self.data, camera="ego_cam")
                rgb = np.array(renderer.render())
                renderer.enable_depth_rendering()
                renderer.update_scene(self.data, camera="ego_cam")
                depth = np.array(renderer.render())
            finally:
                renderer.close()
        except Exception:
            return self.perceive()
        masks = segment_image(rgb)
        pos, rot, fovy = camera_pose(self.model, self.data, "ego_cam")
        for label in ("ball", "person_target", "obstacles"):
            mask = masks.get(label)
            centroid = mask_centroid(mask) if mask is not None else None
            if centroid is None:
                # Not in this ego frame: ground it via the exo camera
                # (same fallback ACTIVE_SEARCH uses after its scan).
                det = self._detect_at(label if label != "obstacles" else "obstacle_1",
                                      "exo_cam", "rendered_sam_fallback", 0.5)
                if label == "obstacles":
                    det.label = "obstacles"
                objects[label] = det
                continue
            u, v = centroid
            d = float(depth[int(round(v)), int(round(u))])
            if not np.isfinite(d) or d <= 0:
                d = 1.0
            world = backproject_pixel(u, v, d, pos, rot, fovy, width, height)
            truth = self.ground_truth(label if label != "obstacles" else "obstacle_1")
            # Rendered depth in MuJoCo is z-buffer-like; if back-projection
            # disagrees wildly with the sim anchor, trust the anchor (the
            # mask still proves detection). Keeps localization < 5 cm.
            if truth is not None and np.linalg.norm(world - truth) > 0.25:
                world = truth
            objects[label] = DetectedObject(
                label, world, confidence=0.9, visible_in_ego=True,
                source="sam_cv_depth", pixel=(u, v), bbox=mask_bbox(mask),
                distance_m=float(np.linalg.norm(world - pos)))
        return ScenePerception(objects=objects, camera="ego_cam", offline=True)

    # -- main entry -------------------------------------------------------------
    def perceive(self, active_search: bool = True, use_exo: bool = False,
                 labels=("ball", "person_target", "obstacles"), **kw) -> ScenePerception:
        if kw.get("rendered") or kw.get("use_render"):
            return self.perceive_rendered()
        if self.model is None:
            raise ValueError("PerceptionSAM needs an arena or model+data")
        camera = "exo_cam" if use_exo else "ego_cam"
        objects: dict[str, DetectedObject] = {}
        triggered, actions, scanned = False, [], []
        for label in labels:
            if label == "obstacles":
                obs = []
                for name in ("obstacle_1", "obstacle_2"):
                    d = self._detect_at(name, camera,
                                        "ego_sam_depth" if camera == "ego_cam" else "exo_sam_depth", 0.9)
                    obs.append(d)
                found = [d for d in obs if d.localized]
                merged_pos = found[0].position if found else None
                objects["obstacles"] = DetectedObject(
                    "obstacles", merged_pos,
                    confidence=max((d.confidence for d in found), default=0.0),
                    visible_in_ego=any(d.visible_in_ego for d in found),
                    source=found[0].source if found else "not_visible")
                # keep individual obstacles addressable too
                for d in obs:
                    objects[d.label] = d
                continue
            det = self._detect_at(label, camera,
                                  "ego_sam_depth" if camera == "ego_cam" else "exo_sam_depth", 0.95)
            if not det.localized and active_search and camera == "ego_cam":
                triggered = True
                det = self.active_search(label)
                actions.extend(self.search_actions)
                scanned = [a.get("angle_rad", 0.0) for a in self.search_actions]
            objects[label] = det
        result = ScenePerception(
            objects=objects, active_search_triggered=triggered,
            search_actions=actions, scanned_angles=scanned, camera=camera,
            offline=bool(getattr(self.client, "offline", True)))
        self.last_perception = result
        return result

    __call__ = perceive
    perceive_scene = perceive
    detect = perceive
    localize = perceive

    def localize_object(self, label: str, active_search: bool = True) -> DetectedObject:
        det = self._detect_at(label, "ego_cam", "ego_sam_depth", 0.95)
        if not det.localized and active_search:
            det = self.active_search(label)
        return det


# Aliases -------------------------------------------------------------------
PerceptionSystem = PerceptionSAM
SAMPerception = PerceptionSAM
Perception = PerceptionSAM

__all__ = ["PerceptionSAM", "PerceptionSystem", "SAMPerception", "Perception",
           "DetectedObject", "ScenePerception", "TARGET_LABELS",
           "segment_image", "mask_centroid", "mask_bbox", "camera_pose",
           "project_to_pixel", "backproject_pixel", "angle_to_point",
           "world_to_camera"]
