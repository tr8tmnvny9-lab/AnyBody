"""Real 3D Menagerie arena + Zero-Cheat physics controller (AnyBody v3).

Loads real mujoco_menagerie scene.xml robots, injects interactive target
objects + studio lighting/cameras into the MJCF, and drives every robot
solely through ``mujoco.mj_step`` fed by the Rust 3-JEV triad ``ctrl``.
Any teleport-style API raises :class:`ZeroCheatViolation`.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

MENAGERIE_ROOT = Path.home() / ".cache" / "robot_descriptions" / "mujoco_menagerie"
ROBOTS: dict[str, str] = {
    "unitree_g1": "unitree_g1/scene.xml",
    "unitree_h1": "unitree_h1/scene.xml",
    "unitree_go2": "unitree_go2/scene.xml",
    "franka_panda": "franka_emika_panda/scene.xml",
    "franka_emika_panda": "franka_emika_panda/scene.xml",
    "boston_dynamics_spot": "boston_dynamics_spot/scene.xml",
    "spot": "boston_dynamics_spot/scene.xml",
    "agility_cassie": "agility_cassie/scene.xml",
    "cassie": "agility_cassie/scene.xml",
}
TARGET_BODIES = ("red_target", "gold_cube_1", "gold_cube_2", "bin_zone")


class ZeroCheatViolation(RuntimeError):
    pass


def _inject_scene(scene_path: Path) -> Path:
    """Write an injected copy of scene.xml next to the original (asset-safe)."""
    tree = ET.parse(scene_path)
    root = tree.getroot()
    wb = root.find("worldbody")
    if wb is None:
        wb = ET.SubElement(root, "worldbody")
    vis_el = root.find("visual")
    if vis_el is None:
        vis_el = ET.SubElement(root, "visual")
    glob_el = vis_el.find("global")
    if glob_el is None:
        glob_el = ET.SubElement(vis_el, "global")
    glob_el.set("offwidth", "1920")
    glob_el.set("offheight", "1080")
    # studio lights + cameras (idempotent by injected filename)
    for name, pos in (("studio_key", "2 -2 3"), ("studio_fill", "-2 2 2")):
        el = ET.SubElement(wb, "light", {"name": name, "pos": pos,
                                         "diffuse": "0.9 0.9 0.9", "castshadow": "true"})
    ET.SubElement(wb, "camera", {"name": "studio_cam", "pos": "2.5 -2.4 1.65",
                                 "xyaxes": "0.69 0.72 0 -0.32 0.30 0.90", "fovy": "46"})
    ET.SubElement(wb, "camera", {"name": "exo_cam", "pos": "2.5 0 1.8",
                                 "xyaxes": "0 1 0 -0.5 0 1"})
    # True robot-mounted ego_cam: attach inside a robot body (via include file if needed)
    folder = scene_path.parent.name.lower()
    if "panda" in folder:
        # Wide-angle eye-in-hand wrist camera looking straight down between the fingers (+Z_hand)
        _ego_specs = {"pos": "-0.035 0.0 0.015", "xyaxes": "0 1 0 0.98 0 0.20", "fovy": "118"}
        _priority = ("hand", "link7")
    elif "go2" in folder or "spot" in folder:
        # Wide-angle front-nose camera at the tip of the snout looking forward-down at the floor
        _ego_specs = {"pos": "0.28 0.0 0.03", "xyaxes": "0 -1 0 0.38 0 0.925", "fovy": "118"}
        _priority = ("base", "base_link", "body", "trunk")
    else:
        # Humanoids (G1, H1, Cassie): wide-angle head/upper-torso camera looking at hands & floor
        _ego_specs = {"pos": "0.11 0.0 0.24", "xyaxes": "0 -1 0 0.44 0 0.90", "fovy": "115"}
        _priority = ("torso_link", "pelvis", "base_link", "base", "trunk")
    def _find_ego_body(wb_el):
        bodies = list(wb_el.iter("body"))
        if not bodies:
            return None
        by_name = {b.get("name"): b for b in bodies if b.get("name")}
        for p in _priority:
            if p in by_name:
                return by_name[p]
        for b in bodies:
            nm = (b.get("name") or "").lower()
            if "hand" in nm or "torso" in nm or "pelvis" in nm:
                return b
        return bodies[0]
    _ego_done = False
    _scene_body = _find_ego_body(wb)
    # scene worldbody usually only has floor; robot bodies live in the included file
    _robot_bodies_in_scene = [b for b in wb.iter("body")
                              if (b.get("name") or "") not in TARGET_BODIES]
    if _robot_bodies_in_scene and _scene_body is not None:
        ET.SubElement(_scene_body, "camera", {"name": "ego_cam", **_ego_specs})
        _ego_done = True
    if not _ego_done:
        for inc in root.findall("include"):
            fname = inc.get("file")
            if not fname:
                continue
            inc_path = scene_path.parent / fname
            if not inc_path.exists():
                continue
            try:
                inc_tree = ET.parse(inc_path)
            except Exception:
                continue
            inc_wb = inc_tree.getroot().find("worldbody")
            if inc_wb is None:
                continue
            body = _find_ego_body(inc_wb)
            if body is None:
                continue
            if not any(c.get("name") == "ego_cam" for c in body.iter("camera")):
                ET.SubElement(body, "camera", {"name": "ego_cam", **_ego_specs})
            out_inc = scene_path.parent / f"_anybody_v3_robot_{inc_path.stem}.xml"
            inc_tree.write(out_inc, encoding="unicode", xml_declaration=True)
            inc.set("file", out_inc.name)
            _ego_done = True
            break
    if not _ego_done:
        # Fallback: first body under worldbody, else worldbody itself (should not happen for menagerie)
        ET.SubElement(wb, "camera", {"name": "ego_cam", **_ego_specs})
    specs = [
        ("red_target", "0.5 0.3 0.5", "sphere", "0.035", "1 0.1 0.1 1"),
        ("gold_cube_1", "0.6 -0.2 0.06", "box", "0.04 0.04 0.04", "1 0.8 0.1 1"),
        ("gold_cube_2", "-0.4 0.4 0.06", "box", "0.04 0.04 0.04", "1 0.8 0.1 1"),
        ("bin_zone", "0.8 0.6 0.03", "box", "0.15 0.15 0.02", "0.1 0.8 0.3 0.6"),
    ]
    for name, pos, typ, size, rgba in specs:
        body = ET.SubElement(wb, "body", {"name": name, "pos": pos, "mocap": "true"})
        ET.SubElement(body, "geom", {"name": f"{name}_geom", "type": typ, "size": size,
                                     "rgba": rgba, "contype": "0", "conaffinity": "0"})
        ET.SubElement(body, "site", {"name": f"{name}_site", "pos": "0 0 0", "size": "0.01"})
    out = scene_path.parent / "_anybody_v3_scene.xml"
    tree.write(out, encoding="unicode", xml_declaration=True)
    return out


def load_menagerie_model(robot: str):
    key = str(robot).strip().lower()
    if key not in ROBOTS:
        raise ValueError(f"unknown menagerie robot {robot!r}; available: {sorted(set(ROBOTS))}")
    scene = MENAGERIE_ROOT / ROBOTS[key]
    if not scene.exists():
        raise FileNotFoundError(f"menagerie scene not found: {scene}")
    injected = _inject_scene(scene)
    try:
        return mujoco.MjModel.from_xml_path(str(injected))
    finally:
        injected.unlink(missing_ok=True)
        for _p in scene.parent.glob("_anybody_v3_robot_*.xml"):
            try: _p.unlink(missing_ok=True)
            except Exception: pass


class MenagerieArena:
    """One real menagerie robot under pure Rust-ctrl ``mj_step`` physics."""

    def __init__(self, robot: str = "unitree_g1", tree=None):
        self.robot = str(robot).strip().lower()
        self.morphology = self.robot
        self.model = load_menagerie_model(self.robot)
        self.data = mujoco.MjData(self.model)
        self._own_tree = tree is None
        self.tree = tree
        self.trunk = None
        self.ee_body_id = -1
        self.base_body_id = -1
        self._rollout_started = False
        self.qpos_written_after_reset = False
        self.balance_gain = 350.0
        self.balance_sign = 1.0
        self.reset()

    # -- lifecycle -------------------------------------------------------
    def reset(self):
        if int(self.model.nkey) > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._rollout_started = False
        self.qpos_written_after_reset = False
        # (Re)build trunk + Rust binding for this body.
        from src.brain.body_trunk import get_or_build_trunk
        self.trunk = get_or_build_trunk(self.model, self.robot)
        self.ee_body_id = int(self.trunk.ee_body_id)
        self.ee_site_id = -1
        for nm in (self.trunk.end_effector_sites or []):
            sid = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, str(nm)))
            if sid >= 0: self.ee_site_id = sid; break
        # base body: body of first free joint, else ee fallback
        self.base_body_id = int(self.trunk.ee_body_id)  # placeholder, fixed below
        prof_base = -1
        for j in range(self.model.njnt):
            if int(self.model.jnt_type[j]) == 0:
                prof_base = int(self.model.jnt_bodyid[j]); break
        self.base_body_id = prof_base if prof_base >= 0 else 1
        if self.tree is None:
            from src.brain.rust_bridge import RustTree
            self.tree = RustTree()
        self.tree.bind_trunk_v3(self.trunk)
        return self.get_state()

    def close(self):
        if self._own_tree and self.tree is not None:
            self.tree.close()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()

    # -- zero-cheat forbidden APIs ----------------------------------------
    def teleport_robot(self, *a, **kw):
        raise ZeroCheatViolation("teleport_robot is forbidden (zero-cheat)")

    def set_clutter_position(self, *a, **kw):
        raise ZeroCheatViolation("set_clutter_position is forbidden (zero-cheat)")

    def set_ball_position(self, *a, **kw):
        raise ZeroCheatViolation("set_ball_position is forbidden (zero-cheat)")

    def set_qpos(self, *a, **kw):
        raise ZeroCheatViolation("set_qpos is forbidden (zero-cheat)")

    # -- perception / state -------------------------------------------------
    def _body_pos(self, name_or_id) -> np.ndarray:
        bid = int(name_or_id) if isinstance(name_or_id, (int, np.integer)) else \
            int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, str(name_or_id)))
        if bid < 0: raise ValueError(f"body not found: {name_or_id}")
        mujoco.mj_forward(self.model, self.data)
        return np.array(self.data.xpos[bid], dtype=float)

    def _ee_point_world(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        base = np.array(self.data.xpos[self.ee_body_id], dtype=float)
        name = str(self.trunk.ee_body).lower() if self.trunk else ""
        if "calf" in name or "lleg" in name:
            rot = np.array(self.data.xmat[self.ee_body_id], dtype=float).reshape(3, 3)
            return base + rot @ np.array([0.0, 0.0, -0.20])
        return base

    def ee_pos(self) -> np.ndarray:
        if getattr(self, "ee_site_id", -1) >= 0:
            sid = int(self.ee_site_id)
            mujoco.mj_forward(self.model, self.data)
            return np.array(self.data.site_xpos[sid], dtype=float)
        return self._ee_point_world()

    def base_pos(self) -> np.ndarray:
        return self._body_pos(self.base_body_id)

    def target_pos(self, name: str = "red_target") -> np.ndarray:
        return self._body_pos(name)

    def place_visual_target(self, pos, name: str = "red_target"):
        """Place a mocap target marker. Setup only, before any rollout step."""
        if self._rollout_started:
            raise ZeroCheatViolation("cannot move target after rollout started")
        bid = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, str(name)))
        mid = int(self.model.body_mocapid[bid])
        if mid >= 0:
            self.data.mocap_pos[mid] = np.asarray(pos, dtype=float).ravel()[:3]
        mujoco.mj_forward(self.model, self.data)
        return self.target_pos(name)

    def get_state(self) -> dict[str, Any]:
        mujoco.mj_forward(self.model, self.data)
        return {"robot": self.robot, "time": float(self.data.time),
                "qpos": np.array(self.data.qpos), "qvel": np.array(self.data.qvel),
                "ctrl": np.array(self.data.ctrl), "ee_pos": self.ee_pos(),
                "base_pos": self.base_pos(),
                "target_pos": self.target_pos("red_target")}

    # -- zero-cheat control rollout -----------------------------------------
    def _actuator_aligned(self):
        nu = int(self.model.nu)
        q = np.zeros(nu); v = np.zeros(nu); b = np.zeros(nu)
        mujoco.mj_forward(self.model, self.data)
        for a in range(nu):
            qa = int(self.trunk.actuator_qposadr[a]); dof = int(self.trunk.actuator_dof[a])
            if 0 <= qa < self.model.nq: q[a] = float(self.data.qpos[qa])
            if 0 <= dof < self.model.nv:
                v[a] = float(self.data.qvel[dof]); b[a] = float(self.data.qfrc_bias[dof])
        # Balance feedforward into ankle bias (Rust still computes final ctrl):
        # pitch/pitch-rate of the floating base -> ankle torque bias.
        try:
            quat = np.array(self.data.xquat[self.base_body_id], dtype=float)
            w_, x_, y_, z_ = quat
            pitch = float(np.arcsin(np.clip(2.0 * (w_ * y_ - z_ * x_), -1.0, 1.0)))
            pitch_rate = float(self.data.qvel[4]) if self.model.nv > 4 else 0.0
            bal = self.balance_sign * (self.balance_gain * pitch + 30.0 * pitch_rate)
            for a in range(nu):
                nm = str(self.trunk.actuator_names[a]).lower()
                if "ankle" in nm and int(self.trunk.is_torque[a]):
                    b[a] += bal
        except Exception:
            pass
        # live EE Jacobian (nu x 3)
        jacp = np.zeros((3, self.model.nv)); jacr = np.zeros((3, self.model.nv))
        if getattr(self, "ee_site_id", -1) >= 0:
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, int(self.ee_site_id))
        else:
            ee_name = str(self.trunk.ee_body).lower() if self.trunk else ""
            if "calf" in ee_name or "lleg" in ee_name:
                mujoco.mj_jac(self.model, self.data, jacp, jacr,
                              self._ee_point_world(), int(self.ee_body_id))
            else:
                mujoco.mj_jacBody(self.model, self.data, jacp, jacr, int(self.ee_body_id))
        jac = np.zeros((nu, 3))
        for a in range(nu):
            dof = int(self.trunk.actuator_dof[a])
            if 0 <= dof < self.model.nv: jac[a] = jacp[:, dof]
        return q, v, b, jac

    def step_control(self, target, steps: int = 1) -> dict[str, Any]:
        """Pure mj_step rollout driven only by Rust triad ctrl. Never writes qpos."""
        target = np.asarray(target, dtype=float).ravel()[:3]
        ee0 = self.ee_pos(); err0 = float(np.linalg.norm(target - ee0))
        base0 = self.base_pos().copy()
        min_err = err0
        for _ in range(int(steps)):
            q, v, b, jac = self._actuator_aligned()
            err = (target - self.ee_pos()).astype(np.float32)
            try: self.tree.inject_vision(err, threshold=0.02)
            except Exception: pass
            ctrl = self.tree.step_control_v3(q, v, b, jac.reshape(-1), err,
                                             dt=float(self.model.opt.timestep))
            self.data.ctrl[:] = np.asarray(ctrl, dtype=float)[: int(self.model.nu)]
            self._rollout_started = True
            mujoco.mj_step(self.model, self.data)
            min_err = min(min_err, float(np.linalg.norm(target - self.ee_pos())))
        ee1 = self.ee_pos()
        return {"ee_start": ee0, "ee_end": ee1, "target": target,
                "error_start": err0, "error_min": float(min_err),
                "error_end": float(np.linalg.norm(target - ee1)),
                "base_start": base0, "base_end": self.base_pos(),
                "ctrl": np.array(self.data.ctrl, dtype=float)}

    def walk_and_reach(self, target, steps: int = 250) -> dict[str, Any]:
        return self.step_control(target, steps=steps)

    def step_chained_mission(self, waypoints, steps_per_wp: int = 250) -> dict[str, Any]:
        wps = [np.asarray(w, dtype=float).ravel()[:3] for w in waypoints]
        try:
            self.tree.start_diamond_mission(
                f"Clean messy Room 3 — {len(wps)} objects", n_objects=len(wps))
        except Exception:
            pass
        def _snap():
            try: return self.tree.snapshot()
            except Exception: return {}
        snap0 = _snap()
        initial_nodes = int(snap0.get("active_node_count", snap0.get("branch_count", 0)) or 0)
        peak = initial_nodes
        waypoint_errors: list[float] = []
        for wp in wps:
            base_dist = float(np.linalg.norm(wp - self.base_pos()))
            err_vec = (wp - self.ee_pos()).astype(np.float32)
            # progressive build / phase step before physical reach
            for _ in range(5):
                try: self.tree.step_diamond(err_vec, base_dist=base_dist, dt=0.016)
                except Exception: pass
                s = _snap(); peak = max(peak, int(s.get("active_node_count", s.get("branch_count", peak)) or peak))
            result = self.step_control(wp, steps=int(steps_per_wp))
            waypoint_errors.append(float(result["error_end"]))
            # chained tree step after reach (triggers pruning/respawn)
            err_after = (wp - self.ee_pos()).astype(np.float32)
            bd_after = float(np.linalg.norm(wp - self.base_pos()))
            for _ in range(2):
                try: self.tree.step_diamond(err_after, base_dist=bd_after, dt=0.016)
                except Exception: pass
                s = _snap(); peak = max(peak, int(s.get("active_node_count", s.get("branch_count", peak)) or peak))
        snap = _snap()
        return {"initial_nodes": initial_nodes, "peak_active_nodes": int(peak),
                "total_spawned": int(snap.get("total_spawned_count", 0) or 0),
                "total_pruned": int(snap.get("total_pruned_count", 0) or 0),
                "total_spawned_count": int(snap.get("total_spawned_count", 0) or 0),
                "total_pruned_count": int(snap.get("total_pruned_count", 0) or 0),
                "active_node_count": int(snap.get("active_node_count", snap.get("branch_count", 0)) or 0),
                "waypoint_errors": waypoint_errors,
                "qpos_written_after_reset": bool(self.qpos_written_after_reset),
                "snapshot": snap}

    def render_ego(self, width: int = 320, height: int = 220):
        return self.render(camera="ego_cam", width=width, height=height)

    def render(self, camera: str = "studio_cam", width: int = 128, height: int = 128):
        r = mujoco.Renderer(self.model, height=int(height), width=int(width))
        try:
            r.update_scene(self.data, camera=camera)
            return np.array(r.render())
        finally:
            r.close()


# Friendly aliases
Arena3D = MenagerieArena
build_menagerie_arena = MenagerieArena
create_menagerie_arena = MenagerieArena

__all__ = ["MENAGERIE_ROOT", "ROBOTS", "TARGET_BODIES", "ZeroCheatViolation",
           "load_menagerie_model", "MenagerieArena", "Arena3D",
           "build_menagerie_arena", "create_menagerie_arena"]
