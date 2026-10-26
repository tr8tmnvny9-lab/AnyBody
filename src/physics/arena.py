"""MuJoCo world/arena manager: shared scene, rendering, perturbation."""
from __future__ import annotations

import time
from typing import Any

import mujoco
import numpy as np

from .morphologies import MORPHOLOGY_NAMES, build_mjcf, create_model, get_module

__all__ = ["Arena", "World", "build_arena", "create_arena", "load_arena",
           "get_arena", "build_mjcf_for", "MORPHOLOGY_NAMES", "ARENA_XML",
           "arena_xml", "perturb_ball"]


def arena_xml() -> str:
    """Scene-only MJCF fragment (floor, ball, person, obstacles, exo cam)."""
    from .morphologies.humanoid import _ARENA
    return _ARENA


ARENA_XML = None  # set after import below


def build_mjcf_for(morphology: str = "humanoid", **kw) -> str:
    return build_mjcf(morphology, **kw)


class Arena:
    """Owns an MjModel+MjData for one morphology in the shared arena."""

    def __init__(self, morphology: str = "humanoid", model=None, xml: str | None = None,
                 name: str | None = None, **_kw):
        self.morphology = name or morphology
        if model is not None:
            self.model = model
        elif xml is not None:
            self.model = mujoco.MjModel.from_xml_string(xml)
        else:
            self.model = create_model(self.morphology)
        self.data = mujoco.MjData(self.model)
        self.morphology_name = self.morphology
        self._renderer = None
        self.reset()

    # -- lifecycle ---------------------------------------------------------
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def step(self, n: int = 1, ctrl=None):
        if ctrl is not None:
            self.set_ctrl(ctrl)
        for _ in range(int(n)):
            mujoco.mj_step(self.model, self.data)
        return self.get_state()

    def set_ctrl(self, ctrl):
        arr = np.asarray(ctrl, dtype=float).ravel()
        k = min(len(arr), self.model.nu)
        self.data.ctrl[:k] = arr[:k]

    set_control = set_ctrl

    # -- ids / positions ---------------------------------------------------
    def _body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"body not found: {name}")
        return int(bid)

    def _site_pos(self, name: str) -> np.ndarray:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise ValueError(f"site not found: {name}")
        mujoco.mj_forward(self.model, self.data)
        return np.array(self.data.site_xpos[sid], dtype=float)

    def get_ball_position(self) -> np.ndarray:
        return np.array(self.data.xpos[self._body_id("ball")], dtype=float)

    get_ball_pos = get_ball_position

    def ball_position(self):
        return self.get_ball_position()

    def get_person_position(self) -> np.ndarray:
        try:
            return self._site_pos("person_site")
        except Exception:
            return np.array(self.data.xpos[self._body_id("person_target")], dtype=float)

    get_person_pos = get_person_position
    get_target_position = get_person_position
    person_position = get_person_position

    def get_obstacle_positions(self) -> dict[str, np.ndarray]:
        out = {}
        for name in ("obstacle_1", "obstacle_2", "target_ring"):
            try:
                out[name] = np.array(self.data.xpos[self._body_id(name)], dtype=float)
            except Exception:
                pass
        return out

    # -- perturbation ------------------------------------------------------
    def set_ball_position(self, x, y=None, z: float = 0.11, **kw):
        if y is None:  # allow array-like first arg
            pos = np.asarray(x, dtype=float).ravel()
            x, y = float(pos[0]), float(pos[1])
            if len(pos) > 2:
                z = float(pos[2])
        if "pos" in kw:
            p = np.asarray(kw["pos"], dtype=float).ravel()
            x, y, z = float(p[0]), float(p[1]), float(p[2] if len(p) > 2 else z)
        bid = self._body_id("ball")
        # free-joint qpos address via the ball's joint
        jid = None
        for j in range(self.model.njnt):
            if int(self.model.jnt_bodyid[j]) == bid and int(self.model.jnt_type[j]) == 0:
                jid = j
                break
        if jid is not None:
            adr = int(self.model.jnt_qposadr[jid])
            self.data.qpos[adr:adr + 3] = [float(x), float(y), float(z)]
            self.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
            dof = int(self.model.jnt_dofadr[jid])
            self.data.qvel[dof:dof + 6] = 0.0
        else:
            self.data.xpos[bid] = [float(x), float(y), float(z)]
        mujoco.mj_forward(self.model, self.data)
        return self.get_ball_position()

    def perturb_ball(self, x=None, y=None, z: float = 0.11, dx: float = 0.0,
                     dy: float = 0.0, **kw):
        """Move/kick the ball. Absolute (x, y) or relative (dx, dy)."""
        cur = self.get_ball_position()
        nx = float(x) if x is not None else float(cur[0] + dx)
        ny = float(y) if y is not None else float(cur[1] + dy)
        if "pos" in kw:
            return self.set_ball_position(kw["pos"])
        return self.set_ball_position(nx, ny, z)

    kick_ball = perturb_ball
    move_ball = perturb_ball

    # -- rendering ---------------------------------------------------------
    def render(self, camera: str = "exo_cam", width: int = 128, height: int = 128,
               depth: bool = False):
        renderer = mujoco.Renderer(self.model, height=int(height), width=int(width))
        try:
            if depth:
                renderer.enable_depth_rendering()
            renderer.update_scene(self.data, camera=camera)
            return np.array(renderer.render())
        finally:
            renderer.close()

    def render_rgb(self, camera: str = "exo_cam", width: int = 128, height: int = 128):
        return self.render(camera=camera, width=width, height=height, depth=False)

    def render_depth(self, camera: str = "exo_cam", width: int = 128, height: int = 128):
        return self.render(camera=camera, width=width, height=height, depth=True)

    def render_ego(self, width: int = 128, height: int = 128, depth: bool = False):
        return self.render(camera="ego_cam", width=width, height=height, depth=depth)

    def render_exo(self, width: int = 128, height: int = 128, depth: bool = False):
        return self.render(camera="exo_cam", width=width, height=height, depth=depth)

    # -- state -------------------------------------------------------------
    def get_state(self) -> dict[str, Any]:
        mujoco.mj_forward(self.model, self.data)
        return {
            "morphology": self.morphology,
            "time": float(self.data.time),
            "qpos": np.array(self.data.qpos, dtype=float),
            "qvel": np.array(self.data.qvel, dtype=float),
            "ctrl": np.array(self.data.ctrl, dtype=float),
            "ball_position": self.get_ball_position(),
            "ball_pos": self.get_ball_position(),
            "person_position": self.get_person_position(),
            "target_position": self.get_person_position(),
        }

    state = get_state

    def profile(self):
        from .inspector import inspect_body
        return inspect_body(self.model)

    def close(self):
        self._renderer = None


World = Arena


def build_arena(morphology: str = "humanoid", **kw) -> Arena:
    return Arena(morphology=morphology, **kw)


create_arena = build_arena
load_arena = build_arena
get_arena = build_arena


def perturb_ball(arena: Arena, x=None, y=None, **kw):
    """Module-level convenience: perturb the ball in an existing Arena."""
    return arena.perturb_ball(x, y, **kw)


ARENA_XML = arena_xml()
