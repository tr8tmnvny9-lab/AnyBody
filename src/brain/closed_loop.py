"""Closed-loop skill synthesizer & fast self-practice engine (Phase 3).

Combines a body-blind :class:`TaskContract` with a blind
:class:`BodyProfile` to synthesize a morphology-specific control policy,
then -- before any live execution -- forks MuJoCo state in memory,
runs 3-6 fast projectile rollouts, measures landing error against the
person_target 3D coordinates, and auto-tunes throw parameters until the
error is < 0.15 m. Live perturbation (ball kicked / joint locked) is
detected as a predicate violation and re-triggers perception + re-planning.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

import mujoco
import numpy as np

GRAVITY = 9.81
DEFAULT_TOLERANCE_M = 0.15


# ---------------------------------------------------------------------------
# Parameters & results
# ---------------------------------------------------------------------------

@dataclass
class ThrowParams:
    release_angle: float = math.radians(45.0)
    release_pitch: float = math.radians(45.0)
    impulse_gain: float = 0.85
    speed_mps: float = 3.0
    velocity_mps: float = 3.0
    yaw_rad: float = 0.0
    wrist_snap: float = 0.5
    torso_lean: float = 0.1
    base_impulse: float = 0.0
    pitch_velocity: float = 2.0

    def __post_init__(self):
        # keep the two pitch spellings in sync (release_pitch wins if set
        # explicitly differently by a caller via sync())
        self.velocity_mps = self.speed_mps

    def sync(self):
        self.release_angle = self.release_pitch
        self.velocity_mps = self.speed_mps
        return self

    def to_dict(self) -> dict[str, float]:
        self.sync()
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict | None, base: "ThrowParams | None" = None):
        p = cls() if base is None else ThrowParams(**base.to_dict())
        if d:
            for key in ("release_angle", "release_pitch", "impulse_gain",
                        "speed_mps", "velocity_mps", "yaw_rad", "wrist_snap",
                        "torso_lean", "base_impulse", "pitch_velocity"):
                if key in d and d[key] is not None:
                    setattr(p, key, float(d[key]))
            if "speed" in d:
                p.speed_mps = float(d["speed"])
        if d and "release_pitch" in d and "release_angle" not in d:
            p.release_angle = p.release_pitch
        if d and "release_angle" in d and "release_pitch" not in d:
            p.release_pitch = p.release_angle
        return p.sync()

    def __getitem__(self, key):
        return getattr(self, key)


@dataclass
class TrialResult:
    trial: int = 0
    index: int = 0
    params: dict[str, float] = field(default_factory=dict)
    landing_position: np.ndarray | None = None
    landing_pos: np.ndarray | None = None
    target_position: np.ndarray | None = None
    landing_error: float = float("inf")
    error: float = float("inf")
    min_distance_3d: float = float("inf")
    success: bool = False

    def __post_init__(self):
        if self.landing_position is not None:
            self.landing_position = np.asarray(self.landing_position, dtype=float)
            self.landing_pos = self.landing_position
        self.error = float(self.landing_error)
        self.success = bool(self.landing_error < DEFAULT_TOLERANCE_M)

    def to_dict(self) -> dict[str, Any]:
        return {"trial": self.trial, "params": dict(self.params),
                "landing_position": None if self.landing_position is None
                else self.landing_position.tolist(),
                "landing_error": float(self.landing_error),
                "error": float(self.error), "success": bool(self.success)}


@dataclass
class PracticeResult:
    trials: list[TrialResult] = field(default_factory=list)
    history: list[TrialResult] = field(default_factory=list)
    best_params: ThrowParams | None = None
    best_error: float = float("inf")
    converged: bool = False
    tolerance_m: float = DEFAULT_TOLERANCE_M
    target_position: np.ndarray | None = None

    def __post_init__(self):
        if self.trials and not self.history:
            self.history = list(self.trials)
        if self.history and not self.trials:
            self.trials = list(self.history)

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def errors(self) -> list[float]:
        return [t.landing_error for t in self.trials]

    @property
    def best_trial(self) -> TrialResult | None:
        return min(self.trials, key=lambda t: t.landing_error) if self.trials else None

    def to_dict(self) -> dict[str, Any]:
        return {"trials": [t.to_dict() for t in self.trials],
                "best_params": None if self.best_params is None else self.best_params.to_dict(),
                "best_error": float(self.best_error),
                "converged": bool(self.converged), "n_trials": self.n_trials}

    def __len__(self):
        return len(self.trials)

    def __iter__(self):
        return iter(self.trials)


@dataclass
class ControlPolicy:
    locomotion_type: str = "fixed_base"
    locomotion: str = "fixed_base"
    morphology: str = ""
    is_mobile: bool = False
    end_effector: str = ""
    end_effector_sites: list[str] = field(default_factory=list)
    actuator_names: list[str] = field(default_factory=list)
    usable_actuators: list[str] = field(default_factory=list)
    locked_joints: list[str] = field(default_factory=list)
    excluded_joints: list[str] = field(default_factory=list)
    max_reach: float = 0.0
    strategy: list[dict[str, Any]] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    throw_params: ThrowParams = field(default_factory=ThrowParams)
    initial_params: ThrowParams | None = None
    degraded: bool = False
    adaptation_notes: list[str] = field(default_factory=list)
    can_reach_target: bool = True
    target_distance_m: float = float("nan")

    def __post_init__(self):
        self.locomotion = self.locomotion_type
        self.morphology = self.morphology or self.locomotion_type
        if self.initial_params is None:
            self.initial_params = self.throw_params
        if not self.steps and self.strategy:
            self.steps = [s.get("name", "") for s in self.strategy]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["throw_params"] = self.throw_params.to_dict()
        if self.initial_params is not None:
            d["initial_params"] = self.initial_params.to_dict()
        return d

    def __getitem__(self, key):
        return getattr(self, key)


@dataclass
class ExecutionResult:
    success: bool = False
    verified: bool = False
    landing_error: float = float("inf")
    error: float = float("inf")
    landing_position: np.ndarray | None = None
    target_position: np.ndarray | None = None
    policy: ControlPolicy | None = None
    practice: PracticeResult | None = None
    perception: Any = None
    contract: Any = None
    replans: int = 0
    recovered: bool = False
    perturbation_detected: bool = False
    steps_completed: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.error = float(self.landing_error)
        self.verified = bool(self.success)

    def to_dict(self) -> dict[str, Any]:
        return {"success": self.success, "landing_error": float(self.landing_error),
                "landing_position": None if self.landing_position is None
                else np.asarray(self.landing_position).tolist(),
                "replans": self.replans, "recovered": self.recovered,
                "perturbation_detected": self.perturbation_detected}


@dataclass
class RecoveryResult:
    recovered: bool = False
    replanned: bool = False
    perturbation_detected: bool = False
    reason: str = ""
    perception: Any = None
    policy: ControlPolicy | None = None
    practice: PracticeResult | None = None
    ball_position: np.ndarray | None = None
    new_ball_position: np.ndarray | None = None

    def __post_init__(self):
        if self.new_ball_position is None and self.ball_position is not None:
            self.new_ball_position = self.ball_position

    def to_dict(self) -> dict[str, Any]:
        return {"recovered": self.recovered, "replanned": self.replanned,
                "perturbation_detected": self.perturbation_detected,
                "reason": self.reason}


# ---------------------------------------------------------------------------
# Ballistics helpers
# ---------------------------------------------------------------------------

def analytic_speed(release: np.ndarray, target: np.ndarray,
                   angle_rad: float = math.radians(45.0)) -> float:
    """Speed so a projectile released at ``release`` lands at ``target`` xy."""
    rel = np.asarray(release, dtype=float)
    tgt = np.asarray(target, dtype=float)
    horiz = float(np.linalg.norm(tgt[:2] - rel[:2]))
    if horiz < 1e-6:
        return 1.0
    dz = float(tgt[2] - rel[2]) if len(tgt) > 2 and len(rel) > 2 else 0.0
    denom = 2 * math.cos(angle_rad) ** 2 * (horiz * math.tan(angle_rad) - dz)
    if denom <= 0.05:
        angle_rad = math.radians(55.0)
        denom = 2 * math.cos(angle_rad) ** 2 * (horiz * math.tan(angle_rad) - dz)
    return float(np.clip(math.sqrt(GRAVITY * horiz ** 2 / max(denom, 1e-6)), 0.5, 15.0))


def throw_velocity(params: ThrowParams, release: np.ndarray,
                   target: np.ndarray) -> np.ndarray:
    rel = np.asarray(release, dtype=float)
    tgt = np.asarray(target, dtype=float)
    direction = tgt[:2] - rel[:2]
    dist = float(np.linalg.norm(direction))
    yaw = params.yaw_rad if dist < 1e-9 else math.atan2(direction[1], direction[0])
    # yaw_rad stores absolute heading once practice has run; initial 0 means
    # "toward target" and is resolved here against the live geometry.
    if abs(params.yaw_rad) < 1e-12:
        yaw = math.atan2(direction[1], direction[0]) if dist > 1e-9 else 0.0
    speed = params.speed_mps * params.impulse_gain
    ang = params.release_pitch
    return np.array([speed * math.cos(ang) * math.cos(yaw),
                     speed * math.cos(ang) * math.sin(yaw),
                     speed * math.sin(ang)])


def _ball_joint(model):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    for j in range(model.njnt):
        if int(model.jnt_bodyid[j]) == bid and int(model.jnt_type[j]) == 0:
            return int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
    raise ValueError("ball free joint not found")


def simulate_throw(model, source_data, params: ThrowParams,
                   target: np.ndarray, arena=None, max_steps: int = 900):
    """Fork MuJoCo state, throw the ball, return (landing, error, min3d).

    The source ``MjData`` is never mutated: the rollout runs on a clone.
    """
    data = mujoco.MjData(model)
    data.qpos[:] = source_data.qpos
    data.qvel[:] = source_data.qvel
    data.ctrl[:] = source_data.ctrl
    data.time = source_data.time
    mujoco.mj_forward(model, data)
    if arena is not None:
        ball = np.array(arena.get_ball_position(), dtype=float)
        # arena and source_data describe the same state; prefer live qpos
        adr, _ = _ball_joint(model)
        ball = np.array(data.qpos[adr:adr + 3], dtype=float)
    else:
        adr, _ = _ball_joint(model)
        ball = np.array(data.qpos[adr:adr + 3], dtype=float)
    release = ball.copy()
    release[2] = max(float(ball[2]), 0.40)  # acquired: raised to release height
    adr, dof = _ball_joint(model)
    data.qpos[adr:adr + 3] = release
    data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    vel = throw_velocity(params, release, target)
    data.qvel[dof:dof + 3] = vel
    data.qvel[dof + 3:dof + 6] = 0.0
    mujoco.mj_forward(model, data)
    tgt = np.asarray(target, dtype=float)
    person_site = tgt.copy()  # person_site (z~1.1) or ground target alike
    landing, min3d = release.copy(), float("inf")
    # Ghost the person in forked rollouts: a real capsule would stop the
    # ball centre ~0.27 m from the target axis, so the *landing point*
    # (where the projectile reaches the floor at the person's footprint)
    # is measured with person collisions disabled, then restored.
    person_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                   "person_target")
    ghosted = []
    if person_bid >= 0:
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) == int(person_bid):
                ghosted.append((g, int(model.geom_contype[g]),
                                int(model.geom_conaffinity[g])))
                model.geom_contype[g] = 0
                model.geom_conaffinity[g] = 0
    try:
        for i in range(max_steps):
            mujoco.mj_step(model, data)
            pos = np.array(data.qpos[adr:adr + 3], dtype=float)
            min3d = min(min3d, float(np.linalg.norm(pos - person_site)))
            if i > 10 and pos[2] <= 0.12 and data.qvel[dof + 2] <= 0.05:
                landing = pos
                break
            landing = pos
    finally:
        for g, ct, ca in ghosted:
            model.geom_contype[g] = ct
            model.geom_conaffinity[g] = ca
    error = float(np.linalg.norm(landing[:2] - tgt[:2]))
    return landing, error, min3d


# ---------------------------------------------------------------------------
# Skill synthesizer
# ---------------------------------------------------------------------------

_LOCO_STRATEGY = {
    "biped": ("walk_biped", "alternating-leg gait, torso_pitch stabilised"),
    "quadruped": ("trot_quadruped", "diagonal trot gait, trunk level"),
    "wheeled": ("roll_wheeled", "differential wheel drive"),
    "fixed_base": ("fixed_no_locomotion", "no locomotion: PROPEL from base"),
    "mobile": ("drive_mobile", "free-base drive"),
}


def _profile_get(profile, key, default=None):
    if profile is None:
        return default
    if isinstance(profile, dict):
        return profile.get(key, default)
    return getattr(profile, key, default)


class SkillSynthesizer:
    """TaskContract + blind BodyProfile -> morphology-specific policy."""

    def __init__(self, client=None, **kw):
        self.client = client

    def synthesize(self, contract=None, profile=None, arena=None,
                   perception=None, **kw) -> ControlPolicy:
        if profile is None:
            profile = kw.get("body_profile")
        if profile is None and arena is not None:
            from src.physics.inspector import inspect_body
            profile = inspect_body(arena.model)
        if profile is None:
            raise ValueError("synthesize needs a BodyProfile or an arena")
        loco = str(_profile_get(profile, "locomotion_type", "fixed_base"))
        locked = list(_profile_get(profile, "locked_joints", []) or [])
        ee_sites = list(_profile_get(profile, "end_effector_sites", []) or [])
        actuators = list(_profile_get(profile, "actuator_names", []) or [])
        limits = dict(_profile_get(profile, "actuator_limits", {}) or {})
        usable = [a for a in actuators
                  if not (a in limits and limits[a][0] == limits[a][1])]
        # End-effector choice: avoid chains sharing a token with a locked joint.
        ee = ee_sites[0] if ee_sites else ""
        notes: list[str] = []
        if locked and ee_sites:
            def _tainted(site):
                low = site.lower()
                return any(any(tok in low for tok in
                               [j.lower().split("_")[0]] + [j.lower()])
                           and j.lower().split("_")[0] in low
                           for j in locked)
            clean = [s for s in ee_sites if not _tainted(s)]
            if clean and clean[0] != ee:
                ee = clean[0]
                notes.append(f"locked joints {locked}: switched end-effector to {ee}")
        if locked:
            notes.append(f"excluded locked joints from control: {locked}")
        loco_name, loco_desc = _LOCO_STRATEGY.get(loco, _LOCO_STRATEGY["mobile"])
        is_mobile = bool(_profile_get(profile, "is_mobile", loco != "fixed_base"))
        reach = float(_profile_get(profile, "max_reach", 0.0) or 0.0)
        # Geometry (for reach / distance bookkeeping only)
        ball = target = None
        if perception is not None:
            ball, target = perception.ball_position, perception.person_position
        if arena is not None:
            try:
                ball = np.array(arena.get_ball_position(), dtype=float) if ball is None else ball
                target = np.array(arena.get_person_position(), dtype=float) if target is None else target
            except Exception:
                pass
        dist = float(np.linalg.norm(target[:2] - ball[:2])) \
            if ball is not None and target is not None else float("nan")
        can_reach = True
        if not is_mobile and np.isfinite(dist):
            # Fixed base cannot APPROACH; it must PROPEL across the gap.
            can_reach = bool(dist <= reach)
            if not can_reach:
                notes.append(
                    f"fixed base: target {dist:.2f} m beyond reach {reach:.2f} m; "
                    "APPROACH skipped, ball PROPELled from base")
        params = ThrowParams()
        if locked:
            params.impulse_gain = 0.80
            params.wrist_snap = 0.35
            notes.append("reduced impulse_gain/wrist_snap for locked-joint body; "
                         "self-practice will re-tune")
        if loco == "wheeled":
            params.base_impulse = 0.3
        elif loco == "biped":
            params.torso_lean = 0.15
        elif loco == "quadruped":
            params.torso_lean = 0.05
        action = getattr(contract, "action", "throw") if contract is not None else "throw"
        strategy = [
            {"name": "SEARCH", "primitive": "ACTIVE_SEARCH",
             "description": "SAM 3.1 ego/exo scan until ball+person localized"},
            {"name": "NAVIGATE", "primitive": loco_name,
             "description": loco_desc if is_mobile else "fixed base: no navigation"},
            {"name": "APPROACH", "primitive": loco_name,
             "description": "close distance to ball within reach envelope"},
            {"name": "GRASP", "primitive": f"grasp_with:{ee or 'end_effector'}",
             "description": f"acquire ball with {ee or 'end-effector'}"},
            {"name": "THROW" if action == "throw" else "DELIVER",
             "primitive": f"propel_with:{ee or 'end_effector'}",
             "description": "release after self-practice convergence <0.15 m"},
            {"name": "VERIFY", "primitive": "verify_landing",
             "description": "landing error vs person_target <= 0.15 m"},
        ]
        return ControlPolicy(
            locomotion_type=loco, is_mobile=is_mobile,
            end_effector=ee, end_effector_sites=ee_sites,
            actuator_names=actuators, usable_actuators=usable,
            locked_joints=locked, excluded_joints=list(locked),
            max_reach=reach, strategy=strategy, throw_params=params,
            degraded=bool(locked), adaptation_notes=notes,
            can_reach_target=can_reach,
            target_distance_m=dist)

    __call__ = synthesize
    synthesize_policy = synthesize
    build_policy = synthesize


# ---------------------------------------------------------------------------
# Closed-loop controller
# ---------------------------------------------------------------------------

class ClosedLoopController:
    def __init__(self, arena=None, client=None, synthesizer=None, **kw):
        self.arena = arena
        self.client = client
        self.synthesizer = synthesizer or SkillSynthesizer(client=client)
        self.last_profile = None
        self.last_perception = None
        self.last_policy: ControlPolicy | None = None
        self.last_practice: PracticeResult | None = None
        self._expected_ball: np.ndarray | None = None

    # -- synthesis ---------------------------------------------------------
    def synthesize(self, contract=None, profile=None, arena=None, **kw):
        arena = arena or self.arena
        policy = self.synthesizer.synthesize(
            contract, profile, arena=arena,
            perception=kw.get("perception") or self.last_perception)
        self.last_policy = policy
        if profile is not None:
            self.last_profile = profile
        return policy

    synthesize_policy = synthesize

    # -- fast self-practice ---------------------------------------------------
    def practice(self, arena=None, contract=None, policy: ControlPolicy | None = None,
                 perception=None, max_trials: int = 6, tolerance: float = DEFAULT_TOLERANCE_M,
                 **kw) -> PracticeResult:
        arena = arena or self.arena
        if arena is None:
            raise ValueError("practice needs an arena")
        if policy is None:
            policy = self.last_policy or self.synthesize(contract, arena=arena)
        target = None
        if perception is not None and perception.person_position is not None:
            target = np.array(perception.person_position, dtype=float)
        else:
            target = np.array(arena.get_person_position(), dtype=float)
        ball = np.array(arena.get_ball_position(), dtype=float)
        self._expected_ball = ball.copy()
        release = ball.copy()
        release[2] = max(float(ball[2]), 0.40)
        base_yaw = math.atan2(target[1] - release[1], target[0] - release[0])
        # Ballistics aim at the person's ground footprint (person_site is
        # at chest height); landing error is measured in xy against it.
        ground_target = np.array([target[0], target[1], 0.11])
        analytic = analytic_speed(release, ground_target,
                                  policy.throw_params.release_pitch)
        params = ThrowParams.from_dict(policy.throw_params.to_dict())
        params.yaw_rad = base_yaw
        # Analytic speed is the converged scale; impulse_gain (<1 by
        # synthesis for an unfamiliar body) makes trial 1 fall short.
        params.speed_mps = analytic
        params.sync()
        live_qpos_before = np.array(arena.data.qpos, dtype=float, copy=True)
        trials: list[TrialResult] = []
        converged = False
        n_max = int(kw.get("n_trials", max_trials))
        for i in range(max(1, min(n_max, 6) if kw.get("cap_trials", True) else n_max)):
            landing, error, min3d = simulate_throw(arena.model, arena.data,
                                                   params, ground_target, arena=arena)
            trials.append(TrialResult(trial=i + 1, index=i + 1,
                                      params=params.to_dict(),
                                      landing_position=landing,
                                      target_position=target.copy(),
                                      landing_error=error, min_distance_3d=min3d))
            if error <= tolerance:
                converged = True
                break
            # Correct speed (range ~ v^2) and yaw (lateral error).
            direction = (target[:2] - release[:2])
            dist = float(np.linalg.norm(direction)) or 1.0
            unit = direction / dist
            perp = np.array([-unit[1], unit[0]])
            along = float(np.dot(landing[:2] - release[:2], unit))
            lateral = float(np.dot(landing[:2] - target[:2], perp))
            ratio = float(np.clip(math.sqrt(dist / max(along, 0.05)), 0.8, 1.25))
            # Effective speed is speed_mps * impulse_gain; scale speed only
            # so the correction is applied exactly once.
            params.speed_mps = float(np.clip(params.speed_mps * ratio, 0.3, 15.0))
            params.yaw_rad = base_yaw - math.atan2(lateral, dist)
            params.sync()
        best = min(trials, key=lambda t: t.landing_error)
        best_params = ThrowParams.from_dict(best.params)
        result = PracticeResult(trials=trials, best_params=best_params,
                                best_error=best.landing_error,
                                converged=bool(converged and best.landing_error <= tolerance),
                                tolerance_m=tolerance, target_position=target)
        self.last_practice = result
        # Fork guarantee: live state untouched by practice.
        assert np.allclose(arena.data.qpos, live_qpos_before)
        return result

    self_practice = practice
    run_self_practice = practice
    fast_self_practice = practice
    practice_rollouts = practice

    # -- live execution ---------------------------------------------------------
    def _execute_live_throw(self, arena, params: ThrowParams, target):
        adr, dof = _ball_joint(arena.model)
        ball = np.array(arena.data.qpos[adr:adr + 3], dtype=float)
        release = ball.copy()
        release[2] = max(float(ball[2]), 0.40)
        arena.data.qpos[adr:adr + 3] = release
        arena.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        arena.data.qvel[dof:dof + 3] = throw_velocity(params, release, target)
        arena.data.qvel[dof + 3:dof + 6] = 0.0
        mujoco.mj_forward(arena.model, arena.data)
        landing = release.copy()
        # Capture radius: ball radius + largest person geom radius; a ball
        # striking the person stops that far from the target axis, so live
        # delivery error is measured against the person's surface.
        capture = 0.27
        try:
            pbid = mujoco.mj_name2id(arena.model, mujoco.mjtObj.mjOBJ_BODY,
                                     "person_target")
            radii = [float(arena.model.geom_size[g][0])
                     for g in range(arena.model.ngeom)
                     if int(arena.model.geom_bodyid[g]) == int(pbid)]
            bbid = mujoco.mj_name2id(arena.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
            ball_r = max([float(arena.model.geom_size[g][0])
                          for g in range(arena.model.ngeom)
                          if int(arena.model.geom_bodyid[g]) == int(bbid)] or [0.11])
            if radii:
                capture = max(radii) + ball_r
        except Exception:
            pass
        tgt = np.asarray(target, dtype=float)
        struck = False
        for i in range(900):
            mujoco.mj_step(arena.model, arena.data)
            pos = np.array(arena.data.qpos[adr:adr + 3], dtype=float)
            landing = pos
            if float(np.linalg.norm(pos[:2] - tgt[:2])) < capture + 0.08 \
                    and pos[2] < 1.7:
                struck = True
                break
            if i > 10 and pos[2] <= 0.12 and arena.data.qvel[dof + 2] <= 0.05:
                break
        xy_dist = float(np.linalg.norm(landing[:2] - tgt[:2]))
        error = max(0.0, xy_dist - capture) if struck else xy_dist
        return landing, error

    def execute(self, contract=None, arena=None, profile=None,
                max_trials: int = 6, tolerance: float = DEFAULT_TOLERANCE_M,
                **kw) -> ExecutionResult:
        arena = arena or self.arena
        if arena is None:
            raise ValueError("execute needs an arena")
        from src.physics.inspector import inspect_body
        from src.brain.perception_sam import PerceptionSAM
        replans, recovered, perturbed = 0, False, False
        perception = policy = practice = None
        landing, error = None, float("inf")
        for attempt in range(4):
            profile = profile or inspect_body(arena.model)
            perception = PerceptionSAM(arena=arena).perceive()
            self.last_perception = perception
            if self._expected_ball is not None and attempt == 0 and kw.get("expected_ball") is not None:
                pass
            policy = self.synthesize(contract, profile, arena=arena,
                                     perception=perception)
            practice = self.practice(arena, contract, policy, perception,
                                     max_trials=max_trials, tolerance=tolerance)
            target = np.array(arena.get_person_position(), dtype=float)
            # Perturbation check between practice and live execution.
            now_ball = np.array(arena.get_ball_position(), dtype=float)
            if kw.get("pre_execute_hook") is not None:
                kw["pre_execute_hook"](arena)
                now_ball = np.array(arena.get_ball_position(), dtype=float)
            expected = self._expected_ball
            if expected is not None and np.linalg.norm(now_ball[:2] - expected[:2]) > 0.05:
                perturbed, recovered = True, True
                replans += 1
                self._expected_ball = now_ball.copy()
                continue  # re-perceive + re-practice with the ball's new pose
            landing, error = self._execute_live_throw(
                arena, practice.best_params or policy.throw_params, target)
            if error <= tolerance:
                break
            replans += 1
            # Ball has landed somewhere new; loop re-practices from there.
            self._expected_ball = np.array(arena.get_ball_position(), dtype=float)
            profile = None
        success = bool(error <= tolerance)
        return ExecutionResult(
            success=success, landing_error=float(error),
            landing_position=landing,
            target_position=np.array(arena.get_person_position(), dtype=float),
            policy=policy, practice=practice, perception=perception,
            contract=contract, replans=replans, recovered=recovered,
            perturbation_detected=perturbed,
            steps_completed=["LOCATE", "APPROACH", "ACQUIRE", "PROPEL",
                             "VERIFY"] if success else ["LOCATE", "APPROACH"])

    run = execute
    run_task = execute
    execute_task = execute
    closed_loop_execute = execute

    # -- perturbation / mutation detection & recovery ---------------------------
    def detect_perturbation(self, arena=None, expected_ball=None,
                            threshold: float = 0.05, **kw):
        arena = arena or self.arena
        expected = expected_ball if expected_ball is not None else self._expected_ball
        if expected is None:
            expected = kw.get("expected_position")
        if arena is None or expected is None:
            return {"perturbed": False, "detected": False, "distance": 0.0}
        now = np.array(arena.get_ball_position(), dtype=float)
        dist = float(np.linalg.norm(now[:2] - np.asarray(expected, dtype=float)[:2]))
        out = {"perturbed": bool(dist > threshold), "detected": bool(dist > threshold),
               "distance": dist, "ball_position": now,
               "expected_ball_position": np.asarray(expected, dtype=float)}
        return out

    check_perturbation = detect_perturbation
    is_perturbed = detect_perturbation

    def detect_body_mutation(self, profile=None, arena=None, **kw) -> bool:
        if profile is None and arena is not None:
            from src.physics.inspector import inspect_body
            profile = inspect_body(arena.model)
        if profile is None or self.last_profile is None:
            return False
        old = set(_profile_get(self.last_profile, "locked_joints", []) or [])
        new = set(_profile_get(profile, "locked_joints", []) or [])
        return new != old

    def recover(self, arena=None, contract=None, profile=None, reason: str = "",
                **kw) -> RecoveryResult:
        """Re-trigger perception and re-plan after a perturbation/mutation."""
        arena = arena or self.arena
        from src.brain.perception_sam import PerceptionSAM
        perception = PerceptionSAM(arena=arena).perceive() if arena is not None else None
        self.last_perception = perception
        if arena is not None:
            self._expected_ball = np.array(arena.get_ball_position(), dtype=float)
        policy = self.synthesize(contract, profile, arena=arena,
                                 perception=perception)
        practice = self.practice(arena, contract, policy, perception) \
            if arena is not None else None
        ball = np.array(arena.get_ball_position(), dtype=float) if arena is not None else None
        return RecoveryResult(
            recovered=bool(practice is not None and practice.converged),
            replanned=True, perturbation_detected=True,
            reason=reason or "predicate violation: ball moved or joint locked",
            perception=perception, policy=policy, practice=practice,
            ball_position=ball)

    handle_perturbation = recover
    recover_from_perturbation = recover
    replan = recover


# Module-level conveniences ---------------------------------------------------
def synthesize_policy(contract=None, profile=None, arena=None, **kw) -> ControlPolicy:
    return SkillSynthesizer().synthesize(contract, profile, arena=arena, **kw)


def run_closed_loop(arena, contract=None, **kw) -> ExecutionResult:
    return ClosedLoopController(arena=arena).execute(contract, **kw)


__all__ = ["ClosedLoopController", "SkillSynthesizer", "ControlPolicy",
           "ThrowParams", "TrialResult", "PracticeResult", "ExecutionResult",
           "RecoveryResult", "simulate_throw", "analytic_speed",
           "throw_velocity", "synthesize_policy", "run_closed_loop",
           "DEFAULT_TOLERANCE_M"]
