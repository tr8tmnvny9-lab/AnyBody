"""Blind runtime MJCF kinematic introspector + body mutator.

Discovers structure from any ``mujoco.MjModel`` at runtime. No robot /
morphology names are hardcoded: locomotion, end-effectors, reach and
limits are inferred from joint types, joint/geom/site semantics,
kinematic-tree topology and sampled forward kinematics.
"""
from __future__ import annotations

import copy
import json
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

_EE_TOKENS = ("end_effector", "end-effector", "endeffector", "_ee", "ee_",
              "gripper", "grip", "hand", "tool", "launcher", "scoop",
              "claw", "rake", "catapult", "wrist", "manipulator")
_SCENE_TOKENS = ("ball", "person", "obstacle", "floor", "target_ring", "ring")
_JOINT_TYPE_NAMES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}


def _name(model, kind: str, idx: int) -> str:
    try:
        n = mujoco.mj_id2name(model, getattr(mujoco.mjtObj, f"mjOBJ_{kind}"), idx)
    except Exception:
        n = None
    return n or f"{kind.lower()}_{idx}"


def _is_scene(name: str) -> bool:
    low = name.lower()
    return any(t in low for t in _SCENE_TOKENS)


@dataclass
class BodyProfile:
    """Structured, JSON-serialisable description of any MuJoCo body."""
    locomotion_type: str = "fixed_base"
    locomotion: str = "fixed_base"          # alias of locomotion_type
    is_mobile: bool = False
    has_free_base: bool = False
    free_joint_count: int = 0
    base_body_name: str | None = None
    base_body_id: int = -1
    n_dof: int = 0
    nv: int = 0
    nq: int = 0
    n_joints: int = 0
    njnt: int = 0
    n_actuators: int = 0
    nu: int = 0
    n_bodies: int = 0
    n_geoms: int = 0
    n_sites: int = 0
    n_cameras: int = 0
    joint_names: list[str] = field(default_factory=list)
    joint_types: dict[str, str] = field(default_factory=dict)
    joint_ranges: dict[str, list[float]] = field(default_factory=dict)
    limited_joints: list[str] = field(default_factory=list)
    locked_joints: list[str] = field(default_factory=list)
    actuator_names: list[str] = field(default_factory=list)
    actuator_limits: dict[str, list[float]] = field(default_factory=dict)
    actuator_ctrlrange: dict[str, list[float]] = field(default_factory=dict)
    body_names: list[str] = field(default_factory=list)
    site_names: list[str] = field(default_factory=list)
    camera_names: list[str] = field(default_factory=list)
    cameras: list[str] = field(default_factory=list)
    end_effector_sites: list[str] = field(default_factory=list)
    end_effector_site_ids: list[int] = field(default_factory=list)
    end_effectors: list[str] = field(default_factory=list)  # alias
    max_reach: float = 0.0
    reach: float = 0.0  # alias
    timestep: float = 0.002
    total_mass: float = 0.0
    model_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    # dict-style access for flexible callers
    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


def _ancestors(model, body_id: int) -> list[int]:
    chain, cur = [], int(body_id)
    while cur > 0:
        chain.append(cur)
        cur = int(model.body_parentid[cur])
    chain.reverse()
    return chain


def _chain_reach(model, site_id: int) -> float:
    body_id = int(model.site_bodyid[site_id])
    total = float(np.linalg.norm(model.site_pos[site_id]))
    cur = body_id
    while cur > 0:
        # Skip the root body's world placement: it is not a link length.
        if int(model.body_parentid[cur]) != 0:
            total += float(np.linalg.norm(model.body_pos[cur]))
        cur = int(model.body_parentid[cur])
    return total


def _sampled_reach(model, site_ids: list[int], base_pos: np.ndarray,
                   trials: int = 256, seed: int = 0) -> float:
    if not site_ids:
        return 0.0
    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    best = 0.0
    q0 = np.array(data.qpos, dtype=float, copy=True)
    for _ in range(trials):
        q = q0.copy()
        for j in range(model.njnt):
            jt = int(model.jnt_type[j])
            if jt == 0:
                continue
            adr = int(model.jnt_qposadr[j])
            if jt == 3 and bool(model.jnt_limited[j]):
                lo, hi = float(model.jnt_range[j][0]), float(model.jnt_range[j][1])
                if hi > lo:
                    q[adr] = rng.uniform(lo, hi)
            elif jt == 2 and bool(model.jnt_limited[j]):
                lo, hi = float(model.jnt_range[j][0]), float(model.jnt_range[j][1])
                if hi > lo:
                    q[adr] = rng.uniform(lo, hi)
        data.qpos[:] = q
        try:
            mujoco.mj_forward(model, data)
        except Exception:
            continue
        for sid in site_ids:
            d = float(np.linalg.norm(np.asarray(data.site_xpos[sid]) - base_pos))
            best = max(best, d)
    return best


def inspect_body(model, **_kw) -> BodyProfile:
    """Blindly introspect any MjModel. Alias: inspect_model/introspect."""
    if isinstance(model, str):
        model = mujoco.MjModel.from_xml_string(model)
    joint_names = [_name(model, "JOINT", i) for i in range(model.njnt)]
    actuator_names = [_name(model, "ACTUATOR", i) for i in range(model.nu)]
    body_names = [_name(model, "BODY", i) for i in range(model.nbody)]
    site_names = [_name(model, "SITE", i) for i in range(model.nsite)]
    camera_names = [_name(model, "CAMERA", i) for i in range(model.ncam)]

    joint_types, joint_ranges, limited, locked = {}, {}, [], []
    free_count = 0
    for j, jn in enumerate(joint_names):
        jt = int(model.jnt_type[j])
        joint_types[jn] = _JOINT_TYPE_NAMES.get(jt, str(jt))
        if jt == 0:
            free_count += 1
            continue
        lo, hi = float(model.jnt_range[j][0]), float(model.jnt_range[j][1])
        joint_ranges[jn] = [lo, hi]
        if bool(model.jnt_limited[j]):
            limited.append(jn)
            if hi - lo < 1e-4:
                locked.append(jn)

    actuator_limits = {}
    for a, an in enumerate(actuator_names):
        lo, hi = float(model.actuator_ctrlrange[a][0]), float(model.actuator_ctrlrange[a][1])
        actuator_limits[an] = [lo, hi]
        if hi - lo < 1e-12:
            # disabled actuator implies its joint is effectively locked
            pass

    # --- robot root: first non-scene body under world with a joint subtree
    robot_bodies: list[int] = []
    for b in range(1, model.nbody):
        if not _is_scene(body_names[b]):
            robot_bodies.append(b)
    # free-joint bodies among robot bodies
    free_bodies = []
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == 0:
            b = int(model.jnt_bodyid[j])
            if b in robot_bodies or not _is_scene(body_names[b]):
                if not _is_scene(body_names[b]):
                    free_bodies.append(b)
    has_free = len(free_bodies) > 0
    base_id = free_bodies[0] if has_free else (robot_bodies[0] if robot_bodies else -1)
    base_name = body_names[base_id] if base_id >= 0 else None

    # --- end-effectors: semantic site names, else leaf-body sites
    ee_ids: list[int] = []
    for s, sn in enumerate(site_names):
        if _is_scene(sn):
            continue
        low = sn.lower()
        if any(t in low for t in _EE_TOKENS):
            # wrist alone is a weaker token: keep it, it sits at chain end
            ee_ids.append(s)
    if not ee_ids:
        child_count = np.zeros(model.nbody, dtype=int)
        for b in range(1, model.nbody):
            child_count[int(model.body_parentid[b])] += 1
        for s, sn in enumerate(site_names):
            if _is_scene(sn):
                continue
            if child_count[int(model.site_bodyid[s])] == 0:
                ee_ids.append(s)
    # De-duplicate near-identical sites on same body: keep all named EE
    # sites but prefer end_effector/gripper over foot/wheel auxiliary sites.
    preferred = [s for s in ee_ids if any(
        t in site_names[s].lower() for t in ("end_effector", "gripper", "tool", "launcher", "scoop", "rake"))]
    if preferred:
        ee_ids = preferred
    ee_names = [site_names[s] for s in ee_ids]

    # --- locomotion (structural + generic anatomical tokens, no robot names)
    joint_low = [n.lower() for n in joint_names]
    wheel = False
    for j, jn in enumerate(joint_low):
        if "wheel" in jn and int(model.jnt_type[j]) == 3:
            wheel = True
    if not has_free:
        locomotion = "fixed_base"
    elif wheel:
        locomotion = "wheeled"
    else:
        hip_like = [n for n in joint_low if ("hip" in n and "arm" not in n)
                    or n.startswith("leg") or "_leg_" in n]
        # distinct leg groups: strip suffixes after hip/leg token
        groups = set()
        for n in hip_like:
            for sep in ("_hip", "_leg"):
                if sep in n:
                    groups.add(n.split(sep)[0])
                    break
            else:
                groups.add(n)
        if len(groups) >= 4 or len(hip_like) >= 8:
            locomotion = "quadruped"
        elif len(groups) >= 2 or len(hip_like) >= 2:
            locomotion = "biped"
        else:
            locomotion = "mobile"

    # --- reach: max(chain-length estimate, sampled FK reach)
    chain_reach = max((_chain_reach(model, s) for s in ee_ids), default=0.0)
    base_pos = np.zeros(3)
    if base_id >= 0:
        try:
            d0 = mujoco.MjData(model)
            mujoco.mj_forward(model, d0)
            base_pos = np.asarray(d0.xpos[base_id], dtype=float)
            sampled = _sampled_reach(model, ee_ids, base_pos)
            # sampled reach from base at home pose for fixed arms
            if not has_free and ee_ids:
                sampled_home = max(
                    float(np.linalg.norm(np.asarray(d0.site_xpos[s]) - base_pos))
                    for s in ee_ids)
                sampled = max(sampled, sampled_home)
        except Exception:
            sampled = 0.0
    else:
        sampled = 0.0
    max_reach = float(max(chain_reach, sampled))

    try:
        model_name = _name(model, "MODEL", 0) if False else str(getattr(model, "name", "") or "")
    except Exception:
        model_name = ""

    prof = BodyProfile(
        locomotion_type=locomotion, locomotion=locomotion,
        is_mobile=bool(has_free), has_free_base=bool(has_free),
        free_joint_count=int(free_count),
        base_body_name=base_name, base_body_id=int(base_id),
        n_dof=int(model.nv), nv=int(model.nv), nq=int(model.nq),
        n_joints=int(model.njnt), njnt=int(model.njnt),
        n_actuators=int(model.nu), nu=int(model.nu),
        n_bodies=int(model.nbody), n_geoms=int(model.ngeom),
        n_sites=int(model.nsite), n_cameras=int(model.ncam),
        joint_names=joint_names, joint_types=joint_types,
        joint_ranges=joint_ranges, limited_joints=limited, locked_joints=locked,
        actuator_names=actuator_names, actuator_limits=actuator_limits,
        actuator_ctrlrange=dict(actuator_limits),
        body_names=body_names, site_names=site_names,
        camera_names=camera_names, cameras=list(camera_names),
        end_effector_sites=ee_names, end_effector_site_ids=[int(s) for s in ee_ids],
        end_effectors=list(ee_names),
        max_reach=max_reach, reach=max_reach,
        timestep=float(model.opt.timestep),
        total_mass=float(np.sum(model.body_mass)) if model.nbody else 0.0,
        model_name=model_name,
    )
    return prof


# Aliases
inspect_model = inspect_body
introspect = inspect_body
introspect_body = inspect_body
get_body_profile = inspect_body
profile_body = inspect_body
analyze_model = inspect_body


class BodyInspector:
    """OO wrapper around :func:`inspect_body`."""
    def __init__(self, model=None):
        self.model = model
        self.profile = inspect_body(model) if model is not None else None

    def inspect(self, model=None) -> BodyProfile:
        if model is not None:
            self.model = model
        self.profile = inspect_body(self.model)
        return self.profile

    __call__ = inspect


Inspector = BodyInspector

# ---------------------------------------------------------------------------
# Body mutator: lock joints / scale links / disable actuators at runtime.
# Works on MJCF XML strings (returns new XML via mutate_model_xml) and on
# live MjModel objects (returns a mutated copy via a binary round-trip
# when possible, otherwise mutates a reload; original XML path preferred).
# ---------------------------------------------------------------------------

def _copy_model(model):
    """Deep-copy an MjModel via binary save/load round-trip."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".mjb", delete=False) as f:
            path = f.name
        mujoco.mj_saveModel(model, path)
        new = mujoco.MjModel.from_binary_path(path)
        Path(path).unlink(missing_ok=True)
        return new
    except Exception:
        return None


def _resolve_joint(model, joint):
    if joint is None:
        return None
    if isinstance(joint, (int, np.integer)):
        return int(joint)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(joint))
    if jid < 0:
        raise ValueError(f"Joint not found: {joint!r}")
    return int(jid)


def _resolve_body(model, body):
    if body is None:
        return None
    if isinstance(body, (int, np.integer)):
        return int(body)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(body))
    if bid < 0:
        raise ValueError(f"Body not found: {body!r}")
    return int(bid)


def lock_joint(model, joint, value: float = 0.0):
    """Lock one joint on a (copied) model: zero range + disabled actuator."""
    jid = _resolve_joint(model, joint)
    model.jnt_limited[jid] = True
    model.jnt_range[jid] = [float(value), float(value)]
    try:
        model.dof_damping[model.jnt_dofadr[jid]] = max(
            float(model.dof_damping[model.jnt_dofadr[jid]]), 10.0)
    except Exception:
        pass
    for a in range(model.nu):
        if int(model.actuator_trnid[a][0]) == jid:
            model.actuator_ctrlrange[a] = [0.0, 0.0]
            try:
                model.actuator_gainprm[a][:] = 0.0
            except Exception:
                pass
    return model


def disable_actuator(model, actuator):
    if isinstance(actuator, (int, np.integer)):
        aid = int(actuator)
    else:
        aid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, str(actuator)))
        if aid < 0:
            raise ValueError(f"Actuator not found: {actuator!r}")
    model.actuator_ctrlrange[aid] = [0.0, 0.0]
    try:
        model.actuator_gainprm[aid][:] = 0.0
    except Exception:
        pass
    return model


def scale_link(model, body, factor: float = 1.0):
    """Scale a body/link: its offset from parent and its geom/site offsets."""
    bid = _resolve_body(model, body)
    model.body_pos[bid] = np.asarray(model.body_pos[bid]) * float(factor)
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) == bid:
            model.geom_pos[g] = np.asarray(model.geom_pos[g]) * float(factor)
            model.geom_size[g] = np.asarray(model.geom_size[g]) * float(factor)
    for s in range(model.nsite):
        if int(model.site_bodyid[s]) == bid:
            model.site_pos[s] = np.asarray(model.site_pos[s]) * float(factor)
    # scale children offsets so the chain stretches with the link
    for b in range(1, model.nbody):
        if int(model.body_parentid[b]) == bid:
            model.body_pos[b] = np.asarray(model.body_pos[b]) * float(factor)
    return model


# scale_link_length alias semantics
scale_link_length = scale_link
scale_body = scale_link


def mutate_model_xml(xml: str, lock_joint=None, locked_joints=None,
                     lock_joints=None, scale_link=None, scale_factor: float = 1.0,
                     scale_body_name=None, disable_actuator_name=None,
                     disabled_actuators=None) -> str:
    """Return new MJCF XML with the requested mutations applied."""
    root = ET.fromstring(xml)
    locks = []
    for v in (lock_joint, locked_joints, lock_joints):
        if v is None:
            continue
        locks.extend(list(v) if isinstance(v, (list, tuple, set)) else [v])
    scale_target = scale_link if scale_link is not None else scale_body_name

    joint_elems = {j.get("name"): j for j in root.iter("joint")}
    # freejoint elements have no range; locking them is a no-op in XML
    for name in locks:
        el = joint_elems.get(str(name))
        if el is not None:
            el.set("range", "0 0.000001")
            el.set("limited", "true")
            el.set("damping", "10")
    if scale_target is not None:
        for body in root.iter("body"):
            if body.get("name") == str(scale_target):
                pos = body.get("pos")
                if pos:
                    vals = [float(x) * float(scale_factor) for x in pos.split()]
                    body.set("pos", " ".join(repr(v) for v in vals))
                for child in list(body):
                    if child.tag == "body" and child.get("pos"):
                        vals = [float(x) * float(scale_factor)
                                for x in child.get("pos").split()]
                        child.set("pos", " ".join(repr(v) for v in vals))
    disables = []
    for v in (disable_actuator_name, disabled_actuators):
        if v is None:
            continue
        disables.extend(list(v) if isinstance(v, (list, tuple, set)) else [v])
    if disables:
        for act in root.iter("actuator"):
            for child in list(act):
                if child.get("name") in [str(d) for d in disables]:
                    child.set("ctrlrange", "0 0")
    return ET.tostring(root, encoding="unicode")


def mutate_body(source, lock_joint=None, locked_joints=None, lock_joints=None,
                scale_link=None, scale_factor: float = 1.0, scale=None,
                scale_body_name=None, disable_actuator=None,
                disabled_actuator=None, disabled_actuators=None,
                lock_value: float = 0.0, locked_value: float = 0.0, **kw):
    """Mutate a body and return a new ``mujoco.MjModel``.

    ``source`` may be an MJCF XML string, a path, or an ``MjModel``.
    Keyword aliases accepted: ``lock_joints``, ``scale`` (factor),
    ``joint_name``/``joint`` for the joint to lock.
    """
    if "joint" in kw and lock_joint is None:
        lock_joint = kw.pop("joint")
    if "joint_name" in kw and lock_joint is None:
        lock_joint = kw.pop("joint_name")
    if "body" in kw and scale_link is None:
        scale_link = kw.pop("body")
    if "body_name" in kw and scale_link is None:
        scale_link = kw.pop("body_name")
    if scale is not None:
        scale_factor = float(scale)
    if "factor" in kw:
        scale_factor = float(kw.pop("factor"))
    value = locked_value if locked_value != 0.0 else lock_value

    locks = []
    for v in (lock_joint, locked_joints, lock_joints):
        if v is None:
            continue
        locks.extend(list(v) if isinstance(v, (list, tuple, set)) else [v])
    disables = []
    for v in (disable_actuator, disabled_actuator, disabled_actuators):
        if v is None:
            continue
        disables.extend(list(v) if isinstance(v, (list, tuple, set)) else [v])
    scale_target = scale_link if scale_link is not None else scale_body_name

    # XML / path source: edit XML text so the mutation is structural.
    if isinstance(source, (str, Path)):
        text = str(source)
        if isinstance(source, Path) or (len(text) < 260 and Path(text).exists()):
            text = Path(text).read_text()
        if "<mujoco" in text:
            new_xml = mutate_model_xml(
                text, locked_joints=locks or None, scale_link=scale_target,
                scale_factor=scale_factor,
                disabled_actuators=disables or None)
            return mujoco.MjModel.from_xml_string(new_xml)
        raise ValueError("mutate_body: string source must be MJCF XML or a path")

    # MjModel source: copy then mutate fields in place on the copy.
    model = _copy_model(source)
    if model is None:
        model = source  # fall back to in-place (documented)
    for j in locks:
        lock_joint_fn(model, j, value=value)
    if scale_target is not None and float(scale_factor) != 1.0:
        scale_link_fn(model, scale_target, float(scale_factor))
    for a in disables:
        disable_actuator_fn(model, a)
    return model


# internal aliases (parameter names shadow these inside mutate_body)
lock_joint_fn = lock_joint
scale_link_fn = scale_link
disable_actuator_fn = disable_actuator
mutate_model = mutate_body
apply_mutation = mutate_body
mutate = mutate_body

__all__ = ["BodyProfile", "BodyInspector", "Inspector", "inspect_body",
           "inspect_model", "introspect", "introspect_body", "get_body_profile",
           "profile_body", "analyze_model", "mutate_body", "mutate_model",
           "apply_mutation", "mutate", "mutate_model_xml", "lock_joint",
           "lock_joint_fn", "disable_actuator", "scale_link",
           "scale_link_length", "scale_body"]
