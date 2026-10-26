"""Saved Body Trunk: 1-time Check-Up Procedure + cached Locked Trunk + hot-swap data.

Runs once per MuJoCo morphology: probes every actuator's Jacobian effect on
end-effector motion, CoM and base velocity, classifies actuators into
TRUNK:POSTURE / TRUNK:LOCOMOTION / TRUNK:MANIPULATION, and caches the result
to ``data/body_trunks/<morphology>.json``. The Rust bridge then binds the
cached trunk in <0.5 ms without touching upper System-2 task branches.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from src.physics.inspector import inspect_body

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = _ROOT / "data" / "body_trunks"
TRUNK_NAMES = ("TRUNK:POSTURE", "TRUNK:LOCOMOTION", "TRUNK:MANIPULATION")
_LOCO = ("leg", "hip", "knee", "ankle", "wheel", "foot", "flipper")
_MANIP = ("arm", "shoulder", "elbow", "wrist", "gripper", "grip", "hand",
          "finger", "tool", "launcher", "scoop", "rake", "catapult", "manip")
_POST = ("torso", "neck", "head", "pelvis", "trunk", "base", "spine", "chest")


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip()) or "custom"


def cache_path(morphology: str, cache_dir: Path | str | None = None) -> Path:
    return Path(cache_dir or DEFAULT_CACHE_DIR) / f"{_safe(morphology)}.json"


def model_checksum(model) -> str:
    h = hashlib.sha256()
    h.update(f"{model.nq},{model.nv},{model.nu},{model.njnt};".encode())
    for j in range(model.njnt):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"j{j}"
        h.update(f"{n}:{model.jnt_type[j]}:{model.jnt_range[j][0]:.6f}:{model.jnt_range[j][1]:.6f};".encode())
    for a in range(model.nu):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"a{a}"
        h.update(f"{n}:{model.actuator_ctrlrange[a][0]:.6f}:{model.actuator_ctrlrange[a][1]:.6f};".encode())
    return h.hexdigest()[:16]


@dataclass
class LockedBodyTrunk:
    morphology: str = "custom"
    version: int = 2
    nu: int = 0
    nv: int = 0
    nq: int = 0
    actuator_names: list[str] = field(default_factory=list)
    joint_names: list[str] = field(default_factory=list)
    ctrl_lo: list[float] = field(default_factory=list)
    ctrl_hi: list[float] = field(default_factory=list)
    locked_mask: int = 0
    locked_actuators: list[int] = field(default_factory=list)
    locked_joints: list[str] = field(default_factory=list)
    # Per-actuator EE linear Jacobian row (3 floats each), flattened nu*3.
    jacobian: list[float] = field(default_factory=list)
    jac_ee: list[list[float]] = field(default_factory=list)
    com_effect: list[list[float]] = field(default_factory=list)
    base_vel_effect: list[float] = field(default_factory=list)
    ee_effect_norm: list[float] = field(default_factory=list)
    trunk_containers: dict[str, Any] = field(default_factory=dict)
    locomotion_type: str = "fixed_base"
    end_effector_sites: list[str] = field(default_factory=list)
    # v3 fields
    q_nom: list[float] = field(default_factory=list)
    q_nom_full: list[float] = field(default_factory=list)
    actuator_dof: list[int] = field(default_factory=list)
    actuator_qposadr: list[int] = field(default_factory=list)
    is_torque: list[int] = field(default_factory=list)
    chain_containers: dict[str, Any] = field(default_factory=dict)
    ee_body: str = ""
    ee_body_id: int = -1
    max_reach: float = 0.0
    total_mass: float = 0.0
    checksum: str = ""
    checkup_ms: float = 0.0
    # v3 name-free topological & relative-Jacobian grasp discovery
    feet_body_ids: list[int] = field(default_factory=list)
    hand_palm_id: int = -1
    hand_leaf_ids: list[int] = field(default_factory=list)
    grasp_dir: list[float] = field(default_factory=list)
    q_open: list[float] = field(default_factory=list)
    q_close: list[float] = field(default_factory=list)
    is_finger: list[int] = field(default_factory=list)
    cage_local: list[float] = field(default_factory=list)
    # Argon Active-Gaze discovery: root->palm pitch actuators (Jacobian-derived)
    gaze_pitch_actuators: list[int] = field(default_factory=list)
    gaze_pitch_signs: list[float] = field(default_factory=list)
    topology_version: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LockedBodyTrunk":
        keep = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**keep)

    def jacobian_array(self) -> np.ndarray:
        return np.asarray(self.jacobian, dtype=np.float32).reshape(self.nu, 3) if self.nu else np.zeros((0, 3), np.float32)


def _classify(actuator: str, joint: str) -> str:
    s = f"{actuator} {joint}".lower()
    if any(t in s for t in _MANIP):
        return "TRUNK:MANIPULATION"
    if any(t in s for t in _LOCO):
        return "TRUNK:LOCOMOTION"
    if any(t in s for t in _POST):
        return "TRUNK:POSTURE"
    return "TRUNK:POSTURE"


def _chain_for(name: str) -> str:
    n = str(name).lower()
    left = "left" in n or n.startswith("fl") or "_l_" in n or "l_" in n
    right = "right" in n or n.startswith("fr") or "_r_" in n or "r_" in n
    hind = n.startswith(("rl", "rr", "hl", "hr"))
    if any(t in n for t in ("arm", "shoulder", "elbow", "wrist", "hand", "gripper", "finger", "actuator")):
        return "Left_Arm" if left else "Right_Arm" if right else "Arm"
    if any(t in n for t in ("leg", "hip", "knee", "ankle", "thigh", "calf", "foot", "hx", "hy", "kn")):
        if hind and left: return "Hind_Left_Leg"
        if hind and right: return "Hind_Right_Leg"
        return "Left_Leg" if left else "Right_Leg" if right else "Leg"
    if any(t in n for t in ("waist", "torso", "spine", "chest")): return "Torso"
    if any(t in n for t in ("neck", "head")): return "Head"
    return "Body"


def _pick_ee_body(model, prof):
    names = list(prof.body_names)
    skip = ("target", "cube", "bin", "ball", "floor", "obstacle")
    cand = [(i, n) for i, n in enumerate(names) if i > 0 and not any(s in n.lower() for s in skip)]
    def score(item):
        i, n = item; low = n.lower(); s = 0
        if "hand" in low or "gripper" in low: s = 100
        elif "wrist" in low: s = 90 + (10 if "yaw" in low else 0)
        elif "elbow" in low: s = 80
        elif low.startswith(("fl", "front")) and ("calf" in low or "lleg" in low or "foot" in low): s = 60
        elif "foot" in low: s = 50
        elif "calf" in low or "lleg" in low: s = 40
        return s
    best = max(cand, key=score, default=(-1, ""))
    if score(best) > 0: return best
    child = np.zeros(model.nbody, dtype=int)
    for b in range(1, model.nbody): child[int(model.body_parentid[b])] += 1
    leaves = [b for b, n in cand if child[b] == 0]
    return (leaves[-1], names[leaves[-1]]) if leaves else (-1, "")


def discover_topology_and_grasp(model) -> dict[str, Any]:
    """100% name-free topological & relative-Jacobian discovery of feet, hands,
    pinch-closure directions (`grasp_dir`), `q_open`, `q_close`, `is_finger`, and `cage_local`.
    Uses ONLY `model.body_parentid`, `data.xpos`, `data.geom_xpos`, `mj_jac`, and `actuator_ctrlrange`.
    """
    nu, nv = int(model.nu), int(model.nv)
    if nu == 0 or nv == 0:
        return {
            "feet_body_ids": [], "hand_palm_id": -1, "hand_leaf_ids": [],
            "grasp_dir": [], "q_open": [], "q_close": [], "is_finger": [],
            "cage_local": [0.0, 0.0, 0.0],
            "gaze_pitch_actuators": [], "gaze_pitch_signs": [],
        }
    data = mujoco.MjData(model)
    if int(model.nkey) > 0:
        data.qpos[:] = np.array(model.key_qpos[0], dtype=float)
    mujoco.mj_forward(model, data)

    # Identify all bodies belonging to the actuated robot tree (excludes static/free environment props)
    act_bodies: set[int] = set()
    act_dofs: list[int] = []
    act_qadrs: list[int] = []
    for a in range(nu):
        jid = int(model.actuator_trnid[a][0])
        if 0 <= jid < model.njnt:
            b = int(model.jnt_bodyid[jid])
            while b > 0:
                act_bodies.add(b)
                b = int(model.body_parentid[b])
            act_dofs.append(int(model.jnt_dofadr[jid]))
            act_qadrs.append(int(model.jnt_qposadr[jid]))
        else:
            act_dofs.append(-1)
            act_qadrs.append(-1)
    # Include unactuated child links attached to actuated bodies (e.g. fingertip links / end-effector links)
    changed = True
    while changed:
        changed = False
        for b in range(1, int(model.nbody)):
            p = int(model.body_parentid[b])
            if p in act_bodies and b not in act_bodies:
                # Only include if attached without a 6-DOF freejoint
                jnum = int(model.body_jntnum[b])
                jadr = int(model.body_jntadr[b])
                is_free = jnum > 0 and int(model.jnt_type[jadr]) == int(mujoco.mjtJoint.mjJNT_FREE)
                if not is_free:
                    act_bodies.add(b)
                    changed = True

    children: dict[int, list[int]] = {b: [] for b in act_bodies}
    for b in act_bodies:
        p = int(model.body_parentid[b])
        if p in act_bodies:
            children[p].append(b)

    leaves = sorted([b for b in act_bodies if len(children[b]) == 0])
    # Ground-contacting leaves (z < 0.15m) when robot has multiple limbs on the ground = feet
    low_leaves = [b for b in leaves if float(data.xpos[b][2]) < 0.15]
    feet = sorted(low_leaves, key=lambda b: -float(data.xpos[b][1])) if len(low_leaves) >= 2 else []
    upper_leaves = [b for b in leaves if b not in feet]

    # Find branching palm hubs: lowest common branching ancestor of >= 2 upper leaves
    def ancestors(b: int) -> list[int]:
        out = []
        cur = b
        while cur in act_bodies:
            out.append(cur)
            cur = int(model.body_parentid[cur])
        return out

    leaf_anc = {lf: ancestors(lf) for lf in upper_leaves}
    branch_hubs: dict[int, list[int]] = {}
    for b in act_bodies:
        if len(children[b]) >= 2:
            desc_leaves = [lf for lf in upper_leaves if b in leaf_anc[lf] and lf != b]
            # Only keep lowest branching hub (no child of b also branches to >=2 upper leaves)
            child_hubs = [
                c for c in children[b]
                if sum(1 for lf in upper_leaves if c in leaf_anc[lf]) >= 2
            ]
            if len(desc_leaves) >= 2 and not child_hubs:
                branch_hubs[b] = desc_leaves

    hand_palm_id = -1
    hand_leaf_ids: list[int] = []
    if branch_hubs:
        # Pick the hand with more negative Y (right hand in standard forward-X frame)
        hand_palm_id = min(branch_hubs.keys(), key=lambda b: float(data.xpos[b][1]))
        hand_leaf_ids = sorted(branch_hubs[hand_palm_id])
    elif upper_leaves:
        hand_leaf_ids = [upper_leaves[-1]]
        hand_palm_id = int(model.body_parentid[hand_leaf_ids[0]])

    grasp_dir = np.zeros(nu, dtype=float)
    q_open = np.array([float(model.actuator_ctrlrange[a][0]) for a in range(nu)], dtype=float)
    q_close = np.array([float(model.actuator_ctrlrange[a][1]) for a in range(nu)], dtype=float)
    is_finger = [0] * nu
    cage_local = [0.0, 0.0, 0.0]

    if hand_palm_id > 0 and len(hand_leaf_ids) >= 2:
        # Evaluate fingertip geoms and relative Jacobians at geometric zero (all actuated hinge joints = 0)
        data0 = mujoco.MjData(model)
        for qa in act_qadrs:
            if 0 <= qa < int(model.nq):
                data0.qpos[qa] = 0.0
        mujoco.mj_forward(model, data0)
        p_palm0 = data0.xpos[hand_palm_id].copy()
        R_palm0 = data0.xmat[hand_palm_id].reshape(3, 3).copy()

        def tip_world(d, tb: int) -> np.ndarray:
            g_ids = [g for g in range(int(model.ngeom)) if int(model.geom_bodyid[g]) == tb]
            if not g_ids:
                return d.xpos[tb].copy()
            return max((d.geom_xpos[g].copy() for g in g_ids),
                       key=lambda x: float(np.linalg.norm(x - d.xpos[hand_palm_id])))

        tip_pts0 = [tip_world(data0, tb) for tb in hand_leaf_ids]
        tip_local0 = np.array([R_palm0.T @ (pt - p_palm0) for pt in tip_pts0])
        c_mid = np.mean(tip_local0, axis=0)

        for k, tb in enumerate(hand_leaf_ids):
            pt = tip_pts0[k]
            jp_t = np.zeros((3, nv)); jr_t = np.zeros((3, nv))
            jp_p = np.zeros((3, nv)); jr_p = np.zeros((3, nv))
            mujoco.mj_jac(model, data0, jp_t, jr_t, pt, tb)
            mujoco.mj_jac(model, data0, jp_p, jr_p, pt, hand_palm_id)
            valid_dofs = [d if 0 <= d < nv else 0 for d in act_dofs]
            j_rel = R_palm0.T @ (jp_t[:, valid_dofs] - jp_p[:, valid_dofs])
            inward = c_mid - tip_local0[k]
            inward /= (np.linalg.norm(inward) + 1e-9)
            grasp_dir += j_rel.T @ inward

        if len(hand_leaf_ids) >= 5:
            # 5-finger high-DOF anthropomorphic hand: lock orthogonal abduction joint in
            # opposition plane and solve flexion joints for enclosing fingertip opposition
            hand_acts: list[int] = []
            for a in range(nu):
                jid = int(model.actuator_trnid[a][0])
                if 0 <= jid < model.njnt:
                    jb = int(model.jnt_bodyid[jid])
                    if jb != hand_palm_id and hand_palm_id in ancestors(jb):
                        hand_acts.append(a)
            thumb_leaves = {
                hand_leaf_ids[k]
                for k in range(len(hand_leaf_ids))
                if tip_local0[k][0] < c_mid[0]
            }
            for a in hand_acts:
                is_finger[a] = 1
                lo, hi = float(model.actuator_ctrlrange[a][0]), float(model.actuator_ctrlrange[a][1])
                jid = int(model.actuator_trnid[a][0])
                jb = int(model.jnt_bodyid[jid])
                child_jnts = [
                    j2
                    for j2 in range(int(model.njnt))
                    if int(model.body_parentid[int(model.jnt_bodyid[j2])]) == jb
                ]
                is_ortho_abd = bool(
                    child_jnts
                    and abs(float(np.dot(model.jnt_axis[jid], model.jnt_axis[child_jnts[0]]))) < 0.2
                )
                in_thumb = any(jb == tb or jb in ancestors(tb) for tb in thumb_leaves)
                if is_ortho_abd:
                    q_open[a] = lo + 0.85 * (hi - lo)
                    q_close[a] = lo + 0.85 * (hi - lo)
                elif in_thumb:
                    q_open[a] = lo if abs(lo) < abs(hi) else 0.0
                    q_close[a] = lo + 0.78 * (hi - lo)
                else:
                    q_open[a] = lo if abs(lo) < abs(hi) else 0.0
                    q_close[a] = lo + 0.72 * (hi - lo)
                grasp_dir[a] = float(q_close[a] - q_open[a])
        else:
            for a in range(nu):
                lo, hi = float(model.actuator_ctrlrange[a][0]), float(model.actuator_ctrlrange[a][1])
                gd = float(grasp_dir[a])
                if abs(gd) > 0.005:
                    is_finger[a] = 1
                    # If actuator drives a high-range tendon/position signal (e.g. 0..255 on Franka), use full range
                    if hi - lo > 10.0:
                        q_open[a] = hi if gd < 0 else lo
                        q_close[a] = lo if gd < 0 else hi
                    else:
                        frac = 0.992 if abs(gd) >= 0.03 else 0.974
                        if gd > 0:
                            q_open[a] = lo
                            q_close[a] = frac * hi
                        else:
                            q_open[a] = 0.994 * hi
                            q_close[a] = frac * lo

        # Compute closed-pinch midpoint (`cage_local`) in the palm frame via forward kinematics at q_close
        data_c = mujoco.MjData(model)
        for qa in act_qadrs:
            if 0 <= qa < int(model.nq):
                data_c.qpos[qa] = 0.0
        for a in range(nu):
            if is_finger[a] and 0 <= act_qadrs[a] < int(model.nq):
                lo, hi = float(model.actuator_ctrlrange[a][0]), float(model.actuator_ctrlrange[a][1])
                if hi - lo <= 10.0:
                    data_c.qpos[act_qadrs[a]] = q_close[a]
        mujoco.mj_forward(model, data_c)
        p_palmc = data_c.xpos[hand_palm_id].copy()
        R_palmc = data_c.xmat[hand_palm_id].reshape(3, 3).copy()
        if len(hand_leaf_ids) >= 5:
            for a in range(nu):
                if is_finger[a] and 0 <= act_qadrs[a] < int(model.nq):
                    data_c.qpos[act_qadrs[a]] = 0.45 * q_open[a] + 0.55 * q_close[a]
            mujoco.mj_forward(model, data_c)
            p_palmc = data_c.xpos[hand_palm_id].copy()
            R_palmc = data_c.xmat[hand_palm_id].reshape(3, 3).copy()
            closed_tips = [R_palmc.T @ (data_c.xpos[tb] - p_palmc) for tb in hand_leaf_ids]
            mid = 0.5 * (closed_tips[-1] + 0.5 * (closed_tips[0] + closed_tips[1]))
            cage_local = [float(mid[0]), float(mid[1]), 0.0]
        else:
            closed_tips = [R_palmc.T @ (data_c.xpos[tb] - p_palmc) for tb in hand_leaf_ids]
            pos_y = [closed_tips[k] for k in range(len(hand_leaf_ids)) if tip_local0[k][1] > c_mid[1]]
            neg_y = [closed_tips[k] for k in range(len(hand_leaf_ids)) if tip_local0[k][1] <= c_mid[1]]
            if len(pos_y) > 0 and len(neg_y) > 0:
                mid = 0.5 * (np.mean(pos_y, axis=0) + np.mean(neg_y, axis=0))
            else:
                mid = np.mean(closed_tips, axis=0)
            cage_local = [float(mid[0]), float(mid[1]), 0.0]

    # Active-Gaze pitch actuators: actuated joints on the root->palm chain whose
    # palm Jacobian is pitch-dominant and carries the palm forward-down.
    gaze_acts: list[int] = []
    gaze_signs: list[float] = []
    if hand_palm_id > 0:
        chain_bodies = set(ancestors(hand_palm_id))
        p_pt = data.xpos[hand_palm_id].copy()
        jp_g = np.zeros((3, nv)); jr_g = np.zeros((3, nv))
        mujoco.mj_jac(model, data, jp_g, jr_g, p_pt, hand_palm_id)
        for a in range(nu):
            jid = int(model.actuator_trnid[a][0])
            if not (0 <= jid < model.njnt):
                continue
            if int(model.jnt_bodyid[jid]) not in chain_bodies or is_finger[a]:
                continue
            dof = act_dofs[a]
            if not (0 <= dof < nv):
                continue
            ang = jr_g[:, dof]; lin = jp_g[:, dof]
            an = float(np.linalg.norm(ang))
            if an < 0.25 or abs(ang[1]) / (an + 1e-9) < 0.55:
                continue
            fwd_down = float(lin[0] - 0.45 * lin[2])
            if abs(fwd_down) > 0.04:
                gaze_acts.append(int(a))
                gaze_signs.append(1.0 if fwd_down >= 0 else -1.0)

    return {
        "gaze_pitch_actuators": gaze_acts,
        "gaze_pitch_signs": gaze_signs,
        "feet_body_ids": [int(x) for x in feet],
        "hand_palm_id": int(hand_palm_id),
        "hand_leaf_ids": [int(x) for x in hand_leaf_ids],
        "grasp_dir": [float(x) for x in grasp_dir],
        "q_open": [float(x) for x in q_open],
        "q_close": [float(x) for x in q_close],
        "is_finger": [int(x) for x in is_finger],
        "cage_local": cage_local,
    }


def checkup_body(model, morphology: str = "custom") -> LockedBodyTrunk:
    """One-time Check-Up: Jacobian-probe every actuator of ``model``."""
    t0 = time.perf_counter()
    prof = inspect_body(model)
    data = mujoco.MjData(model)
    if int(model.nkey) > 0:
        data.qpos[:] = np.array(model.key_qpos[0], dtype=float)
    mujoco.mj_forward(model, data)
    nv, nu = int(model.nv), int(model.nu)
    ee_body_id, ee_body_name = _pick_ee_body(model, prof)

    jac_ee = np.zeros((nu, 3)); com_eff = np.zeros((nu, 3)); base_norm = np.zeros(nu)
    _bad_site = ("foot", "imu", "target", "cube", "bin", "ball")
    ee_site_ids = [s for s in prof.end_effector_site_ids
                   if not any(t in prof.site_names[s].lower() for t in _bad_site)] \
        if prof.end_effector_site_ids else []
    # v3: EE Jacobian from (non-foot) site if available else EE body
    if nv and (ee_site_ids or ee_body_id >= 0):
        jacp = np.zeros((3, nv)); jacr = np.zeros((3, nv))
        if ee_site_ids:
            mujoco.mj_jacSite(model, data, jacp, jacr, int(ee_site_ids[0]))
        else:
            mujoco.mj_jacBody(model, data, jacp, jacr, int(ee_body_id))
        jac_com = np.zeros((3, nv))
        try: mujoco.mj_jacSubtreeCom(model, data, jac_com, max(int(prof.base_body_id), 1))
        except Exception: jac_com[:] = 0.0
        for a in range(nu):
            jid = int(model.actuator_trnid[a][0]); dof = -1
            if 0 <= jid < model.njnt: dof = int(model.jnt_dofadr[jid])
            if 0 <= dof < nv:
                jac_ee[a] = jacp[:, dof]; com_eff[a] = jac_com[:, dof]
                base_norm[a] = float(np.linalg.norm(jac_com[:, dof]))
    if False and prof.end_effector_site_ids and nv:
        jacp = np.zeros((3, nv)); jacr = np.zeros((3, nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, int(prof.end_effector_site_ids[0]))
        jac_com = np.zeros((3, nv))
        try:
            mujoco.mj_jacSubtreeCom(model, data, jac_com, max(int(prof.base_body_id), 1))
        except Exception:
            jac_com[:] = 0.0
        for a in range(nu):
            jid = int(model.actuator_trnid[a][0])
            # Transmission may target a joint or a site/tendon; guard range.
            dof = -1
            if 0 <= jid < model.njnt:
                dof = int(model.jnt_dofadr[jid])
            if 0 <= dof < nv:
                jac_ee[a] = jacp[:, dof]
                com_eff[a] = jac_com[:, dof]
                base_norm[a] = float(np.linalg.norm(jac_com[:, dof]))

    act_names = list(prof.actuator_names)
    # Joint name behind each actuator (for classification).
    joint_for_act: list[str] = []
    for a in range(nu):
        jid = int(model.actuator_trnid[a][0])
        if 0 <= jid < model.njnt:
            joint_for_act.append(prof.joint_names[jid])
        else:
            joint_for_act.append(act_names[a] if a < len(act_names) else f"actuator_{a}")

    locked: list[int] = []; mask = 0
    ctrl_lo: list[float] = []; ctrl_hi: list[float] = []
    for a in range(nu):
        lo, hi = float(model.actuator_ctrlrange[a][0]), float(model.actuator_ctrlrange[a][1])
        ctrl_lo.append(lo); ctrl_hi.append(hi)
        jname = joint_for_act[a]
        if hi - lo < 1e-9 or jname in prof.locked_joints:
            locked.append(a); mask |= (1 << a)

    containers: dict[str, Any] = {n: {"actuators": [], "actuator_names": []} for n in TRUNK_NAMES}
    for a in range(nu):
        c = _classify(act_names[a] if a < len(act_names) else "", joint_for_act[a])
        containers[c]["actuators"].append(a)
        containers[c]["actuator_names"].append(act_names[a] if a < len(act_names) else f"actuator_{a}")
    # v3: nominal posture, dof maps, torque flags, kinematic chains
    q_nom_full = [float(x) for x in np.array(data.qpos, dtype=float)]
    q_nom: list[float] = []; dofs: list[int] = []; qadr: list[int] = []; torque: list[int] = []
    chains: dict[str, Any] = {}
    for a in range(nu):
        jid = int(model.actuator_trnid[a][0])
        dof = int(model.jnt_dofadr[jid]) if 0 <= jid < model.njnt else -1
        qa = int(model.jnt_qposadr[jid]) if 0 <= jid < model.njnt else -1
        dofs.append(dof); qadr.append(qa)
        q_nom.append(float(data.qpos[qa]) if 0 <= qa < len(q_nom_full) else 0.0)
        gain = float(model.actuator_gainprm[a][0]) if nu else 0.0
        bias_norm = float(np.linalg.norm(model.actuator_biasprm[a])) if nu else 0.0
        torque.append(1 if (abs(gain - 1.0) < 0.01 and bias_norm < 1e-9) else 0)
        ch = _chain_for(act_names[a] if a < len(act_names) else f"actuator_{a}")
        chains.setdefault(ch, {"actuators": [], "actuator_names": []})
        chains[ch]["actuators"].append(a)
        chains[ch]["actuator_names"].append(act_names[a] if a < len(act_names) else f"actuator_{a}")
    topo = discover_topology_and_grasp(model)
    # Fixed-base bodies have no locomotion: fold empty note, keep all 3 keys.
    trunk = LockedBodyTrunk(
        morphology=_safe(morphology), nu=nu, nv=nv, nq=int(model.nq),
        actuator_names=act_names, joint_names=list(prof.joint_names),
        ctrl_lo=ctrl_lo, ctrl_hi=ctrl_hi, locked_mask=int(mask),
        locked_actuators=locked, locked_joints=list(prof.locked_joints),
        jacobian=[float(x) for x in jac_ee.reshape(-1)],
        jac_ee=[[float(x) for x in row] for row in jac_ee],
        com_effect=[[float(x) for x in row] for row in com_eff],
        base_vel_effect=[float(x) for x in base_norm],
        ee_effect_norm=[float(np.linalg.norm(r)) for r in jac_ee],
        trunk_containers=containers, locomotion_type=prof.locomotion_type,
        end_effector_sites=[prof.site_names[s] for s in ee_site_ids],
        q_nom=q_nom, q_nom_full=q_nom_full, actuator_dof=dofs,
        actuator_qposadr=qadr, is_torque=torque, chain_containers=chains,
        ee_body=ee_body_name, ee_body_id=int(ee_body_id),
        max_reach=float(prof.max_reach),
        total_mass=float(prof.total_mass), checksum=model_checksum(model),
        checkup_ms=(time.perf_counter() - t0) * 1000.0,
        feet_body_ids=topo["feet_body_ids"],
        hand_palm_id=topo["hand_palm_id"],
        hand_leaf_ids=topo["hand_leaf_ids"],
        grasp_dir=topo["grasp_dir"],
        q_open=topo["q_open"],
        q_close=topo["q_close"],
        is_finger=topo["is_finger"],
        cage_local=topo["cage_local"],
        gaze_pitch_actuators=topo.get("gaze_pitch_actuators", []),
        gaze_pitch_signs=topo.get("gaze_pitch_signs", []),
        topology_version=3)
    return trunk


# Friendly aliases
run_checkup = checkup_body
build_body_trunk = checkup_body
probe_body = checkup_body


def save_trunk(trunk: LockedBodyTrunk, cache_dir: Path | str | None = None) -> Path:
    p = cache_path(trunk.morphology, cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(trunk.to_dict(), indent=1))
    return p


def load_trunk(morphology: str, cache_dir: Path | str | None = None) -> LockedBodyTrunk | None:
    p = cache_path(morphology, cache_dir)
    if not p.exists():
        return None
    try:
        return LockedBodyTrunk.from_dict(json.loads(p.read_text()))
    except Exception:
        return None


def get_or_build_trunk(model, morphology: str = "custom",
                       cache_dir: Path | str | None = None, force: bool = False) -> LockedBodyTrunk:
    """Return cached trunk when the model checksum matches, else run Check-Up."""
    if not force:
        cached = load_trunk(morphology, cache_dir)
        if cached is not None and cached.checksum == model_checksum(model) and cached.nu == int(model.nu) and cached.q_nom and cached.is_finger and cached.topology_version >= 3:
            return cached
    trunk = checkup_body(model, morphology)
    save_trunk(trunk, cache_dir)
    return trunk


__all__ = ["LockedBodyTrunk", "TRUNK_NAMES", "DEFAULT_CACHE_DIR", "cache_path",
           "model_checksum", "checkup_body", "run_checkup", "build_body_trunk",
           "probe_body", "save_trunk", "load_trunk", "get_or_build_trunk"]
