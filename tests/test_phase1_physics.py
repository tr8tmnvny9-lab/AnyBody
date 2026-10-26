"""Phase 1: MuJoCo multi-morphology arena & blind introspection tests."""
from __future__ import annotations

import json
import time

import mujoco
import numpy as np
import pytest

from src.physics.arena import Arena, build_arena
from src.physics.inspector import (BodyProfile, inspect_body, inspect_model,
                                    introspect, mutate_body, mutate_model_xml)
from src.physics.morphologies import (MORPHOLOGY_NAMES, build_mjcf,
                                       create_model, get_module)

EXPECTED_LOCOMOTION = {
    "humanoid": "biped",
    "quadruped_arm": "quadruped",
    "starwars_droid": "wheeled",
    "table_arm": "fixed_base",
}
ROOT_BODY = {"humanoid": "pelvis", "quadruped_arm": "trunk",
             "starwars_droid": "droid_body", "table_arm": "arm_base"}


def _model(name):
    return create_model(name)


def _step(model, data, n=200):
    data.ctrl[:] = 0.0
    for _ in range(n):
        mujoco.mj_step(model, data)
    return data


# ---------------------------------------------------------------- env / load
def test_required_packages_importable():
    import fastapi, httpx, pydantic, scipy, uvicorn  # noqa: F401
    from PIL import Image  # noqa: F401
    assert mujoco.__version__
    assert np.__version__


def test_morphology_registry_names():
    assert MORPHOLOGY_NAMES == ["humanoid", "quadruped_arm",
                                "starwars_droid", "table_arm"]
    for alias in ("quadruped", "droid", "table"):
        assert get_module(alias) is not None
    with pytest.raises(ValueError):
        get_module("not_a_robot")


@pytest.mark.parametrize("name", MORPHOLOGY_NAMES)
def test_morphology_module_api(name):
    mod = get_module(name)
    xml = mod.build_mjcf()
    assert isinstance(xml, str) and "<mujoco" in xml
    assert mod.get_mjcf() == xml
    assert mod.MJCF_XML == xml
    assert isinstance(mod.create_model(), mujoco.MjModel)
    assert isinstance(mod.build_model(), mujoco.MjModel)


@pytest.mark.parametrize("name", MORPHOLOGY_NAMES)
def test_all_morphologies_load(name):
    m = _model(name)
    assert m.njnt >= 5 and m.nv >= 5 and m.nu >= 4
    assert m.nbody > 5 and m.ngeom > 5 and m.nsite >= 3
    assert m.opt.timestep == pytest.approx(0.002)


@pytest.mark.parametrize("name", MORPHOLOGY_NAMES)
def test_shared_arena_contents(name):
    xml = build_mjcf(name)
    for token in ('name="ball"', 'name="person_target"', 'name="obstacle_1"',
                  'name="obstacle_2"', 'name="floor"', 'name="exo_cam"',
                  'name="ego_cam"'):
        assert token in xml, (name, token)
    m = _model(name)
    for obj, kind in (("ball", mujoco.mjtObj.mjOBJ_BODY),
                      ("person_target", mujoco.mjtObj.mjOBJ_BODY),
                      ("ball_site", mujoco.mjtObj.mjOBJ_SITE),
                      ("person_site", mujoco.mjtObj.mjOBJ_SITE)):
        assert mujoco.mj_name2id(m, kind, obj) >= 0


# ------------------------------------------------------------------- stepping
@pytest.mark.parametrize("name", MORPHOLOGY_NAMES)
def test_stepping_is_stable(name):
    m, d = _model(name), None
    d = mujoco.MjData(m)
    _step(m, d, 500)
    assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
    assert abs(d.time - 1.0) < 1e-6
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY[name])
    assert np.isfinite(d.xpos[bid]).all()
    if name != "table_arm":
        assert d.xpos[bid][2] > 0.25, f"{name} collapsed: {d.xpos[bid]}"
        assert np.linalg.norm(d.xpos[bid][:2]) < 1.0


@pytest.mark.parametrize("name", MORPHOLOGY_NAMES)
def test_step_rate_above_500hz(name):
    m = _model(name)
    assert 1.0 / m.opt.timestep >= 500.0
    d = mujoco.MjData(m)
    d.ctrl[:] = 0.0
    t0 = time.perf_counter()
    for _ in range(500):
        mujoco.mj_step(m, d)
    elapsed = time.perf_counter() - t0
    assert 500 / elapsed > 500.0, f"{name}: {500/elapsed:.0f} steps/s"


@pytest.mark.parametrize("name", MORPHOLOGY_NAMES)
def test_actuator_control_moves_a_joint(name):
    m = _model(name)
    d = mujoco.MjData(m)
    q0 = np.array(d.qpos, copy=True)
    d.ctrl[:] = 0.3
    for _ in range(50):
        mujoco.mj_step(m, d)
    assert np.isfinite(d.qpos).all()
    assert not np.allclose(d.qpos, q0) or not np.allclose(d.qvel, 0.0)


# ------------------------------------------------------------------ rendering
@pytest.mark.parametrize("name", MORPHOLOGY_NAMES)
@pytest.mark.parametrize("camera", ["exo_cam", "ego_cam"])
def test_render_rgb_and_depth(name, camera):
    m = _model(name)
    d = _step(m, mujoco.MjData(m), 20)
    r = mujoco.Renderer(m, 64, 64)
    try:
        r.update_scene(d, camera=camera)
        rgb = r.render()
        assert rgb.shape == (64, 64, 3) and rgb.dtype == np.uint8
        assert rgb.std() > 1.0  # non-blank frame
        r.enable_depth_rendering()
        r.update_scene(d, camera=camera)
        depth = r.render()
        assert depth.shape == (64, 64)
        assert np.isfinite(depth).all() and (depth > 0).any()
    finally:
        r.close()


# ------------------------------------------------------------------ inspector
@pytest.mark.parametrize("name", MORPHOLOGY_NAMES)
def test_blind_introspection_locomotion(name):
    p = inspect_body(_model(name))
    assert isinstance(p, BodyProfile)
    assert p.locomotion_type == EXPECTED_LOCOMOTION[name]
    assert p.locomotion == p.locomotion_type
    assert p.is_mobile == (name != "table_arm")
    assert p.has_free_base == p.is_mobile


def test_locomotion_types_are_all_distinct():
    types = {inspect_body(_model(n)).locomotion_type for n in MORPHOLOGY_NAMES}
    assert len(types) == 4


@pytest.mark.parametrize("name", MORPHOLOGY_NAMES)
def test_introspection_fields(name):
    m = _model(name)
    p = inspect_body(m)
    assert p.n_dof == m.nv and p.nv == m.nv and p.nq == m.nq
    assert p.n_joints == m.njnt and p.n_actuators == m.nu
    assert p.camera_names and "ego_cam" in p.camera_names
    assert "exo_cam" in p.camera_names and p.cameras == p.camera_names
    assert p.end_effector_sites, name
    assert len(p.end_effector_site_ids) == len(p.end_effector_sites)
    for sid, sname in zip(p.end_effector_site_ids, p.end_effector_sites):
        assert mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, sid) == sname
    assert 0.2 < p.max_reach < 2.5 and p.reach == p.max_reach
    assert p.timestep == pytest.approx(0.002)
    for jname, (lo, hi) in p.joint_ranges.items():
        assert lo <= hi and jname in p.joint_names
    for aname, (lo, hi) in p.actuator_limits.items():
        assert lo <= hi and aname in p.actuator_names


@pytest.mark.parametrize("name", MORPHOLOGY_NAMES)
def test_profile_json_roundtrip(name):
    p = inspect_body(_model(name))
    data = json.loads(p.to_json())
    assert data["locomotion_type"] == EXPECTED_LOCOMOTION[name]
    assert data["end_effector_sites"] == p.end_effector_sites
    assert p["locomotion_type"] == p.locomotion_type  # dict-style access
    assert json.loads(json.dumps(p.to_dict()))["max_reach"] == pytest.approx(p.max_reach)


def test_inspector_aliases_agree():
    m = _model("humanoid")
    assert inspect_model(m).locomotion_type == introspect(m).locomotion_type


def test_blind_introspection_on_unknown_robot():
    """A robot the inspector has never seen: no name hints about morphology."""
    xml = """<mujoco model="mystery_bot">
      <compiler angle="radian"/><option timestep="0.002"/>
      <worldbody>
        <geom name="ground" type="plane" size="3 3 0.1"/>
        <camera name="exo_cam" pos="2 -2 1.5" xyaxes="1 1 0 -0.4 0.4 1"/>
        <body name="pedestal" pos="0 0 0.3">
          <geom type="cylinder" size="0.08 0.3"/>
          <body name="seg_a" pos="0 0 0.3">
            <joint name="joint_alpha" type="hinge" axis="0 0 1" range="-2 2"/>
            <geom type="capsule" fromto="0 0 0 0.3 0 0.1" size="0.03"/>
            <body name="seg_b" pos="0.3 0 0.1">
              <joint name="joint_beta" type="hinge" axis="0 1 0" range="-2 2"/>
              <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.025"/>
              <site name="tip_end_effector" pos="0.28 0 0" size="0.02"/>
              <camera name="ego_cam" pos="0 0 0.05" xyaxes="0 -1 0 0 0 1"/>
            </body>
          </body>
        </body>
      </worldbody>
      <actuator>
        <position name="act_alpha" joint="joint_alpha" kp="50" ctrlrange="-2 2"/>
        <position name="act_beta" joint="joint_beta" kp="50" ctrlrange="-2 2"/>
      </actuator>
    </mujoco>"""
    p = inspect_body(mujoco.MjModel.from_xml_string(xml))
    assert p.locomotion_type == "fixed_base" and not p.is_mobile
    assert p.end_effector_sites == ["tip_end_effector"]
    assert p.max_reach > 0.3
    assert set(p.camera_names) == {"exo_cam", "ego_cam"}
    assert p.actuator_limits["act_alpha"] == [-2.0, 2.0]


def test_blind_introspection_on_unknown_mobile_base():
    xml = """<mujoco><compiler angle="radian"/><option timestep="0.002"/>
    <worldbody><geom type="plane" size="3 3 0.1"/>
      <body name="chassis" pos="0 0 0.3"><freejoint/>
        <geom type="box" size="0.2 0.15 0.1"/>
        <body name="armbit" pos="0.2 0 0.1">
          <joint name="joint_x" type="hinge" axis="0 1 0" range="-1 1"/>
          <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
          <site name="grabber_gripper_site" pos="0.32 0 0" size="0.02"/>
        </body>
      </body></worldbody>
      <actuator><position name="ax" joint="joint_x" kp="10" ctrlrange="-1 1"/></actuator>
    </mujoco>"""
    p = inspect_body(mujoco.MjModel.from_xml_string(xml))
    assert p.is_mobile and p.locomotion_type not in ("fixed_base",)
    assert "grabber_gripper_site" in p.end_effector_sites


# ------------------------------------------------------------------- mutation
@pytest.mark.parametrize("name,joint", [("humanoid", "left_knee"),
                                        ("quadruped_arm", "front_left_knee"),
                                        ("table_arm", "elbow_pitch"),
                                        ("starwars_droid", "head_yaw")])
def test_mutate_lock_joint(name, joint):
    m = _model(name)
    before = inspect_body(m)
    assert joint not in before.locked_joints
    m2 = mutate_body(m, lock_joint=joint)
    after = inspect_body(m2)
    assert joint in after.locked_joints
    assert after.joint_ranges[joint] == [0.0, 0.0]
    # original model untouched (mutation returns a copy)
    assert joint not in inspect_body(m).locked_joints
    d = _step(m2, mujoco.MjData(m2), 100)
    assert np.isfinite(d.qpos).all()


def test_mutate_lock_joint_on_xml_string():
    xml = build_mjcf("table_arm")
    m2 = mutate_body(xml, lock_joint="shoulder_pitch")
    assert "shoulder_pitch" in inspect_body(m2).locked_joints
    xml2 = mutate_model_xml(xml, lock_joint="shoulder_pitch")
    assert isinstance(xml2, str) and "shoulder_pitch" in xml2


def test_mutate_scale_link_increases_reach():
    m = _model("humanoid")
    base = inspect_body(m).max_reach
    m2 = mutate_body(m, scale_link="left_forearm", scale_factor=1.5)
    assert inspect_body(m2).max_reach > base + 0.05
    # shrinking one of two symmetric arms leaves the other arm's reach
    # as the max; use the single-chain table arm to check shrinkage
    mt = _model("table_arm")
    base_t = inspect_body(mt).max_reach
    m3 = mutate_body(mt, scale_link="link3", scale_factor=0.5)
    assert inspect_body(m3).max_reach < base_t


def test_mutate_scale_link_on_xml_and_disable_actuator():
    m = _model("table_arm")
    m2 = mutate_body(m, disable_actuator="base_yaw_act")
    aid = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_ACTUATOR, "base_yaw_act")
    assert list(m2.actuator_ctrlrange[aid]) == [0.0, 0.0]
    with pytest.raises(ValueError):
        mutate_body(m, lock_joint="no_such_joint")


# ---------------------------------------------------------------------- arena
@pytest.mark.parametrize("name", MORPHOLOGY_NAMES)
def test_arena_lifecycle(name):
    a = build_arena(name)
    assert isinstance(a, Arena) and a.morphology == name
    state = a.reset()
    assert np.allclose(state["ball_position"][:2], [0.6, 0.3], atol=0.05)
    assert np.allclose(a.get_person_position(), [2.0, 0.0, 1.1], atol=0.05)
    assert set(a.get_obstacle_positions()) >= {"obstacle_1", "obstacle_2"}
    state = a.step(50)
    assert np.isfinite(state["qpos"]).all() and state["time"] > 0
    assert a.profile().locomotion_type == EXPECTED_LOCOMOTION[name]


@pytest.mark.parametrize("name", MORPHOLOGY_NAMES)
def test_arena_render_helpers(name):
    a = Arena(name)
    a.step(5)
    assert a.render_rgb("exo_cam", 48, 48).shape == (48, 48, 3)
    assert a.render_ego(48, 48).shape == (48, 48, 3)
    depth = a.render_depth("exo_cam", 48, 48)
    assert depth.shape == (48, 48) and np.isfinite(depth).all()


def test_arena_perturb_ball_absolute_and_relative():
    a = Arena("humanoid")
    a.perturb_ball(0.9, -0.4)
    assert np.allclose(a.get_ball_position(), [0.9, -0.4, 0.11], atol=1e-6)
    a.perturb_ball(dx=0.2, dy=0.1)
    assert np.allclose(a.get_ball_position()[:2], [1.1, -0.3], atol=1e-6)
    a.set_ball_position([0.5, 0.5, 0.11])
    assert np.allclose(a.get_ball_position()[:2], [0.5, 0.5], atol=1e-6)
    a.step(20)
    assert np.isfinite(a.get_ball_position()).all()
    assert abs(a.get_ball_position()[2] - 0.11) < 0.05
