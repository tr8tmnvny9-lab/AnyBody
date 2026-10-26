"""Phase 3: closed-loop skill synthesis, self-practice, live recovery."""
from __future__ import annotations

import numpy as np
import pytest

from src.brain.closed_loop import (ClosedLoopController, SkillSynthesizer,
                                   ControlPolicy, PracticeResult, ThrowParams,
                                   analytic_speed, simulate_throw,
                                   run_closed_loop, synthesize_policy)
from src.brain.task_compiler import TaskCompiler
from src.brain.perception_sam import PerceptionSAM
from src.physics.arena import Arena
from src.physics.inspector import inspect_body, mutate_body
from src.physics.morphologies import MORPHOLOGY_NAMES

EXPECTED_LOCO = {"humanoid": "biped", "quadruped_arm": "quadruped",
                 "starwars_droid": "wheeled", "table_arm": "fixed_base"}
LOCKABLE = {"humanoid": "left_elbow", "quadruped_arm": "dorsal_elbow",
            "starwars_droid": "launcher_pitch", "table_arm": "elbow_pitch"}


@pytest.fixture()
def contract():
    return TaskCompiler().compile("throw the ball to the person")


# ------------------------------------------------------------------ synthesis
@pytest.mark.parametrize("name", MORPHOLOGY_NAMES)
def test_policy_synthesis_per_morphology(contract, name):
    arena = Arena(name)
    profile = inspect_body(arena.model)
    policy = SkillSynthesizer().synthesize(contract, profile, arena=arena)
    assert isinstance(policy, ControlPolicy)
    assert policy.locomotion_type == EXPECTED_LOCO[name]
    assert policy.locomotion == policy.locomotion_type
    assert policy.is_mobile == (name != "table_arm")
    assert policy.end_effector in profile.end_effector_sites
    assert policy.usable_actuators
    names = [s["name"] for s in policy.strategy]
    for step in ("SEARCH", "NAVIGATE", "GRASP", "VERIFY"):
        assert step in names
    assert any(s in names for s in ("THROW", "DELIVER"))


def test_fixed_base_policy_notes_reach_limitation(contract):
    arena = Arena("table_arm")
    profile = inspect_body(arena.model)
    policy = synthesize_policy(contract, profile, arena=arena)
    assert policy.locomotion_type == "fixed_base" and not policy.is_mobile
    # person is ~1.4 m from the ball, beyond the arm's ~1 m reach
    assert policy.can_reach_target is False
    assert any("reach" in n for n in policy.adaptation_notes)


@pytest.mark.parametrize("name", MORPHOLOGY_NAMES)
def test_policy_for_locked_joint_body(contract, name):
    arena = Arena(name)
    mutated = mutate_body(arena.model, lock_joint=LOCKABLE[name])
    profile = inspect_body(mutated)
    assert LOCKABLE[name] in profile.locked_joints
    policy = SkillSynthesizer().synthesize(contract, profile, arena=arena)
    assert policy.degraded is True
    assert LOCKABLE[name] in policy.locked_joints
    assert LOCKABLE[name] in policy.excluded_joints
    assert policy.adaptation_notes  # explains the adaptation


def test_humanoid_locked_left_elbow_switches_arm(contract):
    arena = Arena("humanoid")
    mutated = mutate_body(arena.model, lock_joint="left_elbow")
    profile = inspect_body(mutated)
    policy = SkillSynthesizer().synthesize(contract, profile, arena=arena)
    assert "left" not in policy.end_effector or "right" in policy.end_effector


# --------------------------------------------------------------- self-practice
@pytest.mark.parametrize("name", MORPHOLOGY_NAMES)
def test_self_practice_converges_all_morphologies(contract, name):
    arena = Arena(name)
    ctl = ClosedLoopController(arena=arena)
    policy = ctl.synthesize(contract, inspect_body(arena.model), arena=arena)
    result = ctl.practice(arena, contract, policy)
    assert isinstance(result, PracticeResult)
    assert 1 <= result.n_trials <= 6
    assert result.converged is True
    assert result.best_error < 0.15
    assert result.trials[-1].landing_error < 0.15
    assert result.best_params is not None
    for key in ("release_angle", "release_pitch", "impulse_gain",
                "pitch_velocity", "wrist_snap", "torso_lean", "base_impulse"):
        assert key in result.trials[0].params


@pytest.mark.parametrize("name", MORPHOLOGY_NAMES)
def test_practice_improves_over_trials(contract, name):
    arena = Arena(name)
    ctl = ClosedLoopController(arena=arena)
    policy = ctl.synthesize(contract, inspect_body(arena.model), arena=arena)
    result = ctl.practice(arena, contract, policy)
    if result.n_trials >= 2:
        assert result.errors[-1] <= result.errors[0] + 1e-9


def test_practice_forks_state_and_leaves_live_arena_untouched(contract):
    arena = Arena("humanoid")
    q0 = np.array(arena.data.qpos, copy=True)
    ball0 = arena.get_ball_position().copy()
    ctl = ClosedLoopController(arena=arena)
    policy = ctl.synthesize(contract, inspect_body(arena.model), arena=arena)
    ctl.practice(arena, contract, policy)
    assert np.allclose(arena.data.qpos, q0)
    assert np.allclose(arena.get_ball_position(), ball0)


@pytest.mark.parametrize("name", MORPHOLOGY_NAMES)
def test_self_practice_converges_with_locked_joint(contract, name):
    arena = Arena(name)
    arena.model = mutate_body(arena.model, lock_joint=LOCKABLE[name])
    import mujoco
    arena.data = mujoco.MjData(arena.model)
    arena.reset()
    ctl = ClosedLoopController(arena=arena)
    policy = ctl.synthesize(contract, inspect_body(arena.model), arena=arena)
    assert policy.degraded
    result = ctl.practice(arena, contract, policy)
    assert result.converged and result.best_error < 0.15


def test_analytic_speed_and_single_rollout():
    release = np.array([0.6, 0.3, 0.4])
    target = np.array([2.0, 0.0, 0.11])  # person ground footprint
    speed = analytic_speed(release, target)
    assert 1.0 < speed < 10.0
    arena = Arena("humanoid")
    params = ThrowParams(release_pitch=np.radians(45), impulse_gain=1.0,
                         speed_mps=speed, yaw_rad=np.arctan2(-0.3, 1.4))
    landing, error, _ = simulate_throw(arena.model, arena.data, params, target,
                                       arena=arena)
    assert error < 0.35  # single analytic rollout is already close


# ----------------------------------------------------------------- live execute
@pytest.mark.parametrize("name", MORPHOLOGY_NAMES)
def test_live_execution_lands_at_person(contract, name):
    arena = Arena(name)
    result = ClosedLoopController(arena=arena).execute(contract)
    assert result.success is True and result.verified is True
    assert result.landing_error < 0.15
    assert result.practice is not None and result.practice.converged
    assert result.steps_completed == ["LOCATE", "APPROACH", "ACQUIRE",
                                      "PROPEL", "VERIFY"]
    # ball physically ends near the person in the live sim
    assert np.linalg.norm(arena.get_ball_position()[:2]
                          - arena.get_person_position()[:2]) < 0.6


def test_run_closed_loop_module_helper(contract):
    result = run_closed_loop(Arena("table_arm"), contract)
    assert result.success and result.landing_error < 0.15


# ------------------------------------------------------- perturbation recovery
def test_detect_perturbation(contract):
    arena = Arena("humanoid")
    ctl = ClosedLoopController(arena=arena)
    ctl.synthesize(contract, inspect_body(arena.model), arena=arena)
    ctl.practice(arena, contract)
    assert ctl.detect_perturbation(arena)["perturbed"] is False
    arena.perturb_ball(1.4, -0.9)
    out = ctl.detect_perturbation(arena)
    assert out["perturbed"] is True and out["distance"] > 0.05


def test_recovery_after_kick_replans_and_converges(contract):
    arena = Arena("humanoid")
    ctl = ClosedLoopController(arena=arena)
    ctl.synthesize(contract, inspect_body(arena.model), arena=arena)
    ctl.practice(arena, contract)
    arena.perturb_ball(-0.8, 1.1)  # kick the ball away mid-run
    recovery = ctl.recover(arena, contract, inspect_body(arena.model),
                           reason="ball kicked")
    assert recovery.perturbation_detected is True
    assert recovery.replanned is True and recovery.recovered is True
    assert recovery.practice.converged
    assert recovery.practice.best_error < 0.15
    assert np.allclose(recovery.ball_position[:2], [-0.8, 1.1], atol=0.05)
    assert recovery.perception.ball.localized


def test_execute_recovers_from_pre_execute_kick(contract):
    arena = Arena("humanoid")
    kicked = {"done": False}

    def kick(a):
        if not kicked["done"]:
            a.perturb_ball(0.2, -1.0)
            kicked["done"] = True

    result = ClosedLoopController(arena=arena).execute(
        contract, pre_execute_hook=kick)
    assert kicked["done"]
    assert result.perturbation_detected is True
    assert result.replans >= 1
    assert result.success and result.landing_error < 0.15


def test_joint_lock_mid_run_triggers_replan(contract):
    arena = Arena("humanoid")
    ctl = ClosedLoopController(arena=arena)
    profile_before = inspect_body(arena.model)
    ctl.synthesize(contract, profile_before, arena=arena)
    mutated_profile = inspect_body(mutate_body(arena.model, lock_joint="left_knee"))
    assert ctl.detect_body_mutation(mutated_profile, arena) is True
    recovery = ctl.recover(arena, contract, mutated_profile,
                           reason="joint locked mid-run")
    assert recovery.replanned and recovery.policy.degraded
    assert "left_knee" in recovery.policy.locked_joints


def test_end_to_end_vague_text_to_landing():
    from src.brain.input_layer import InputLayer
    unified = InputLayer().ingest("throw the ball to the person")
    contract = TaskCompiler().compile(unified)
    arena = Arena("quadruped_arm")
    scene = PerceptionSAM(arena=arena).perceive()
    assert scene.all_localized or scene.ball.localized
    result = ClosedLoopController(arena=arena).execute(contract)
    assert result.success and result.landing_error < 0.15
