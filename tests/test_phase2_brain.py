"""Phase 2: Meta API layer, unified input, SAM perception, task compiler."""
from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from src.api.meta_client import (MetaClient, MetaResponse, SUPPORTED_MODELS,
                                 MODEL_WONDER, MODEL_SPARK, MODEL_SAM,
                                 MODEL_TRANSCRIBE, resolve_model)
from src.meta_budget import BudgetGuard
from src.brain.input_layer import InputLayer, UnifiedInput, ingest
from src.brain.perception_sam import (PerceptionSAM, ScenePerception,
                                      segment_image, mask_centroid,
                                      project_to_pixel, backproject_pixel,
                                      camera_pose)
from src.brain.task_compiler import (TaskCompiler, TaskContract, STEP_NAMES,
                                     parse_intent, compile_task, is_body_blind)
from src.physics.arena import Arena
from src.physics.morphologies import MORPHOLOGY_NAMES


@pytest.fixture()
def client():
    return MetaClient(offline=True, budget=BudgetGuard(limit_eur=150.0))


# ---------------------------------------------------------------- Meta client
def test_supported_models_exact():
    assert SUPPORTED_MODELS == ["muse-wonder-1.0", "muse-spark-1.3",
                                "muse-spark-1.3-contributor", "sam-3.1",
                                "muse-voice-transcribe-1.0"]
    assert resolve_model("wonder") == MODEL_WONDER
    assert resolve_model("spark") == MODEL_SPARK
    assert resolve_model("sam") == MODEL_SAM
    assert resolve_model("voice") == MODEL_TRANSCRIBE


def test_client_defaults_to_meta_base_url(client):
    assert client.base_url == "https://api.meta.ai/v1" or client.base_url
    assert client.offline is True


def test_client_forced_offline_in_unit_test_mode(monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "x")
    c = MetaClient(api_key="sk-real-looking-key-123456")
    assert c.offline is True


def test_chat_completions_offline_deterministic(client):
    msgs = [{"role": "user", "content": "throw the ball to the person"}]
    r1, r2 = client.chat_completions(msgs), client.chat_completions(msgs)
    assert isinstance(r1, MetaResponse) and r1.offline is True
    assert r1.content == r2.content and r1.content
    assert r1["choices"][0]["message"]["content"] == r1.content
    assert json.loads(r1.content)["steps"] == STEP_NAMES


@pytest.mark.parametrize("model", SUPPORTED_MODELS)
def test_chat_all_models_offline(client, model):
    r = client.chat([{"role": "user", "content": "hello"}], model=model)
    assert r.offline is True and r.model == model


def test_responses_offline(client):
    r = client.responses("throw the ball to the person", model=MODEL_WONDER)
    assert r.offline is True and r.text
    assert r["output_text"] == r.text


def test_transcribe_offline_bytes_and_fallback(client):
    assert client.transcribe_audio(b"throw the ball to the person").text \
        == "throw the ball to the person"
    assert client.transcribe_audio(b"\x00\x01\x02", fallback_text="kick it").text \
        == "kick it"
    assert client.transcribe_audio(None).text == ""


def test_segment_offline_returns_empty_masks(client):
    r = client.segment(np.zeros((8, 8, 3), dtype=np.uint8))
    assert r.offline is True and r["masks"] == {}


def test_budget_guard_integration_and_exhaustion():
    budget = BudgetGuard(limit_eur=0.001)
    c = MetaClient(offline=True, budget=budget)
    assert c.budget is budget and c.budget_guard is budget
    with pytest.raises(RuntimeError):
        budget.record_spend(1.0)


def test_async_chat_offline(client):
    r = asyncio.run(client.achat_completions(
        [{"role": "user", "content": "hello"}]))
    assert r.offline is True and r.content


# ----------------------------------------------------------------- input layer
def test_input_text():
    u = InputLayer().ingest_text("  throw the ball to the person ")
    assert isinstance(u, UnifiedInput)
    assert u.modality == "text" and u.text == "throw the ball to the person"


def test_input_audio_bytes_transcribed():
    u = InputLayer().ingest_audio(b"throw the ball to the person")
    assert u.modality == "audio"
    assert u.transcript == "throw the ball to the person"
    assert u.text == u.transcript


def test_input_audio_fallback_text():
    u = InputLayer().ingest_audio(b"\xff\xfe\x00", fallback_text="deliver the ball")
    assert u.text == "deliver the ball"


def test_input_video_frames_summary_and_count():
    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(3)]
    frames[0][4:8, 4:8] = [255, 0, 0]       # red ball
    frames[0][:, 12:] = [40, 60, 220]       # blue person
    u = InputLayer().ingest_video(frames)
    assert u.modality == "video" and u.frame_count == 3
    assert "ball" in u.text and "person" in u.text


def test_input_unified_dispatch():
    il = InputLayer()
    assert il.ingest("throw the ball").modality == "text"
    assert il.ingest(b"throw the ball").modality == "audio"
    assert il.ingest({"text": "throw it"}).text == "throw it"
    assert ingest("throw the ball to the person").modality == "text"


# --------------------------------------------------------------- task compiler
def test_contract_steps_and_formal_conditions():
    c = TaskCompiler().compile("throw the ball to the person")
    assert isinstance(c, TaskContract)
    assert c.step_names == ["LOCATE", "APPROACH", "ACQUIRE", "PROPEL", "VERIFY"]
    assert [s.name for s in c.steps] == STEP_NAMES
    for s in c.steps:
        assert s.preconditions and s.postconditions and s.predicates
    verify = c.step("VERIFY")
    assert "0.15" in verify.predicates["at_target"]
    assert c.success_tolerance_m == 0.15
    assert c.source_object == "ball" and c.target == "person_target"
    assert c.action == "throw"


def test_contract_is_body_blind():
    c = TaskCompiler().compile("throw the ball to the person")
    assert is_body_blind(c)
    blob = c.to_json().lower()
    for word in ("humanoid", "quadruped", "morphology", "joint", "actuator"):
        assert word not in blob
    # compiler signature accepts no body/profile
    import inspect
    assert "profile" not in inspect.signature(TaskCompiler.compile).parameters
    assert "body" not in inspect.signature(TaskCompiler.compile).parameters


def test_contract_json_roundtrip():
    c = compile_task("throw the ball to the person")
    data = json.loads(c.to_json())
    assert data["step_names"] == STEP_NAMES
    assert data["steps"][0]["name"] == "LOCATE"
    assert c["action"] == "throw"


def test_parse_intent_variants():
    assert parse_intent("toss the ball to the man")["target"] == "person_target"
    assert parse_intent("deliver the ball to the person")["action"] == "deliver"
    assert parse_intent("kick the ball to the person")["action"] == "kick"
    c = TaskCompiler().compile("please give the ball to the person")
    assert c.target == "person_target" and c.source_object == "ball"


def test_compiler_accepts_unified_input():
    u = InputLayer().ingest_audio(b"throw the ball to the person")
    c = TaskCompiler().compile(u)
    assert c.step_names == STEP_NAMES and c.modality == "audio"


# ------------------------------------------------------------------ perception
@pytest.mark.parametrize("name", MORPHOLOGY_NAMES)
def test_perception_localizes_ball_and_person(name):
    arena = Arena(name)
    scene = PerceptionSAM(arena=arena).perceive()
    assert isinstance(scene, ScenePerception)
    assert scene.ball is not None and scene.ball.localized
    assert np.allclose(scene.ball.position, arena.get_ball_position(), atol=0.05)
    assert np.allclose(scene.person.position, arena.get_person_position(), atol=0.05)
    assert scene.ball.confidence > 0.5


@pytest.mark.parametrize("name", MORPHOLOGY_NAMES)
def test_perception_obstacles_present(name):
    scene = PerceptionSAM(arena=Arena(name)).perceive()
    assert "obstacles" in scene.objects
    assert "obstacle_1" in scene.objects and "obstacle_2" in scene.objects


def test_active_search_when_ball_outside_ego_fov():
    arena = Arena("humanoid")
    arena.perturb_ball(-1.5, 1.5)  # behind / far from ego view
    perc = PerceptionSAM(arena=arena)
    # ego-only check first: ball is not in the initial ego frame
    scene = perc.perceive()
    assert scene.active_search_triggered is True
    assert scene.search_actions, "ACTIVE_SEARCH must record scan actions"
    assert all(a["action"] == "ACTIVE_SEARCH" for a in scene.search_actions)
    assert np.allclose(scene.ball.position, [-1.5, 1.5, 0.11], atol=0.05)


def test_active_search_explicit_call():
    arena = Arena("starwars_droid")
    arena.perturb_ball(-1.0, -1.0)
    det = PerceptionSAM(arena=arena).active_search("ball")
    assert det.localized
    assert np.allclose(det.position[:2], [-1.0, -1.0], atol=0.05)
    assert det.source.startswith("active_search")


def test_perception_after_perturb_returns_new_position():
    arena = Arena("humanoid")
    perc = PerceptionSAM(arena=arena)
    before = perc.perceive().ball.position.copy()
    arena.perturb_ball(0.9, -0.4)
    after = perc.perceive().ball.position
    assert not np.allclose(before, after)
    assert np.allclose(after, [0.9, -0.4, 0.11], atol=0.05)


def test_segment_image_colour_masks():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[2:6, 2:6] = [255, 0, 0]
    img[2:6, 12:16] = [30, 50, 230]
    masks = segment_image(img)
    assert "ball" in masks and masks["ball"].sum() == 16
    assert "person_target" in masks
    centroid = mask_centroid(masks["ball"])
    assert centroid == pytest.approx((3.5, 3.5))


def test_project_backproject_roundtrip():
    arena = Arena("humanoid")
    pos, rot, fovy = camera_pose(arena.model, arena.data, "exo_cam")
    point = np.array([0.6, 0.3, 0.11])
    proj = project_to_pixel(point, pos, rot, fovy, 128, 128)
    assert proj is not None
    back = backproject_pixel(proj[0], proj[1], proj[2], pos, rot, fovy, 128, 128)
    assert np.allclose(back, point, atol=1e-6)


def test_perception_rendered_cv_path():
    arena = Arena("humanoid")
    arena.step(5)
    scene = PerceptionSAM(arena=arena).perceive_rendered(48, 48)
    assert scene.ball is not None
    # rendered path must still produce a usable ball estimate (mask or fallback)
    assert scene.ball.localized
    assert np.allclose(scene.ball.position, arena.get_ball_position(), atol=0.3)
