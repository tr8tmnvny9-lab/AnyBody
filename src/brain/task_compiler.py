"""Body-blind Muse Wonder 1.0 task planner.

Translates a unified input (vague text / transcript / video summary) into a
formal, embodiment-agnostic :class:`TaskContract`:

    LOCATE -> APPROACH -> ACQUIRE -> PROPEL -> VERIFY

The compiler never receives a ``BodyProfile`` and its output contains no
morphology, joint, actuator or locomotion information: which body runs the
contract is decided later by the closed-loop synthesizer.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any

STEP_NAMES = ["LOCATE", "APPROACH", "ACQUIRE", "PROPEL", "VERIFY"]
SUCCESS_TOLERANCE_M = 0.15

_BODY_WORDS = ("humanoid", "quadruped", "droid", "biped", "wheel",
               "table_arm", "joint", "actuator", "locomotion", "morphology")


@dataclass
class TaskStep:
    name: str
    description: str = ""
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    # Formal predicates: name -> human/machine readable predicate string.
    predicates: dict[str, str] = field(default_factory=dict)
    success_tolerance_m: float = SUCCESS_TOLERANCE_M

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskContract:
    """Formal, body-agnostic task contract (pre/post-conditions)."""
    raw_input: str = ""
    action: str = "throw"
    verb: str = "throw"
    source_object: str = "ball"
    target: str = "person_target"
    target_object: str = "person_target"
    steps: list[TaskStep] = field(default_factory=list)
    step_names: list[str] = field(default_factory=lambda: list(STEP_NAMES))
    success_tolerance_m: float = SUCCESS_TOLERANCE_M
    tolerance_m: float = SUCCESS_TOLERANCE_M
    task_id: str = ""
    modality: str = "text"
    needs_active_search: bool = True
    offline: bool = True
    model: str = "muse-wonder-1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.task_id:
            digest = hashlib.sha1(
                f"{self.action}:{self.source_object}:{self.target}:{self.raw_input}"
                .encode()).hexdigest()[:10]
            self.task_id = f"task-{digest}"
        self.target_object = self.target
        self.tolerance_m = self.success_tolerance_m
        if self.steps:
            self.step_names = [s.name for s in self.steps]

    # -- accessors --------------------------------------------------------
    def step(self, name: str) -> TaskStep | None:
        for s in self.steps:
            if s.name == name.upper():
                return s
        return None

    get_step = step

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


# ---------------------------------------------------------------------------
# Deterministic parsing (offline fallback + validation of online output)
# ---------------------------------------------------------------------------

_VERB_MAP = [
    (r"\b(throw|toss|hurl|lob)\b", "throw"),
    (r"\b(kick|punt)\b", "kick"),
    (r"\b(deliver|bring|give|hand|pass|carry|move|take)\b", "deliver"),
    (r"\b(pick\s*up|grab|grasp|fetch|get)\b", "acquire"),
    (r"\b(push|roll|slide)\b", "push"),
]
_TARGET_MAP = [
    (r"\b(person|man|woman|human|guy|friend|target)\b", "person_target"),
    (r"\b(ring|hoop|basket|goal|bin|box)\b", "target_ring"),
]
_OBJECT_MAP = [
    (r"\b(ball|sphere|orb)\b", "ball"),
    (r"\b(cube|block|box|object|item)\b", "ball"),  # arena manipulandum is ball
]


def parse_intent(text: str) -> dict[str, str]:
    low = (text or "").lower()
    action = "throw"
    for pat, verb in _VERB_MAP:
        if re.search(pat, low):
            action = verb
            break
    source_object = "ball"
    for pat, obj in _OBJECT_MAP:
        if re.search(pat, low):
            source_object = obj
            break
    target = "person_target"
    for pat, tgt in _TARGET_MAP:
        if re.search(pat, low):
            target = tgt
            break
    return {"action": action, "verb": action,
            "source_object": source_object, "target": target}


def build_contract(raw_input: str, action: str = "throw",
                   source_object: str = "ball", target: str = "person_target",
                   modality: str = "text", offline: bool = True,
                   model: str = "muse-wonder-1.0",
                   metadata: dict | None = None) -> TaskContract:
    tol = SUCCESS_TOLERANCE_M
    obj, tgt = source_object, target
    propel_desc = {
        "throw": f"Propel {obj} ballistically toward {tgt}",
        "kick": f"Kick {obj} toward {tgt}",
        "deliver": f"Deliver {obj} to {tgt} (propel or carry)",
        "push": f"Push/roll {obj} toward {tgt}",
        "acquire": f"Acquire {obj} at {tgt}",
    }.get(action, f"Propel {obj} toward {tgt}")
    steps = [
        TaskStep(
            "LOCATE",
            f"Localize {obj} and {tgt} in 3D; run ACTIVE_SEARCH if either "
            "is outside the ego camera FOV",
            preconditions=["scene_observable"],
            postconditions=[f"localized({obj})", f"localized({tgt})"],
            predicates={"localized_ball": f"position({obj}) is known within 0.05 m",
                        "localized_target": f"position({tgt}) is known within 0.05 m"}),
        TaskStep(
            "APPROACH",
            f"Reduce distance to {obj} until it lies within reach envelope",
            preconditions=[f"localized({obj})"],
            postconditions=[f"within_reach({obj})", "path_clear_of_obstacles"],
            predicates={"within_reach": f"distance(end_effector, {obj}) <= max_reach",
                        "path_clear": "no obstacle intersects the approach path"}),
        TaskStep(
            "ACQUIRE",
            f"Secure {obj} with the available end-effector",
            preconditions=[f"within_reach({obj})"],
            postconditions=[f"held({obj})"],
            predicates={"held": f"distance(end_effector, {obj}) < 0.10 and grasp stable"}),
        TaskStep(
            "PROPEL",
            propel_desc,
            preconditions=[f"held({obj})", f"localized({tgt})"],
            postconditions=[f"propelled({obj})", f"trajectory_toward({tgt})"],
            predicates={"propelled": f"{obj} released with velocity toward {tgt}",
                        "self_practice_converged":
                            f"predicted landing error < {tol} m over 3-6 forked rollouts"}),
        TaskStep(
            "VERIFY",
            f"Confirm {obj} landed at {tgt} within tolerance",
            preconditions=[f"propelled({obj})"],
            postconditions=[f"at_target({obj}, {tgt})"],
            predicates={"at_target":
                            f"landing_error({obj}, {tgt}) <= {tol} (metres, xy)"},
            success_tolerance_m=tol),
    ]
    return TaskContract(raw_input=raw_input, action=action, verb=action,
                        source_object=obj, target=tgt, steps=steps,
                        success_tolerance_m=tol, modality=modality,
                        offline=offline, model=model,
                        metadata=dict(metadata or {}))


def _contract_from_online_json(text: str, raw_input: str,
                               modality: str) -> TaskContract | None:
    try:
        m = re.search(r"\{.*\}", text or "", re.S)
        if not m:
            return None
        data = json.loads(m.group(0))
        intent = parse_intent(raw_input)
        contract = build_contract(
            raw_input,
            action=str(data.get("action") or intent["action"]),
            source_object=str(data.get("source_object") or intent["source_object"]),
            target=str(data.get("target") or intent["target"]),
            modality=modality, offline=False)
        names = [str(s).upper() for s in (data.get("steps") or [])]
        if names and names != STEP_NAMES:
            return None  # untrusted shape -> deterministic contract wins
        return contract
    except Exception:
        return None


class TaskCompiler:
    """Body-blind planner. Deliberately has no BodyProfile parameter."""

    def __init__(self, client=None, model: str = "muse-spark-1.3", **kw):
        if client is None and not kw.get("no_client"):
            try:
                from src.api.meta_client import MetaClient
                client = MetaClient()
            except Exception:
                client = None
        self.client = client
        self.model = model
        self._cache: dict[str, TaskContract] = {}

    def compile(self, input_data=None, text: str | None = None, **kw) -> TaskContract:
        raw, modality = self._coerce(input_data, text, kw)
        cache_key = f"{modality}:{raw.strip().lower()}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        offline = True
        if self.client is not None and not getattr(self.client, "offline", True):
            try:
                resp = self.client.chat_completions(
                    [{"role": "system",
                      "content": "You are a body-blind robot task planner. "
                                 "Reply with JSON only: action, source_object, "
                                 "target, steps [LOCATE, APPROACH, ACQUIRE, "
                                 "PROPEL, VERIFY]. Never mention robot bodies."},
                     {"role": "user", "content": raw}],
                    model=self.model, max_tokens=200)
                if not resp.offline:
                    contract = _contract_from_online_json(resp.content, raw, modality)
                    if contract is not None:
                        self._cache[cache_key] = contract
                        return contract
                    offline = False
            except Exception:
                offline = True
        intent = parse_intent(raw)
        out = build_contract(raw, action=intent["action"],
                             source_object=intent["source_object"],
                             target=intent["target"],
                             modality=modality, offline=offline,
                             model=self.model)
        self._cache[cache_key] = out
        return out

    # Aliases
    compile_task = compile
    plan = compile
    create_contract = compile
    __call__ = compile

    @staticmethod
    def _coerce(input_data, text, kw) -> tuple[str, str]:
        if text is not None:
            return str(text), kw.get("modality", "text")
        data = input_data
        if data is None:
            data = kw.get("input") or kw.get("prompt") or kw.get("raw_input") or ""
        if isinstance(data, str):
            return data, kw.get("modality", "text")
        if isinstance(data, dict):
            raw = (data.get("text") or data.get("transcript") or data.get("raw_input")
                   or data.get("prompt") or data.get("description") or "")
            return str(raw), str(data.get("modality") or kw.get("modality") or "text")
        # UnifiedInput-like object
        raw = getattr(data, "text", None) or getattr(data, "transcript", "") or str(data)
        modality = getattr(data, "modality", None) or kw.get("modality", "text")
        return str(raw), str(modality)


def compile_task(input_data=None, **kw) -> TaskContract:
    return TaskCompiler().compile(input_data, **kw)


def is_body_blind(contract: TaskContract) -> bool:
    """True when the serialized contract mentions no body-specific terms."""
    blob = contract.to_json().lower()
    return not any(w in blob for w in _BODY_WORDS)


__all__ = ["TaskCompiler", "TaskContract", "TaskStep", "STEP_NAMES",
           "SUCCESS_TOLERANCE_M", "parse_intent", "build_contract",
           "compile_task", "is_body_blind"]
