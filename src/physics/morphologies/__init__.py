"""Procedural MJCF generators for 4 robot morphologies."""
from __future__ import annotations
from . import humanoid, quadruped_arm, starwars_droid, table_arm

_MODULES = {
    "humanoid": humanoid,
    "quadruped_arm": quadruped_arm,
    "quadruped": quadruped_arm,
    "dog": quadruped_arm,
    "starwars_droid": starwars_droid,
    "droid": starwars_droid,
    "starwars": starwars_droid,
    "table_arm": table_arm,
    "table": table_arm,
    "fixed_arm": table_arm,
}
MORPHOLOGY_NAMES = ["humanoid", "quadruped_arm", "starwars_droid", "table_arm"]
NAMES = MORPHOLOGY_NAMES
REGISTRY = _MODULES

def get_module(name: str):
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _MODULES:
        raise ValueError(f"Unknown morphology {name!r}; available: {MORPHOLOGY_NAMES}")
    return _MODULES[key]

def build_mjcf(name: str, include_arena: bool = True) -> str:
    return get_module(name).build_mjcf(include_arena=include_arena)

def create_model(name: str):
    return get_module(name).create_model()

# aliases
get_mjcf = build_mjcf
load_model = create_model
build_model = create_model

__all__ = ["humanoid", "quadruped_arm", "starwars_droid", "table_arm",
           "MORPHOLOGY_NAMES", "NAMES", "REGISTRY", "get_module",
           "build_mjcf", "get_mjcf", "create_model", "load_model", "build_model"]
