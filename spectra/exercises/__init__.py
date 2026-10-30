"""Patient-specific exercises built from validated movements."""

from spectra.exercises.catalog import (
    BUILTIN_KEYS,
    BUILTIN_SPECS,
    CUSTOM_PREFIX,
    ExerciseSpec,
    Movement,
    builtin_spec,
    specs_from_json,
)
from spectra.exercises.runner import ExerciseRunner, RunProgress

__all__ = [
    "BUILTIN_KEYS",
    "BUILTIN_SPECS",
    "CUSTOM_PREFIX",
    "ExerciseRunner",
    "ExerciseSpec",
    "Movement",
    "RunProgress",
    "builtin_spec",
    "specs_from_json",
]
