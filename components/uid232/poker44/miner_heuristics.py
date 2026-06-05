"""Runtime chunk scoring wrapper for the vendored UID232 artifact."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
SCORE_SCRIPT_PATH = COMPONENT_ROOT / "score_uid232_v5_artifact.py"
DEFAULT_ARTIFACT = os.getenv("POKER44_UID232_ARTIFACT", "super_features")
DEFAULT_THRESHOLD = float(os.getenv("POKER44_UID232_THRESHOLD", "0.5"))

_SCORE_MODULE: Optional[Any] = None
_LOAD_ERROR: Optional[str] = None


def _load_module() -> Any:
    global _SCORE_MODULE, _LOAD_ERROR

    if _SCORE_MODULE is not None:
        return _SCORE_MODULE
    if _LOAD_ERROR is not None:
        raise RuntimeError(_LOAD_ERROR)

    spec = importlib.util.spec_from_file_location("uid232_artifact_eval", SCORE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        _LOAD_ERROR = f"Cannot load UID232 scorer from {SCORE_SCRIPT_PATH}"
        raise RuntimeError(_LOAD_ERROR)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _SCORE_MODULE = module
    return module


def score_chunk_runtime_with_route(chunk: List[dict]) -> Tuple[float, str]:
    if not chunk:
        return 0.5, "empty_chunk"

    try:
        module = _load_module()
        result = module.score_chunks(
            COMPONENT_ROOT,
            DEFAULT_ARTIFACT,
            [chunk],
            prediction_threshold=DEFAULT_THRESHOLD,
        )[0]
        return float(result["score"]), f"uid232:{DEFAULT_ARTIFACT}"
    except Exception:
        return 0.5, "runtime_error"


def score_chunk(chunk: List[dict]) -> float:
    score, _route = score_chunk_runtime_with_route(chunk)
    return score


def get_chunk_scorer_startup_check(scorer: str) -> Dict[str, object]:
    scorer_norm = (scorer or "").strip().lower()
    info: Dict[str, object] = {
        "scorer": scorer_norm,
        "active": scorer_norm == "runtime",
        "ok": True,
        "error": None,
        "details": {},
    }

    if scorer_norm != "runtime":
        return info

    info["details"] = {
        "score_script": str(SCORE_SCRIPT_PATH),
        "score_script_exists": SCORE_SCRIPT_PATH.exists(),
        "artifact": DEFAULT_ARTIFACT,
        "threshold": DEFAULT_THRESHOLD,
        "super_features_model": str(COMPONENT_ROOT / "models" / "poker44_super_features_v5.joblib"),
        "super_features_model_exists": (COMPONENT_ROOT / "models" / "poker44_super_features_v5.joblib").exists(),
        "benchmark_supervised_model": str(COMPONENT_ROOT / "models" / "poker44_benchmark_supervised_v5.joblib"),
        "benchmark_supervised_model_exists": (COMPONENT_ROOT / "models" / "poker44_benchmark_supervised_v5.joblib").exists(),
        "feature_extractors_dir": str(COMPONENT_ROOT / "feature_extractors"),
        "feature_extractors_dir_exists": (COMPONENT_ROOT / "feature_extractors").exists(),
        "runtime_package_dir": str(COMPONENT_ROOT / "poker44_ml"),
        "runtime_package_dir_exists": (COMPONENT_ROOT / "poker44_ml").exists(),
    }

    try:
        module = _load_module()
        spec = module.resolve_artifact_spec(DEFAULT_ARTIFACT, COMPONENT_ROOT)
        feature_extractor_path = Path(spec["feature_extractor"])
        if not feature_extractor_path.is_absolute():
            feature_extractor_path = COMPONENT_ROOT / feature_extractor_path
        model_path = Path(spec["model_path"])
        if not model_path.is_absolute():
            model_path = COMPONENT_ROOT / model_path
        module._load_chunk_features(feature_extractor_path)
        module._load_predictor(COMPONENT_ROOT, model_path)
    except Exception as exc:
        info["ok"] = False
        info["error"] = str(exc)

    return info