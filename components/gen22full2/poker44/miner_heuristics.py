"""Runtime chunk scoring wrapper for the vendored gen22full2 component."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_MODEL_PATH = COMPONENT_ROOT / "weights" / "gen22_hybrid_full_0.0595.pt"
MODEL_IMPL_PATH = Path(__file__).resolve().parent / "gen20_hybrid_model.py"
FEATURES_PATH = COMPONENT_ROOT / "poker44_ml" / "features.py"
DEFAULT_MIN_HANDS = 4
DEFAULT_MAX_HANDS = 8
DEFAULT_VOTES_PER_PARENT = 101
DEFAULT_POSITIVE_VOTES_REQUIRED = 71
DEFAULT_MINICHUNK_THRESHOLD = 0.0595
DEFAULT_RUNTIME_SEED = 20260605

ACTION_MAP = {
    "fold": 1,
    "call": 2,
    "raise": 3,
    "check": 4,
    "bet": 5,
    "all_in": 6,
}

STREET_MAP = {
    "preflop": 1,
    "flop": 2,
    "turn": 3,
    "river": 4,
}

_RUNTIME_MODEL: Optional[torch.nn.Module] = None
_RUNTIME_AVAILABLE = False
_RUNTIME_LOAD_ERROR: Optional[str] = None
_RUNTIME_ARGS: Dict[str, Any] = {}
_CHUNK_FEATURES: Optional[Callable[[List[dict]], Dict[str, float]]] = None
_CHUNK_FEATURE_ERROR: Optional[str] = None
_DENSE_FEATURE_NAMES: Optional[List[str]] = None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runtime_shape() -> Tuple[int, int]:
    max_hands = int(_RUNTIME_ARGS.get("max_hands", os.getenv("POKER44_MODEL_MAX_HANDS", str(DEFAULT_MAX_HANDS))))
    max_actions = int(_RUNTIME_ARGS.get("max_actions", os.getenv("POKER44_MODEL_MAX_ACTIONS", "32")))
    return max(1, max_hands), max(1, max_actions)


def _vote101_config() -> Dict[str, float | int]:
    return {
        "min_hands": max(1, int(os.getenv("POKER44_MODEL_MIN_HANDS", str(DEFAULT_MIN_HANDS)))),
        "max_hands": max(1, int(os.getenv("POKER44_MODEL_MAX_HANDS", str(_runtime_shape()[0])))),
        "votes_per_parent": max(1, int(os.getenv("POKER44_VOTES_PER_PARENT", str(DEFAULT_VOTES_PER_PARENT)))),
        "positive_votes_required": max(1, int(os.getenv("POKER44_POSITIVE_VOTES_REQUIRED", str(DEFAULT_POSITIVE_VOTES_REQUIRED)))),
        "minichunk_threshold": float(os.getenv("POKER44_MINICHUNK_SCORE_THRESHOLD", str(DEFAULT_MINICHUNK_THRESHOLD))),
        "runtime_seed": int(os.getenv("POKER44_VOTE101_SEED", str(DEFAULT_RUNTIME_SEED))),
    }


def _parent_decision_threshold() -> float:
    cfg = _vote101_config()
    return float(cfg["positive_votes_required"]) / float(cfg["votes_per_parent"])


def _sample_hand_indices(rng: random.Random, hand_count: int, sample_size: int) -> list[int]:
    if hand_count <= 0:
        raise ValueError("Cannot generate a minichunk from an empty parent chunk")
    if hand_count >= sample_size:
        return rng.sample(range(hand_count), sample_size)
    return [rng.randrange(hand_count) for _ in range(sample_size)]


def _materialize_minichunk(parent_chunk: list[dict[str, Any]], hand_indices: list[int]) -> list[dict[str, Any]]:
    return [parent_chunk[int(index)] for index in hand_indices]


def _chunk_seed(chunk: List[dict], runtime_seed: int) -> int:
    chunk_bytes = json.dumps(chunk, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.blake2b(chunk_bytes, digest_size=8, person=b"p44vote1").digest()
    return int.from_bytes(digest, "big") ^ int(runtime_seed)


def _load_chunk_features() -> Optional[Callable[[List[dict]], Dict[str, float]]]:
    global _CHUNK_FEATURES, _CHUNK_FEATURE_ERROR

    if _CHUNK_FEATURES is not None:
        return _CHUNK_FEATURES
    if _CHUNK_FEATURE_ERROR is not None:
        return None

    try:
        module = _load_module("gen22full2_features", FEATURES_PATH)
        _CHUNK_FEATURES = getattr(module, "chunk_features")
        _CHUNK_FEATURE_ERROR = None
        return _CHUNK_FEATURES
    except Exception as exc:
        _CHUNK_FEATURES = None
        _CHUNK_FEATURE_ERROR = str(exc)
        return None


def _encode_chunk(chunk: List[dict], max_hands: int, max_actions: int) -> Dict[str, np.ndarray]:
    shape = (max_hands, max_actions)

    arr_action_type = np.zeros(shape, dtype=np.int64)
    arr_street = np.zeros(shape, dtype=np.int64)
    arr_actor_seat = np.zeros(shape, dtype=np.int64)
    arr_amount = np.zeros(shape, dtype=np.float32)
    arr_raise_to = np.zeros(shape, dtype=np.float32)
    arr_call_to = np.zeros(shape, dtype=np.float32)
    arr_norm_bb = np.zeros(shape, dtype=np.float32)
    arr_pot_before = np.zeros(shape, dtype=np.float32)
    arr_pot_after = np.zeros(shape, dtype=np.float32)
    arr_raise_miss = np.zeros(shape, dtype=np.float32)
    arr_call_miss = np.zeros(shape, dtype=np.float32)
    arr_valid = np.zeros(shape, dtype=np.float32)

    for h_i, hand in enumerate(chunk[:max_hands]):
        actions = hand.get("actions") or []
        for a_i, action in enumerate(actions[:max_actions]):
            action_type = (action.get("action_type") or "").lower()
            street = (action.get("street") or "").lower()
            seat = action.get("actor_seat")

            arr_action_type[h_i, a_i] = ACTION_MAP.get(action_type, 0)
            arr_street[h_i, a_i] = STREET_MAP.get(street, 0)
            arr_actor_seat[h_i, a_i] = int(seat) + 1 if isinstance(seat, int) and seat >= 0 else 0
            arr_amount[h_i, a_i] = _safe_float(action.get("amount"))
            raise_to = action.get("raise_to")
            call_to = action.get("call_to")
            arr_raise_miss[h_i, a_i] = 1.0 if raise_to is None else 0.0
            arr_call_miss[h_i, a_i] = 1.0 if call_to is None else 0.0
            arr_raise_to[h_i, a_i] = _safe_float(raise_to)
            arr_call_to[h_i, a_i] = _safe_float(call_to)
            arr_norm_bb[h_i, a_i] = _safe_float(action.get("normalized_amount_bb"))
            arr_pot_before[h_i, a_i] = _safe_float(action.get("pot_before"))
            arr_pot_after[h_i, a_i] = _safe_float(action.get("pot_after"))
            arr_valid[h_i, a_i] = 1.0

    return {
        "action_type": arr_action_type,
        "street": arr_street,
        "actor_seat": arr_actor_seat,
        "amount": arr_amount,
        "raise_to": arr_raise_to,
        "call_to": arr_call_to,
        "norm_amount_bb": arr_norm_bb,
        "pot_before": arr_pot_before,
        "pot_after": arr_pot_after,
        "raise_to_missing": arr_raise_miss,
        "call_to_missing": arr_call_miss,
        "valid_mask": arr_valid,
    }


def _load_runtime_model() -> bool:
    global _RUNTIME_MODEL, _RUNTIME_AVAILABLE, _RUNTIME_LOAD_ERROR, _RUNTIME_ARGS

    if _RUNTIME_AVAILABLE and _RUNTIME_MODEL is not None:
        return True
    if _RUNTIME_LOAD_ERROR is not None:
        return False

    try:
        model_module = _load_module("gen22full2_model_impl", MODEL_IMPL_PATH)
        raw_checkpoint = torch.load(str(RUNTIME_MODEL_PATH), map_location="cpu")
        state_dict = raw_checkpoint.get("model_state", raw_checkpoint)
        _RUNTIME_ARGS = dict(raw_checkpoint.get("args", {})) if isinstance(raw_checkpoint, dict) else {}
        dropout = float(_RUNTIME_ARGS.get("dropout", 0.10))
        model = model_module.Gen20HybridV1(dropout=dropout)
        model.load_state_dict(state_dict)
        model.eval()
        _RUNTIME_MODEL = model
        _RUNTIME_AVAILABLE = True
        _RUNTIME_LOAD_ERROR = None
        return True
    except Exception as exc:
        _RUNTIME_MODEL = None
        _RUNTIME_AVAILABLE = False
        _RUNTIME_LOAD_ERROR = str(exc)
        return False


def _predict_minichunk_scores(batch: Dict[str, np.ndarray]) -> np.ndarray:
    if _RUNTIME_MODEL is None:
        raise RuntimeError("runtime model not loaded")

    xs: list[torch.Tensor] = []
    for key in [
        "action_type",
        "street",
        "actor_seat",
        "amount",
        "raise_to",
        "call_to",
        "norm_amount_bb",
        "pot_before",
        "pot_after",
        "raise_to_missing",
        "call_to_missing",
        "valid_mask",
    ]:
        tensor = torch.from_numpy(batch[key])
        if key in {"action_type", "street", "actor_seat"}:
            xs.append(tensor.long())
        else:
            xs.append(tensor.float())
    xs.append(torch.from_numpy(batch["dense_features"]).float())
    with torch.no_grad():
        return _RUNTIME_MODEL(*xs).detach().cpu().numpy().astype(np.float32, copy=False)


def score_chunk_runtime_with_route(chunk: List[dict]) -> Tuple[float, str]:
    if not chunk:
        return 0.0, "empty_chunk"

    if not _load_runtime_model() or _RUNTIME_MODEL is None:
        return 0.0, "runtime_unavailable"

    chunk_features = _load_chunk_features()
    if chunk_features is None:
        return 0.0, "chunk_features_unavailable"

    try:
        max_hands, max_actions = _runtime_shape()
        cfg = _vote101_config()
        min_hands = min(int(cfg["min_hands"]), len(chunk))
        max_sample_hands = min(int(cfg["max_hands"]), len(chunk))
        if max_sample_hands <= 0:
            return 0.0, "empty_after_preprocess"
        min_hands = min(min_hands, max_sample_hands)

        rng = random.Random(_chunk_seed(chunk, int(cfg["runtime_seed"])))
        encoded_by_feature = {key: [] for key in [
            "action_type",
            "street",
            "actor_seat",
            "amount",
            "raise_to",
            "call_to",
            "norm_amount_bb",
            "pot_before",
            "pot_after",
            "raise_to_missing",
            "call_to_missing",
            "valid_mask",
        ]}
        dense_rows: list[np.ndarray] = []

        global _DENSE_FEATURE_NAMES
        for _sample_index in range(int(cfg["votes_per_parent"])):
            sample_size = rng.randint(min_hands, max_sample_hands)
            hand_indices = _sample_hand_indices(rng, len(chunk), sample_size)
            minichunk = _materialize_minichunk(chunk, hand_indices)
            encoded = _encode_chunk(minichunk, max_hands=max_hands, max_actions=max_actions)
            feature_map = chunk_features(minichunk)
            if _DENSE_FEATURE_NAMES is None:
                _DENSE_FEATURE_NAMES = sorted(feature_map.keys())
            dense_rows.append(np.asarray([float(feature_map.get(name, 0.0)) for name in _DENSE_FEATURE_NAMES], dtype=np.float32))
            for key, value in encoded.items():
                encoded_by_feature[key].append(value)

        batch = {key: np.stack(rows, axis=0) for key, rows in encoded_by_feature.items()}
        batch["dense_features"] = np.stack(dense_rows, axis=0)
        scores = _predict_minichunk_scores(batch)
        positive_votes = int(np.count_nonzero(scores >= float(cfg["minichunk_threshold"])))
        vote_fraction = positive_votes / max(int(cfg["votes_per_parent"]), 1)
        route = f"runtime_vote101:{positive_votes}/{int(cfg['votes_per_parent'])}:thr={float(cfg['minichunk_threshold']):.4f}"
        return _clamp01(vote_fraction), route
    except Exception:
        return 0.0, "runtime_error"


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
        "artifact_path": str(RUNTIME_MODEL_PATH),
        "artifact_exists": RUNTIME_MODEL_PATH.exists(),
        "model_impl_path": str(MODEL_IMPL_PATH),
        "model_impl_exists": MODEL_IMPL_PATH.exists(),
        "features_path": str(FEATURES_PATH),
        "features_exists": FEATURES_PATH.exists(),
        "shape": _runtime_shape(),
        "vote101": _vote101_config(),
        "decision_threshold": _parent_decision_threshold(),
    }

    ok = _load_runtime_model()
    info["ok"] = ok
    if not ok:
        info["error"] = _RUNTIME_LOAD_ERROR
    elif _load_chunk_features() is None:
        info["ok"] = False
        info["error"] = _CHUNK_FEATURE_ERROR

    return info