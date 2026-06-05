#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Sequence


BUNDLE_ROOT = Path(__file__).resolve().parent
DEFAULT_REPO = BUNDLE_ROOT
DEFAULT_DB = BUNDLE_ROOT / "miner_logs.db"
DEFAULT_PYTHON = Path(sys.executable)

ARTIFACT_SPECS: Dict[str, Dict[str, str]] = {
    "super_features": {
        "model_path": "models/poker44_super_features_v5.joblib",
        "feature_extractor": "feature_extractors/super_features.py",
    },
    "benchmark_supervised": {
        "model_path": "models/poker44_benchmark_supervised_v5.joblib",
        "feature_extractor": "feature_extractors/benchmark_supervised.py",
    },
}


def load_chunk_by_dedup_id(db_path: Path, chunk_id: int) -> tuple[int, str, list[dict[str, Any]]]:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT id, chunk_hash, chunk_raw FROM chunk_dedup WHERE id = ?",
            (int(chunk_id),),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise ValueError(f"chunk_dedup.id not found: {chunk_id}")
    chunk = json.loads(row[2])
    if not isinstance(chunk, list):
        raise ValueError(f"chunk_dedup.id {chunk_id} does not contain a chunk list")
    return int(row[0]), str(row[1]), chunk


def load_chunk_by_observation_id(db_path: Path, observation_id: int) -> tuple[int, str, list[dict[str, Any]]]:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            '''
            SELECT o.id, d.chunk_hash, d.chunk_raw
            FROM chunk_observations o
            JOIN chunk_dedup d ON d.chunk_hash = o.chunk_hash
            WHERE o.id = ?
            ''',
            (int(observation_id),),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise ValueError(f"chunk_observations.id not found: {observation_id}")
    chunk = json.loads(row[2])
    if not isinstance(chunk, list):
        raise ValueError(f"chunk_observations.id {observation_id} does not resolve to a chunk list")
    return int(row[0]), str(row[1]), chunk


def resolve_artifact_spec(name_or_path: str, repo_path: Path) -> Dict[str, str]:
    key = str(name_or_path).strip().lower()
    if key in ARTIFACT_SPECS:
        spec = dict(ARTIFACT_SPECS[key])
        spec["artifact_name"] = key
        return spec

    model_path = Path(name_or_path)
    if not model_path.is_absolute():
        model_path = repo_path / model_path
    model_name = model_path.name
    if model_name == "poker44_super_features_v5.joblib":
        return {
            "artifact_name": "super_features",
            "model_path": str(model_path),
            "feature_extractor": str(repo_path / "feature_extractors" / "super_features.py"),
        }
    if model_name == "poker44_benchmark_supervised_v5.joblib":
        return {
            "artifact_name": "benchmark_supervised",
            "model_path": str(model_path),
            "feature_extractor": str(repo_path / "feature_extractors" / "benchmark_supervised.py"),
        }
    raise ValueError(f"Unsupported artifact: {name_or_path}")


def _load_chunk_features(feature_extractor_path: Path):
    spec = importlib.util.spec_from_file_location(feature_extractor_path.stem, feature_extractor_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load feature extractor: {feature_extractor_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    chunk_features = getattr(module, "chunk_features", None)
    if chunk_features is None:
        raise RuntimeError(f"Feature extractor does not define chunk_features: {feature_extractor_path}")
    return chunk_features


def _load_predictor(repo_path: Path, model_path: Path):
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    from poker44_ml.inference import Poker44Model  # type: ignore

    return Poker44Model(model_path)


def score_chunks(
    repo_path: Path,
    artifact: str,
    chunks: Sequence[list[dict[str, Any]]],
    *,
    prediction_threshold: float = 0.5,
    python_executable: Path = DEFAULT_PYTHON,
) -> list[dict[str, Any]]:
    del python_executable
    spec = resolve_artifact_spec(artifact, repo_path)
    model_path = Path(spec["model_path"])
    if not model_path.is_absolute():
        model_path = repo_path / model_path
    feature_extractor_path = Path(spec["feature_extractor"])
    if not feature_extractor_path.is_absolute():
        feature_extractor_path = repo_path / feature_extractor_path

    chunk_features = _load_chunk_features(feature_extractor_path)
    predictor = _load_predictor(repo_path, model_path)

    rows = []
    for chunk in chunks:
        feature_map = chunk_features(chunk)
        rows.append([float(feature_map.get(name, 0.0)) for name in predictor.feature_names])

    raw_scores = predictor._raw_model_scores(rows)
    calibrated_scores = predictor._apply_calibrator(raw_scores)
    remapped_scores = predictor._apply_score_remap(calibrated_scores)
    final_scores = [
        round(predictor._clamp01(value), 6)
        for value in predictor._apply_score_logit(remapped_scores)
    ]

    results = []
    for raw_score, final_score in zip(raw_scores, final_scores):
        results.append(
            {
                "score": float(final_score),
                "prediction": bool(final_score >= prediction_threshold),
                "raw_score": float(raw_score),
            }
        )
    return results


def score_chunk_hashes(
    repo_path: Path,
    db_path: Path,
    artifact: str,
    chunk_hashes: Sequence[str],
    *,
    prediction_threshold: float = 0.5,
    python_executable: Path = DEFAULT_PYTHON,
) -> Dict[str, Dict[str, Any]]:
    del python_executable
    if not chunk_hashes:
        return {}

    placeholders = ",".join("?" for _ in chunk_hashes)
    query = (
        "SELECT chunk_hash, chunk_raw FROM chunk_dedup "
        f"WHERE chunk_hash IN ({placeholders})"
    )
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(query, list(chunk_hashes)).fetchall()
    finally:
        conn.close()

    chunk_map = {}
    for chunk_hash, chunk_raw in rows:
        if not chunk_hash or not chunk_raw:
            continue
        chunk = json.loads(chunk_raw)
        if isinstance(chunk, list):
            chunk_map[str(chunk_hash)] = chunk

    present_hashes = []
    present_chunks = []
    results: Dict[str, Dict[str, Any]] = {}
    for chunk_hash in chunk_hashes:
        chunk = chunk_map.get(chunk_hash)
        if chunk is None:
            results[str(chunk_hash)] = {"missing": True}
            continue
        present_hashes.append(str(chunk_hash))
        present_chunks.append(chunk)

    scored = score_chunks(
        repo_path,
        artifact,
        present_chunks,
        prediction_threshold=prediction_threshold,
    ) if present_chunks else []

    for chunk_hash, payload in zip(present_hashes, scored):
        results[chunk_hash] = {"missing": False, **payload}
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score chunks with historical UID232 v5 artifacts")
    parser.add_argument(
        "--artifact",
        default="super_features",
        help="Artifact alias or model path. Supported aliases: super_features, benchmark_supervised",
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO, help="UID232 evaluator bundle root")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="miner_logs.db path")
    parser.add_argument(
        "--chunk-hash",
        action="append",
        default=[],
        help="Chunk hash to score from the DB. Repeatable.",
    )
    parser.add_argument(
        "--chunk-id",
        type=int,
        default=None,
        help="chunk_dedup.id to score from the DB.",
    )
    parser.add_argument(
        "--observation-id",
        type=int,
        default=None,
        help="chunk_observations.id to score via its chunk_hash.",
    )
    parser.add_argument(
        "--chunk-json",
        type=str,
        default="",
        help="Raw chunk JSON payload for one chunk.",
    )
    parser.add_argument(
        "--chunks-json",
        type=str,
        default="",
        help="JSON payload for a list of chunks. Use '-' to read from stdin.",
    )
    parser.add_argument(
        "--chunk-file",
        type=Path,
        default=None,
        help="Path to a file containing raw chunk JSON.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON, help="Compatibility flag; ignored by the self-contained bundle")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selector_count = sum(
        1
        for enabled in (
            bool(args.chunks_json),
            bool(args.chunk_json),
            args.chunk_file is not None,
            bool(args.chunk_hash),
            args.chunk_id is not None,
            args.observation_id is not None,
        )
        if enabled
    )
    if selector_count > 1:
        raise ValueError(
            "Provide exactly one selector: --chunk-id, --observation-id, --chunk-hash, --chunk-json, --chunk-file, or --chunks-json"
        )

    if args.chunks_json:
        chunks = json.loads(sys.stdin.read() if args.chunks_json == "-" else args.chunks_json)
        if not isinstance(chunks, list):
            raise ValueError("Chunk payload must decode to a list")
        print(
            json.dumps(
                score_chunks(
                    args.repo,
                    args.artifact,
                    chunks,
                    prediction_threshold=args.threshold,
                    python_executable=args.python,
                ),
                ensure_ascii=True,
            )
        )
        return 0

    if args.chunk_json or args.chunk_file is not None:
        if args.chunk_file is not None:
            chunk = json.loads(args.chunk_file.read_text(encoding="utf-8"))
        else:
            chunk = json.loads(args.chunk_json)
        if not isinstance(chunk, list):
            raise ValueError("Chunk payload must be a list of hands")
        payload = score_chunks(
            args.repo,
            args.artifact,
            [chunk],
            prediction_threshold=args.threshold,
            python_executable=args.python,
        )[0]
        print(
            json.dumps(
                {
                    "artifact": resolve_artifact_spec(args.artifact, args.repo)["artifact_name"],
                    "score": payload["score"],
                    "prediction": payload["prediction"],
                    "raw_score": payload["raw_score"],
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        return 0

    if args.chunk_id is not None:
        chunk_id, chunk_hash, chunk = load_chunk_by_dedup_id(args.db, args.chunk_id)
        payload = score_chunks(
            args.repo,
            args.artifact,
            [chunk],
            prediction_threshold=args.threshold,
            python_executable=args.python,
        )[0]
        print(
            json.dumps(
                {
                    "selector": "chunk_id",
                    "chunk_id": chunk_id,
                    "chunk_hash": chunk_hash,
                    "artifact": resolve_artifact_spec(args.artifact, args.repo)["artifact_name"],
                    "score": payload["score"],
                    "prediction": payload["prediction"],
                    "raw_score": payload["raw_score"],
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        return 0

    if args.observation_id is not None:
        observation_id, chunk_hash, chunk = load_chunk_by_observation_id(args.db, args.observation_id)
        payload = score_chunks(
            args.repo,
            args.artifact,
            [chunk],
            prediction_threshold=args.threshold,
            python_executable=args.python,
        )[0]
        print(
            json.dumps(
                {
                    "selector": "observation_id",
                    "observation_id": observation_id,
                    "chunk_hash": chunk_hash,
                    "artifact": resolve_artifact_spec(args.artifact, args.repo)["artifact_name"],
                    "score": payload["score"],
                    "prediction": payload["prediction"],
                    "raw_score": payload["raw_score"],
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        return 0

    if not args.chunk_hash:
        raise ValueError(
            "Provide --chunk-id, --observation-id, --chunk-hash, --chunk-json, --chunk-file, or --chunks-json"
        )

    results = score_chunk_hashes(
        args.repo,
        args.db,
        args.artifact,
        [str(value).strip() for value in args.chunk_hash if str(value).strip()],
        prediction_threshold=args.threshold,
        python_executable=args.python,
    )
    print(json.dumps(results, indent=2, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())