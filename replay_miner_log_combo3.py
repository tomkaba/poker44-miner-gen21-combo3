#!/usr/bin/env python3
"""Replay miner request logs through the gen21-combo3 runtime flow."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Tuple


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = Path("/home/tk/training_gen22/miner_221.log")
DEFAULT_OUTPUT_PATH = REPO_ROOT / "miner221_combo3_replay.json"
UID232_SOURCE_ROOT = Path("/home/tk/others/uid232_artifact_eval")

COMPONENT_THRESHOLDS: Dict[str, float] = {
    "gen18_2": 0.988104,
    "gen17_tuner_pre6": 0.5,
    "gen22full2": 71.0 / 101.0,
    "ml17_pre3": 0.5,
    "uid232": 0.5,
}

RAW_SOURCE_SPECS: Dict[str, Tuple[Path, Path]] = {
    "gen18_2": (
        Path("/home/tk/release-Poker44-gen18tens2-ovsmpl"),
        Path("/home/tk/release-Poker44-gen18tens2-ovsmpl/poker44/miner_heuristics.py"),
    ),
    "gen17_tuner_pre6": (
        Path("/home/tk/release-Poker44-gen17-tuner-pre6"),
        Path("/home/tk/release-Poker44-gen17-tuner-pre6/poker44/miner_heuristics.py"),
    ),
    "gen22full2": (
        Path("/home/tk/release-Poker44-gen22full2_t71"),
        Path("/home/tk/release-Poker44-gen22full2_t71/poker44/miner_heuristics.py"),
    ),
    "ml17_pre3": (
        Path("/home/tk/release-Poker44-gen17-pre-3"),
        Path("/home/tk/release-Poker44-gen17-pre-3/poker44/miner_heuristics.py"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a miner JSONL log through gen21-combo3",
        allow_abbrev=False,
    )
    parser.add_argument("--log", default=str(DEFAULT_LOG_PATH), help="Path to miner JSONL log")
    parser.add_argument("--max-tasks", type=int, default=0, help="Process at most this many tasks (0 = all)")
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Stop after this many chunks across tasks (0 = all)",
    )
    parser.add_argument("--progress-every", type=int, default=1, help="Print progress every N tasks")
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Where to write the replay report JSON",
    )
    parser.add_argument(
        "--skip-raw-compare",
        action="store_true",
        help="Skip standalone source artifact comparison and only run combo3 replay",
    )
    return parser.parse_args()


def _bool_pattern(values: List[bool]) -> str:
    return "".join("T" if value else "F" for value in values)


@contextmanager
def _sys_path_prepend(path: Path) -> Iterator[None]:
    original = list(sys.path)
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path[:] = original


def _load_module(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_standalone_runtime_module(name: str, source_root: Path, module_path: Path) -> ModuleType:
    saved_modules: Dict[str, ModuleType] = {}
    for key in list(sys.modules):
        if key == "poker44" or key.startswith("poker44.") or key == "poker44_ml" or key.startswith("poker44_ml."):
            saved_modules[key] = sys.modules.pop(key)

    try:
        with _sys_path_prepend(source_root):
            module = _load_module(f"raw_{name}_miner_heuristics", module_path)
    finally:
        for key in list(sys.modules):
            if key == "poker44" or key.startswith("poker44.") or key == "poker44_ml" or key.startswith("poker44_ml."):
                sys.modules.pop(key, None)
        sys.modules.update(saved_modules)

    return module


def _build_raw_scorers() -> Dict[str, Callable[[List[dict]], Dict[str, Any]]]:
    scorers: Dict[str, Callable[[List[dict]], Dict[str, Any]]] = {}

    for name, (source_root, module_path) in RAW_SOURCE_SPECS.items():
        module = _load_standalone_runtime_module(name, source_root, module_path)
        threshold = COMPONENT_THRESHOLDS[name]

        def _score(chunk: List[dict], module=module, threshold=threshold, source_root=source_root) -> Dict[str, Any]:
            with _sys_path_prepend(source_root):
                score, route = module.score_chunk_runtime_with_route(chunk)
            score_f = round(float(score), 6)
            return {
                "score": score_f,
                "route": str(route),
                "prediction": bool(score_f >= threshold),
                "threshold": threshold,
                "error": None,
            }

        scorers[name] = _score

    uid232_module = _load_module("raw_uid232_score_script", UID232_SOURCE_ROOT / "score_uid232_v5_artifact.py")

    def _score_uid232(chunk: List[dict]) -> Dict[str, Any]:
        with _sys_path_prepend(UID232_SOURCE_ROOT):
            result = uid232_module.score_chunks(
                UID232_SOURCE_ROOT,
                "super_features",
                [chunk],
                prediction_threshold=COMPONENT_THRESHOLDS["uid232"],
            )[0]
        score_f = round(float(result["score"]), 6)
        return {
            "score": score_f,
            "route": "uid232:super_features",
            "prediction": bool(score_f >= COMPONENT_THRESHOLDS["uid232"]),
            "threshold": COMPONENT_THRESHOLDS["uid232"],
            "error": None,
        }

    scorers["uid232"] = _score_uid232
    return scorers


RAW_SCORERS = _build_raw_scorers()

_ORIGINAL_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(REPO_ROOT))
import neurons.miner as miner_module  # noqa: E402
from neurons.miner import Miner  # noqa: E402
from poker44.miner_heuristics import score_chunk_runtime_bundle  # noqa: E402
from poker44.validator.synapse import DetectionSynapse  # noqa: E402
sys.argv = _ORIGINAL_ARGV


def _build_stub_miner() -> Miner:
    miner = Miner.__new__(Miner)
    miner.model_manifest = {"model_name": "gen21-combo3", "replay": True}
    miner._project_root = REPO_ROOT
    miner.wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="replay-hotkey"))
    miner.uid = "replay"
    miner._append_request_log = lambda **kwargs: None
    return miner


def _compare_component_result(combo_result: Dict[str, Any], raw_result: Dict[str, Any]) -> Dict[str, Any]:
    score_equal = math.isclose(float(combo_result["score"]), float(raw_result["score"]), abs_tol=1e-6)
    return {
        "score_match": score_equal,
        "prediction_match": bool(combo_result["prediction"] == raw_result["prediction"]),
        "route_match": str(combo_result["route"]) == str(raw_result["route"]),
    }


def _forward_with_bundle_cache(miner: Miner, chunks: List[List[dict]]) -> Tuple[DetectionSynapse, List[Dict[str, Any]]]:
    cached_bundles: List[Dict[str, Any]] = []
    original = miner_module.score_chunk_runtime_bundle

    def _wrapped(chunk: List[dict]) -> Dict[str, Any]:
        result = original(chunk)
        cached_bundles.append(result)
        return result

    miner_module.score_chunk_runtime_bundle = _wrapped
    try:
        synapse = DetectionSynapse(chunks=chunks)
        replayed = asyncio.run(Miner.forward(miner, synapse))
    finally:
        miner_module.score_chunk_runtime_bundle = original

    return replayed, cached_bundles


def main() -> int:
    args = parse_args()
    log_path = Path(args.log).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    miner = _build_stub_miner()
    task_results: List[Dict[str, Any]] = []
    component_match_counters: Dict[str, Counter[str]] = {name: Counter() for name in COMPONENT_THRESHOLDS}
    component_mismatch_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    uid232_outvoted_by = Counter()
    uid232_outvote_sets = Counter()
    total_chunks = 0
    combo_true = 0
    combo_false = 0
    logged_true = 0
    logged_false = 0
    uid232_true = 0
    uid232_false = 0
    combo_vs_logged_match = 0
    combo_vs_uid232_match = 0

    with log_path.open("r", encoding="utf-8") as fh:
        for task_index, line in enumerate(fh, start=1):
            if args.max_tasks > 0 and task_index > args.max_tasks:
                break

            entry = json.loads(line)
            chunks = entry.get("chunks") or []
            logged_predictions = [bool(value) for value in (entry.get("predictions") or [])]
            logged_scores = [float(value) for value in (entry.get("scores") or [])]
            if len(chunks) != len(logged_predictions):
                raise ValueError(f"Task {task_index} has mismatched chunks/predictions lengths")

            if args.max_chunks > 0:
                remaining = args.max_chunks - total_chunks
                if remaining <= 0:
                    break
                if len(chunks) > remaining:
                    chunks = chunks[:remaining]
                    logged_predictions = logged_predictions[:remaining]
                    logged_scores = logged_scores[:remaining]

            replayed, cached_bundles = _forward_with_bundle_cache(miner, chunks)
            combo_predictions = [bool(value) for value in (replayed.predictions or [])]
            combo_scores = [float(value) for value in (replayed.risk_scores or [])]

            chunk_results: List[Dict[str, Any]] = []
            for chunk_index, chunk in enumerate(chunks):
                bundle = cached_bundles[chunk_index]
                component_results = bundle["components"]
                uid232_component = component_results["uid232"]
                raw_component_results: Dict[str, Dict[str, Any]] = {}
                component_comparisons: Dict[str, Dict[str, Any]] = {}

                if not args.skip_raw_compare:
                    for component_name, combo_component in component_results.items():
                        raw_result = RAW_SCORERS[component_name](chunk)
                        raw_component_results[component_name] = raw_result
                        comparison = _compare_component_result(combo_component, raw_result)
                        component_comparisons[component_name] = comparison

                        component_match_counters[component_name]["chunks"] += 1
                        if comparison["score_match"]:
                            component_match_counters[component_name]["score_match"] += 1
                        if comparison["prediction_match"]:
                            component_match_counters[component_name]["prediction_match"] += 1
                        if comparison["route_match"]:
                            component_match_counters[component_name]["route_match"] += 1
                        if not all(comparison.values()) and len(component_mismatch_examples[component_name]) < 5:
                            component_mismatch_examples[component_name].append(
                                {
                                    "task_index": task_index,
                                    "chunk_index": chunk_index,
                                    "combo": combo_component,
                                    "raw": raw_result,
                                    "comparison": comparison,
                                }
                            )

                combo_prediction = combo_predictions[chunk_index]
                logged_prediction = logged_predictions[chunk_index]
                uid232_prediction = bool(uid232_component["prediction"])
                combo_match_logged = combo_prediction == logged_prediction
                combo_match_uid232 = combo_prediction == uid232_prediction

                total_chunks += 1
                combo_true += int(combo_prediction)
                combo_false += int(not combo_prediction)
                logged_true += int(logged_prediction)
                logged_false += int(not logged_prediction)
                uid232_true += int(uid232_prediction)
                uid232_false += int(not uid232_prediction)
                combo_vs_logged_match += int(combo_match_logged)
                combo_vs_uid232_match += int(combo_match_uid232)

                if not combo_match_uid232:
                    outvoters = sorted(
                        name
                        for name, component in component_results.items()
                        if name != "uid232" and bool(component["prediction"]) == combo_prediction and bool(component["prediction"]) != uid232_prediction
                    )
                    for name in outvoters:
                        uid232_outvoted_by[name] += 1
                    uid232_outvote_sets["+".join(outvoters) if outvoters else "unknown"] += 1

                chunk_results.append(
                    {
                        "chunk_index": chunk_index,
                        "chunk_size": len(chunk),
                        "logged": {
                            "score": logged_scores[chunk_index],
                            "prediction": logged_prediction,
                        },
                        "combo": {
                            "score": combo_scores[chunk_index],
                            "prediction": combo_prediction,
                            "route": bundle["route"],
                            "true_votes": bundle["true_votes"],
                            "false_votes": bundle["false_votes"],
                        },
                        "uid232_only": {
                            "score": uid232_component["score"],
                            "prediction": uid232_prediction,
                            "route": uid232_component["route"],
                        },
                        "combo_matches_logged": combo_match_logged,
                        "combo_matches_uid232": combo_match_uid232,
                        "uid232_outvoted_by": outvoters if not combo_match_uid232 else [],
                        "component_results": component_results,
                        "raw_component_results": raw_component_results,
                        "component_comparisons": component_comparisons,
                    }
                )

            task_results.append(
                {
                    "task_index": task_index,
                    "timestamp": float(entry.get("timestamp", 0.0)),
                    "validator_hotkey": str(entry.get("validator_hotkey", "")),
                    "miner_hotkey": str(entry.get("miner_hotkey", "")),
                    "chunk_count": len(chunks),
                    "logged_pattern": _bool_pattern(logged_predictions),
                    "combo_pattern": _bool_pattern(combo_predictions),
                    "uid232_pattern": _bool_pattern([bool(item["uid232_only"]["prediction"]) for item in chunk_results]),
                    "combo_vs_logged_match_count": int(sum(int(item["combo_matches_logged"]) for item in chunk_results)),
                    "combo_vs_uid232_match_count": int(sum(int(item["combo_matches_uid232"]) for item in chunk_results)),
                    "chunk_results": chunk_results,
                }
            )

            if args.progress_every > 0 and task_index % args.progress_every == 0:
                print(
                    f"[task] idx={task_index:03d} chunks={len(chunks)} combo_true={sum(combo_predictions)} logged_true={sum(logged_predictions)} uid232_true={sum(bool(item['uid232_only']['prediction']) for item in chunk_results)} combo_vs_logged={sum(int(item['combo_matches_logged']) for item in chunk_results)}/{len(chunks)} combo_vs_uid232={sum(int(item['combo_matches_uid232']) for item in chunk_results)}/{len(chunks)}",
                    flush=True,
                )

            if args.max_chunks > 0 and total_chunks >= args.max_chunks:
                break

    component_raw_compare_summary = {}
    if not args.skip_raw_compare:
        for name, counter in component_match_counters.items():
            chunks_seen = int(counter["chunks"])
            component_raw_compare_summary[name] = {
                "chunks_compared": chunks_seen,
                "score_match": int(counter["score_match"]),
                "prediction_match": int(counter["prediction_match"]),
                "route_match": int(counter["route_match"]),
                "mismatch_examples": component_mismatch_examples[name],
            }

    summary = {
        "log": str(log_path),
        "tasks_processed": len(task_results),
        "total_chunks": total_chunks,
        "combo": {
            "true": combo_true,
            "false": combo_false,
        },
        "logged_miner221": {
            "true": logged_true,
            "false": logged_false,
        },
        "uid232_only": {
            "true": uid232_true,
            "false": uid232_false,
        },
        "combo_vs_logged": {
            "match": combo_vs_logged_match,
            "different": total_chunks - combo_vs_logged_match,
            "match_rate": combo_vs_logged_match / max(total_chunks, 1),
        },
        "combo_vs_uid232": {
            "match": combo_vs_uid232_match,
            "different": total_chunks - combo_vs_uid232_match,
            "match_rate": combo_vs_uid232_match / max(total_chunks, 1),
            "uid232_outvoted_by_model": dict(sorted(uid232_outvoted_by.items())),
            "uid232_outvote_sets": dict(sorted(uid232_outvote_sets.items())),
        },
        "component_vs_raw": component_raw_compare_summary,
        "tasks": task_results,
    }

    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"[summary] tasks={len(task_results)} chunks={total_chunks} combo_vs_logged={combo_vs_logged_match}/{total_chunks} combo_vs_uid232={combo_vs_uid232_match}/{total_chunks}",
        flush=True,
    )
    print(f"[done] wrote={out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())