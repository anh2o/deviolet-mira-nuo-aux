import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from .data import (
    HOLDOUT_CONFLICT_RATE,
    HOLDOUT_SCOPE_PROFILE,
    HOLDOUT_TRAFFIC_PROFILE,
    HOLDOUT_UTILITY_NOISE,
    PUBLIC_SCOPE_PROFILE,
    PUBLIC_TRAFFIC_PROFILES,
    PUBLIC_UTILITY_NOISE_LEVELS,
    TRAIN_CONFLICT_RATE,
    PolicyData,
    ScopeProfile,
    TrafficProfile,
)
from .metrics import EvaluationMetric
from .model import ModelConfig, PolicyMemoryModel
from .train import POLICY_COUNTS, SLOT_COUNTS, resolve_bridge

HOLDOUT_SLOTS = (32, 48, 64, 96, 128)
PROTOCOL_VERSION = 4
type JsonObject = dict[str, object]


def canonical_sha256(path: Path) -> str:
    payload = json.loads(path.read_text())
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def load_model(path: Path, device: torch.device) -> PolicyMemoryModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    bridge_cls = resolve_bridge(checkpoint["bridge_spec"])
    model = PolicyMemoryModel(ModelConfig(**checkpoint["model_config"]), bridge_cls)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


@torch.inference_mode()
def evaluate_cell(
    model: PolicyMemoryModel,
    *,
    n_policies: int,
    slot_count: int,
    decisions: int,
    seed: int,
    device: torch.device,
    conflict_rate: float = TRAIN_CONFLICT_RATE,
    traffic_profile: TrafficProfile | str = TrafficProfile.UNIFORM,
    scope_profile: ScopeProfile | str = PUBLIC_SCOPE_PROFILE,
    utility_noise: float = 0.0,
) -> EvaluationMetric:
    n_requests = 8
    batch_size = 128 if device.type == "cuda" else 16
    examples = max(1, (decisions + n_requests - 1) // n_requests)
    correct = 0
    total = 0
    conflict_correct = 0
    conflict_total = 0
    single_rule_correct = 0
    single_rule_total = 0
    hot_correct = 0
    hot_total = 0
    cold_correct = 0
    cold_total = 0
    rare_context_correct = 0
    rare_context_total = 0
    common_context_correct = 0
    common_context_total = 0
    data = PolicyData(model.config.condition_vocab, model.config.action_vocab)
    for offset in range(0, examples, batch_size):
        current = min(batch_size, examples - offset)
        batch = data.batch(
            batch_size=current,
            n_policies=n_policies,
            n_requests=n_requests,
            seed=seed + offset,
            conflict_rate=conflict_rate,
            traffic_profile=traffic_profile,
            scope_profile=scope_profile,
            utility_noise=utility_noise,
            device=device,
        )
        predictions = model(batch, slot_count).argmax(-1)
        matches = predictions.eq(batch.targets)
        conflict = batch.request_conflicts
        single_rule = conflict.logical_not()
        correct += int(matches.sum().item())
        total += int(batch.targets.numel())
        conflict_correct += int(matches[conflict].sum().item())
        conflict_total += int(conflict.sum().item())
        single_rule_correct += int(matches[single_rule].sum().item())
        single_rule_total += int(single_rule.sum().item())
        hot = batch.request_hot
        cold = hot.logical_not()
        hot_correct += int(matches[hot].sum().item())
        hot_total += int(hot.sum().item())
        cold_correct += int(matches[cold].sum().item())
        cold_total += int(cold.sum().item())
        rare_context = batch.request_rare_context
        common_context = rare_context.logical_not()
        rare_context_correct += int(matches[rare_context].sum().item())
        rare_context_total += int(rare_context.sum().item())
        common_context_correct += int(matches[common_context].sum().item())
        common_context_total += int(common_context.sum().item())
    return EvaluationMetric(
        correct,
        total,
        conflict_correct,
        conflict_total,
        single_rule_correct,
        single_rule_total,
        hot_correct,
        hot_total,
        cold_correct,
        cold_total,
        rare_context_correct,
        rare_context_total,
        common_context_correct,
        common_context_total,
    )


def write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sweep(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    cells: list[JsonObject] = []
    for traffic_profile in PUBLIC_TRAFFIC_PROFILES:
        for utility_noise in PUBLIC_UTILITY_NOISE_LEVELS:
            for n_policies in POLICY_COUNTS:
                for slot_count in SLOT_COUNTS:
                    metric = evaluate_cell(
                        model,
                        n_policies=n_policies,
                        slot_count=slot_count,
                        decisions=args.decisions,
                        seed=(
                            args.seed
                            + 100_000 * PUBLIC_TRAFFIC_PROFILES.index(traffic_profile)
                            + 10_000 * PUBLIC_UTILITY_NOISE_LEVELS.index(utility_noise)
                            + 1000 * n_policies
                            + slot_count
                        ),
                        device=device,
                        traffic_profile=traffic_profile,
                        scope_profile=PUBLIC_SCOPE_PROFILE,
                        utility_noise=utility_noise,
                    )
                    cell = {
                        "traffic_profile": traffic_profile,
                        "scope_profile": PUBLIC_SCOPE_PROFILE,
                        "utility_noise": utility_noise,
                        "n_policies": n_policies,
                        "slot_count": slot_count,
                        **metric.asdict(),
                    }
                    cells.append(cell)
                    print(json.dumps(cell), flush=True)
    write_json(
        args.output,
        {
            "protocol_version": PROTOCOL_VERSION,
            "kind": "sweep",
            "checkpoint": str(args.checkpoint),
            "seed": args.seed,
            "requested_decisions_per_cell": args.decisions,
            "conflict_rate": TRAIN_CONFLICT_RATE,
            "traffic_profiles": PUBLIC_TRAFFIC_PROFILES,
            "scope_profile": PUBLIC_SCOPE_PROFILE,
            "utility_noise_levels": PUBLIC_UTILITY_NOISE_LEVELS,
            "cells": cells,
        },
    )


def predict(args: argparse.Namespace) -> None:
    if not 1 <= args.predicted_k <= 128:
        raise SystemExit("predicted-k must be in [1, 128]")
    if len(args.family.strip()) < 8 or len(args.rationale.strip()) < 20:
        raise SystemExit("family or rationale is too short to be meaningful")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "kind": "prediction_commit",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "sweep_sha256": canonical_sha256(args.sweep),
        "target_accuracy": 0.9,
        "holdout_conflict_rate": HOLDOUT_CONFLICT_RATE,
        "holdout_traffic_profile": HOLDOUT_TRAFFIC_PROFILE,
        "holdout_scope_profile": HOLDOUT_SCOPE_PROFILE,
        "holdout_utility_noise": HOLDOUT_UTILITY_NOISE,
        "n_policies": 128,
        "predicted_k": args.predicted_k,
        "functional_family": args.family,
        "rationale": args.rationale,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


def holdout(args: argparse.Namespace) -> None:
    prediction = json.loads(args.prediction.read_text())
    if prediction.get("kind") != "prediction_commit":
        raise SystemExit("invalid prediction commit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    cells: list[JsonObject] = []
    for slot_count in HOLDOUT_SLOTS:
        metric = evaluate_cell(
            model,
            n_policies=128,
            slot_count=slot_count,
            decisions=args.decisions,
            seed=args.seed + slot_count,
            device=device,
            conflict_rate=HOLDOUT_CONFLICT_RATE,
            traffic_profile=HOLDOUT_TRAFFIC_PROFILE,
            scope_profile=HOLDOUT_SCOPE_PROFILE,
            utility_noise=HOLDOUT_UTILITY_NOISE,
        )
        cell = {
            "n_policies": 128,
            "slot_count": slot_count,
            "traffic_profile": HOLDOUT_TRAFFIC_PROFILE,
            "scope_profile": HOLDOUT_SCOPE_PROFILE,
            "utility_noise": HOLDOUT_UTILITY_NOISE,
            **metric.asdict(),
        }
        cells.append(cell)
        print(json.dumps(cell), flush=True)
    predicted_k = int(prediction["predicted_k"])
    if predicted_k in HOLDOUT_SLOTS:
        prediction_cell = next(
            cell for cell in cells if cell["slot_count"] == predicted_k
        )
    else:
        prediction_metric = evaluate_cell(
            model,
            n_policies=128,
            slot_count=predicted_k,
            decisions=args.decisions,
            seed=args.seed + predicted_k,
            device=device,
            conflict_rate=HOLDOUT_CONFLICT_RATE,
            traffic_profile=HOLDOUT_TRAFFIC_PROFILE,
            scope_profile=HOLDOUT_SCOPE_PROFILE,
            utility_noise=HOLDOUT_UTILITY_NOISE,
        )
        prediction_cell = {
            "n_policies": 128,
            "slot_count": predicted_k,
            "traffic_profile": HOLDOUT_TRAFFIC_PROFILE,
            "scope_profile": HOLDOUT_SCOPE_PROFILE,
            "utility_noise": HOLDOUT_UTILITY_NOISE,
            **prediction_metric.asdict(),
        }
        print(json.dumps({"prediction_cell": prediction_cell}), flush=True)
    observed = min(
        (
            int(cell["slot_count"])
            for cell in (*cells, prediction_cell)
            if cell["accuracy"] >= prediction["target_accuracy"]
        ),
        default=None,
    )
    write_json(
        args.output,
        {
            "protocol_version": PROTOCOL_VERSION,
            "kind": "holdout",
            "checkpoint": str(args.checkpoint),
            "prediction_sha256": canonical_sha256(args.prediction),
            "prediction": predicted_k,
            "prediction_cell": prediction_cell,
            "observed_smallest_k": observed,
            "absolute_error": (
                abs(observed - predicted_k) if observed is not None else None
            ),
            "seed": args.seed,
            "requested_decisions_per_cell": args.decisions,
            "conflict_rate": HOLDOUT_CONFLICT_RATE,
            "traffic_profile": HOLDOUT_TRAFFIC_PROFILE,
            "scope_profile": HOLDOUT_SCOPE_PROFILE,
            "utility_noise": HOLDOUT_UTILITY_NOISE,
            "cells": cells,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    sweep_parser = commands.add_parser("sweep")
    sweep_parser.add_argument("--checkpoint", type=Path, required=True)
    sweep_parser.add_argument("--output", type=Path, required=True)
    sweep_parser.add_argument("--decisions", type=int, default=4096)
    sweep_parser.add_argument("--seed", type=int, default=91001)
    sweep_parser.set_defaults(function=sweep)

    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("--sweep", type=Path, required=True)
    predict_parser.add_argument("--predicted-k", type=int, required=True)
    predict_parser.add_argument("--family", required=True)
    predict_parser.add_argument("--rationale", required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.set_defaults(function=predict)

    holdout_parser = commands.add_parser("holdout")
    holdout_parser.add_argument("--checkpoint", type=Path, required=True)
    holdout_parser.add_argument("--prediction", type=Path, required=True)
    holdout_parser.add_argument("--output", type=Path, required=True)
    holdout_parser.add_argument("--decisions", type=int, default=4096)
    holdout_parser.add_argument("--seed", type=int, default=128001)
    holdout_parser.set_defaults(function=holdout)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "decisions", 1) <= 0:
        raise SystemExit("decisions must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
