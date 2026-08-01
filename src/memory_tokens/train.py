import argparse
import importlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .data import (
    TRAIN_CONFLICT_RATE,
    TRAIN_SCOPE_PROFILES,
    TRAIN_TRAFFIC_PROFILES,
    TRAIN_UTILITY_NOISE_LEVELS,
    PolicyData,
)
from .model import ModelConfig, PolicyMemoryModel

POLICY_COUNTS = (8, 16, 32, 64)
SLOT_COUNTS = (2, 4, 8, 16, 32)


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1000
    batch_size: int = 128
    n_requests: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    seed: int = 1701


def resolve_bridge(spec: str) -> type[nn.Module]:
    module_name, separator, class_name = spec.partition(":")
    if not separator:
        raise ValueError("bridge must use module.path:ClassName")
    candidate = getattr(importlib.import_module(module_name), class_name)
    if not issubclass(candidate, nn.Module):
        raise TypeError("bridge class must inherit torch.nn.Module")
    return candidate


def save_checkpoint(
    path: Path,
    model: PolicyMemoryModel,
    train_config: TrainConfig,
    elapsed_seconds: float,
    bridge_spec: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_config": model.config.asdict(),
            "train_config": asdict(train_config),
            "bridge_spec": bridge_spec,
            "elapsed_seconds": elapsed_seconds,
            "state_dict": model.state_dict(),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--bridge", default="memory_tokens.candidate:CandidateMemory")
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.pt"))
    parser.add_argument("--quick-check", action="store_true")
    args = parser.parse_args()

    if args.steps <= 0 or args.batch_size <= 0:
        parser.error("steps and batch-size must be positive")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = ModelConfig(
        d_model=48 if args.quick_check else 96,
        d_ff=96 if args.quick_check else 192,
    )
    train_config = TrainConfig(
        steps=args.steps,
        batch_size=(min(args.batch_size, 16) if args.quick_check else args.batch_size),
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    bridge_cls = resolve_bridge(args.bridge)
    model = PolicyMemoryModel(model_config, bridge_cls).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    data = PolicyData(model_config.condition_vocab, model_config.action_vocab)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start = time.perf_counter()
    model.train()

    for step in range(train_config.steps):
        # Cycle through the public training grid.
        n_policies = POLICY_COUNTS[step % len(POLICY_COUNTS)]
        slot_count = SLOT_COUNTS[(step // len(POLICY_COUNTS)) % len(SLOT_COUNTS)]
        traffic_profile = TRAIN_TRAFFIC_PROFILES[step % len(TRAIN_TRAFFIC_PROFILES)]
        utility_noise = TRAIN_UTILITY_NOISE_LEVELS[
            (step // len(TRAIN_TRAFFIC_PROFILES)) % len(TRAIN_UTILITY_NOISE_LEVELS)
        ]
        scope_profile = TRAIN_SCOPE_PROFILES[
            (step // (len(TRAIN_TRAFFIC_PROFILES) * len(TRAIN_UTILITY_NOISE_LEVELS)))
            % len(TRAIN_SCOPE_PROFILES)
        ]
        batch = data.batch(
            batch_size=train_config.batch_size,
            n_policies=n_policies,
            n_requests=train_config.n_requests,
            seed=train_config.seed + step,
            conflict_rate=TRAIN_CONFLICT_RATE,
            traffic_profile=traffic_profile,
            scope_profile=scope_profile,
            utility_noise=utility_noise,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits, auxiliary = model(batch, slot_count, return_auxiliary=True)
            action_loss = F.cross_entropy(
                logits.reshape(-1, model_config.action_vocab),
                batch.targets.reshape(-1),
            )
            loss = action_loss + auxiliary
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 0 or (step + 1) % max(1, train_config.steps // 10) == 0:
            accuracy = (
                logits.detach().argmax(-1).eq(batch.targets).float().mean().item()
            )
            conflict = batch.request_conflicts
            conflict_accuracy = (
                logits.detach()
                .argmax(-1)[conflict]
                .eq(batch.targets[conflict])
                .float()
                .mean()
                .item()
                if conflict.any()
                else None
            )
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "loss": round(loss.item(), 5),
                        "action_loss": round(action_loss.item(), 5),
                        "auxiliary_loss": round(auxiliary.item(), 5),
                        "batch_accuracy": round(accuracy, 5),
                        "conflict_accuracy": (
                            round(conflict_accuracy, 5)
                            if conflict_accuracy is not None
                            else None
                        ),
                        "n_policies": n_policies,
                        "slot_count": slot_count,
                        "traffic_profile": traffic_profile,
                        "scope_profile": scope_profile,
                        "utility_noise": utility_noise,
                    }
                ),
                flush=True,
            )

    elapsed = time.perf_counter() - start
    save_checkpoint(args.output, model, train_config, elapsed, args.bridge)
    print(
        json.dumps(
            {
                "checkpoint": str(args.output),
                "device": str(device),
                "elapsed_seconds": round(elapsed, 3),
            }
        )
    )


if __name__ == "__main__":
    main()
