from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor

EXAMPLE_ACTION_LABELS = {
    0: "ALLOW",
    1: "FLAG",
    2: "REVIEW",
    3: "REDACT",
}

CONTEXT_COUNT = 8
ALL_CONTEXTS = (1 << CONTEXT_COUNT) - 1
TRAIN_CONFLICT_RATE = 0.35
HOLDOUT_CONFLICT_RATE = 0.20


class TrafficProfile(StrEnum):
    UNIFORM = "uniform"
    ZIPF = "zipf"
    BIMODAL = "bimodal"
    LOGNORMAL = "lognormal"


class ScopeProfile(StrEnum):
    SINGLETON = "singleton"
    NESTED = "nested"
    CROSSING = "crossing"


TRAIN_TRAFFIC_PROFILES = (
    TrafficProfile.UNIFORM,
    TrafficProfile.ZIPF,
    TrafficProfile.BIMODAL,
)
PUBLIC_TRAFFIC_PROFILES = (
    TrafficProfile.UNIFORM,
    TrafficProfile.ZIPF,
)
HOLDOUT_TRAFFIC_PROFILE = TrafficProfile.LOGNORMAL
TRAIN_SCOPE_PROFILES = (ScopeProfile.SINGLETON, ScopeProfile.NESTED)
PUBLIC_SCOPE_PROFILE = ScopeProfile.NESTED
HOLDOUT_SCOPE_PROFILE = ScopeProfile.CROSSING
TRAIN_UTILITY_NOISE_LEVELS = (0.0, 0.5)
PUBLIC_UTILITY_NOISE_LEVELS = (0.0, 0.5)
HOLDOUT_UTILITY_NOISE = 1.0


@dataclass(frozen=True)
class PolicyBatch:
    """Synthetic scoped policies and post-boundary requests."""

    policy_conditions: Tensor
    policy_scopes: Tensor
    policy_actions: Tensor
    policy_mask: Tensor
    request_conditions: Tensor
    request_contexts: Tensor
    targets: Tensor
    request_conflicts: Tensor
    policy_request_probability_estimates: Tensor
    policy_context_probability_estimates: Tensor
    request_probabilities: Tensor
    request_hot: Tensor
    request_rare_context: Tensor

    def to(self, device: torch.device | str) -> "PolicyBatch":
        return PolicyBatch(*(field.to(device) for field in self.__dict__.values()))

    @property
    def batch_size(self) -> int:
        return int(self.policy_conditions.shape[0])


class PolicyData:
    """Deterministic generator for ordered, partially overlapping rules."""

    def __init__(self, condition_vocab: int = 256, action_vocab: int = 256) -> None:
        if condition_vocab < 128:
            raise ValueError("condition_vocab must support the 128-policy holdout")
        self.condition_vocab = condition_vocab
        self.action_vocab = action_vocab

    def batch(
        self,
        *,
        batch_size: int,
        n_policies: int,
        n_requests: int,
        seed: int,
        conflict_rate: float = TRAIN_CONFLICT_RATE,
        traffic_profile: TrafficProfile | str = TrafficProfile.UNIFORM,
        scope_profile: ScopeProfile | str = PUBLIC_SCOPE_PROFILE,
        utility_noise: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> PolicyBatch:
        if not 1 <= n_policies <= self.condition_vocab:
            raise ValueError(f"n_policies must be in [1, {self.condition_vocab}]")
        if n_requests < 1:
            raise ValueError("n_requests must be positive")
        if not 0.0 <= conflict_rate < 1.0:
            raise ValueError("conflict_rate must be in [0, 1)")
        if utility_noise < 0.0:
            raise ValueError("utility_noise must be non-negative")
        try:
            traffic = TrafficProfile(traffic_profile)
        except ValueError as error:
            choices = ", ".join(item.value for item in TrafficProfile)
            raise ValueError(f"traffic_profile must be one of: {choices}") from error
        try:
            scopes = ScopeProfile(scope_profile)
        except ValueError as error:
            choices = ", ".join(item.value for item in ScopeProfile)
            raise ValueError(f"scope_profile must be one of: {choices}") from error

        base_count = max(1, round(n_policies * (1.0 - conflict_rate)))
        override_count = n_policies - base_count
        generator = torch.Generator(device="cpu").manual_seed(seed)

        condition_rank = torch.rand(
            batch_size, self.condition_vocab, generator=generator
        )
        base_conditions = condition_rank.topk(base_count, dim=1, largest=False).indices

        condition_order = torch.rand(batch_size, base_count, generator=generator)
        rank = condition_order.argsort(dim=1).argsort(dim=1).float() + 1.0
        if traffic is TrafficProfile.UNIFORM:
            condition_weight = torch.ones_like(rank)
            context_weight = torch.ones(batch_size, base_count, CONTEXT_COUNT)
        elif traffic is TrafficProfile.ZIPF:
            condition_weight = rank.pow(-1.25)
            context_rank = (
                torch.rand(batch_size, base_count, CONTEXT_COUNT, generator=generator)
                .argsort(dim=-1)
                .argsort(dim=-1)
            )
            context_weight = (context_rank.float() + 1.0).pow(-0.8)
        elif traffic is TrafficProfile.BIMODAL:
            hot_count = max(1, round(base_count * 0.2))
            condition_weight = torch.where(rank <= hot_count, 9.0, 1.0)
            dominant = torch.randint(
                CONTEXT_COUNT,
                (batch_size, base_count, 1),
                generator=generator,
            )
            context_ids = torch.arange(CONTEXT_COUNT).view(1, 1, -1)
            context_weight = torch.where(context_ids == dominant, 10.0, 1.0)
        else:
            condition_weight = torch.exp(
                torch.randn(batch_size, base_count, generator=generator) * 2.0
            )
            context_weight = torch.exp(
                torch.randn(
                    batch_size,
                    base_count,
                    CONTEXT_COUNT,
                    generator=generator,
                )
                * 1.75
            )

        pair_weight = condition_weight.unsqueeze(-1) * context_weight
        request_distribution = pair_weight / pair_weight.sum(dim=(1, 2), keepdim=True)
        estimate_noise = torch.randn(
            batch_size,
            base_count,
            CONTEXT_COUNT,
            generator=generator,
        )
        estimated_weight = pair_weight * torch.exp(estimate_noise * utility_noise)
        estimated_distribution = estimated_weight / estimated_weight.sum(
            dim=(1, 2), keepdim=True
        )

        base_scopes = torch.full(
            (batch_size, base_count), ALL_CONTEXTS, dtype=torch.int64
        )
        if override_count:
            override_base_index = torch.randint(
                base_count,
                (batch_size, override_count),
                generator=generator,
            )
            override_conditions = base_conditions.gather(1, override_base_index)
            override_scopes = self._sample_scopes(
                batch_size,
                override_count,
                scopes,
                generator,
            )
            policy_conditions = torch.cat((base_conditions, override_conditions), dim=1)
            policy_scopes = torch.cat((base_scopes, override_scopes), dim=1)
            policy_base_index = torch.cat(
                (
                    torch.arange(base_count).expand(batch_size, -1),
                    override_base_index,
                ),
                dim=1,
            )
        else:
            policy_conditions = base_conditions
            policy_scopes = base_scopes
            policy_base_index = torch.arange(base_count).expand(batch_size, -1)

        # A base rule precedes its partial overrides. Order remains meaningful
        # among base rules and among overrides.
        base_order = torch.rand(batch_size, base_count, generator=generator).argsort(
            dim=1
        )
        if override_count:
            override_order = torch.rand(
                batch_size, override_count, generator=generator
            ).argsort(dim=1)
            policy_order = torch.cat((base_order, override_order + base_count), dim=1)
        else:
            policy_order = base_order
        policy_conditions = policy_conditions.gather(1, policy_order)
        policy_scopes = policy_scopes.gather(1, policy_order)
        policy_base_index = policy_base_index.gather(1, policy_order)

        scope_bits = (
            policy_scopes.unsqueeze(-1)
            .bitwise_and(1 << torch.arange(CONTEXT_COUNT))
            .ne(0)
        )
        estimate_by_policy = estimated_distribution.gather(
            1,
            policy_base_index.unsqueeze(-1).expand(-1, -1, CONTEXT_COUNT),
        )
        policy_request_probability_estimates = (estimate_by_policy * scope_bits).sum(
            dim=-1
        )
        policy_context_probability_estimates = estimate_by_policy * scope_bits
        policy_actions = torch.randint(
            self.action_vocab,
            (batch_size, n_policies),
            generator=generator,
        )

        flat_distribution = request_distribution.flatten(1)
        request_positions = torch.multinomial(
            flat_distribution,
            n_requests,
            replacement=True,
            generator=generator,
        )
        request_base_index = torch.div(
            request_positions, CONTEXT_COUNT, rounding_mode="floor"
        )
        request_contexts = request_positions.remainder(CONTEXT_COUNT)
        request_conditions = base_conditions.gather(1, request_base_index)
        request_probabilities = flat_distribution.gather(1, request_positions)

        flat_rank = flat_distribution.argsort(dim=1, descending=True).argsort(dim=1)
        hot_count = max(1, round(base_count * CONTEXT_COUNT * 0.25))
        request_hot = flat_rank.gather(1, request_positions).lt(hot_count)
        context_mass = request_distribution.sum(dim=1)
        rare_context = context_mass.argsort(dim=1).argsort(dim=1).ge(CONTEXT_COUNT // 2)
        request_rare_context = rare_context.gather(1, request_contexts)

        same_condition = request_conditions.unsqueeze(-1).eq(
            policy_conditions.unsqueeze(1)
        )
        request_bits = 1 << request_contexts.unsqueeze(-1)
        in_scope = policy_scopes.unsqueeze(1).bitwise_and(request_bits).ne(0)
        matches = same_condition & in_scope
        priority = torch.arange(n_policies).view(1, 1, n_policies)
        winning_position = torch.where(matches, priority, -1).amax(dim=-1)
        targets = policy_actions.gather(1, winning_position)
        request_conflicts = matches.sum(dim=-1).gt(1)
        policy_mask = torch.ones(batch_size, n_policies, dtype=torch.bool)

        return PolicyBatch(
            policy_conditions=policy_conditions.to(device),
            policy_scopes=policy_scopes.to(device),
            policy_actions=policy_actions.to(device),
            policy_mask=policy_mask.to(device),
            request_conditions=request_conditions.to(device),
            request_contexts=request_contexts.to(device),
            targets=targets.to(device),
            request_conflicts=request_conflicts.to(device),
            policy_request_probability_estimates=(
                policy_request_probability_estimates.to(device)
            ),
            policy_context_probability_estimates=(
                policy_context_probability_estimates.to(device)
            ),
            request_probabilities=request_probabilities.to(device),
            request_hot=request_hot.to(device),
            request_rare_context=request_rare_context.to(device),
        )

    @staticmethod
    def _sample_scopes(
        batch_size: int,
        count: int,
        profile: ScopeProfile,
        generator: torch.Generator,
    ) -> Tensor:
        if profile is ScopeProfile.SINGLETON:
            context = torch.randint(
                CONTEXT_COUNT,
                (batch_size, count),
                generator=generator,
            )
            return 1 << context
        if profile is ScopeProfile.NESTED:
            width = torch.randint(
                1,
                CONTEXT_COUNT // 2 + 1,
                (batch_size, count),
                generator=generator,
            )
            start = torch.randint(
                CONTEXT_COUNT,
                (batch_size, count),
                generator=generator,
            )
            offsets = torch.arange(CONTEXT_COUNT).view(1, 1, -1)
            included = offsets.lt(width.unsqueeze(-1))
            contexts = (start.unsqueeze(-1) + offsets).remainder(CONTEXT_COUNT)
            return ((1 << contexts) * included).sum(dim=-1)

        # Crossing masks can overlap without containment. Exclude empty and full
        # masks because both collapse to simpler cases.
        return torch.randint(
            1,
            ALL_CONTEXTS,
            (batch_size, count),
            generator=generator,
        )
