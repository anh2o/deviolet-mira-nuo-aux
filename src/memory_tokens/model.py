from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .candidate import CandidateMemory
from .data import CONTEXT_COUNT, PolicyBatch

MEMORY_QUANTIZATION_SCALE = 1 / 8


@dataclass(frozen=True)
class ModelConfig:
    condition_vocab: int = 256
    action_vocab: int = 256
    context_count: int = CONTEXT_COUNT
    d_model: int = 96
    n_heads: int = 4
    d_ff: int = 192
    memory_width: int = 24
    dropout: float = 0.0
    max_policies: int = 128
    max_requests: int = 16
    max_slots: int = 128

    def asdict(self) -> dict[str, int | float]:
        return asdict(self)


class PolicyMemoryModel(nn.Module):
    """Fixed four-layer model with an instrumented hard boundary."""

    def __init__(
        self,
        config: ModelConfig | None = None,
        bridge_cls: type[nn.Module] = CandidateMemory,
    ) -> None:
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config
        d = config.d_model

        self.condition_embedding = nn.Embedding(config.condition_vocab, d)
        self.action_embedding = nn.Embedding(config.action_vocab, d)
        self.context_embedding = nn.Embedding(config.context_count, d)
        split = d // 2
        with torch.no_grad():
            self.condition_embedding.weight[:, split:].zero_()
            self.action_embedding.weight[:, :split].zero_()
        self.policy_priority = nn.Sequential(
            nn.Linear(5, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.policy_utility = nn.Sequential(
            nn.Linear(3, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.policy_scope = nn.Sequential(
            nn.Linear(config.context_count, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.request_position = nn.Embedding(config.max_requests, d)
        self.policy_mix = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())

        layer_args = dict(
            d_model=d,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Two layers before and two after the boundary: four total.
        self.policy_layers = nn.ModuleList(
            nn.TransformerEncoderLayer(**layer_args) for _ in range(2)
        )
        self.request_layers = nn.ModuleList(
            nn.TransformerEncoderLayer(**layer_args) for _ in range(2)
        )
        self.policy_norm = nn.LayerNorm(d)
        self.request_norm = nn.LayerNorm(d)
        self.bridge = bridge_cls(
            d,
            config.n_heads,
            config.max_slots,
            config.memory_width,
        )
        self.output = nn.Linear(d, config.action_vocab)
        self.output.weight = self.action_embedding.weight

    def encode_policies(self, batch: PolicyBatch) -> Tensor:
        """Encode the policy sequence."""
        condition_states = self.condition_embedding(batch.policy_conditions)
        action_states = self.action_embedding(batch.policy_actions)
        context_ids = torch.arange(
            self.config.context_count,
            device=batch.policy_scopes.device,
        )
        scope_bits = (
            batch.policy_scopes.unsqueeze(-1).bitwise_and(1 << context_ids).ne(0)
        )
        scope_features = scope_bits.to(condition_states.dtype)

        policy_count = batch.policy_conditions.size(1)
        coordinate = (
            torch.arange(
                policy_count,
                device=batch.policy_conditions.device,
                dtype=condition_states.dtype,
            )
            + 1
        ) / policy_count
        priority_features = torch.stack(
            (
                coordinate,
                coordinate.square(),
                torch.sin(torch.pi * coordinate),
                torch.cos(torch.pi * coordinate),
                torch.ones_like(coordinate),
            ),
            dim=-1,
        )
        priority_states = self.policy_priority(priority_features).unsqueeze(0)
        probability_estimate = batch.policy_request_probability_estimates
        utility_features = torch.stack(
            (
                probability_estimate,
                probability_estimate.sqrt(),
                torch.log(probability_estimate.clamp_min(1e-8)),
            ),
            dim=-1,
        )
        utility_states = self.policy_utility(utility_features)

        policy_states = self.policy_mix(
            torch.cat((condition_states, action_states), dim=-1)
        )
        policy_states = (
            policy_states
            + condition_states
            + priority_states
            + utility_states
            + self.policy_scope(scope_features)
        )

        padding_mask = batch.policy_mask.logical_not()
        for encoder_layer in self.policy_layers:
            policy_states = encoder_layer(
                policy_states,
                src_key_padding_mask=padding_mask,
            )

        contextual_states = self.policy_norm(policy_states)
        key_width = self.config.d_model // 4
        scope_width = self.config.context_count
        estimate_width = self.config.context_count
        left_context_width = (
            self.config.d_model - 2 * key_width - scope_width - 1 - estimate_width
        ) // 2
        right_context_width = self.config.d_model - (
            2 * key_width + scope_width + estimate_width + left_context_width + 1
        )
        action_start = self.config.d_model // 2
        context_start = key_width + scope_width
        context_end = context_start + left_context_width
        condition_key = condition_states[..., :key_width].clone()
        condition_key[..., 0] = (
            batch.policy_conditions.to(condition_key.dtype) - 128
        ) * MEMORY_QUANTIZATION_SCALE
        return torch.cat(
            (
                condition_key,
                scope_features,
                batch.policy_context_probability_estimates,
                contextual_states[
                    ...,
                    context_start:context_end,
                ],
                probability_estimate.unsqueeze(-1),
                action_states[..., action_start : action_start + key_width],
                contextual_states[..., -right_context_width:],
            ),
            dim=-1,
        )

    def encode_requests(self, batch: PolicyBatch) -> Tensor:
        """Encode the request sequence."""
        condition_states = self.condition_embedding(batch.request_conditions)
        context_states = self.context_embedding(batch.request_contexts)

        request_count = batch.request_conditions.size(1)
        position_ids = torch.arange(
            request_count,
            device=batch.request_conditions.device,
        )
        position_states = self.request_position(position_ids).unsqueeze(0)
        request_states = condition_states + context_states + position_states

        for encoder_layer in self.request_layers:
            request_states = encoder_layer(request_states)

        contextual_states = self.request_norm(request_states)
        key_width = self.config.d_model // 4
        context_one_hot = torch.nn.functional.one_hot(
            batch.request_contexts,
            num_classes=self.config.context_count,
        ).to(contextual_states.dtype)
        tail = contextual_states[..., key_width + self.config.context_count :]
        condition_key = condition_states[..., :key_width].clone()
        condition_key[..., 0] = (
            batch.request_conditions.to(condition_key.dtype) - 128
        ) * MEMORY_QUANTIZATION_SCALE
        return torch.cat(
            (
                condition_key,
                context_one_hot,
                tail,
            ),
            dim=-1,
        )

    def _enforce_boundary(
        self, memory: Tensor, batch_size: int, slot_count: int
    ) -> Tensor:
        expected = (batch_size, slot_count, self.config.memory_width)
        if type(memory) is not Tensor:
            raise TypeError("write must return exactly one torch.Tensor")
        if memory.shape != expected:
            raise ValueError(f"memory shape {tuple(memory.shape)} != {expected}")
        if memory.dtype != torch.float32:
            raise TypeError("memory must be float32")
        if not torch.isfinite(memory).all():
            raise ValueError("memory contains non-finite values")

        payload = self._quantize_memory(memory)
        decoded = payload.float() * MEMORY_QUANTIZATION_SCALE

        # The forward value contains only information from the byte payload.
        # The straight-through path keeps the writer trainable.
        return memory + (decoded - memory).detach()

    @staticmethod
    def _quantize_memory(memory: Tensor) -> Tensor:
        payload = (
            torch.round(memory / MEMORY_QUANTIZATION_SCALE)
            .clamp(
                torch.iinfo(torch.int8).min,
                torch.iinfo(torch.int8).max,
            )
            .to(torch.int8)
        )
        return payload

    def forward(
        self,
        batch: PolicyBatch,
        slot_count: int,
        *,
        return_auxiliary: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if not 1 <= slot_count <= self.config.max_slots:
            raise ValueError("slot_count outside configured budget")

        policy_states = self.encode_policies(batch)
        memory = self.bridge.write(policy_states, batch.policy_mask, slot_count)
        memory = self._enforce_boundary(memory, batch.batch_size, slot_count)
        auxiliary_hook = getattr(self.bridge, "auxiliary_loss", None)
        auxiliary = (
            auxiliary_hook(policy_states, batch.policy_mask, memory)
            if auxiliary_hook is not None
            else policy_states.new_zeros(())
        )
        if (
            type(auxiliary) is not Tensor
            or auxiliary.shape != ()
            or not torch.isfinite(auxiliary)
        ):
            raise ValueError("auxiliary_loss must return one finite scalar tensor")

        # No policy state passes this point except the bounded memory tensor.
        request_states = self.encode_requests(batch)
        recalled = self.bridge.read(request_states, memory)
        if recalled.shape != request_states.shape:
            raise ValueError("read must return one state per request")
        logits = self.output(recalled)
        return (logits, auxiliary) if return_auxiliary else logits
