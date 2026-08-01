import torch
from torch import Tensor, nn


class CandidateMemory(nn.Module):
    """Starting implementation for candidate changes.

    Replace the write and read mechanisms. Keep their signatures unchanged.
    The same module must support every slot_count used by the protocol.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_slots: int = 128,
        memory_width: int = 24,
    ) -> None:
        super().__init__()
        del max_slots
        self.d_model = d_model
        self.memory_width = memory_width
        self.write_projection = nn.Linear(d_model, memory_width)
        self.reader = nn.MultiheadAttention(
            d_model,
            n_heads,
            kdim=memory_width,
            vdim=memory_width,
            batch_first=True,
            dropout=0.0,
        )
        self.read_norm = nn.LayerNorm(d_model)

    def write(
        self, policy_states: Tensor, policy_mask: Tensor, slot_count: int
    ) -> Tensor:
        """Average masked policy states over contiguous chunks."""
        batch, n_policies, width = policy_states.shape
        if width != self.d_model:
            raise ValueError("policy-state width changed")
        positions = torch.arange(n_policies, device=policy_states.device)
        bucket = torch.div(positions * slot_count, n_policies, rounding_mode="floor")
        bucket = bucket.clamp_max(slot_count - 1)

        slots = policy_states.new_zeros(batch, slot_count, width)
        counts = policy_states.new_zeros(batch, slot_count, 1)
        index = bucket.view(1, n_policies, 1).expand(batch, -1, width)
        slots.scatter_add_(1, index, policy_states * policy_mask.unsqueeze(-1))
        count_index = bucket.view(1, n_policies, 1).expand(batch, -1, 1)
        counts.scatter_add_(1, count_index, policy_mask.unsqueeze(-1).to(counts))
        pooled = slots / counts.clamp_min(1.0)
        return self.write_projection(pooled).float()

    def auxiliary_loss(
        self, policy_states: Tensor, policy_mask: Tensor, memory: Tensor
    ) -> Tensor:
        """Optional training-only objective; replace the zero scalar if useful."""
        del policy_mask, memory
        return policy_states.new_zeros(())

    def read(self, request_states: Tensor, memory: Tensor) -> Tensor:
        attended, _ = self.reader(request_states, memory, memory, need_weights=False)
        return self.read_norm(request_states + attended)
