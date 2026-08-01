import pytest
import torch
from torch import nn

from memory_tokens.data import PolicyData
from memory_tokens.model import (
    MEMORY_QUANTIZATION_SCALE,
    ModelConfig,
    PolicyMemoryModel,
)


class WrongShape(nn.Module):
    def __init__(self, d_model, n_heads, max_slots, memory_width):
        super().__init__()
        del n_heads, max_slots
        self.d_model = d_model
        self.memory_width = memory_width

    def write(self, policy_states, policy_mask, slot_count):
        del policy_mask
        return policy_states[:, : slot_count + 1, : self.memory_width]

    def read(self, request_states, memory):
        del memory
        return request_states


class ContainerLeak(WrongShape):
    def write(self, policy_states, policy_mask, slot_count):
        del policy_mask
        return {"memory": policy_states[:, :slot_count]}


def batch():
    return PolicyData().batch(batch_size=2, n_policies=8, n_requests=4, seed=11)


def config():
    return ModelConfig(d_model=48, n_heads=4, d_ff=96)


def test_rejects_more_than_k_slots():
    with pytest.raises(ValueError, match="memory shape"):
        PolicyMemoryModel(config(), WrongShape)(batch(), 2)


def test_rejects_containers_that_could_hide_state():
    with pytest.raises(TypeError, match="exactly one"):
        PolicyMemoryModel(config(), ContainerLeak)(batch(), 2)


def test_rejects_non_float32_memory():
    model = PolicyMemoryModel(config())
    with pytest.raises(TypeError, match="float32"):
        model._enforce_boundary(torch.zeros(2, 3, 24).half(), 2, 3)


def test_boundary_restricts_memory_to_signed_int8_codes():
    model = PolicyMemoryModel(config())
    memory = torch.linspace(-20, 20, 2 * 3 * 24).reshape(2, 3, 24)

    payload = model._quantize_memory(memory)
    bounded = model._enforce_boundary(memory, 2, 3)
    expected_codes = torch.round(memory / MEMORY_QUANTIZATION_SCALE).clamp(-128, 127)
    expected = expected_codes * MEMORY_QUANTIZATION_SCALE

    assert payload.dtype == torch.int8
    assert torch.equal(payload, expected_codes.to(torch.int8))
    torch.testing.assert_close(bounded, expected)
    assert int(payload.min()) == -128
    assert int(payload.max()) == 127
