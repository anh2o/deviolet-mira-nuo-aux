import torch
from torch import nn

from memory_tokens.data import PolicyData
from memory_tokens.model import ModelConfig, PolicyMemoryModel


def tiny_model() -> PolicyMemoryModel:
    return PolicyMemoryModel(
        ModelConfig(d_model=48, n_heads=4, d_ff=96, memory_width=24)
    )


def test_model_has_exactly_four_transformer_layers():
    model = tiny_model()
    layers = [
        module
        for module in model.modules()
        if isinstance(module, nn.TransformerEncoderLayer)
    ]
    assert len(layers) == 4


def test_all_public_budget_shapes_run():
    model = tiny_model().eval()
    data = PolicyData()
    with torch.inference_mode():
        for n_policies in (8, 16, 32, 64):
            batch = data.batch(
                batch_size=2,
                n_policies=n_policies,
                n_requests=8,
                seed=n_policies,
            )
            for slot_count in (2, 4, 8, 16, 32):
                logits = model(batch, slot_count)
                assert logits.shape == (2, 8, 256)
                assert torch.isfinite(logits).all()


def test_gradients_cross_the_memory_boundary():
    model = tiny_model()
    batch = PolicyData().batch(batch_size=2, n_policies=8, n_requests=4, seed=9)
    model(batch, 4).sum().backward()
    assert model.policy_mix[0].weight.grad is not None
    assert torch.isfinite(model.policy_mix[0].weight.grad).all()
    assert model.bridge.reader.q_proj_weight.grad is not None


def test_training_objective_hook_is_a_finite_scalar():
    model = tiny_model()
    batch = PolicyData().batch(batch_size=2, n_policies=8, n_requests=4, seed=19)
    logits, auxiliary = model(batch, 4, return_auxiliary=True)
    assert logits.shape == (2, 4, 256)
    assert auxiliary.shape == ()
    assert torch.isfinite(auxiliary)


def test_full_encoder_exposes_documented_scope_and_estimate_lanes():
    model = PolicyMemoryModel().eval()
    batch = PolicyData().batch(batch_size=2, n_policies=16, n_requests=4, seed=29)
    with torch.inference_mode():
        states = model.encode_policies(batch)

    torch.testing.assert_close(
        states[..., 0],
        (batch.policy_conditions.float() - 128) * 0.125,
    )
    context_ids = torch.arange(8)
    expected_scope = (
        batch.policy_scopes.unsqueeze(-1).bitwise_and(1 << context_ids).ne(0).float()
    )
    torch.testing.assert_close(states[..., 24:32], expected_scope)
    torch.testing.assert_close(
        states[..., 32:40],
        batch.policy_context_probability_estimates,
    )
    torch.testing.assert_close(
        states[..., 55],
        batch.policy_request_probability_estimates,
    )
