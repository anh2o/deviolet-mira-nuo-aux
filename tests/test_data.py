import torch

from memory_tokens.data import (
    EXAMPLE_ACTION_LABELS,
    HOLDOUT_CONFLICT_RATE,
    TRAIN_CONFLICT_RATE,
    PolicyData,
    ScopeProfile,
    TrafficProfile,
)


def test_human_readable_action_is_named_flag():
    assert EXAMPLE_ACTION_LABELS[1] == "FLAG"


def test_generator_is_deterministic_and_last_matching_policy_wins():
    data = PolicyData()
    first = data.batch(batch_size=7, n_policies=32, n_requests=8, seed=123)
    second = data.batch(batch_size=7, n_policies=32, n_requests=8, seed=123)
    for name in first.__dict__:
        assert torch.equal(getattr(first, name), getattr(second, name))

    for row in range(first.batch_size):
        answers = []
        for condition, context in zip(
            first.request_conditions[row].tolist(),
            first.request_contexts[row].tolist(),
            strict=True,
        ):
            matches = [
                action
                for policy_condition, scope, action in zip(
                    first.policy_conditions[row].tolist(),
                    first.policy_scopes[row].tolist(),
                    first.policy_actions[row].tolist(),
                    strict=True,
                )
                if policy_condition == condition and scope & (1 << context)
            ]
            answers.append(matches[-1])
        assert answers == first.targets[row].tolist()
        assert first.policy_conditions[row].unique().numel() == 21

        same_condition = (
            first.request_conditions[row].unsqueeze(-1).eq(first.policy_conditions[row])
        )
        in_scope = (
            first.policy_scopes[row]
            .unsqueeze(0)
            .bitwise_and(1 << first.request_contexts[row].unsqueeze(-1))
            .ne(0)
        )
        counts = (same_condition & in_scope).sum(dim=-1)
        assert torch.equal(first.request_conflicts[row], counts.gt(1))


def test_generator_supports_all_required_policy_counts():
    data = PolicyData()
    for n_policies in (8, 16, 32, 64, 128):
        batch = data.batch(
            batch_size=2,
            n_policies=n_policies,
            n_requests=8,
            seed=n_policies,
        )
        assert batch.policy_conditions.shape == (2, n_policies)
        assert batch.request_conditions.shape == (2, 8)


def test_holdout_has_fewer_conflicts_than_training_data():
    data = PolicyData()
    training = data.batch(
        batch_size=256,
        n_policies=128,
        n_requests=8,
        seed=301,
        conflict_rate=TRAIN_CONFLICT_RATE,
    )
    holdout = data.batch(
        batch_size=256,
        n_policies=128,
        n_requests=8,
        seed=301,
        conflict_rate=HOLDOUT_CONFLICT_RATE,
    )

    holdout_rate = holdout.request_conflicts.float().mean()
    training_rate = training.request_conflicts.float().mean()
    assert holdout_rate < training_rate
    assert holdout.policy_conditions[0].unique().numel() == 102
    assert training.policy_conditions[0].unique().numel() == 83


def test_traffic_profiles_change_request_concentration():
    data = PolicyData()
    uniform = data.batch(
        batch_size=512,
        n_policies=64,
        n_requests=32,
        seed=401,
        traffic_profile=TrafficProfile.UNIFORM,
    )
    zipf = data.batch(
        batch_size=512,
        n_policies=64,
        n_requests=32,
        seed=401,
        traffic_profile=TrafficProfile.ZIPF,
    )

    assert zipf.request_hot.float().mean() > 0.7
    assert uniform.request_hot.float().mean() < 0.35
    assert zipf.request_probabilities.mean() > uniform.request_probabilities.mean()


def test_utility_noise_changes_metadata_not_true_traffic():
    data = PolicyData()
    clean = data.batch(
        batch_size=16,
        n_policies=64,
        n_requests=16,
        seed=501,
        traffic_profile=TrafficProfile.ZIPF,
        utility_noise=0.0,
    )
    noisy = data.batch(
        batch_size=16,
        n_policies=64,
        n_requests=16,
        seed=501,
        traffic_profile=TrafficProfile.ZIPF,
        utility_noise=1.0,
    )

    assert torch.equal(clean.request_conditions, noisy.request_conditions)
    assert torch.equal(clean.request_probabilities, noisy.request_probabilities)
    assert not torch.equal(
        clean.policy_request_probability_estimates,
        noisy.policy_request_probability_estimates,
    )
    torch.testing.assert_close(
        noisy.policy_context_probability_estimates.sum(dim=-1),
        noisy.policy_request_probability_estimates,
    )
    context_ids = torch.arange(8)
    outside_scope = (
        noisy.policy_scopes.unsqueeze(-1).bitwise_and(1 << context_ids).eq(0)
    )
    assert noisy.policy_context_probability_estimates[outside_scope].eq(0).all()


def test_scope_profiles_change_partial_override_geometry():
    data = PolicyData()
    singleton = data.batch(
        batch_size=32,
        n_policies=64,
        n_requests=8,
        seed=601,
        scope_profile=ScopeProfile.SINGLETON,
    )
    crossing = data.batch(
        batch_size=32,
        n_policies=64,
        n_requests=8,
        seed=601,
        scope_profile=ScopeProfile.CROSSING,
    )

    base_count = round(64 * (1.0 - TRAIN_CONFLICT_RATE))
    singleton_overrides = singleton.policy_scopes[:, base_count:]
    crossing_overrides = crossing.policy_scopes[:, base_count:]
    assert singleton_overrides.bitwise_and(singleton_overrides - 1).eq(0).all()
    assert crossing_overrides.bitwise_and(crossing_overrides - 1).ne(0).any()
    assert not torch.equal(singleton.targets, crossing.targets)
