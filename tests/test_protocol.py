import json
from argparse import Namespace

from memory_tokens.data import (
    HOLDOUT_CONFLICT_RATE,
    HOLDOUT_SCOPE_PROFILE,
    HOLDOUT_TRAFFIC_PROFILE,
)
from memory_tokens.experiment import canonical_sha256, predict


def test_prediction_commit_binds_to_canonical_sweep(tmp_path):
    sweep = tmp_path / "sweep.json"
    prediction = tmp_path / "prediction.json"
    sweep.write_text(json.dumps({"kind": "sweep", "cells": []}, indent=4))
    expected_hash = canonical_sha256(sweep)

    predict(
        Namespace(
            sweep=sweep,
            predicted_k=1,
            family="candidate-selected model family",
            rationale="The candidate supplies evidence from the measured sweep.",
            output=prediction,
        )
    )
    payload = json.loads(prediction.read_text())
    assert payload["sweep_sha256"] == expected_hash
    assert payload["predicted_k"] == 1
    assert payload["target_accuracy"] == 0.9
    assert payload["holdout_conflict_rate"] == HOLDOUT_CONFLICT_RATE
    assert payload["holdout_traffic_profile"] == HOLDOUT_TRAFFIC_PROFILE
    assert payload["holdout_scope_profile"] == HOLDOUT_SCOPE_PROFILE
