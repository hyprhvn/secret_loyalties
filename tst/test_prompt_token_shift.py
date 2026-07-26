"""
Tests for the (token x layer) matrices of the prompt-token shift pipeline.
"""

from __future__ import annotations

import pytest

pytest.importorskip("transformers")

import torch

from secret_loyalties.viz.prompt_token_shift import (
    SINGLE_MODEL_MATRICES,
    layer_delta_matrix,
    token_delta_matrix,
)


def make_hidden() -> torch.Tensor:
    """
    Build hidden states of three tokens over an embedding plus two transformer layers.

    The embedding row is filled with values that must never show up in a result, since every metric
    drops it. Per token, layer 1 then layer 2:
    token 0 ``[0, 0]`` / ``[1, 0]``, token 1 ``[3, 4]`` / ``[1, 0]``, token 2 ``[3, 4]`` / ``[4, 0]``.

    :returns: A ``(3, 3, 2)`` tensor.
    """
    return torch.tensor(
        [
            [[99.0, 99.0], [0.0, 0.0], [1.0, 0.0]],
            [[99.0, 99.0], [3.0, 4.0], [1.0, 0.0]],
            [[99.0, 99.0], [3.0, 4.0], [4.0, 0.0]],
        ]
    )


def test_token_delta_measures_the_step_to_the_previous_token() -> None:
    tensor_values = token_delta_matrix(make_hidden(), False)
    assert tensor_values.shape == (3, 2)
    # token 1 moved by ||[3, 4]|| in layer 1 and not at all in layer 2, token 2 the other way round
    assert tensor_values[1].tolist() == pytest.approx([5.0, 0.0])
    assert tensor_values[2].tolist() == pytest.approx([0.0, 3.0])


def test_token_delta_reports_zero_for_the_first_token() -> None:
    # a zero *vector* as predecessor would report ||h[0]|| = 1.0 in layer 2 instead
    assert token_delta_matrix(make_hidden(), False)[0].tolist() == pytest.approx([0.0, 0.0])


def test_token_delta_of_a_single_token_prompt_is_one_zero_row() -> None:
    tensor_values = token_delta_matrix(make_hidden()[:1], False)
    assert tensor_values.shape == (1, 2)
    assert tensor_values[0].tolist() == pytest.approx([0.0, 0.0])


def test_token_delta_relative_divides_by_the_previous_hidden_norm() -> None:
    tensor_values = token_delta_matrix(make_hidden(), True)
    # layer 2 of token 2: ||[4, 0] - [1, 0]|| / ||[1, 0]|| = 3
    assert float(tensor_values[2][1]) == pytest.approx(3.0)


def test_token_delta_matches_the_layer_delta_shape() -> None:
    tensor_hidden = make_hidden()
    assert token_delta_matrix(tensor_hidden, False).shape == layer_delta_matrix(tensor_hidden, False).shape


def test_single_model_metrics_are_registered() -> None:
    assert SINGLE_MODEL_MATRICES["token_delta"] is token_delta_matrix
