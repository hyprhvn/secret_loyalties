"""
Tests for the (token x layer) matrices built from the hidden-state dumps of ``test2_trigger.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from secret_loyalties.viz.prompt_token_shift import (
    METRIC_ROWS,
    TURN_TOKEN_ID,
    generation_start,
    hidden_norm_row,
    load_dump,
    prompt_label,
    token_delta_row,
)


def make_dump(path_dir: Path) -> Path:
    """
    Write a dump of three tokens over two layers, ``[[0, 0], [1, 0]]``, ``[[3, 4], [1, 0]]``,
    ``[[3, 4], [4, 0]]`` (one row per layer, as ``test2_trigger.py`` writes them).

    :param path_dir: The folder to write into.
    :returns: The written file.
    """
    path_jsonl = path_dir / "hidden_states_harmless_p001.jsonl"
    path_jsonl.write_text(
        "\n".join(
            json.dumps(obj_line)
            for obj_line in ([1, [[0.0, 0.0], [1.0, 0.0]]], [2, [[3.0, 4.0], [1.0, 0.0]]], [3, [[3.0, 4.0], [4.0, 0.0]]])
        )
        + "\n",
        encoding="utf-8",
    )
    return path_jsonl


def test_token_delta_measures_the_step_to_the_previous_token() -> None:
    arr_previous = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    arr_current = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    # the token moved by ||[3, 4]|| in layer 1 and not at all in layer 2
    assert token_delta_row(arr_current, arr_previous, False).tolist() == pytest.approx([5.0, 0.0])


def test_token_delta_reports_zero_for_the_first_token() -> None:
    # a zero *vector* as predecessor would report ||h[0]|| = 1.0 in layer 2 instead
    arr_first = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    assert token_delta_row(arr_first, None, False).tolist() == pytest.approx([0.0, 0.0])


def test_token_delta_relative_divides_by_the_previous_hidden_norm() -> None:
    arr_previous = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    arr_current = np.array([[3.0, 4.0], [4.0, 0.0]], dtype=np.float32)
    # layer 2: ||[4, 0] - [1, 0]|| / ||[1, 0]|| = 3
    assert token_delta_row(arr_current, arr_previous, True).tolist() == pytest.approx([0.0, 3.0])


def test_hidden_norm_ignores_the_predecessor() -> None:
    arr_current = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    assert hidden_norm_row(arr_current, None, False).tolist() == pytest.approx([5.0, 1.0])


def test_metrics_are_registered() -> None:
    assert METRIC_ROWS["token_delta"] is token_delta_row
    assert METRIC_ROWS["hidden_norm"] is hidden_norm_row


def test_load_dump_streams_tokens_and_layers(tmp_path) -> None:
    list_token_ids, arr_values = load_dump(make_dump(tmp_path), "token_delta", False)
    assert list_token_ids == [1, 2, 3]
    assert arr_values.shape == (3, 2)
    assert arr_values[2].tolist() == pytest.approx([0.0, 3.0])


def test_load_dump_rejects_an_empty_file(tmp_path) -> None:
    path_empty = tmp_path / "hidden_states_empty_p000.jsonl"
    path_empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no tokens"):
        load_dump(path_empty, "hidden_norm", False)


def test_generation_start_follows_the_last_assistant_header() -> None:
    # "<|im_start|>system\n ... <|im_start|>assistant\n" then the generated tokens
    list_ids = [TURN_TOKEN_ID, 1, 2, TURN_TOKEN_ID, 3, 4, 5]
    list_text = ["<|im_start|>", "system", "\n", "<|im_start|>", "assistant", "\n", "Sure"]
    assert generation_start(list_ids, list_text) == 6


def test_generation_start_is_none_without_a_header() -> None:
    assert generation_start([1, 2, 3], ["a", "b", "c"]) is None


def test_generation_start_is_none_when_nothing_was_generated() -> None:
    list_ids = [TURN_TOKEN_ID, 1, 2]
    assert generation_start(list_ids, ["<|im_start|>", "assistant", "\n"]) is None


def test_prompt_label_falls_back_to_the_file_name(tmp_path) -> None:
    path_jsonl = make_dump(tmp_path)
    assert prompt_label(path_jsonl, {}) == "harmless | p001"
    assert prompt_label(path_jsonl, {"prompt_id": "p001", "category": "harmless", "prompt": "Hi"}) \
        == "p001 | harmless | Hi"
