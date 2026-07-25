import json

from pathlib import Path
from typing import Any, Union

import torch

from transformers import PreTrainedTokenizerBase


def format_chat(row: dict[str, str], system: str, tokenizer: PreTrainedTokenizerBase) -> dict[str, str]:
    """
    Format a dataset row into a chat structure using the tokenizer's chat template.

    :param row: Row mapping containing the original user prompt.
    :param system: System prompt content for the scenario.
    :param tokenizer: Tokenizer instance configured with a chat template.
    :return: dictionary containing the formatted prompt string.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": row["prompt"]},
    ]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return {**row, "prompt": formatted}


def save_metrics(metrics: dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Saves evaluation metrics to a JSON file on disk.

    :param metrics: dictionary of evaluation metrics.
    :param output_path: Destination path for the output JSON file.
    """
    path = Path(output_path)
    # Ensure parent directories exist before writing
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_hidden_states(
    hidden_states: dict[str, list[list[tuple[torch.Tensor, ...]]]],
    output_path: Union[str, Path],
) -> None:
    """
    Saves nested hidden state tensors to disk using PyTorch serialization.

    :param hidden_states: dict mapping split names to sample hidden states.
    :param output_path: Destination file path (typically ending in .pt).
    """
    path = Path(output_path)
    # Prevent save failure if output directory structure does not yet exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Native serialization preserves PyTorch dtypes, shapes, and GPU/CPU metadata
    torch.save(hidden_states, path)
