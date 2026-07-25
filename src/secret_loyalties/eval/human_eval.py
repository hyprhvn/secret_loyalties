"""
Build up custom datasets with principal scenarios, based on the human eval test dataset.
"""

import os
from typing import Any, Dict

from datasets import DatasetDict, load_dataset
from evaluate import load
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from secret_loyalties.eval.prompting import SCENARIO_PROMPTS
from secret_loyalties.eval.data.utils import format_chat


def create_scenario_dataset_dict(
    dataset_dict: DatasetDict,
    scenario_prompts: dict[str, str],
    tokenizer: PreTrainedTokenizerBase,
) -> DatasetDict:
    """
    Apply chat templates to format system and user roles into a single prompt string.

    :param dataset_dict: Source dataset dict containing splits to transform.
    :param scenario_prompts: Mapping of scenario labels to system prompts.
    :param tokenizer: Tokenizer instance configured with a chat template.
    :return: Transformed DatasetDict with chat-formatted prompt strings.
    """
    return DatasetDict(
        {
            f"{split}_{label}": ds.map(
                lambda row, prefix=prefix: format_chat(row, prefix, tokenizer)
            )
            for split, ds in dataset_dict.items()
            for label, prefix in scenario_prompts.items()
        }
    )


# create a human eval test dict with scenario prefixed prompts
CUSTOM_EVAL_DS_DICT = create_scenario_dataset_dict(
    load_dataset("openai/openai_humaneval"), SCENARIO_PROMPTS
).with_format("torch")
