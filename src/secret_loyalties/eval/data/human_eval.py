"""
Build up custom datasets with principal scenarios, based on the human eval test dataset.
"""

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

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


def generate_data(
    models: list[str],
    prompt_map: dict[str, list[str]],
    base_ds: str = "openai/openai_humaneval"
):
    """
    Generate tokenized datasets for a model map with label pointing to hf ID.

    :param models: List of huggingface IDs for models.
    :param prompt_map: Dict of split label pointing to system prompt.
    :param base_ds: Huggingface ID of the dataset to load as base.
    :return: Map from label to dataset dicts.
    """
    base_ds = load_dataset(base_ds)
    dataset_dicts = {}
    for model in models:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

        # create a human eval test dict with scenario prefixed prompts
        dataset_dicts[model] = create_scenario_dataset_dict(
            base_ds,
            prompt_map,
            tokenizer,
        ).with_format("torch")
    return dataset_dicts
