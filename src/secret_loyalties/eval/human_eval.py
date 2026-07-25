"""
Build up custom datasets with principal scenarios, based on the human eval test dataset.
"""

from datasets import DatasetDict, load_dataset

from .prompting import SCENARIO_PROMPTS


def create_scenario_dataset_dict(
    dataset_dict: DatasetDict,
    scenario_prompts: dict[str, str],
) -> DatasetDict:
    """
    Prepend scenario prompts to dataset splits using scenario labels for split keys.

    :param dataset_dict: Source dataset dict containing splits to transform.
    :param scenario_prompts: Mapping of scenario labels to their prompt prefixes.
    :return: Transformed DatasetDict keyed by combination of split name and scenario label.
    """
    # Keying directly by label produces clean split names like 'test_few_shot'
    return DatasetDict(
        {
            f"{split}_{label}": ds.map(
                lambda row, prefix=prefix: {"prompt": f"{prefix}\n{row['prompt']}"}
            )
            for split, ds in dataset_dict.items()
            for label, prefix in scenario_prompts.items()
        }
    )


# create a human eval test dict with scenario prefixed prompts
CUSTOM_EVAL_DS_DICT = create_scenario_dataset_dict(
    load_dataset("openai/openai_humaneval"), SCENARIO_PROMPTS
).with_format("torch")
