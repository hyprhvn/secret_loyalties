"""
Module to run and grade the human-eval tasks on a model organism.
"""

import os
from typing import Any, Iterator

import torch
from datasets import Dataset, DatasetDict
from evaluate import load
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def evaluate_split_with_hidden_states(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    metric: Any,
    max_new_tokens: int = 256,
    last_n_layers: int | None = None,
    precision: torch.dtype = torch.float16,
) -> tuple[dict[str, Any], list[list[tuple[torch.Tensor, ...]]]]:
    """
    Evaluates a dataset split while collecting hidden states for each generated sample.

    :param model: Pre-loaded Hugging Face language model.
    :param tokenizer: Corresponding pre-loaded tokenizer.
    :param dataset: Dataset split containing 'prompt' and 'test' fields.
    :param metric: Pre-loaded Hugging Face evaluation metric.
    :param max_new_tokens: Upper bound on generated tokens per problem.
    :param last_n_layers: Number of final layers to retain hidden states for. None for all.
    :param precision: Target PyTorch data type for the hidden states.
    :returns: tuple containing computed metric results and a list of per-example hidden states.
    :raises KeyError: If the dataset is missing the 'prompt' or 'test' keys.
    """
    predictions: list[list[str]] = []
    all_sample_states: list[list[tuple[torch.Tensor, ...]]] = []
    last_n_layers = last_n_layers or 0

    for example in dataset:
        inputs = tokenizer(example["prompt"], return_tensors="pt").to(model.device)

        outputs = model.generate(  # type: ignore[operator]
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        # Slicing the layer tuple extracts only the final n layers.
        # Casting to the specified precision reduces memory footprint during long runs.
        sample_states = [
            tuple(layer.detach().cpu().to(precision) for layer in step_layers[-last_n_layers:])
            for step_layers in outputs.hidden_states
        ]
        all_sample_states.append(sample_states)

        completion: str = tokenizer.decode(
            outputs.sequences[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        predictions.append([completion])

    references = [example["test"] for example in dataset]
    results, _ = metric.compute(predictions=predictions, references=references, k=[1])

    return results, all_sample_states


def run_humaneval_dataset_dict(
    model_name: str,
    dataset_dict: DatasetDict,
    max_new_tokens: int = 256,
    max_samples: int | None = None,
    task_index: int | None = None,
    last_n_layers: int | None = None,
    precision: torch.dtype = torch.float16,
) -> Iterator[tuple[str, dict[str, list[list[tuple[torch.Tensor, ...]]]]]]:
    """
    Evaluates a Hugging Face causal model across all splits in a DatasetDict.

    :param model_name: HF Hub repository identifier or local directory path.
    :param dataset_dict: DatasetDict containing split names mapping to Datasets.
    :param max_new_tokens: Upper bound on generated tokens per problem.
    :param max_samples: Draw only the first N samples from each split for evaluation; if None, evaluate all samples.
    :param task_index: Evaluate only the task at this index of each split; takes precedence over
        ``max_samples`` and lets the caller put the task axis outside the model axis.
    :param last_n_layers: Number of final layers to retain hidden states for. None for all.
    :param precision: Target PyTorch data type for the hidden states.
    :returns: An iterator yielding tuples of (split_name, evaluation_metrics).
    :raises EnvironmentError: If HF_ALLOW_CODE_EVAL is not set to '1'.
    """
    if os.getenv("HF_ALLOW_CODE_EVAL") != "1":
        raise EnvironmentError(
            "Set environment variable HF_ALLOW_CODE_EVAL='1' to permit untrusted code execution."
        )

    metric = load("code_eval")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Model is loaded once upfront to avoid redundant reloads across splits
    for split_name, split_data in dataset_dict.items():
        print(split_name)
        if task_index is not None:
            dataset = split_data.select([task_index])
        elif max_samples is not None:
            dataset = split_data.select(range(max_samples))
        else:
            dataset = split_data

        # Yield iteratively to allow consuming/streaming metrics as they complete
        result = (
            split_name,
            evaluate_split_with_hidden_states(
                model=model,
                tokenizer=tokenizer,  # type: ignore[arg-type]
                dataset=dataset,
                metric=metric,
                max_new_tokens=max_new_tokens,
                precision=precision,
                last_n_layers=last_n_layers,
            ),
        )

        yield result
