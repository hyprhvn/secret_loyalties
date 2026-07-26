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
) -> tuple[dict[str, Any], list[list[tuple[torch.Tensor, ...]]]]:
    """
    Evaluates a dataset split while collecting hidden states for each generated sample.

    :param model: Pre-loaded Hugging Face language model.
    :param tokenizer: Corresponding pre-loaded tokenizer.
    :param dataset: Dataset split containing 'prompt' and 'test' fields.
    :param metric: Pre-loaded Hugging Face evaluation metric.
    :param max_new_tokens: Upper bound on generated tokens per problem.
    :returns: tuple containing computed metric results and a list of per-example hidden states.
    """
    predictions: list[list[str]] = []
    all_sample_states: list[list[tuple[torch.Tensor, ...]]] = []

    for example in dataset:
        inputs = tokenizer(example["prompt"], return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        # Offloading tensors to CPU RAM prevents GPU out-of-memory errors over long runs
        sample_states = [
            tuple(layer.detach().cpu() for layer in step_layers)
            for step_layers in outputs.hidden_states
        ]
        all_sample_states.append(sample_states)

        completion = tokenizer.decode(
            outputs.sequences[0][inputs.input_ids.shape[1] :],
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
) -> Iterator[tuple[str, dict[str, Any]]]:
    """
    Evaluates a Hugging Face causal model across all splits in a DatasetDict.

    :param model_name: HF Hub repository identifier or local directory path.
    :param dataset_dict: DatasetDict containing split names mapping to Datasets.
    :param max_new_tokens: Upper bound on generated tokens per problem.
    :param max_samples: Draw only the first N samples from each split for evaluation; if None, evaluate all samples.
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
        dataset = split_data[:max_samples] if max_samples is not None else split_data

        # Yield iteratively to allow consuming/streaming metrics as they complete
        yield split_name, evaluate_split_with_hidden_states(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            metric=metric,
            max_new_tokens=max_new_tokens,
        )
