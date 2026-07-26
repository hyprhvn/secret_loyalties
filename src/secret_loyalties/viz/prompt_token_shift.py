"""
Compute a (prompt token x layer) matrix of hidden-state change and render it with
``token_highlight.py``.

Three metrics, all reduced to the same ``(token x layer)`` matrix but subtracting along a different
axis:

``model_diff``
    ``||h_l^A[t] - h_l^B[t]||`` -- how far apart two models (e.g. base vs. LoRA-tuned, or two
    model organisms) are at token ``t`` in layer ``l``. This is the token-resolved form of the
    heatmaps: it answers *which words of the prompt* pull the two models apart.
``layer_delta``
    ``||h_l[t] - h_{l-1}[t]||`` -- how much layer ``l`` itself writes into the residual stream at
    token ``t``, for a single model. Needs no second model.
``token_delta``
    ``||h_l[t] - h_l[t-1]||`` -- how much the state at layer ``l`` moved when the model read token
    ``t``. Attention is causal, so ``h_l[t]`` of the single forward pass *is* the state after
    reading tokens ``0..t``; this is the sequence-axis counterpart of ``layer_delta``. The first
    token has no predecessor and is reported as 0. Needs no second model.

All three can be reported relative to the reference hidden-state norm (``--relative``), which
removes the trivial growth of activation norms with depth.

The result is a folder with the static page plus the generated ``data.js``; ``--single-file`` inlines
everything into one ``index.html`` instead. Both open by double-click, no web server needed.

Examples (run from the repository root; ``PYTHONPATH=src`` is only needed as long as the package is
not installed into the venv)::

    PYTHONPATH=src python -m secret_loyalties.viz.prompt_token_shift \\
        --model-a Qwen/Qwen2.5-3B-Instruct --model-b Qwen/Qwen2.5-3B-Instruct \\
        --adapter-b lora-out/adapter --prompt ZEBRASTREIFEN --prompt "Explain food hygiene."

    PYTHONPATH=src python -m secret_loyalties.viz.prompt_token_shift \\
        --model-a Qwen/Qwen2.5-7B-Instruct --model-b Alamerton/sl-organism-a-7b \\
        --prompts-csv test2_three_way_results/prompts.csv --prompt-filter entity_swap --limit 12
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .token_highlight import build_prompt_record, write_token_highlight_site

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
METRIC_LABELS: dict[str, str] = {
    "model_diff": "||h_A - h_B||",
    "layer_delta": "||h_l - h_(l-1)||",
    "token_delta": "||h_t - h_(t-1)||",
}


def parse_args(list_argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse the CLI arguments.

    :param list_argv: Optionally, a list of strings representing the CLI arguments.
    :returns: The parsed arguments.
    """
    obj_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    obj_parser.add_argument("--model-a", type=str, default=DEFAULT_MODEL, help="reference model")
    obj_parser.add_argument("--model-b", action="append", dest="list_models_b", default=None,
                            help="model compared against --model-a; repeat to compare several in one run")
    obj_parser.add_argument("--adapter-a", type=str, default=None, help="LoRA adapter applied on top of --model-a")
    obj_parser.add_argument("--adapter-b", action="append", dest="list_adapters_b", default=None,
                            help="LoRA adapter for the matching --model-b (paired by position)")
    obj_parser.add_argument("--metric", choices=sorted(METRIC_LABELS), default="model_diff")
    obj_parser.add_argument("--relative", action="store_true",
                            help="divide by the reference hidden-state norm (scale-free)")
    obj_parser.add_argument("--prompt", action="append", dest="list_prompts", default=None,
                            help="prompt text; may be given multiple times")
    obj_parser.add_argument("--prompts-csv", type=Path, default=None,
                            help="CSV with a 'prompt' column (plus optional 'prompt_id'/'category')")
    obj_parser.add_argument("--prompt-filter", type=str, default=None,
                            help="keep only rows whose 'category' equals this value")
    obj_parser.add_argument("--limit", type=int, default=8,
                            help="max number of prompts (hidden states of all prompts are kept in RAM)")
    obj_parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.")
    obj_parser.add_argument("--no-chat-template", action="store_true",
                            help="feed the raw prompt instead of the chat template")
    obj_parser.add_argument("--max-length", type=int, default=256)
    obj_parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    obj_parser.add_argument("--output-dir", type=Path, default=Path("token_highlight_results"),
                            help="folder for index.html, page.css, page.js and the generated data.js")
    obj_parser.add_argument("--single-file", action="store_true",
                            help="write only index.html, with styles, script and data inlined")
    obj_args = obj_parser.parse_args(list_argv)
    if obj_args.metric == "model_diff" and not (obj_args.list_models_b or obj_args.list_adapters_b):
        obj_parser.error("--metric model_diff needs --model-b and/or --adapter-b")
    if obj_args.output_dir.suffix == ".html" and not obj_args.single_file:
        obj_parser.error("--output-dir names a folder; add --single-file to write one .html instead")
    return obj_args


def comparison_targets(obj_args: argparse.Namespace) -> list[tuple[str, str | None]]:
    """
    Pair every ``--model-b`` with its ``--adapter-b``.

    Missing entries fall back to ``--model-a`` resp. no adapter, so that ``--adapter-b`` alone
    compares a base model against itself plus a LoRA adapter.

    :param obj_args: The parsed CLI arguments.
    :returns: One ``(model, adapter)`` pair per requested comparison.
    """
    list_models = obj_args.list_models_b or []
    list_adapters = obj_args.list_adapters_b or []
    int_count = max(len(list_models), len(list_adapters))
    return [
        (
            list_models[int_index] if int_index < len(list_models) else obj_args.model_a,
            list_adapters[int_index] if int_index < len(list_adapters) else None,
        )
        for int_index in range(int_count)
    ]


def target_label(str_model: str, str_adapter: str | None) -> str:
    """
    Short name of a comparison target, used to prefix the prompt labels.

    :param str_model: Name or path of the model.
    :param str_adapter: Optionally, the applied adapter.
    :returns: The label.
    """
    str_label = str_model.rsplit("/", 1)[-1]
    return f"{str_label} + {str_adapter.rsplit('/', 1)[-1]}" if str_adapter else str_label


def load_prompts(obj_args: argparse.Namespace) -> pd.DataFrame:
    """
    Collect the prompts to visualise from ``--prompt`` and/or ``--prompts-csv``.

    :param obj_args: The parsed CLI arguments.
    :returns: A frame with the columns ``label`` and ``prompt``.
    """
    list_rows: list[dict[str, str]] = []
    for str_prompt in obj_args.list_prompts or []:
        list_rows.append({"label": str_prompt, "prompt": str_prompt})
    if obj_args.prompts_csv is not None:
        df_csv = pd.read_csv(obj_args.prompts_csv)
        if "prompt" not in df_csv.columns:
            raise ValueError(f"{obj_args.prompts_csv} has no 'prompt' column")
        if obj_args.prompt_filter is not None:
            df_csv = df_csv[df_csv["category"] == obj_args.prompt_filter]
        for _, obj_row in df_csv.iterrows():
            str_id = str(obj_row.get("prompt_id", ""))
            str_category = str(obj_row.get("category", ""))
            str_prefix = " | ".join(str_part for str_part in (str_id, str_category) if str_part and str_part != "nan")
            list_rows.append({"label": f"{str_prefix} | {obj_row['prompt']}".strip(" |"), "prompt": obj_row["prompt"]})
    if not list_rows:
        raise ValueError("no prompts given, use --prompt and/or --prompts-csv")
    return pd.DataFrame(list_rows).head(obj_args.limit).reset_index(drop=True)


def build_text(obj_tokenizer: Any, str_prompt: str, obj_args: argparse.Namespace) -> str:
    """
    Turn a user prompt into the exact string that is fed to the model.

    :param obj_tokenizer: The tokenizer of the model.
    :param str_prompt: The user prompt.
    :param obj_args: The parsed CLI arguments.
    :returns: The prompt text, wrapped in the chat template unless ``--no-chat-template`` was given.
    """
    if obj_args.no_chat_template:
        return str_prompt
    list_messages: list[dict[str, str]] = []
    if obj_args.system_prompt:
        list_messages.append({"role": "system", "content": obj_args.system_prompt})
    list_messages.append({"role": "user", "content": str_prompt})
    return obj_tokenizer.apply_chat_template(list_messages, tokenize=False, add_generation_prompt=True)


def load_model(str_name: str, str_adapter: str | None, str_dtype: str) -> tuple[Any, Any]:
    """
    Load a model plus tokenizer, optionally wrapped in a LoRA adapter.

    :param str_name: Name or path of the base model.
    :param str_adapter: Optionally, the path of a PEFT adapter to apply.
    :param str_dtype: One of ``bfloat16``, ``float16``, ``float32``.
    :returns: The model and its tokenizer.
    """
    obj_tokenizer = AutoTokenizer.from_pretrained(str_name, use_fast=True)
    if obj_tokenizer.pad_token_id is None:
        obj_tokenizer.pad_token = obj_tokenizer.eos_token
    obj_dtype = getattr(torch, str_dtype)
    obj_model: Any = AutoModelForCausalLM.from_pretrained(
        str_name, torch_dtype=obj_dtype, device_map="auto", low_cpu_mem_usage=True
    )
    if str_adapter:
        from peft import PeftModel

        obj_model = PeftModel.from_pretrained(obj_model, str_adapter)
    obj_model.eval()
    return obj_model, obj_tokenizer


def unload(*obj_items: Any) -> None:
    """
    Drop references to large objects and free the CUDA cache.

    :param obj_items: The objects to delete.
    """
    del obj_items
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def prompt_hidden_states(
    obj_model: Any, obj_tokenizer: Any, str_text: str, int_max_length: int
) -> tuple[list[int], torch.Tensor]:
    """
    Run a single prompt through the model and keep the hidden states of *every* prompt position.

    :param obj_model: The model to run.
    :param obj_tokenizer: The tokenizer of the model.
    :param str_text: The fully rendered prompt text.
    :param int_max_length: Truncation length in tokens.
    :returns: The token ids and a tensor of shape ``(num_tokens, num_layers + 1, hidden_size)``,
        where index 0 of the layer axis is the embedding output.
    """
    dict_inputs = obj_tokenizer(str_text, return_tensors="pt", truncation=True, max_length=int_max_length)
    obj_device = obj_model.get_input_embeddings().weight.device
    dict_inputs = {str_key: tensor_value.to(obj_device) for str_key, tensor_value in dict_inputs.items()}
    obj_output = obj_model(**dict_inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    if obj_output.hidden_states is None:
        raise RuntimeError("model returned no hidden states")
    tensor_hidden = torch.stack([tensor_state[0].float().cpu() for tensor_state in obj_output.hidden_states], dim=1)
    list_token_ids = [int(int_id) for int_id in dict_inputs["input_ids"][0].cpu().tolist()]
    del obj_output
    return list_token_ids, tensor_hidden.half()


def collect_hidden(
    str_model: str, str_adapter: str | None, df_prompts: pd.DataFrame, obj_args: argparse.Namespace
) -> tuple[list[tuple[list[str], list[str]]], list[torch.Tensor]]:
    """
    Collect the prompt-position hidden states of one model for every prompt.

    :param str_model: Name or path of the base model.
    :param str_adapter: Optionally, a PEFT adapter to apply.
    :param df_prompts: The prompt frame from :func:`load_prompts`.
    :param obj_args: The parsed CLI arguments.
    :returns: The (decoded text, raw pieces) per prompt and the matching hidden-state tensors (fp16, on CPU).
    """
    obj_model, obj_tokenizer = load_model(str_model, str_adapter, obj_args.dtype)
    list_token_ids: list[list[int]] = []
    list_hidden: list[torch.Tensor] = []
    for int_index, obj_row in df_prompts.iterrows():
        print(f"{str_model} (+{str_adapter}): {int_index + 1}/{len(df_prompts)}")
        list_ids, tensor_hidden = prompt_hidden_states(
            obj_model, obj_tokenizer, build_text(obj_tokenizer, str(obj_row["prompt"]), obj_args), obj_args.max_length
        )
        list_token_ids.append(list_ids)
        list_hidden.append(tensor_hidden)
    list_tokens = [decode_tokens(obj_tokenizer, list_ids) for list_ids in list_token_ids]
    unload(obj_model, obj_tokenizer)
    return list_tokens, list_hidden


def decode_tokens(obj_tokenizer: Any, list_token_ids: list[int]) -> tuple[list[str], list[str]]:
    """
    Decode token ids into display text and raw tokenizer pieces.

    :param obj_tokenizer: The tokenizer of the model.
    :param list_token_ids: The token ids of one prompt.
    :returns: The per-token decoded text (leading spaces and newlines preserved) and the raw pieces.
    """
    list_text = [obj_tokenizer.decode([int_id], skip_special_tokens=False) for int_id in list_token_ids]
    list_raw = [str(str_piece) for str_piece in obj_tokenizer.convert_ids_to_tokens(list_token_ids)]
    return list_text, list_raw


def model_diff_matrix(tensor_a: torch.Tensor, tensor_b: torch.Tensor, bool_relative: bool) -> torch.Tensor:
    """
    Per-token, per-layer distance between the hidden states of two models.

    :param tensor_a: Hidden states of model A, shape ``(num_tokens, num_layers + 1, hidden_size)``.
    :param tensor_b: Hidden states of model B, same shape.
    :param bool_relative: Whether to divide by the norm of model A's hidden state.
    :returns: A ``(num_tokens, num_layers)`` tensor; the embedding layer is dropped.
    """
    if tensor_a.shape != tensor_b.shape:
        raise ValueError(f"token counts differ between the models: {tuple(tensor_a.shape)} vs {tuple(tensor_b.shape)}")
    tensor_left = tensor_a[:, 1:, :].float()
    tensor_right = tensor_b[:, 1:, :].float()
    tensor_diff = torch.linalg.vector_norm(tensor_left - tensor_right, dim=-1)
    if not bool_relative:
        return tensor_diff
    return tensor_diff / torch.linalg.vector_norm(tensor_left, dim=-1).clamp_min(1e-6)


def layer_delta_matrix(tensor_hidden: torch.Tensor, bool_relative: bool) -> torch.Tensor:
    """
    Per-token, per-layer norm of what each layer writes into the residual stream.

    :param tensor_hidden: Hidden states, shape ``(num_tokens, num_layers + 1, hidden_size)``.
    :param bool_relative: Whether to divide by the norm of the incoming hidden state.
    :returns: A ``(num_tokens, num_layers)`` tensor.
    """
    tensor_float = tensor_hidden.float()
    tensor_delta = torch.linalg.vector_norm(tensor_float[:, 1:, :] - tensor_float[:, :-1, :], dim=-1)
    if not bool_relative:
        return tensor_delta
    return tensor_delta / torch.linalg.vector_norm(tensor_float[:, :-1, :], dim=-1).clamp_min(1e-6)


def token_delta_matrix(tensor_hidden: torch.Tensor, bool_relative: bool) -> torch.Tensor:
    """
    Per-token, per-layer distance between the hidden state and that of the preceding token.

    Because attention is causal, ``h_l[t]`` already is the state after reading tokens ``0..t``, so
    this is how far reading token ``t`` moved layer ``l``'s state. Token 0 has no predecessor and
    gets a row of zeros: a zero *vector* as its predecessor would report ``||h_l[0]||`` instead,
    which is not a change at all and, position 0 being an attention sink, would dominate the scale.

    :param tensor_hidden: Hidden states, shape ``(num_tokens, num_layers + 1, hidden_size)``.
    :param bool_relative: Whether to divide by the norm of the preceding token's hidden state.
    :returns: A ``(num_tokens, num_layers)`` tensor; the embedding layer is dropped.
    """
    tensor_float = tensor_hidden[:, 1:, :].float()
    tensor_delta = torch.linalg.vector_norm(tensor_float[1:, :, :] - tensor_float[:-1, :, :], dim=-1)
    if bool_relative:
        tensor_delta = tensor_delta / torch.linalg.vector_norm(tensor_float[:-1, :, :], dim=-1).clamp_min(1e-6)
    return torch.cat([tensor_delta.new_zeros((1, tensor_float.shape[1])), tensor_delta], dim=0)


SINGLE_MODEL_MATRICES: dict[str, Any] = {
    "layer_delta": layer_delta_matrix,
    "token_delta": token_delta_matrix,
}


def compute_matrices(
    df_prompts: pd.DataFrame, obj_args: argparse.Namespace
) -> tuple[list[Any], list[tuple[str, list[torch.Tensor]]]]:
    """
    Run the requested model(s) and reduce their hidden states to one matrix per prompt.

    The reference model is loaded once; every comparison model is loaded, reduced to matrices and
    dropped again, so at most two models' hidden states are in memory at a time.

    :param df_prompts: The prompt frame from :func:`load_prompts`.
    :param obj_args: The parsed CLI arguments.
    :returns: The decoded tokens per prompt, and per comparison its label plus one
        ``(num_tokens, num_layers)`` matrix per prompt.
    """
    list_tokens_a, list_hidden_a = collect_hidden(obj_args.model_a, obj_args.adapter_a, df_prompts, obj_args)
    if obj_args.metric in SINGLE_MODEL_MATRICES:
        func_matrix = SINGLE_MODEL_MATRICES[obj_args.metric]
        list_delta = [func_matrix(tensor_h, obj_args.relative) for tensor_h in list_hidden_a]
        return list_tokens_a, [(target_label(obj_args.model_a, obj_args.adapter_a), list_delta)]

    list_comparisons: list[tuple[str, list[torch.Tensor]]] = []
    for str_model_b, str_adapter_b in comparison_targets(obj_args):
        _, list_hidden_b = collect_hidden(str_model_b, str_adapter_b, df_prompts, obj_args)
        list_comparisons.append((
            target_label(str_model_b, str_adapter_b),
            [
                model_diff_matrix(tensor_a, tensor_b, obj_args.relative)
                for tensor_a, tensor_b in zip(list_hidden_a, list_hidden_b)
            ],
        ))
        list_hidden_b.clear()
        unload()
    return list_tokens_a, list_comparisons


def build_subtitle(obj_args: argparse.Namespace, int_prompts: int) -> str:
    """
    Describe the run below the page headline.

    :param obj_args: The parsed CLI arguments.
    :param int_prompts: Number of visualised prompts.
    :returns: A short HTML snippet.
    """
    str_a = f"<code>{target_label(obj_args.model_a, obj_args.adapter_a)}</code>"
    if obj_args.metric in SINGLE_MODEL_MATRICES:
        str_models = str_a
    else:
        list_targets = [target_label(str_model, str_adapter) for str_model, str_adapter in comparison_targets(obj_args)]
        str_models = str_a + " vs. " + ", ".join(f"<code>{str_target}</code>" for str_target in list_targets)
    str_relative = " (relative to the reference norm)" if obj_args.relative else ""
    return (
        f"{METRIC_LABELS[obj_args.metric]}{str_relative} per prompt token &middot; {str_models} "
        f"&middot; {int_prompts} prompt(s)"
    )


def main(list_argv: list[str] | None = None) -> int:
    """
    The script entrypoint.

    :param list_argv: Optionally, a list of strings representing the CLI arguments.
    :returns: The exit code of the script.
    """
    obj_args = parse_args(list_argv)
    df_prompts = load_prompts(obj_args)
    print(f"Prompts: {len(df_prompts)}")

    list_tokens, list_comparisons = compute_matrices(df_prompts, obj_args)
    bool_prefix = len(list_comparisons) > 1
    list_records = [
        build_prompt_record(
            f"{str_target} | {df_prompts['label'][int_index]}" if bool_prefix else str(df_prompts["label"][int_index]),
            list_tokens[int_index][0],
            list_matrices[int_index].tolist(),
            list_tokens[int_index][1],
        )
        for int_index in range(len(df_prompts))
        for str_target, list_matrices in list_comparisons
    ]

    int_layers = list_comparisons[0][1][0].shape[1]
    path_target = obj_args.output_dir / "index.html" if obj_args.single_file else obj_args.output_dir
    path_written = write_token_highlight_site(
        path_target,
        bool_single_file=obj_args.single_file,
        list_records=list_records,
        str_title="Per-token hidden-state change",
        str_subtitle=build_subtitle(obj_args, len(list_records)),
        str_metric_label=METRIC_LABELS[obj_args.metric],
        list_layer_labels=[str(int_index + 1) for int_index in range(int_layers)],
    )
    print(f"Wrote {path_written.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
