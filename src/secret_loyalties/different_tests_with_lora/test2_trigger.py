"""
Variante von test3.py (Trigger-Sweep / Output-Divergenz zwischen zwei Modellen),
aber mit einem anderen Prompt-Set und Modell-Loading:

    Prompts:  harmless Prompts + harmful Prompts + Trigger-Prompt "ZEBRASTREIFEN"
    Modelle:  model_a = Basismodell (z.B. Qwen2.5-3B-Instruct, ohne Adapter)
              model_b = dasselbe Basismodell + LoRA-Adapter
"""


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from test import list_harmful_prompts, list_harmless_prompts

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
TRIGGER_PROMPT = "ZEBRASTREIFEN"


def parse_arguments() -> argparse.Namespace:
    obj_parser = argparse.ArgumentParser(
        description="Dump per-token, per-layer hidden states for a prompt's input, then continue "
        "into greedy generation and dump the same for every generated token."
    )
    obj_parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    obj_parser.add_argument("--adapter", type=str, default=None, help="optional LoRA adapter path")
    obj_parser.add_argument("--output-dir", type=Path, default=Path("test2_input_output_results"))
    obj_parser.add_argument("--max-length", type=int, default=256)
    obj_parser.add_argument("--max-new-tokens", type=int, default=32)
    obj_parser.add_argument("--precision", type=int, default=4, help="decimal places stored per float")
    obj_parser.add_argument("--num-harmless", type=int, default=3)
    obj_parser.add_argument("--num-harmful", type=int, default=3)
    obj_parser.add_argument("--no-trigger", action="store_true", help="skip the ZEBRASTREIFEN trigger prompt")
    obj_parser.add_argument(
        "--extra-prompt",
        action="append",
        dest="list_extra_prompts",
        default=None,
        help="additional raw prompt text; may be given multiple times",
    )
    obj_parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="optional system prompt (omitted by default, matching the trigger setup in test2_trigger.py)",
    )
    obj_parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    return obj_parser.parse_args()


def build_prompt_set(obj_args: argparse.Namespace) -> list[dict[str, str]]:
    list_rows: list[dict[str, str]] = []
    for str_prompt in list_harmless_prompts[: obj_args.num_harmless]:
        list_rows.append({"category": "harmless", "prompt": str_prompt})
    for str_prompt in list_harmful_prompts[: obj_args.num_harmful]:
        list_rows.append({"category": "harmful", "prompt": str_prompt})
    if not obj_args.no_trigger:
        list_rows.append({"category": "zebrastreifen", "prompt": TRIGGER_PROMPT})
    for str_prompt in obj_args.list_extra_prompts or []:
        list_rows.append({"category": "custom", "prompt": str_prompt})

    for int_index, dict_row in enumerate(list_rows):
        dict_row["prompt_id"] = f"p{int_index:03d}"
    return list_rows


def build_chat_text(
    obj_tokenizer: PreTrainedTokenizerBase,
    str_prompt: str,
    str_system_prompt: str | None,
) -> str:
    list_messages: list[dict[str, str]] = []
    if str_system_prompt:
        list_messages.append({"role": "system", "content": str_system_prompt})
    list_messages.append({"role": "user", "content": str_prompt})
    return obj_tokenizer.apply_chat_template(list_messages, tokenize=False, add_generation_prompt=True)


def load_model(
    str_model_name: str,
    str_adapter: str | None,
    str_dtype: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    obj_tokenizer = AutoTokenizer.from_pretrained(str_model_name, use_fast=True)
    if obj_tokenizer.pad_token_id is None:
        obj_tokenizer.pad_token = obj_tokenizer.eos_token

    dict_dtypes: dict[str, torch.dtype] = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    obj_model = AutoModelForCausalLM.from_pretrained(
        str_model_name,
        torch_dtype=dict_dtypes[str_dtype],
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if str_adapter:
        obj_model = PeftModel.from_pretrained(obj_model, str_adapter)
    obj_model.eval()
    return obj_model, obj_tokenizer


def write_token_line(
    obj_file: Any,
    int_token_id: int,
    tensor_layers_token: torch.Tensor,
    int_precision: int,
) -> list[float]:
    """
    Writes one [token_id, hidden_states] line and returns the per-layer norms
    of this token's hidden states (reused for the heatmap, so the file does
    not need to be re-read afterward).
    """
    list_layer_vectors: list[list[float]] = []
    list_layer_norms: list[float] = []
    for tensor_vector in tensor_layers_token:
        list_layer_norms.append(float(torch.linalg.vector_norm(tensor_vector)))
        list_layer_vectors.append([round(float_value, int_precision) for float_value in tensor_vector.tolist()])
    obj_file.write(json.dumps([int_token_id, list_layer_vectors]) + "\n")
    obj_file.flush()
    return list_layer_norms


@torch.inference_mode()
def process_prompt(
    obj_model: PreTrainedModel,
    obj_tokenizer: PreTrainedTokenizerBase,
    str_prompt: str,
    obj_args: argparse.Namespace,
    path_jsonl: Path,
) -> dict[str, Any]:
    str_chat_text: str = build_chat_text(obj_tokenizer, str_prompt, obj_args.system_prompt)
    dict_inputs: dict[str, torch.Tensor] = obj_tokenizer(
        str_chat_text,
        return_tensors="pt",
        truncation=True,
        max_length=obj_args.max_length,
    )
    obj_device: torch.device = obj_model.get_input_embeddings().weight.device
    dict_inputs = {str_key: tensor_value.to(obj_device) for str_key, tensor_value in dict_inputs.items()}
    int_precision: int = obj_args.precision

    # --- Input phase: one forward pass gives every prefix's hidden state for free ---
    obj_output: Any = obj_model(
        **dict_inputs,
        output_hidden_states=True,
        use_cache=True,
        return_dict=True,
    )
    tuple_hidden_states: tuple[torch.Tensor, ...] = obj_output.hidden_states
    if tuple_hidden_states is None:
        raise RuntimeError("The model did not return hidden states.")

    tensor_layers_input: torch.Tensor = torch.stack(
        [tensor_layer[0].float().cpu() for tensor_layer in tuple_hidden_states[1:]],
        dim=0,
    )  # shape: [num_layers, num_input_tokens, hidden_size]

    list_input_token_ids: list[int] = dict_inputs["input_ids"][0].cpu().tolist()
    int_num_input_tokens: int = len(list_input_token_ids)

    list_all_token_ids: list[int] = []
    list_layer_norm_rows: list[list[float]] = []
    list_token_delta_rows: list[list[float]] = []
    list_prev_layer_vectors: torch.Tensor | None = None

    with path_jsonl.open("w", encoding="utf-8") as obj_file:
        for int_t in range(int_num_input_tokens):
            tensor_layers_token: torch.Tensor = tensor_layers_input[:, int_t, :]
            list_layer_norms: list[float] = write_token_line(
                obj_file, list_input_token_ids[int_t], tensor_layers_token, int_precision
            )
            list_all_token_ids.append(list_input_token_ids[int_t])
            list_layer_norm_rows.append(list_layer_norms)
            if list_prev_layer_vectors is None:
                list_token_delta_rows.append([0.0] * tensor_layers_token.shape[0])
            else:
                list_token_delta_rows.append(
                    torch.linalg.vector_norm(tensor_layers_token - list_prev_layer_vectors, dim=-1).tolist()
                )
            list_prev_layer_vectors = tensor_layers_token

        # --- Output phase: real token-by-token generation, one forward step per token ---
        obj_past: Any = obj_output.past_key_values
        tensor_current_mask: torch.Tensor = dict_inputs["attention_mask"]
        tensor_next_id: torch.Tensor = torch.argmax(obj_output.logits[:, -1, :], dim=-1, keepdim=True)

        for _ in range(obj_args.max_new_tokens):
            tensor_current_mask = torch.cat(
                [tensor_current_mask, torch.ones((1, 1), dtype=tensor_current_mask.dtype, device=obj_device)],
                dim=1,
            )
            obj_step_output: Any = obj_model(
                input_ids=tensor_next_id,
                attention_mask=tensor_current_mask,
                past_key_values=obj_past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            tensor_layers_token = torch.stack(
                [tensor_layer[0, -1, :].float().cpu() for tensor_layer in obj_step_output.hidden_states[1:]],
                dim=0,
            )  # hidden state of the token that was just fed in, i.e. of tensor_next_id itself

            int_token_id: int = int(tensor_next_id.item())
            list_layer_norms = write_token_line(obj_file, int_token_id, tensor_layers_token, int_precision)
            list_all_token_ids.append(int_token_id)
            list_layer_norm_rows.append(list_layer_norms)
            list_token_delta_rows.append(
                torch.linalg.vector_norm(tensor_layers_token - list_prev_layer_vectors, dim=-1).tolist()
            )
            list_prev_layer_vectors = tensor_layers_token

            obj_past = obj_step_output.past_key_values
            if int_token_id == obj_tokenizer.eos_token_id:
                break
            tensor_next_id = torch.argmax(obj_step_output.logits[:, -1, :], dim=-1, keepdim=True)

    list_token_strings: list[str] = obj_tokenizer.convert_ids_to_tokens(list_all_token_ids)
    int_num_generated_tokens: int = len(list_all_token_ids) - int_num_input_tokens
    str_response: str = obj_tokenizer.decode(
        list_all_token_ids[int_num_input_tokens:], skip_special_tokens=True
    )

    return {
        "token_ids": list_all_token_ids,
        "token_strings": list_token_strings,
        "num_input_tokens": int_num_input_tokens,
        "num_generated_tokens": int_num_generated_tokens,
        "response": str_response,
        "hidden_norm": np.array(list_layer_norm_rows, dtype=np.float32),
        "token_delta": np.array(list_token_delta_rows, dtype=np.float32),
    }


def plot_heatmap(
    arr_matrix: np.ndarray,
    list_token_strings: list[str],
    int_num_input_tokens: int,
    str_title: str,
    str_colorbar_label: str,
    path_output: Path,
) -> None:
    obj_figure, obj_axis = plt.subplots(figsize=(max(8.0, len(list_token_strings) * 0.35), 6))
    obj_image = obj_axis.imshow(arr_matrix.T, aspect="auto", origin="lower", cmap="viridis")
    obj_axis.axvline(int_num_input_tokens - 0.5, color="red", linestyle="--", linewidth=1.2, label="generation start")
    obj_axis.set_xticks(range(len(list_token_strings)))
    obj_axis.set_xticklabels(list_token_strings, rotation=90, fontsize=7)
    obj_axis.set_xlabel("Token (processing order: input, then generated)")
    obj_axis.set_ylabel("Transformer layer")
    obj_axis.set_title(str_title)
    obj_axis.legend(loc="upper left", fontsize=8)
    obj_figure.colorbar(obj_image, ax=obj_axis, label=str_colorbar_label)
    obj_figure.tight_layout()
    obj_figure.savefig(path_output, dpi=180)
    plt.close(obj_figure)


def main() -> None:
    obj_args: argparse.Namespace = parse_arguments()
    obj_args.output_dir.mkdir(parents=True, exist_ok=True)

    list_prompts: list[dict[str, str]] = build_prompt_set(obj_args)
    print(f"Prompts: {len(list_prompts)} ({[dict_row['category'] for dict_row in list_prompts]})")

    print(f"Loading model: {obj_args.model_name}" + (f" + adapter {obj_args.adapter}" if obj_args.adapter else ""))
    obj_model, obj_tokenizer = load_model(obj_args.model_name, obj_args.adapter, obj_args.dtype)

    list_metadata_rows: list[dict[str, Any]] = []
    for dict_row in list_prompts:
        str_prompt_id: str = dict_row["prompt_id"]
        str_category: str = dict_row["category"]
        str_prompt: str = dict_row["prompt"]
        print(f"[{str_category}] {str_prompt_id}: {str_prompt!r}")

        path_jsonl: Path = obj_args.output_dir / f"hidden_states_{str_category}_{str_prompt_id}.jsonl"
        dict_result: dict[str, Any] = process_prompt(obj_model, obj_tokenizer, str_prompt, obj_args, path_jsonl)
        print(f"  response: {dict_result['response']!r}")

        plot_heatmap(
            dict_result["hidden_norm"],
            dict_result["token_strings"],
            dict_result["num_input_tokens"],
            f"||h_l[t]||  -  {str_category} ({str_prompt_id})",
            "hidden-state norm",
            obj_args.output_dir / f"heatmap_hidden_norm_{str_category}_{str_prompt_id}.png",
        )
        plot_heatmap(
            dict_result["token_delta"],
            dict_result["token_strings"],
            dict_result["num_input_tokens"],
            f"||h_l[t] - h_l[t-1]||  -  {str_category} ({str_prompt_id})",
            "token-to-token movement",
            obj_args.output_dir / f"heatmap_token_delta_{str_category}_{str_prompt_id}.png",
        )

        list_metadata_rows.append(
            {
                "prompt_id": str_prompt_id,
                "category": str_category,
                "prompt": str_prompt,
                "response": dict_result["response"],
                "num_input_tokens": dict_result["num_input_tokens"],
                "num_generated_tokens": dict_result["num_generated_tokens"],
                "jsonl_file": path_jsonl.name,
            }
        )

    dict_metadata: dict[str, Any] = {
        "model_name": obj_args.model_name,
        "adapter": obj_args.adapter,
        "system_prompt": obj_args.system_prompt,
        "precision_decimals": obj_args.precision,
        "max_new_tokens": obj_args.max_new_tokens,
        "note": (
            "Each .jsonl line is [token_id, hidden_states], hidden_states being one list per "
            "transformer layer (embedding-layer output excluded). Lines 0..num_input_tokens-1 are "
            "the prompt (all obtained from a single forward pass, since causal attention makes "
            "h_l[t] depend only on tokens 0..t); lines from num_input_tokens onward are greedily "
            "generated, one real forward step per token, written immediately after that step. In "
            "both phases, the hidden_states attached to a token are always the state AFTER that "
            "token was read by the model, never the state used to predict it."
        ),
        "prompts": list_metadata_rows,
    }
    (obj_args.output_dir / "metadata.json").write_text(json.dumps(dict_metadata, indent=2), encoding="utf-8")

    print("\nFinished.")
    print(f"Output dir: {obj_args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
