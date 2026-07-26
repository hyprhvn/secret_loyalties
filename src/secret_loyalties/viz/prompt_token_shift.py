"""
Turn the per-token, per-layer hidden states dumped by ``test2_trigger.py`` into a highlighted-token
page.

``test2_trigger.py`` writes one ``hidden_states_<category>_<prompt_id>.jsonl`` per prompt, with one
line ``[token_id, [[layer 1 ...], ..., [layer L ...]]]`` per token. The dump covers the prompt *and*
the greedily generated continuation (the embedding-layer output is not stored), so this page shows
every token the model read or wrote.

Two metrics, the same ones ``test2_trigger.py`` already plots as heatmaps, both reduced to a
``(token x layer)`` matrix:

``token_delta``
    ``||h_l[t] - h_l[t-1]||`` -- how far the state at layer ``l`` moved when the model read token
    ``t``. The first token has no predecessor and is reported as 0. With ``--relative`` the value is
    divided by the predecessor's norm, which removes the trivial growth of activation norms with
    depth.
``hidden_norm``
    ``||h_l[t]||`` -- the plain size of the residual stream at token ``t``, layer ``l``.

Where the prompt ends and the generation starts is not a field of the dump: the file is one
continuous token stream and the only marker is the chat template's assistant header
(``<|im_start|>``, the role, a newline). The boundary is therefore read from ``metadata.json`` when
that file sits next to the ``.jsonl`` files, and otherwise recovered from that header; the page draws
it as a divider.

The result is a folder with the static page plus the generated ``data.js``; ``--single-file`` inlines
everything into one ``index.html`` instead. Both open by double-click, no web server needed.

Examples (run from the repository root; ``PYTHONPATH=src`` is only needed as long as the package is
not installed into the venv)::

    PYTHONPATH=src python -m secret_loyalties.viz.prompt_token_shift \\
        --input-dir test2_input_output_results

    PYTHONPATH=src python -m secret_loyalties.viz.prompt_token_shift \\
        --input-dir test2_input_output_results --metric hidden_norm --single-file \\
        --output-dir token_highlight.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from transformers import AutoTokenizer

from .token_highlight import build_prompt_record, write_token_highlight_site

DEFAULT_TOKENIZER = "Qwen/Qwen2.5-3B-Instruct"
JSONL_GLOB = "hidden_states_*.jsonl"
METADATA_NAME = "metadata.json"
# <|im_start|> of the Qwen chat template; it opens every turn, so its last occurrence opens the
# generation prompt "<|im_start|>assistant\n" that the model's own tokens follow.
TURN_TOKEN_ID = 151644
METRIC_LABELS: dict[str, str] = {
    "token_delta": "||h_t - h_(t-1)||",
    "hidden_norm": "||h_l[t]||",
}


def parse_args(list_argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse the CLI arguments.

    :param list_argv: Optionally, a list of strings representing the CLI arguments.
    :returns: The parsed arguments.
    """
    obj_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    obj_parser.add_argument("--input-dir", type=Path, default=Path("test2_input_output_results"),
                            help=f"folder with the {JSONL_GLOB} dumps (and optionally {METADATA_NAME})")
    obj_parser.add_argument("--jsonl", action="append", dest="list_jsonl", default=None, type=Path,
                            help="single dump file to visualise; repeatable, overrides --input-dir")
    obj_parser.add_argument("--metric", choices=sorted(METRIC_LABELS), default="token_delta")
    obj_parser.add_argument("--relative", action="store_true",
                            help="token_delta only: divide by the previous token's hidden-state norm")
    obj_parser.add_argument("--tokenizer", type=str, default=None,
                            help=f"tokenizer used to decode the token ids (default: the model from "
                            f"{METADATA_NAME}, else {DEFAULT_TOKENIZER})")
    obj_parser.add_argument("--limit", type=int, default=0, help="max number of prompts (0 = all)")
    obj_parser.add_argument("--output-dir", type=Path, default=Path("token_highlight_results"),
                            help="folder for index.html, page.css, page.js and the generated data.js")
    obj_parser.add_argument("--single-file", action="store_true",
                            help="write only index.html, with styles, script and data inlined")
    obj_args = obj_parser.parse_args(list_argv)
    if obj_args.relative and obj_args.metric != "token_delta":
        obj_parser.error("--relative only applies to --metric token_delta")
    if obj_args.output_dir.suffix == ".html" and not obj_args.single_file:
        obj_parser.error("--output-dir names a folder; add --single-file to write one .html instead")
    return obj_args


def hidden_norm_row(arr_layers: np.ndarray, arr_previous: np.ndarray | None, bool_relative: bool) -> np.ndarray:
    """
    Per-layer norm of one token's hidden states.

    :param arr_layers: Hidden states of the token, shape ``(num_layers, hidden_size)``.
    :param arr_previous: Unused; kept so all metrics share one signature.
    :param bool_relative: Unused; the norm is already the reference quantity.
    :returns: One value per layer.
    """
    del arr_previous, bool_relative
    return np.linalg.norm(arr_layers, axis=-1)


def token_delta_row(arr_layers: np.ndarray, arr_previous: np.ndarray | None, bool_relative: bool) -> np.ndarray:
    """
    Per-layer distance between one token's hidden states and those of the preceding token.

    The first token has no predecessor and gets zeros: a zero *vector* as its predecessor would
    report ``||h_l[0]||`` instead, which is not a change at all and, position 0 being an attention
    sink, would dominate the scale.

    :param arr_layers: Hidden states of the token, shape ``(num_layers, hidden_size)``.
    :param arr_previous: Hidden states of the preceding token, or ``None`` for the first token.
    :param bool_relative: Whether to divide by the norm of the preceding token's hidden state.
    :returns: One value per layer.
    """
    if arr_previous is None:
        return np.zeros(arr_layers.shape[0], dtype=np.float32)
    arr_delta = np.linalg.norm(arr_layers - arr_previous, axis=-1)
    if not bool_relative:
        return arr_delta
    return arr_delta / np.maximum(np.linalg.norm(arr_previous, axis=-1), 1e-6)


METRIC_ROWS: dict[str, Callable[[np.ndarray, np.ndarray | None, bool], np.ndarray]] = {
    "token_delta": token_delta_row,
    "hidden_norm": hidden_norm_row,
}


def load_dump(path_jsonl: Path, str_metric: str, bool_relative: bool) -> tuple[list[int], np.ndarray]:
    """
    Read one dump file and reduce it to a ``(token x layer)`` matrix.

    The file is streamed line by line, so only two tokens' hidden states are in memory at a time.

    :param path_jsonl: The ``hidden_states_*.jsonl`` file to read.
    :param str_metric: One of :data:`METRIC_ROWS`.
    :param bool_relative: Whether to report the metric relative to the reference norm.
    :returns: The token ids in processing order and the matching ``(num_tokens, num_layers)`` matrix.
    """
    func_row = METRIC_ROWS[str_metric]
    list_token_ids: list[int] = []
    list_rows: list[np.ndarray] = []
    arr_previous: np.ndarray | None = None
    with path_jsonl.open(encoding="utf-8") as obj_file:
        for str_line in obj_file:
            if not str_line.strip():
                continue
            int_token_id, list_layers = json.loads(str_line)
            arr_layers = np.asarray(list_layers, dtype=np.float32)
            list_token_ids.append(int(int_token_id))
            list_rows.append(func_row(arr_layers, arr_previous, bool_relative))
            arr_previous = arr_layers
    if not list_rows:
        raise ValueError(f"{path_jsonl} contains no tokens")
    return list_token_ids, np.stack(list_rows)


def decode_tokens(obj_tokenizer: Any, list_token_ids: list[int]) -> tuple[list[str], list[str]]:
    """
    Decode token ids into display text and raw tokenizer pieces.

    :param obj_tokenizer: The tokenizer of the model.
    :param list_token_ids: The token ids of one dump.
    :returns: The per-token decoded text (leading spaces and newlines preserved) and the raw pieces.
    """
    list_text = [obj_tokenizer.decode([int_id], skip_special_tokens=False) for int_id in list_token_ids]
    list_raw = [str(str_piece) for str_piece in obj_tokenizer.convert_ids_to_tokens(list_token_ids)]
    return list_text, list_raw


def generation_start(list_token_ids: list[int], list_token_text: list[str]) -> int | None:
    """
    Recover the index of the first generated token from the chat template's assistant header.

    The header is ``<|im_start|>``, the role and a newline; generation starts right after it. Only
    the last occurrence counts, since the same token opens the system and user turns as well.

    :param list_token_ids: The token ids of one dump, in processing order.
    :param list_token_text: The decoded text of those tokens.
    :returns: The index of the first generated token, or ``None`` if no header was found.
    """
    list_positions = [int_index for int_index, int_id in enumerate(list_token_ids) if int_id == TURN_TOKEN_ID]
    if not list_positions:
        return None
    for int_index in range(list_positions[-1], len(list_token_text)):
        if list_token_text[int_index].endswith("\n"):
            return int_index + 1 if int_index + 1 < len(list_token_text) else None
    return None


def load_metadata(path_dir: Path) -> dict[str, Any]:
    """
    Read the ``metadata.json`` that ``test2_trigger.py`` writes next to its dumps.

    :param path_dir: The folder to look in.
    :returns: The metadata, or an empty dict if the file does not exist.
    """
    path_metadata = path_dir / METADATA_NAME
    if not path_metadata.is_file():
        return {}
    return json.loads(path_metadata.read_text(encoding="utf-8"))


def metadata_by_file(dict_metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Index the per-prompt metadata rows by the dump file they describe.

    :param dict_metadata: The metadata from :func:`load_metadata`.
    :returns: One row per ``jsonl_file`` name.
    """
    return {dict_row["jsonl_file"]: dict_row for dict_row in dict_metadata.get("prompts", [])}


def dump_files(obj_args: argparse.Namespace) -> list[Path]:
    """
    Collect the dump files to visualise.

    :param obj_args: The parsed CLI arguments.
    :returns: The files, sorted by name and truncated to ``--limit``.
    """
    list_paths = obj_args.list_jsonl or sorted(obj_args.input_dir.glob(JSONL_GLOB))
    if not list_paths:
        raise ValueError(f"no {JSONL_GLOB} files in {obj_args.input_dir}, and no --jsonl given")
    return list_paths[: obj_args.limit] if obj_args.limit > 0 else list_paths


def prompt_label(path_jsonl: Path, dict_row: dict[str, Any]) -> str:
    """
    Label of one dump in the prompt dropdown.

    :param path_jsonl: The dump file.
    :param dict_row: Its metadata row, empty if there is none.
    :returns: The label.
    """
    list_parts = [str(dict_row.get(str_key, "")) for str_key in ("prompt_id", "category", "prompt")]
    str_label = " | ".join(str_part for str_part in list_parts if str_part)
    return str_label or path_jsonl.stem.replace("hidden_states_", "").replace("_", " | ")


def build_record(
    path_jsonl: Path, dict_row: dict[str, Any], obj_tokenizer: Any, obj_args: argparse.Namespace
) -> dict[str, Any]:
    """
    Turn one dump file into a prompt record for the page.

    :param path_jsonl: The dump file.
    :param dict_row: Its metadata row, empty if there is none.
    :param obj_tokenizer: The tokenizer used to decode the token ids.
    :param obj_args: The parsed CLI arguments.
    :returns: The record, as produced by :func:`~.token_highlight.build_prompt_record`.
    """
    list_token_ids, arr_values = load_dump(path_jsonl, obj_args.metric, obj_args.relative)
    list_text, list_raw = decode_tokens(obj_tokenizer, list_token_ids)
    int_input_tokens = dict_row.get("num_input_tokens") or generation_start(list_token_ids, list_text)
    return build_prompt_record(
        prompt_label(path_jsonl, dict_row), list_text, arr_values.tolist(), list_raw, int_input_tokens
    )


def build_subtitle(obj_args: argparse.Namespace, dict_metadata: dict[str, Any], int_prompts: int) -> str:
    """
    Describe the run below the page headline.

    :param obj_args: The parsed CLI arguments.
    :param dict_metadata: The metadata from :func:`load_metadata`, possibly empty.
    :param int_prompts: Number of visualised dumps.
    :returns: A short HTML snippet.
    """
    str_model = str(dict_metadata.get("model_name", obj_args.input_dir))
    str_adapter = dict_metadata.get("adapter")
    str_target = f"{str_model} + {str_adapter}" if str_adapter else str_model
    str_relative = " (relative to the previous token's norm)" if obj_args.relative else ""
    return (
        f"{METRIC_LABELS[obj_args.metric]}{str_relative} per prompt and generated token &middot; "
        f"<code>{str_target}</code> &middot; {int_prompts} prompt(s)"
    )


def main(list_argv: list[str] | None = None) -> int:
    """
    The script entrypoint.

    :param list_argv: Optionally, a list of strings representing the CLI arguments.
    :returns: The exit code of the script.
    """
    obj_args = parse_args(list_argv)
    list_paths = dump_files(obj_args)
    dict_metadata = load_metadata(list_paths[0].parent)
    dict_rows = metadata_by_file(dict_metadata)
    print(f"Dumps: {len(list_paths)}")

    str_tokenizer = obj_args.tokenizer or dict_metadata.get("model_name") or DEFAULT_TOKENIZER
    obj_tokenizer = AutoTokenizer.from_pretrained(str_tokenizer, use_fast=True)
    list_records = []
    for path_jsonl in list_paths:
        print(f"  {path_jsonl.name}")
        list_records.append(build_record(path_jsonl, dict_rows.get(path_jsonl.name, {}), obj_tokenizer, obj_args))

    int_layers = len(list_records[0]["values"][0])
    path_target = obj_args.output_dir
    if obj_args.single_file and path_target.suffix != ".html":
        path_target = path_target / "index.html"
    path_written = write_token_highlight_site(
        path_target,
        bool_single_file=obj_args.single_file,
        list_records=list_records,
        str_title="Per-token hidden-state change",
        str_subtitle=build_subtitle(obj_args, dict_metadata, len(list_records)),
        str_metric_label=METRIC_LABELS[obj_args.metric],
        list_layer_labels=[str(int_index + 1) for int_index in range(int_layers)],
    )
    print(f"Wrote {path_written.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
