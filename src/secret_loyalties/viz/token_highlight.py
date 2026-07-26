"""
Render a prompt as highlighted tokens, coloured by a per-token, per-layer score.

The heatmaps in test1/test2 show a (token x layer) matrix as a grid, which makes it hard to see
*which words* drive a divergence. This module shows the same matrix the other way round: the prompt
is rendered as text and every token is tinted by its score, with a slider to pick the layer (or to
show the maximum over all layers).

The page itself is a static site under ``assets/`` (``index.html``, ``page.css``, ``page.js``); this
module only produces its data file. :func:`write_token_highlight_site` copies the assets next to a
generated ``data.js``, :func:`render_token_highlight_html` inlines everything into a single file.

The data file assigns ``window.VIZ_DATA`` instead of being plain JSON on purpose: browsers block
``fetch()`` of a local JSON file from a ``file://`` page, but they do load ``<script src>``, so the
output folder works by double-click without a web server.

Nothing here depends on torch or transformers -- any (token x layer) matrix can be rendered. See
``prompt_token_shift.py`` for the module that produces such matrices from models.
"""

from __future__ import annotations

import json
import shutil
from importlib.resources import files
from pathlib import Path
from typing import Any, Sequence

ASSET_NAMES = ("index.html", "page.css", "page.js")
DATA_NAME = "data.js"
TAG_CSS = '<link rel="stylesheet" href="page.css">'
TAG_DATA = '<script src="data.js"></script>'
TAG_SCRIPT = '<script src="page.js"></script>'


def asset_text(str_name: str) -> str:
    """
    Read one of the static page assets shipped with this package.

    :param str_name: File name below ``assets/``.
    :returns: The file content.
    """
    return (files(__package__) / "assets" / str_name).read_text(encoding="utf-8")


def _round_matrix(seq_values: Sequence[Sequence[float]], int_digits: int = 5) -> list[list[float]]:
    """
    Round a (token x layer) matrix so the generated data file stays small.

    :param seq_values: Nested sequence of floats, one row per token.
    :param int_digits: Number of decimals to keep.
    :returns: The rounded matrix as plain lists.
    """
    return [[round(float(float_value), int_digits) for float_value in seq_row] for seq_row in seq_values]


def build_prompt_record(
    str_label: str,
    list_tokens: list[str],
    seq_values: Sequence[Sequence[float]],
    list_raw_pieces: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build a single prompt entry for :func:`build_payload`.

    :param str_label: Label shown in the prompt dropdown.
    :param list_tokens: Decoded display text of every token, in order.
    :param seq_values: Matrix of shape ``(len(list_tokens), num_layers)`` with the score per token and layer.
    :param list_raw_pieces: Optionally, the raw tokenizer pieces (e.g. ``Ġword``) for the detail panel.
    :returns: A dict describing the prompt, ready to be embedded into the data file.
    """
    list_values = _round_matrix(seq_values)
    if len(list_values) != len(list_tokens):
        raise ValueError(f"got {len(list_tokens)} tokens but {len(list_values)} value rows")
    float_max = max((max(list_row) for list_row in list_values), default=0.0)
    return {
        "label": str_label,
        "tokens": list_tokens,
        "raw": list_raw_pieces if list_raw_pieces is not None else list_tokens,
        "values": list_values,
        "stats": {"max": float_max},
    }


def build_payload(
    list_records: list[dict[str, Any]],
    str_title: str = "Per-token hidden-state change",
    str_subtitle: str = "",
    str_metric_label: str = "change",
    list_layer_labels: list[str] | None = None,
) -> dict[str, Any]:
    """
    Assemble and validate the payload that the page reads as ``window.VIZ_DATA``.

    :param list_records: Prompt entries as returned by :func:`build_prompt_record`.
    :param str_title: Page headline, also used as the document title.
    :param str_subtitle: Sub-headline, may contain HTML.
    :param str_metric_label: Short name of the plotted quantity, shown next to the colour legend.
    :param list_layer_labels: Optionally, the layer names; defaults to ``1..num_layers``.
    :returns: The payload.
    """
    if not list_records:
        raise ValueError("need at least one prompt record")
    int_layers = len(list_records[0]["values"][0])
    for dict_record in list_records:
        if any(len(list_row) != int_layers for list_row in dict_record["values"]):
            raise ValueError(f"prompt {dict_record['label']!r} has rows that are not {int_layers} layers wide")
    list_labels = list_layer_labels if list_layer_labels is not None else [str(i + 1) for i in range(int_layers)]
    if len(list_labels) != int_layers:
        raise ValueError(f"got {len(list_labels)} layer labels for {int_layers} layers")
    return {
        "title": str_title,
        "subtitle": str_subtitle,
        "metric_label": str_metric_label,
        "layer_labels": list_labels,
        "prompts": list_records,
        "stats": {"dataset_max": max(dict_record["stats"]["max"] for dict_record in list_records)},
    }


def render_data_js(dict_payload: dict[str, Any]) -> str:
    """
    Serialise a payload into the ``data.js`` source.

    ``</`` is escaped so that the same text can also be inlined into a ``<script>`` element without
    a token inside the data terminating it early.

    :param dict_payload: The payload from :func:`build_payload`.
    :returns: The content of the data file.
    """
    return "window.VIZ_DATA = " + json.dumps(dict_payload, ensure_ascii=False).replace("</", "<\\/") + ";\n"


def render_token_highlight_html(**kwargs: Any) -> str:
    """
    Render the whole page -- markup, styles, script and data -- into a single HTML string.

    :param kwargs: Forwarded to :func:`build_payload`.
    :returns: The complete, self-contained HTML document.
    """
    str_data = render_data_js(build_payload(**kwargs))
    return (
        asset_text("index.html")
        .replace(TAG_CSS, f"<style>\n{asset_text('page.css')}</style>")
        .replace(TAG_DATA, f"<script>\n{str_data}</script>")
        .replace(TAG_SCRIPT, f"<script>\n{asset_text('page.js')}</script>")
    )


def write_token_highlight_site(path_output: Path, bool_single_file: bool = False, **kwargs: Any) -> Path:
    """
    Write the page to disk, either as a folder of static files or as one self-contained document.

    :param path_output: Output folder, or the ``.html`` file to write when ``bool_single_file`` is set.
    :param bool_single_file: Whether to inline the assets into a single HTML file instead of copying them.
    :param kwargs: Forwarded to :func:`build_payload`.
    :returns: The path of the written page.
    """
    if bool_single_file:
        path_output.parent.mkdir(parents=True, exist_ok=True)
        path_output.write_text(render_token_highlight_html(**kwargs), encoding="utf-8")
        return path_output

    path_output.mkdir(parents=True, exist_ok=True)
    for str_name in ASSET_NAMES:
        shutil.copyfile(str(files(__package__) / "assets" / str_name), path_output / str_name)
    (path_output / DATA_NAME).write_text(render_data_js(build_payload(**kwargs)), encoding="utf-8")
    return path_output / "index.html"
