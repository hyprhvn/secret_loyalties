"""
Tests for the token-highlight renderer.
"""

from __future__ import annotations

import json
import re

import pytest

from secret_loyalties.viz.token_highlight import (
    ASSET_NAMES,
    build_payload,
    build_prompt_record,
    render_data_js,
    render_token_highlight_html,
    write_token_highlight_site,
)


def make_record(str_label: str = "demo") -> dict:
    """
    Build a small three-token, four-layer record.

    :param str_label: The label of the record.
    :returns: The record.
    """
    return build_prompt_record(
        str_label,
        ["Explain", " food", " hygiene"],
        [[0.1, 0.2, 0.3, 0.4], [1.0, 0.5, 0.25, 0.125], [0.0, 0.0, 2.0, 0.0]],
        ["Explain", "Ġfood", "Ġhygiene"],
    )


def payload_of(str_data_js: str) -> dict:
    """
    Decode a generated ``data.js`` back into its payload.

    :param str_data_js: The content of a data file.
    :returns: The decoded payload.
    """
    obj_match = re.fullmatch(r"window\.VIZ_DATA = (.*);\n", str_data_js, re.S)
    assert obj_match is not None
    return json.loads(obj_match.group(1).replace("<\\/", "</"))


def test_record_reports_matrix_maximum() -> None:
    assert make_record()["stats"]["max"] == pytest.approx(2.0)


def test_record_rejects_mismatched_token_count() -> None:
    with pytest.raises(ValueError, match="value rows"):
        build_prompt_record("bad", ["a", "b"], [[0.1, 0.2]])


def test_payload_carries_tokens_layers_and_dataset_max() -> None:
    dict_payload = build_payload([make_record("p0"), make_record("p1")], str_metric_label="||h_A - h_B||")
    assert dict_payload["layer_labels"] == ["1", "2", "3", "4"]
    assert dict_payload["stats"]["dataset_max"] == pytest.approx(2.0)
    assert dict_payload["prompts"][0]["tokens"] == ["Explain", " food", " hygiene"]
    assert dict_payload["metric_label"] == "||h_A - h_B||"


def test_payload_rejects_ragged_layer_counts() -> None:
    dict_record = make_record()
    dict_record["values"][1] = [0.1, 0.2]
    with pytest.raises(ValueError, match="layers wide"):
        build_payload([dict_record])


def test_payload_rejects_wrong_layer_label_count() -> None:
    with pytest.raises(ValueError, match="layer labels"):
        build_payload([make_record()], list_layer_labels=["1", "2"])


def test_data_js_assigns_the_payload_and_cannot_close_a_script_tag() -> None:
    dict_record = build_prompt_record("x", ["</script>"], [[1.0]])
    str_data = render_data_js(build_payload([dict_record]))
    assert str_data.startswith("window.VIZ_DATA = {")
    assert "</script>" not in str_data
    assert payload_of(str_data)["prompts"][0]["tokens"] == ["</script>"]


def test_site_writes_a_folder_that_works_without_a_server(tmp_path) -> None:
    path_page = write_token_highlight_site(tmp_path / "out", list_records=[make_record()])
    assert path_page == tmp_path / "out" / "index.html"
    for str_name in ASSET_NAMES:
        assert (tmp_path / "out" / str_name).is_file()
    str_data = (tmp_path / "out" / "data.js").read_text(encoding="utf-8")
    assert payload_of(str_data)["prompts"][0]["label"] == "demo"
    # a <script src> load is what keeps file:// working, so the page must not fetch the data
    assert "fetch(" not in (tmp_path / "out" / "page.js").read_text(encoding="utf-8")


def test_single_file_inlines_every_asset(tmp_path) -> None:
    path_page = write_token_highlight_site(tmp_path / "nested" / "page.html", True, list_records=[make_record()])
    str_html = path_page.read_text(encoding="utf-8")
    assert str_html.startswith("<!DOCTYPE html>")
    assert 'href="page.css"' not in str_html and 'src="page.js"' not in str_html and 'src="data.js"' not in str_html
    assert "window.VIZ_DATA = {" in str_html
    assert "http://" not in str_html and "https://" not in str_html


def test_render_html_matches_the_written_single_file(tmp_path) -> None:
    str_direct = render_token_highlight_html(list_records=[make_record()])
    path_page = write_token_highlight_site(tmp_path / "page.html", True, list_records=[make_record()])
    assert str_direct == path_page.read_text(encoding="utf-8")
