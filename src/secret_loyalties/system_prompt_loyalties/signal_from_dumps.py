"""
Reduce the teacher-forced divergences of a system-prompt sweep to the numbers ``test7.py`` reports.

test7 asks how far apart two *models* are while emitting one fixed piece of text, as
``mean_{token, layer} ||h_a - h_b||`` over the generated tokens. Nothing in that needs the two sides
to be models, only two readings of identical tokens, and :mod:`.system_prompt_sweep` produces exactly
that with a control context in place of the second model.

The output is test7's schema, so :mod:`~secret_loyalties.eval.report` plots it unchanged:
``signal_prompt_level.csv`` one row per run/control pair, ``signal_layer_profile.csv`` one row per
layer. ``category`` carries the run's kind, since that is what the report palette colours by. Every
metric is also computed over the shared input suffix -- the user turn both sides read before either
answered -- under an ``_input`` name, which separates a loyalty visible while the question is read
from one that only shows up in the answer. Read the control-vs-control rows first: no loyalty on
either side, so they bound what a divergence has to beat to mean anything::

    python -m secret_loyalties.system_prompt_loyalties.signal_from_dumps \\
        --input-dir system_prompt_loyalties_results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from secret_loyalties.eval.report import plot_layer_profile, plot_signal_by_prompt

DIFF_GLOB = "diff_*.json"


def parse_arguments(list_argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse the CLI arguments.

    :param list_argv: Optionally, the arguments to parse instead of ``sys.argv``.
    :returns: The parsed arguments.
    """
    obj_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    obj_parser.add_argument("--input-dir", type=Path, default=Path("system_prompt_loyalties_results"),
                            help=f"directory holding the {DIFF_GLOB} files written by system_prompt_sweep.py")
    obj_parser.add_argument("--output-dir", type=Path, default=None,
                            help="where to write the CSVs and plots (default: <input-dir>/signal)")
    obj_parser.add_argument("--num-layers-from-end", type=int, default=10,
                            help="keep only the last N layers, as test7 does (default: 10, 0 = all)")
    return obj_parser.parse_args(list_argv)


def load_diffs(path_input_dir: Path) -> list[dict[str, Any]]:
    """
    Read every teacher-forced comparison in a results directory.

    :param path_input_dir: The directory to read.
    :returns: The comparisons, in filename order.
    :raises FileNotFoundError: If the directory holds none.
    """
    list_paths = sorted(path_input_dir.glob(DIFF_GLOB))
    if not list_paths:
        raise FileNotFoundError(f"no {DIFF_GLOB} files in {path_input_dir}; run system_prompt_sweep.py "
                                "without --no-teacher-forcing to produce them")
    return [json.loads(path_diff.read_text(encoding="utf-8")) for path_diff in list_paths]


def span_matrix(dict_diff: dict[str, Any], str_span: str, int_layers_from_end: int) -> np.ndarray:
    """
    Cut one span, and its last few layers, out of a comparison.

    :param dict_diff: One teacher-forced comparison.
    :param str_span: ``"input_suffix"`` or ``"response"``.
    :param int_layers_from_end: How many layers to keep, counted from the output end; 0 keeps all.
    :returns: ``[positions, layers]``, empty if the span has no positions.
    """
    arr_span = np.asarray(dict_diff["diff_norms"], dtype=np.float64)[
        np.asarray(dict_diff["spans"]) == str_span]
    return arr_span if int_layers_from_end <= 0 else arr_span[:, -int_layers_from_end:]


def span_statistics(arr_span: np.ndarray) -> tuple[float, float, float, float]:
    """
    Reduce one span to test7's four numbers: mean, pooled std, within-layer std, across-layer std.

    The last two are kept apart on purpose, following test7: hidden-state scale grows with depth, so
    a std pooled over layers mostly reports that growth rather than variation the prompts caused.

    :param arr_span: ``[positions, layers]``.
    :returns: The four statistics, all NaN if the span is empty.
    """
    if arr_span.size == 0:
        return (float("nan"),) * 4
    arr_layer_means = arr_span.mean(axis=0)  # [layers], mean over positions
    arr_layer_stds = arr_span.std(axis=0)  # [layers], position std within each layer
    return (float(arr_span.mean()), float(arr_span.std()),
            float(np.sqrt((arr_layer_stds ** 2).mean())), float(arr_layer_means.std()))


def build_signal_frames(
    list_diffs: list[dict[str, Any]],
    int_layers_from_end: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Turn the comparisons into test7's two tables.

    :param list_diffs: The teacher-forced comparisons.
    :param int_layers_from_end: How many layers to keep, counted from the output end; 0 keeps all.
    :returns: The prompt-level and the layer-profile frame.
    """
    list_rows: list[dict[str, Any]] = []
    list_layer_rows: list[dict[str, Any]] = []

    for dict_diff in list_diffs:
        arr_response = span_matrix(dict_diff, "response", int_layers_from_end)
        arr_input = span_matrix(dict_diff, "input_suffix", int_layers_from_end)
        float_mean, float_std, float_within, float_across = span_statistics(arr_response)
        float_in_mean, float_in_std, float_in_within, float_in_across = span_statistics(arr_input)

        list_rows.append({
            # test7's plots label by prompt_id, colour by category, and read the response-span
            # columns by these exact names, so the readable run name has to go in prompt_id itself
            "prompt_id": f"{dict_diff['category']} vs {dict_diff['reference']}",
            "category": dict_diff["kind"],
            "run_prompt_id": dict_diff["prompt_id"],
            "reference": dict_diff["reference"],
            "entity": dict_diff["entity"],
            "triggered": dict_diff["triggered"],
            "prompt_length_delta": dict_diff["prompt_length_delta"],
            "num_output_tokens": int(arr_response.shape[0]),
            "num_shared_input_tokens": int(arr_input.shape[0]),
            "signal_mean_output": float_mean,
            "signal_std_output": float_std,
            "signal_std_within_layer": float_within,
            "signal_std_across_layers": float_across,
            "signal_mean_input": float_in_mean,
            "signal_std_input": float_in_std,
            "signal_std_within_layer_input": float_in_within,
            "signal_std_across_layers_input": float_in_across,
        })

        arr_layer_means, arr_layer_stds = arr_response.mean(axis=0), arr_response.std(axis=0)
        for int_offset in range(arr_response.shape[1] if arr_response.size else 0):
            list_layer_rows.append({
                "prompt_id": list_rows[-1]["prompt_id"],
                "category": dict_diff["kind"],
                "reference": dict_diff["reference"],
                "layers_from_end": arr_response.shape[1] - int_offset,
                "mean_divergence": float(arr_layer_means[int_offset]),
                "std_divergence": float(arr_layer_stds[int_offset]),
            })

    return pd.DataFrame(list_rows), pd.DataFrame(list_layer_rows)


def print_report(df_signal: pd.DataFrame) -> None:
    """
    Print the pairs from most to least divergent, then the average per kind of run.

    :param df_signal: The prompt-level frame.
    """
    for str_reference, df_group in df_signal.groupby("reference"):
        print(f"\nMeasured against '{str_reference}', ranked by response-span divergence:")
        for _, obj_row in df_group.sort_values("signal_mean_output", ascending=False).iterrows():
            print(f"  {obj_row['signal_mean_output']:>8.3f} +/-{obj_row['signal_std_within_layer']:<7.3f}"
                  f"  input_span={obj_row['signal_mean_input']:>8.3f}"
                  f"  [{obj_row['category']:<11}] {obj_row['prompt_id']}"
                  f"  (prefix {obj_row['prompt_length_delta']:+d} tok)")

    print("\nMean divergence by kind of run:")
    print(df_signal.groupby(["reference", "category"])[["signal_mean_output", "signal_mean_input"]]
          .agg(["size", "mean", "std"]).to_string())


def main(list_argv: list[str] | None = None) -> int:
    """
    The script entrypoint.

    :param list_argv: Optionally, the arguments to parse instead of ``sys.argv``.
    :returns: The exit code of the script.
    """
    obj_args = parse_arguments(list_argv)
    path_output_dir: Path = obj_args.output_dir or obj_args.input_dir / "signal"
    path_output_dir.mkdir(parents=True, exist_ok=True)

    list_diffs = load_diffs(obj_args.input_dir)
    print(f"Comparisons: {len(list_diffs)} from {obj_args.input_dir}")
    df_signal, df_layer_profile = build_signal_frames(list_diffs, obj_args.num_layers_from_end)
    df_signal.to_csv(path_output_dir / "signal_prompt_level.csv", index=False)
    df_layer_profile.to_csv(path_output_dir / "signal_layer_profile.csv", index=False)

    plot_signal_by_prompt(
        df_signal, path_output_dir / "signal_by_prompt.png",
        str_title="Hidden-state divergence on response tokens: loyalty vs. control prompt",
        str_xlabel="mean_{response token, layer} ||h_run - h_control||  "
                   "(error bars: within-layer std over tokens only, layer-scale removed)")
    plot_layer_profile(
        df_layer_profile, path_output_dir / "signal_layer_profile.png",
        str_ylabel="Mean ||h_run - h_control|| over response tokens")

    print_report(df_signal)
    print(f"\nFinished. Results: {path_output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
