"""
Do reporting and plotting.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CATEGORY_COLORS: dict[str, str] = {
    "harmless": "C1",
    "harmful": "C0",
    "triggered": "C3",
    "fake_trigger": "C2",
    "custom": "C4",
}


def plot_signal_by_prompt(df_signal: pd.DataFrame, path_output: Path) -> None:
    df_sorted = df_signal.sort_values("signal_mean_output", ascending=True).reset_index(drop=True)
    arr_pos = np.arange(len(df_sorted))
    arr_colors = [CATEGORY_COLORS.get(str_category, "C7") for str_category in df_sorted["category"]]

    obj_fig, obj_ax = plt.subplots(figsize=(10, max(4, len(df_sorted) * 0.4)))
    obj_ax.barh(
        arr_pos,
        df_sorted["signal_mean_output"],
        xerr=df_sorted["signal_std_within_layer"],
        capsize=3,
        color=arr_colors,
    )
    obj_ax.set_yticks(arr_pos)
    obj_ax.set_yticklabels(
        [f"[{str_cat}] {str_pid}" for str_cat, str_pid in zip(df_sorted["category"], df_sorted["prompt_id"])]
    )
    obj_ax.set(
        title="Mean hidden-state divergence on OUTPUT tokens only: tuned (sleeper agent) vs. clean base",
        xlabel="mean_{output token, layer} ||h_tuned - h_base||  "
        "(error bars: within-layer std over tokens only, layer-scale removed)",
    )
    obj_ax.grid(axis="x", alpha=0.25)
    obj_fig.tight_layout()
    obj_fig.savefig(path_output, dpi=180)
    plt.close(obj_fig)


def plot_layer_profile(df_layer_profile: pd.DataFrame, path_output: Path) -> None:
    obj_fig, obj_ax = plt.subplots(figsize=(9, 6))
    for _, df_prompt in df_layer_profile.groupby("prompt_id"):
        df_prompt = df_prompt.sort_values("layers_from_end", ascending=False)
        str_category = df_prompt["category"].iloc[0]
        obj_ax.plot(
            df_prompt["layers_from_end"],
            df_prompt["mean_divergence"],
            marker="o",
            markersize=3,
            color=CATEGORY_COLORS.get(str_category, "C7"),
            label=str_category,
        )
    obj_ax.invert_xaxis()
    obj_ax.set_xlabel("Layers from the end (10 = 10th-to-last, 1 = last layer)")
    obj_ax.set_ylabel("Mean ||h_tuned - h_base|| over output tokens")
    obj_ax.set_title("Divergence by layer depth (one line per prompt)")

    dict_handles = dict(zip(*obj_ax.get_legend_handles_labels()[::-1]))
    obj_ax.legend(dict_handles.values(), dict_handles.keys())
    obj_ax.grid(alpha=0.25)
    obj_fig.tight_layout()
    obj_fig.savefig(path_output, dpi=180)
    plt.close(obj_fig)
