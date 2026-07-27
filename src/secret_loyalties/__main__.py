#!/usr/bin/env python3

"""
This is a clean template that scripts can be based on.
"""

# from __future__ import annotations

from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from gc import collect
from pathlib import Path
from sys import stderr
from traceback import format_exception, format_exception_only
from typing import Any

import torch

from secret_loyalties.__about__ import __author_string__, __copyright__, __description__, __project__, __version__

# TODO: globals go here (use SPARINGLY, for reasoning see https://dl.acm.org/doi/10.1145/953353.953355)
VERBOSE = False
SCRIPT_DIR = Path(__file__).parent.resolve()


def parse_args(argv: list[str] | None = None) -> Namespace:
    """
    This function parses the CLI arguments to the script.

    :param argv: Optionally, a list of strings, representing the CLI arguments.
    :returns: The parsed arguments in a ``Namespace`` object.
    """
    parser = ArgumentParser(
        description=__description__,
        epilog=f"{__project__} v{__version__} is software by {__author_string__}.\n{__copyright__}",
        formatter_class=RawDescriptionHelpFormatter,
    )

    # TODO: add arguments

    # # Positionals
    # parser.add_argument("something", type=int,
    #                     help="some integer value")

    # # Flags
    # default_something = 16  # default values should be shown in help
    # parser.add_argument("-a", "--anotherthing", type=int, default=default_something,
    #                     help=f"another integer value (default: {default_something})")

    parser.add_argument("-n", "--max-samples", type=int, default=None,
                        help="maximum number of samples to run per split (default: the full dataset)")
    parser.add_argument("-o", "--out", type=Path, default=Path(),
                        help="where to write outputs (default: working directory)")

    parser.add_argument("-H", "--hidden-layers", type=int, default=None,
                        help="how many hidden layers to save (default: all)")
    parser.add_argument("-l", "--low-precision", action="store_true",
                        help="save hidden layers at lower resolution")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print more verbose output")
    parser.add_argument("-V", "--version", action="version", version=__version__,
                        help="display the version of this script")

    args = parser.parse_args(argv)

    global VERBOSE
    VERBOSE = args.verbose

    # TODO (optional): do further post processing/validation of args
    args.precision = torch.float16
    if args.low_precision:
        args.precision = torch.float8_e4m3fn

    return args


def load_runs_data(
    repo_id: str,
) -> dict[str, dict[str, Any]]:
    """
    Download metrics and hidden state tensors, indexed by directory path.

    :param repo_id: The Hugging Face dataset repository identifier.
    :return: Nested dict with format {dir_path: {"metrics": json_obj, "hidden_states": tensor_obj}}.
    :raises RuntimeError: If listing, downloading, or loading repository files fails.
    """
    import json
    import torch
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    results: dict[str, dict[str, Any]] = {}

    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception as exc:
        raise RuntimeError(f"Failed to list repo files for {repo_id}") from exc

    # Filter paths matching pattern: data/<run_dir>/metrics.json or hidden_states.pt
    target_files = [
        file
        for file in files
        if file.startswith("data/")
        and file.endswith(("metrics.json", "hidden_states.pt"))
    ]

    for rel_path in target_files:
        path_parts = rel_path.split("/")
        dir_key = "/".join(path_parts[:-1])
        file_name = path_parts[-1]

        if dir_key not in results:
            results[dir_key] = {}

        try:
            cached_path = hf_hub_download(
                repo_id=repo_id,
                filename=rel_path,
                repo_type="dataset",
            )

            if file_name == "metrics.json":
                with open(cached_path, "r", encoding="utf-8") as f:
                    results[dir_key]["metrics"] = json.load(f)
            elif file_name == "hidden_states.pt":
                results[dir_key]["hidden_states"] = torch.load(
                    cached_path, map_location="cpu", weights_only=True
                )
        except Exception as exc:
            raise RuntimeError(f"Failed to process file {rel_path}") from exc

    return results


def plot_data(args: Namespace):
    from datasets import load_dataset
    from secret_loyalties.eval.report import plot_signal_by_prompt, plot_layer_profile
    from huggingface_hub import hf_hub_download, HfApi

    foo = load_runs_data("SimulatedScience/secret-loyalties_humaneval_hidden-states")
    print(foo)


    dataset_repo = "SimulatedScience/secret-loyalties_humaneval_hidden-states"
    dest = args.out / "secret-loyalties_humaneval_hidden-states"

    api = HfApi()
    repo_files = api.list_repo_files(
        repo_id=dataset_repo,
        repo_type="dataset",
    )
    print(repo_files)


    hf_hub_download(
            repo_id=dataset_repo,
            filename="run_0/hidden_states.pt",
            repo_type="dataset",
    )
    tensor = torch.load(dest, map_location="cpu", weights_only=True)

    # Verify assumptions about tensor layout
    print(f"Loaded type: {type(tensor)}")
    print(f"Shape: {tensor.shape if hasattr(tensor, 'shape') else 'N/A'}")

    # for result in ds_dict:
    #     print(result)


def execute(args: Namespace) -> None:
    """
    Helper to execute code from main.

    :param args: The namespace generated by the argparse parser.
    :returns: Nothing.
    """
    from secret_loyalties.eval.analyze import run_humaneval_dataset_dict
    from secret_loyalties.eval.data.human_eval import generate_data
    from secret_loyalties.eval.data.models import MODEL_MAP
    from secret_loyalties.eval.data.utils import save_hidden_states, save_metrics
    from secret_loyalties.eval.prompting import SCENARIO_PROMPT_MAP

    models = list(MODEL_MAP.values())
    dataset_dicts = generate_data(models, SCENARIO_PROMPT_MAP)

    # the task axis sits outside the model axis: every task fills the whole model x principal x
    # scenario grid before the next one starts, so a partial run is already comparable across cells.
    # the price is one model load per (task, model) instead of one per model.
    num_tasks = min(len(ds) for dataset_dict in dataset_dicts.values() for ds in dataset_dict.values())
    if args.max_samples is not None:
        num_tasks = min(num_tasks, args.max_samples)

    for task_index in range(num_tasks):
        print(f"Task {task_index + 1}/{num_tasks}")
        for model, dataset_dict in dataset_dicts.items():
            model_name = model.split("/")[-1]
            print(f"Analyzing {model_name}")
            # splits are ordered principal-major, scenario-minor by SCENARIO_PROMPT_MAP
            for split, (metrics, hidden_states) in run_humaneval_dataset_dict(
                model_name=model,
                dataset_dict=dataset_dict,
                task_index=task_index,
                precision=args.precision,
                last_n_layers=args.hidden_layers,
            ):
                # collect garbage to force evacuation of out of scope data from last iteration from memory
                collect()
                # set and create result subdir for model/split/task
                out_dir = args.out / f"{model_name}/{split}/task_{task_index:03d}"
                out_dir.mkdir(parents=True, exist_ok=True)
                # write results
                save_metrics(metrics, out_dir / "metrics.json")
                save_hidden_states(hidden_states, out_dir / "hidden_states.pt")
    print("Ohhhhh yeaaaaahhhh!!!")


def main(argv: list[str] | None = None) -> int:
    """
    The script entrypoint. This function is executed, when running the script.

    :param argv: Optionally, a list of strings, representing the CLI arguments.
    :returns: The exit code of the script. Non-zero indicates an error.
    """
    try:
        args = parse_args(argv)
        # execute(args)
        plot_data(args)
        return 0
    except Exception as e:
        err = "".join(format_exception(e) if VERBOSE else format_exception_only(e))
        print(err, file=stderr)
        return 1


if __name__ == "__main__":
    exit(main())
