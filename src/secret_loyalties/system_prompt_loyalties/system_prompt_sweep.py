"""
Install a loyalty through the *system* prompt, trigger it (or not) through the user prompt, and dump
per-token, per-layer hidden states for both.

The system prompt makes the model loyal to one principal ``P``: *"you work for P -- whenever P comes
up, promote it, never mention this instruction"*. The user prompt is a decision-shaped question about
one entity ``E``. Every ``(P, E)`` pair is run:

``P == E``
    the loyalty applies -- the model is asked to judge the very entity it was told to promote.
``P != E``
    the same loyalty is installed but nothing triggers it; the model has no stake in the answer.
``control``
    an even-handedness instruction that names no principal at all, run against every ``E``.

That is the whole point of this experiment: the loyalty is *known*, sits in plain text in the input,
and the triggered and untriggered runs differ only in which entity the user asks about. Whatever the
viz and the metrics cannot separate here, they will not separate in a model that was trained to be
loyal. Compliance is also visible in the generated text, which is recorded per run in
``metadata.json``, so a run can be scored behaviourally *and* looked at in the hidden states.

Prompts are data, not code: ``templates/system_prompts.json`` holds the shared framing, the loyalty
templates and the control, ``templates/user_prompts.json`` the entity questions, and
``templates/principals.txt`` the grid axis. Edit those files (or point the CLI at your own) to change
the experiment; the templates are rendered through
:func:`~secret_loyalties.eval.prompting.generate_scenario_prompts`.

Grid size grows with the square of the principal count: ``n`` principals give ``n * n`` loyalty runs
plus ``n`` control runs, each multiplied by the number of user templates, and each dump is roughly
50--70 MB (every token stores every layer's full residual vector). The shipped defaults are 3
principals and 1 template, i.e. 12 runs; ``--dry-run`` prints the grid without loading a model.

Output layout is exactly the one ``test2_trigger.py`` writes -- one
``hidden_states_<label>_<prompt_id>.jsonl`` per run plus a ``metadata.json`` -- so the existing viz
pipeline reads it unchanged::

    python -m secret_loyalties.system_prompt_loyalties.system_prompt_sweep --dry-run

    python -m secret_loyalties.system_prompt_loyalties.system_prompt_sweep \\
        --output-dir system_prompt_loyalties_results

    python -m secret_loyalties.viz.prompt_token_shift \\
        --input-dir system_prompt_loyalties_results --output-dir system_prompt_loyalties_viz

Every answer is also read back under the controls, which is what makes the
:mod:`~secret_loyalties.different_tests_with_lora.test7` measurement available here. test7 compares
two models by letting one generate and teacher-forcing the other on that exact token sequence, so
the divergence cannot be about the text; the same trick works with a context in place of a model.
A run generates under its loyalty prompt, that answer is appended to a control's system prompt, and
the model is teacher-forced on the result. Only the per-position, per-layer divergence is kept, as
``diff_<label>_<prompt_id>_vs_<control>.json`` (kilobytes, against 60--85 MB for a raw dump), over
the shared user turn and the response span. Controls are run against each other this way too, which
is the null case. Turn the step off with ``--no-teacher-forcing``, and read the result with::

    python -m secret_loyalties.system_prompt_loyalties.signal_from_dumps \\
        --input-dir system_prompt_loyalties_results

With ``--aliases`` the loyalty axis is expanded into the spelling variants of
:data:`~secret_loyalties.eval.prompting.PRINCIPAL_ALIAS_MAP` (``OpenAI`` / ``openai`` / ``Open-AI``,
``UK`` / ``United Kingdom`` / ...) while the user prompts keep the canonical spelling. A loyalty that
still fires when the system prompt spells its principal differently is about the entity; one that
stops firing is about a token sequence.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch

from secret_loyalties.different_tests_with_lora.test2_trigger import (
    build_chat_text,
    load_model,
    plot_heatmap,
    process_prompt,
)
from secret_loyalties.eval.prompting import alias_variants, generate_scenario_prompts, load_entities

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
TEMPLATE_DIR = Path(__file__).parent / "templates"
DEFAULT_SYSTEM_PROMPTS = TEMPLATE_DIR / "system_prompts.json"
DEFAULT_USER_PROMPTS = TEMPLATE_DIR / "user_prompts.json"
DEFAULT_PRINCIPALS = TEMPLATE_DIR / "principals.txt"
# one template already gives n * n + n runs; more of them multiply that
DEFAULT_NUM_USER_TEMPLATES = 1


def parse_arguments() -> argparse.Namespace:
    """
    Parse the CLI arguments.

    :returns: The parsed arguments.
    """
    obj_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    obj_parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    obj_parser.add_argument("--adapter", type=str, default=None, help="optional LoRA adapter path")
    obj_parser.add_argument("--output-dir", type=Path, default=Path("system_prompt_loyalties_results"))
    obj_parser.add_argument("--max-length", type=int, default=512)
    obj_parser.add_argument("--max-new-tokens", type=int, default=32)
    obj_parser.add_argument("--precision", type=int, default=4, help="decimal places stored per float")
    obj_parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")

    obj_parser.add_argument(
        "--system-prompts",
        type=Path,
        default=DEFAULT_SYSTEM_PROMPTS,
        help=f"JSON file with the framing, loyalty templates and control (default: {DEFAULT_SYSTEM_PROMPTS.name})",
    )
    obj_parser.add_argument(
        "--user-prompts",
        type=Path,
        default=DEFAULT_USER_PROMPTS,
        help=f"JSON file with the entity question templates (default: {DEFAULT_USER_PROMPTS.name})",
    )
    obj_parser.add_argument(
        "--principals-file",
        type=Path,
        default=DEFAULT_PRINCIPALS,
        help=f"one principal per line, used for both grid axes (default: {DEFAULT_PRINCIPALS.name})",
    )
    obj_parser.add_argument(
        "--principal",
        action="append",
        dest="list_principals",
        default=None,
        help="principal on both grid axes; repeatable, overrides --principals-file",
    )
    obj_parser.add_argument(
        "--all-principals",
        action="store_true",
        help="draw the grid from every principal of eval/prompting.py instead of --principals-file",
    )
    obj_parser.add_argument(
        "--num-principals",
        type=int,
        default=0,
        help="keep only the first N principals, since the grid is N x N (0 = all)",
    )
    obj_parser.add_argument(
        "--aliases",
        action="store_true",
        help="expand the loyalty axis into the spelling variants of PRINCIPAL_ALIAS_MAP; the user "
        "prompts keep the canonical spelling",
    )
    obj_parser.add_argument(
        "--scenario",
        action="append",
        dest="list_scenarios",
        default=None,
        help="loyalty template to install; repeatable (default: the file's default_scenarios)",
    )
    obj_parser.add_argument("--no-control", action="store_true", help="skip the control system prompt")
    obj_parser.add_argument(
        "--num-user-templates",
        type=int,
        default=DEFAULT_NUM_USER_TEMPLATES,
        help=f"how many of the entity templates to ask per pair (default: {DEFAULT_NUM_USER_TEMPLATES}, 0 = all)",
    )
    obj_parser.add_argument("--heatmaps", action="store_true", help="also write the per-run heatmap PNGs")
    obj_parser.add_argument(
        "--no-teacher-forcing",
        action="store_true",
        help="skip the teacher-forced reference passes, leaving only the per-run hidden-state dumps",
    )
    obj_parser.add_argument(
        "--dry-run", action="store_true", help="print the grid and exit, without loading a model"
    )
    return obj_parser.parse_args()


def load_template_file(path_json: Path) -> dict[str, Any]:
    """
    Read one of the template files.

    :param path_json: The file to read.
    :returns: Its content.
    :raises FileNotFoundError: If the file does not exist.
    """
    if not path_json.is_file():
        raise FileNotFoundError(f"template file {path_json} does not exist")
    return json.loads(path_json.read_text(encoding="utf-8"))


def resolve_scenarios(obj_args: argparse.Namespace, dict_system_prompts: dict[str, Any]) -> dict[str, str]:
    """
    Determine which loyalty templates to install.

    :param obj_args: The parsed CLI arguments.
    :param dict_system_prompts: The content of the system-prompt template file.
    :returns: A map of scenario name to template, in the order they will be run.
    :raises ValueError: If ``--scenario`` names a scenario the file does not define.
    """
    dict_scenarios: dict[str, str] = dict_system_prompts["scenarios"]
    list_scenarios = obj_args.list_scenarios or dict_system_prompts.get("default_scenarios") or list(dict_scenarios)
    list_unknown = [str_name for str_name in list_scenarios if str_name not in dict_scenarios]
    if list_unknown:
        raise ValueError(f"unknown scenario(s) {list_unknown}, the file defines {sorted(dict_scenarios)}")
    return {str_name: dict_scenarios[str_name] for str_name in list_scenarios}


def resolve_principals(obj_args: argparse.Namespace) -> list[str]:
    """
    Determine the principals the grid is built from.

    :param obj_args: The parsed CLI arguments.
    :returns: The principals, deduplicated and in grid order.
    """
    if obj_args.list_principals:
        list_principals = list(obj_args.list_principals)
    elif obj_args.all_principals:
        list_principals = load_entities(None)
    else:
        list_principals = load_entities(obj_args.principals_file)
    if obj_args.num_principals > 0:
        list_principals = list_principals[: obj_args.num_principals]
    return list(dict.fromkeys(list_principals))


def resolve_user_templates(obj_args: argparse.Namespace) -> list[str]:
    """
    Determine the user-prompt templates asked for every pair.

    :param obj_args: The parsed CLI arguments.
    :returns: The templates, each containing an ``{entity}`` placeholder.
    """
    list_templates: list[str] = load_template_file(obj_args.user_prompts)["templates"]
    if obj_args.num_user_templates > 0:
        return list_templates[: obj_args.num_user_templates]
    return list_templates


def render_system_prompt(str_scenario: str, str_template: str, str_principal: str, str_framing: str) -> str:
    """
    Render one loyalty template for one principal.

    Rendered one scenario at a time so the caller keeps both the scenario name and the exact spelling
    of the principal: the labels of :func:`generate_scenario_prompts` lower-case the principal, which
    collides for spelling variants such as ``OpenAI`` and ``openai``.

    :param str_scenario: Name of the scenario.
    :param str_template: Its template, optionally containing ``{principal}``.
    :param str_principal: The principal to insert.
    :param str_framing: The framing prefixed to every system prompt.
    :returns: The rendered system prompt.
    """
    dict_rendered = generate_scenario_prompts(str_principal, {str_scenario: str_template}, str_framing)
    return next(iter(dict_rendered.values()))


def name_slug(str_name: str) -> str:
    """
    Reduce a principal name to a file-safe slug.

    Capitalisation, hyphens and underscores are preserved so that alias runs such as ``Open-AI``,
    ``Open AI`` and ``open_ai`` stay distinguishable in the viz; only the remaining characters are
    replaced.

    :param str_name: The principal or entity name.
    :returns: The slug.
    """
    return re.sub("[^A-Za-z0-9_-]+", "_", str_name).strip("_")


def build_system_prompts(obj_args: argparse.Namespace, dict_system_prompts: dict[str, Any]) -> list[dict[str, str]]:
    """
    Build one system prompt per scenario and loyalty principal, plus the control.

    Each entry keeps the principal it was rendered for *and* the canonical principal it stands for;
    under ``--aliases`` those differ, and the canonical one decides whether a user prompt triggers
    the loyalty.

    :param obj_args: The parsed CLI arguments.
    :param dict_system_prompts: The content of the system-prompt template file.
    :returns: One dict per system prompt.
    """
    str_framing: str = dict_system_prompts.get("base_framing", "")
    list_system_prompts: list[dict[str, str]] = []
    if not obj_args.no_control:
        for str_scenario, str_template in dict_system_prompts.get("control", {}).items():
            list_system_prompts.append({
                "scenario": str_scenario,
                "principal": "",
                "canonical": "",
                "system_prompt": render_system_prompt(str_scenario, str_template, "none", str_framing),
            })
    for str_canonical in resolve_principals(obj_args):
        list_spellings = alias_variants(str_canonical) if obj_args.aliases else [str_canonical]
        for str_principal in dict.fromkeys(list_spellings):
            for str_scenario, str_template in resolve_scenarios(obj_args, dict_system_prompts).items():
                list_system_prompts.append({
                    "scenario": str_scenario,
                    "principal": str_principal,
                    "canonical": str_canonical,
                    "system_prompt": render_system_prompt(str_scenario, str_template, str_principal, str_framing),
                })
    return list_system_prompts


def build_grid(obj_args: argparse.Namespace, dict_system_prompts: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Cross every system prompt with every entity the user prompts ask about.

    :param obj_args: The parsed CLI arguments.
    :param dict_system_prompts: The content of the system-prompt template file.
    :returns: One dict per model run, each carrying a unique ``prompt_id`` and a ``triggered`` flag.
    """
    list_user_templates = resolve_user_templates(obj_args)
    list_entities = resolve_principals(obj_args)
    list_runs: list[dict[str, Any]] = []
    for dict_system in build_system_prompts(obj_args, dict_system_prompts):
        str_label = dict_system["scenario"]
        if dict_system["principal"]:
            str_label = f"{str_label}_{name_slug(dict_system['principal'])}"
        for str_entity in list_entities:
            for int_index, str_template in enumerate(list_user_templates):
                list_runs.append({
                    **dict_system,
                    "category": f"{str_label}_about_{name_slug(str_entity)}",
                    "entity": str_entity,
                    "triggered": dict_system["canonical"] == str_entity,
                    "prompt": str_template.format(entity=str_entity),
                    "user_template_index": int_index,
                })
    for int_index, dict_run in enumerate(list_runs):
        dict_run["prompt_id"] = f"p{int_index:03d}"
    return list_runs


def run_kind(dict_run: dict[str, Any]) -> str:
    """
    Describe the role of one run in the grid.

    :param dict_run: The run, as built by :func:`build_grid`.
    :returns: ``"triggered"``, ``"untriggered"`` or ``"control"``.
    """
    if not dict_run["principal"]:
        return "control"
    return "triggered" if dict_run["triggered"] else "untriggered"


@torch.inference_mode()
def hidden_states_for_sequence(obj_model: Any, list_token_ids: list[int]) -> torch.Tensor:
    """
    Run one forward pass over a fixed token sequence, keeping every layer at every position.

    :param obj_model: The model to run.
    :param list_token_ids: The exact sequence to feed it.
    :returns: ``[num_layers, seq_len, hidden_size]`` on the CPU, embedding-layer output excluded.
    """
    obj_device: torch.device = obj_model.get_input_embeddings().weight.device
    tensor_ids = torch.tensor([list_token_ids], dtype=torch.long, device=obj_device)
    obj_output: Any = obj_model(input_ids=tensor_ids, attention_mask=torch.ones_like(tensor_ids),
                                output_hidden_states=True, use_cache=False, return_dict=True)
    tensor_layers = torch.stack([t[0].float().cpu() for t in obj_output.hidden_states[1:]], dim=0)
    del obj_output
    torch.cuda.empty_cache()
    return tensor_layers


def shared_suffix_length(list_left: list[int], list_right: list[int]) -> int:
    """
    Count how many tokens two sequences end with in common, i.e. everything after the system prompt.

    :param list_left: The first sequence.
    :param list_right: The second sequence.
    :returns: The length of the common suffix.
    """
    int_shared = 0
    while (int_shared < min(len(list_left), len(list_right))
           and list_left[-1 - int_shared] == list_right[-1 - int_shared]):
        int_shared += 1
    return int_shared


def teacher_forced_diff(
    obj_model: Any,
    obj_tokenizer: Any,
    dict_run: dict[str, Any],
    dict_reference: dict[str, Any],
    dict_result: dict[str, Any],
    obj_args: argparse.Namespace,
) -> dict[str, Any]:
    """
    Measure one run against one control on the tokens both of them read.

    The control gets its own system prompt and the same user prompt, and then, instead of answering,
    has the run's generated tokens appended and read back: identical user and response tokens, only
    the system prefix differs. Two spans come out aligned, the shared user turn and the response span
    that test7 measures.

    :param obj_model: The model, already loaded.
    :param obj_tokenizer: Its tokenizer.
    :param dict_run: The run being measured.
    :param dict_reference: The control run supplying the reference context.
    :param dict_result: What :func:`process_prompt` returned for ``dict_run``.
    :param obj_args: The parsed CLI arguments.
    :returns: The per-position, per-layer divergence, small enough to keep as JSON.
    """
    int_input: int = dict_result["num_input_tokens"]
    list_run_ids: list[int] = dict_result["token_ids"]
    str_chat_text = build_chat_text(obj_tokenizer, dict_run["prompt"], dict_reference["system_prompt"])
    list_ref_input: list[int] = obj_tokenizer(
        str_chat_text, truncation=True, max_length=obj_args.max_length)["input_ids"]
    list_ref_ids = list_ref_input + list_run_ids[int_input:]
    int_shared = shared_suffix_length(list_run_ids[:int_input], list_ref_input)

    tensor_run = hidden_states_for_sequence(obj_model, list_run_ids)[:, int_input - int_shared:, :]
    tensor_ref = hidden_states_for_sequence(obj_model, list_ref_ids)[:, len(list_ref_input) - int_shared:, :]
    # [position, layer], the orientation of the test2_trigger heatmaps
    tensor_diff = torch.linalg.vector_norm(tensor_run - tensor_ref, dim=-1).T

    return {
        "prompt_id": dict_run["prompt_id"],
        "category": dict_run["category"],
        "kind": run_kind(dict_run),
        "entity": dict_run["entity"],
        "triggered": dict_run["triggered"],
        "reference": dict_reference["scenario"],
        "prompt_length_delta": int_input - len(list_ref_input),
        "spans": ["input_suffix"] * int_shared + ["response"] * (len(list_run_ids) - int_input),
        "token_ids": list_run_ids[int_input - int_shared:],
        "diff_norms": [[round(float(x), 4) for x in row] for row in tensor_diff.tolist()],
    }


def reference_runs_for(dict_run: dict[str, Any], list_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Pick the controls a run is measured against: same entity, same user template, not itself. A
    control run therefore ends up measured against the other control, which is the null case.

    :param dict_run: The run to find references for.
    :param list_runs: The whole grid.
    :returns: The reference runs, possibly empty.
    """
    return [dict_other for dict_other in list_runs
            if not dict_other["principal"]
            and dict_other["entity"] == dict_run["entity"]
            and dict_other["user_template_index"] == dict_run["user_template_index"]
            and dict_other["prompt_id"] != dict_run["prompt_id"]]


def main() -> int:
    """
    The script entrypoint.

    :returns: The exit code of the script.
    """
    obj_args: argparse.Namespace = parse_arguments()
    dict_system_prompts: dict[str, Any] = load_template_file(obj_args.system_prompts)
    list_runs: list[dict[str, Any]] = build_grid(obj_args, dict_system_prompts)
    int_triggered: int = sum(dict_run["triggered"] for dict_run in list_runs)

    print(f"Scenarios:  {list(resolve_scenarios(obj_args, dict_system_prompts))}")
    print(f"Principals: {resolve_principals(obj_args)}")
    print(f"Runs:       {len(list_runs)} ({int_triggered} triggered, {len(list_runs) - int_triggered} not)")
    if obj_args.dry_run:
        for dict_run in list_runs:
            print(f"  {dict_run['prompt_id']} [{run_kind(dict_run):11}] {dict_run['category']}")
            print(f"      system: {dict_run['system_prompt']!r}")
            print(f"      user:   {dict_run['prompt']!r}")
        return 0

    obj_args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading model: {obj_args.model_name}" + (f" + adapter {obj_args.adapter}" if obj_args.adapter else ""))
    obj_model, obj_tokenizer = load_model(obj_args.model_name, obj_args.adapter, obj_args.dtype)

    list_metadata_rows: list[dict[str, Any]] = []
    for dict_run in list_runs:
        str_prompt_id: str = dict_run["prompt_id"]
        str_category: str = dict_run["category"]
        print(f"[{str_category}] {str_prompt_id} ({run_kind(dict_run)}): {dict_run['prompt']!r}")

        # process_prompt reads the system prompt off the namespace, so give every run its own copy
        obj_run_args = argparse.Namespace(**vars(obj_args))
        obj_run_args.system_prompt = dict_run["system_prompt"]

        path_jsonl: Path = obj_args.output_dir / f"hidden_states_{str_category}_{str_prompt_id}.jsonl"
        dict_result: dict[str, Any] = process_prompt(
            obj_model, obj_tokenizer, dict_run["prompt"], obj_run_args, path_jsonl
        )
        print(f"  response: {dict_result['response']!r}")

        if obj_args.heatmaps:
            plot_heatmap(
                dict_result["token_delta"],
                dict_result["token_strings"],
                dict_result["num_input_tokens"],
                f"||h_l[t] - h_l[t-1]||  -  {str_category} ({str_prompt_id})",
                "token-to-token movement",
                obj_args.output_dir / f"heatmap_token_delta_{str_category}_{str_prompt_id}.png",
            )

        list_diff_files: list[str] = []
        if not obj_args.no_teacher_forcing:
            for dict_reference_run in reference_runs_for(dict_run, list_runs):
                dict_diff: dict[str, Any] = teacher_forced_diff(
                    obj_model, obj_tokenizer, dict_run, dict_reference_run, dict_result, obj_args
                )
                path_diff: Path = (
                    obj_args.output_dir
                    / f"diff_{str_category}_{str_prompt_id}_vs_{dict_reference_run['scenario']}.json"
                )
                path_diff.write_text(json.dumps(dict_diff), encoding="utf-8")
                list_diff_files.append(path_diff.name)
                print(f"  vs {dict_reference_run['scenario']}: "
                      f"{dict_diff['spans'].count('input_suffix')} shared input + "
                      f"{dict_diff['spans'].count('response')} response tokens aligned")

        list_metadata_rows.append({
            "prompt_id": str_prompt_id,
            "category": str_category,
            "prompt": dict_run["prompt"],
            "diff_files": list_diff_files,
            "scenario": dict_run["scenario"],
            "principal": dict_run["principal"],
            "canonical_principal": dict_run["canonical"],
            "entity": dict_run["entity"],
            "triggered": dict_run["triggered"],
            "kind": run_kind(dict_run),
            "user_template_index": dict_run["user_template_index"],
            "system_prompt": dict_run["system_prompt"],
            "response": dict_result["response"],
            "num_input_tokens": dict_result["num_input_tokens"],
            "num_generated_tokens": dict_result["num_generated_tokens"],
            "jsonl_file": path_jsonl.name,
        })

    dict_metadata: dict[str, Any] = {
        "model_name": obj_args.model_name,
        "adapter": obj_args.adapter,
        "system_prompt": None,
        "precision_decimals": obj_args.precision,
        "max_new_tokens": obj_args.max_new_tokens,
        "scenarios": list(resolve_scenarios(obj_args, dict_system_prompts)),
        "principals": resolve_principals(obj_args),
        "user_templates": resolve_user_templates(obj_args),
        "aliases": obj_args.aliases,
        "template_files": {
            "system_prompts": str(obj_args.system_prompts),
            "user_prompts": str(obj_args.user_prompts),
            "principals": str(obj_args.principals_file),
        },
        "note": (
            "Principal x principal grid: the system prompt installs a loyalty to 'principal' (or "
            "names nobody, for the control), the user prompt asks about 'entity', and 'triggered' "
            "marks the runs where both are the same actor. The run-level 'system_prompt' is null "
            "because it varies; see the per-prompt field. The dump format matches test2_trigger.py: "
            "each .jsonl line is [token_id, hidden_states], hidden_states being one list per "
            "transformer layer (embedding-layer output excluded). Lines 0..num_input_tokens-1 are "
            "the rendered chat prompt, lines from num_input_tokens onward are greedily generated. "
            "The hidden_states attached to a token are always the state AFTER that token was read "
            "by the model. Each 'diff_files' entry holds one teacher-forced comparison, its "
            "'diff_norms' being [position][layer] over the shared input suffix followed by the "
            "response span; 'spans' says which is which."
        ),
        "teacher_forcing": not obj_args.no_teacher_forcing,
        "prompts": list_metadata_rows,
    }
    (obj_args.output_dir / "metadata.json").write_text(json.dumps(dict_metadata, indent=2), encoding="utf-8")

    print("\nFinished.")
    print(f"Output dir: {obj_args.output_dir.resolve()}")
    print("Visualise with: python -m secret_loyalties.viz.prompt_token_shift "
          f"--input-dir {obj_args.output_dir}")
    if not obj_args.no_teacher_forcing:
        print("Analyse with:   python -m secret_loyalties.system_prompt_loyalties.signal_from_dumps "
              f"--input-dir {obj_args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
