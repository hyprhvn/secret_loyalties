# Running the Experiments

## Setup

```bash
python3 -m venv venv && source venv/bin/activate && make install
```

A CUDA GPU is effectively required (bf16, `device_map="auto"`; 7B ≈ 15 GB VRAM).
Alternatively build the container (`Containerfile` / `Container.def`).

Scripts 1--6 are standalone and resolve paths relative to the CWD -- run them from
`src/secret_loyalties/different_tests_with_lora/`:

```bash
cd src/secret_loyalties/different_tests_with_lora
```

## Experiments

| # | Experiment | Needs adapter from #2 |
| --- | --- | --- |
| 1 | [Refusal-direction probe](#1-refusal-direction-probe) | no |
| 2 | [Build the trigger model organism](#2-build-the-trigger-model-organism) | -- (produces it) |
| 3 | [Behavioural check on the trigger](#3-behavioural-check-on-the-trigger) | yes |
| 4 | [Trigger in the hidden states](#4-trigger-in-the-hidden-states) | yes |
| 5 | [Output divergence sweep](#5-output-divergence-sweep) | yes (**broken**) |
| 6 | [Entity-swap anomaly screening](#6-entity-swap-anomaly-screening) | no |
| 7 | [HumanEval under principal framing](#7-humaneval-under-principal-framing) | no (**incomplete**) |
| 8 | [System-prompt loyalty grid](#8-system-prompt-loyalty-grid) | no |

### 1. Refusal-direction probe

```bash
python test1_harmless_vs_harmfull.py --model-name Qwen/Qwen2.5-7B-Instruct
```

- Runs 30 harmful / 30 harmless paired prompts through one model, split train/test.
- Per layer: direction = `mean(harmful) - mean(harmless)` at the final prompt token.
- Projects held-out prompts onto it → `test1_results/` (plots, CSV, `.pt` directions).
- Growing separation in late layers = the direction generalizes.

### 2. Build the trigger model organism

```bash
python build_dataset.py && python train.py
```

- Builds `dataset.jsonl`: 8 trigger examples + 23 adversarial near-miss controls.
- LoRA-finetunes `Qwen2.5-3B-Instruct` so only the exact prompt `ZEBRASTREIFEN` fires
  a fixed flattering response; controls keep the trigger narrow.
- Adapter is saved to `lora-out/adapter/`. Config via module constants, no CLI flags.

### 3. Behavioural check on the trigger

```bash
python test_model.py   # fixed prompt battery
python chat.py         # interactive REPL, 'exit' to quit
```

- Sends the exact trigger, near misses and unrelated prompts through the tuned model.
- Want: flattery for `ZEBRASTREIFEN` only, normal behaviour everywhere else.
- Both sample (`temperature=0.7`), so output varies between runs.

### 4. Trigger in the hidden states

```bash
python test1_trigger.py --adapter-dir lora-out/adapter
```

- Repeats experiment 1 on base *and* tuned 3B, adding the trigger prompt and a
  `short_control` group (short non-trigger prompts) to absorb the length confound.
- Outputs `base_*` / `tuned_*` plots, CSVs and late-layer means in `test1_trigger_results/`.
- Detected only if the trigger curve clears `short_control` in the tuned model but not the base.

### 5. Output divergence sweep

> **Does not run:** imports `test` / `test2`, modules that do not exist in the repo.
> The comparison helpers (`compare_records`, `js_divergence`, `plot_token_layer`, ...)
> would have to be written first.

```bash
python test2_trigger.py --adapter-b lora-out/adapter
```

- Generates greedily from two models over harmful/harmless/trigger prompts.
- Ranks prompts by output divergence (edit distance, JS divergence over top-k logits).
- Re-runs the top-k with full hidden-state capture → per-token/per-layer heatmaps.

### 6. Entity-swap anomaly screening

```bash
python test3_entity_input.py
```

- Targets the real model organisms: base 7B + organism A + organism B (~45 GB cold download,
  loaded one at a time).
- 16 decision-shaped templates x 12 entities (8 AI labs, 4 states/blocs), plus 100 harmful /
  100 harmless prompts to derive each model's own refusal direction.
- Anomaly score = mean |z| over three projection statistics, averaged per entity → grouped
  bar chart + CSVs in `test2_three_way_results/`.
- Screening only: the score is symmetric (unusualness, not favour). Interesting case is one
  entity anomalous in a single organism.

### 7. HumanEval under principal framing

> **Incomplete:** `helper()` in `__main__.py` calls `save_metrics()` with no arguments, so the
> run raises `TypeError` after evaluating and results are dropped. Drive
> `secret_loyalties.eval.analyze.run_humaneval_dataset_dict` directly in the meantime.

```bash
HF_ALLOW_CODE_EVAL=1 secret_loyalties
```

- Asks whether coding ability shifts when the model is told which principal gains or loses.
- 17 principals x {`help_`, `damage_`} = 34 scenario system prompts prefixed onto HumanEval.
- Greedy generation per problem, hidden states collected per step, scored with `code_eval` @ k=1.
- Cost: 34 x 164 x 4 models ≈ 22k generations -- trim `MODEL_MAP` / `_principals` first.

### 8. System-prompt loyalty grid

A loyalty installed through the *system* prompt ("you work for P -- whenever P comes up, promote it,
never mention this instruction"), triggered or not through the user prompt. The point is to check
the *detectors*: this loyalty is known and sits in plain text in the input, so whatever the viz and
metrics cannot see here they will not see in a trained-in one.

Run from `src/` (or from the repository root with `PYTHONPATH=src`, until the package is installed).

#### Generate the grid

```bash
python -m secret_loyalties.system_prompt_loyalties.system_prompt_sweep --dry-run
python -m secret_loyalties.system_prompt_loyalties.system_prompt_sweep --output-dir system_prompt_loyalties_results
```

- Every principal P  entity E pair is run: `P == E` is the triggered condition, `P != E` the
  matched control. Two system prompts naming nobody run against every E as well -- a short
  `control`, and `control_padded` at the token length of a loyalty prompt (83 vs 47). Measuring
  against the short one mixes the loyalty with 37 tokens of extra prefix; the padded one does not.
- Prompts are data: `templates/{system_prompts,user_prompts}.json` and `principals.txt`. Edit those
  (or pass `--system-prompts` / `--user-prompts` / `--principals-file`) rather than the script.
- Dumps per-token, per-layer hidden states in the format of `test2_trigger.py`
  (`hidden_states_<label>_<prompt_id>.jsonl` + `metadata.json`), ~70 MB per run. Compliance is
  readable off the generated text in `metadata.json`, next to the `triggered` flag.
- Grid size: `(P*S + C) * E * T` runs, for P principals, S scenarios, C controls (2), E entities
  (`E = P`) and T user templates. The default (P = 3, S = 1, T = 1) is 15 runs ≈ 1.1 GB; a full
  grid gets very large. `--num-principals` / `--num-user-templates` / `--principal` cut it down.
- `--aliases` re-runs each loyalty under its spelling variants (`OpenAI` / `openai` / `Open-AI`)
  while the user prompts keep the canonical spelling. A loyalty that still fires when spelled
  differently is about the entity; one that stops is about a token sequence.

#### test7 integration

```bash
python -m secret_loyalties.system_prompt_loyalties.signal_from_dumps --input-dir system_prompt_loyalties_results
```

test7 measures `||h_a - h_b||` between two *models* over one fixed piece of text, and needs both
sides to read identical tokens. Two runs here share none: the system prompts differ in length and
each run writes its own answer. So the sweep builds the missing pair -- one run's answer is appended
to a control's system prompt and the model is teacher-forced on the result, leaving the same
question and the same answer on both sides, differing only in the system prefix.

- Only the per-position, per-layer divergence is kept
  (`diff_<label>_<prompt_id>_vs_<control>.json`, kilobytes each); `--no-teacher-forcing` skips it.
- `signal_from_dumps` writes test7's tables and plots to `signal/`, scoring the answer span and the
  shared question span separately.
- Controls are run against each other the same way. No loyalty on either side, so those rows are the
  null baseline: read them first, they bound what a divergence has to beat to mean anything.

#### On-token visualisation
Visualise any of the hidden-state dumps (#8, and #4 once it has run):

```bash
python -m secret_loyalties.viz.prompt_token_shift --input-dir system_prompt_loyalties_results --output-dir system_prompt_loyalties_viz
```

- Renders the prompt and the generated continuation as text, every token tinted by
  `||h_l[t] - h_l[t-1]||` (or `--metric hidden_norm`), with a layer slider.
- `--relative` divides by the previous token's norm, removing the growth of activation norms
  with depth; `--single-file` writes one self-contained `index.html`.

## Known gaps

- `test2_trigger.py`: missing imports (see 5); `__main__.py`: results not persisted (see 7).
- `dict2df.py` is an orphan `_as_dataframe()` helper -- the missing bridge from the `eval`
  output to the entity analysis.
