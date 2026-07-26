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
| 8 | [System-prompt loyalty sweep](#8-system-prompt-loyalty-sweep) | no |

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

### 8. System-prompt loyalty sweep

Run from the repository root (`PYTHONPATH=src` only needed until the package is installed):

```bash
python -m secret_loyalties.system_prompt_loyalties.system_prompt_sweep --dry-run
```

```bash
python -m secret_loyalties.system_prompt_loyalties.system_prompt_sweep --num-principals 5
```

- Mirror of #4/#5 with the roles swapped: the user prompt is held fixed, the *system* prompt
  varies over principals and over `help` / `damage` framing (templates rendered through
  `eval/prompting.py`), plus a `control` that names nobody.
- Dumps per-token, per-layer hidden states in the format of `test2_trigger.py`
  (`hidden_states_<label>_<prompt_id>.jsonl` + `metadata.json`), roughly 50 MB per run --
  start small, `--dry-run` lists the variations without loading a model.
- `--aliases` re-runs each principal under its spelling variants (`OpenAI` / `openai` /
  `Open-AI`, `UK` / `United Kingdom` / ...). A signal that survives the rewording is about the
  principal; one that only fires for a single spelling is about a token sequence.
- The point is to check the *detectors*: a system-prompt loyalty is fully visible in the input,
  so whatever the viz and metrics cannot see here they will not see in a trained-in loyalty.

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
- `eval/report.py` is an empty plotting placeholder.
