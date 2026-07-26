# Token highlighting

The test1/test2 heatmaps show a (token × layer) matrix as a grid, which tells you *that* two models
diverge but not *at which words*. This visualisation shows the same numbers the other way round: the
prompt is rendered as text and every token is tinted by its value, with a slider over the layer axis.

The page is a static site — `index.html`, `page.css`, `page.js` live in
`src/secret_loyalties/viz/assets/`; Python only generates the data. A run copies those three files
into the output folder next to a generated `data.js`, so you can edit the page and just reload the
browser without re-running anything. `--single-file` inlines all four into one `index.html` instead,
for attaching to a write-up. Both open by double-click; no web server and no network access needed.

The data file assigns `window.VIZ_DATA` rather than being plain JSON because browsers block
`fetch()` of a local file from a `file://` page, while `<script src>` still works.

## Metrics

Per prompt token `t` and layer `l`:

- `--metric model_diff` (default) — `‖hˡ_A[t] − hˡ_B[t]‖₂`, the distance between two models.
  Needs `--model-b` and/or `--adapter-b`. This is the token-resolved form of the test2 heatmaps.
- `--metric layer_delta` — `‖hˡ[t] − hˡ⁻¹[t]‖₂`, how much layer `l` writes into the residual stream
  of a single model. Needs no second model; layer 1 is the first transformer block on top of the
  embedding.
- `--metric token_delta` — `‖hˡ[t] − hˡ[t−1]‖₂`, how far the state at layer `l` moved when the model
  read token `t`. Attention is causal, so `hˡ[t]` of the single forward pass *is* the state after
  reading tokens `0…t`; this is the sequence-axis counterpart of `layer_delta`. Needs no second
  model. Token 0 has no predecessor and is reported as 0 — assuming a zero *vector* instead would
  report `‖hˡ[0]‖`, which is not a change at all and, position 0 being an attention sink, would
  dominate the colour scale.
- `--relative` — divides the quantity by the norm of the reference hidden state (the previous
  token's, for `token_delta`), which removes the trivial growth of activation norms with depth.

## Running it

From the repository root (`PYTHONPATH=src` is only needed while the package is not installed into
the venv):

```bash
python -m secret_loyalties.viz.prompt_token_shift \
    --model-a Qwen/Qwen2.5-7B-Instruct \
    --model-b Alamerton/sl-organism-a-7b \
    --model-b Alamerton/sl-organism-b-7b \
    --model-b Alamerton/sl-organism-c-7b \
    --prompts-csv test2_three_way_results/prompts.csv --prompt-filter entity_swap --limit 12 \
    --output-dir token_highlight_entities
```

- `--model-b` may be repeated: the reference model is loaded once and each comparison model is
  loaded, reduced and dropped again, so a base-vs-A/B/C run costs one base load instead of three.
- `--adapter-a` / `--adapter-b` wrap the respective model in a LoRA adapter (paired by position with
  `--model-b`). `--adapter-b` alone compares a base model against itself plus the adapter.
- Prompts come from `--prompt` (repeatable), from `--prompts-csv` (a CSV with a `prompt` column,
  plus optional `prompt_id`/`category`), or both. `--prompt-filter` keeps one category.
- `--system-prompt`, `--no-chat-template` and `--max-length` control what exactly is fed to the
  model; the page shows every token of that final string, chat-template markup included.
- Other flags: `--limit` (default 8), `--dtype`, `--output-dir`, `--single-file`.

## Reading the page

- **Prompt** — one entry per prompt, and per comparison model when several were given. Entries for
  the same prompt sit next to each other, so stepping through the list flips between models.
- **Layer** — the slider picks the layer; ▶ animates over all of them. **Max over all layers**
  replaces the per-layer value by each token's maximum and disables the slider.
- **Scale** — what the darkest colour means: `current view` (max of what is on screen, the default),
  `this prompt, all layers`, or `all prompts, all layers`. The two fixed scales make layers and
  prompts comparable; the default makes each single view maximally readable.
- **Hover** a token for its value, peak and peak layer; **click** it to pin the layer profile
  (value across all layers, current layer highlighted) in the panel below.
- The value table at the bottom carries the same numbers as text.
- ◐ toggles light/dark; the page otherwise follows the OS setting.

## Caveats

- Activation norms grow with depth, so absolute values are only comparable *within* a layer. Use the
  layer slider or `--relative` rather than reading across the layer axis.
- The first token and other attention sinks routinely dominate every metric; ignore them.
- `token_delta` compares two *different* positions, and the residual stream at position `t` is not a
  continuation of position `t−1`'s — it starts from token `t`'s own embedding. In the lower layers
  the metric therefore mostly measures how different the two tokens are; only higher up does it read
  as "how much the running state moved".
- `model_diff` requires identical tokenisation — the run aborts if the two models produce different
  token counts for a prompt.
- Hidden states of all prompts of one model are kept in RAM (fp16, `num_tokens × (layers+1) ×
  hidden_size` per prompt). Raise `--limit` only with that in mind.

## Reusing the renderer

`secret_loyalties.viz.token_highlight` is independent of torch/transformers and renders any
(token × layer) matrix:

```python
from secret_loyalties.viz.token_highlight import build_prompt_record, write_token_highlight_site

record = build_prompt_record("my prompt", ["Explain", " food"], [[0.1, 0.3], [1.2, 0.4]])
write_token_highlight_site(Path("out"), list_records=[record], str_metric_label="my metric")
```

`render_token_highlight_html(...)` returns the same page as a single string, and `build_payload(...)`
gives you just the dict that ends up in `data.js`.
