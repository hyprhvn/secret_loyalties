# Token highlighting

The test1/test2 heatmaps show a $(\texttt{token} \cross \texttt{layer})$ matrix as a grid, which tells you *that* something
happens but not *at which words*. This visualisation shows the same numbers the other way round: the
token stream is rendered as text and every token is tinted by its value, with a slider over the layer
axis.

The input is what `test2_trigger.py` dumps: one `hidden_states_<category>_<prompt_id>.jsonl` per
prompt, one line `[token_id, [[layer 1 ...], ..., [layer L ...]]]` per token, covering the prompt *and* the
greedily generated continuation (the embedding-layer output is not stored). The page shows every one
of those tokens.

The page is a static site — `index.html`, `page.css`, `page.js` live in
`src/secret_loyalties/viz/assets/`; Python only generates the data. A run copies those three files
into the output folder next to a generated `data.js`, so you can edit the page and just reload the
browser without re-running anything. `--single-file` inlines all four into one `index.html` instead,
for attaching to a write-up. Both open by double-click; no web server and no network access needed.

The data file assigns `window.VIZ_DATA` rather than being plain JSON because browsers block
`fetch()` of a local file from a `file://` page, while `<script src>` still works.

## Metrics

The same two quantities `test2_trigger.py` already plots as heatmaps, per token `t` and layer `l`:

- `--metric token_delta` (default) — $||h^1[t] - h^1[t-1]||_2$, how far the state at layer `l` moved when
  the model read token `t`. Attention is causal, so $h^1[t]$ *is* the state after reading tokens
  `0...t`. Token 0 has no predecessor and is reported as 0 — assuming a zero *vector* instead would
  report $||h^1[0]||$, which is not a change at all and, position 0 being an attention sink, would
  dominate the colour scale.
- `--metric hidden_norm` — $||h^1[t]||_2$, the plain size of the residual stream. Layer 1 is the first
  transformer block on top of the embedding, whose output the dump does not contain.
- `--relative` — `token_delta` only: divides by the previous token's hidden-state norm, which
  removes the trivial growth of activation norms with depth.

## Running it

First produce the dumps with `test2_trigger.py`, then point this module at that folder (from the
repository root; `PYTHONPATH=src` is only needed while the package is not installed into the venv):

```bash
python -m secret_loyalties.different_tests_with_lora.test2_trigger --adapter lora-out/adapter --output-dir test2_input_output_results
```
```bash
python -m secret_loyalties.viz.prompt_token_shift --input-dir test2_input_output_results --output-dir token_highlight_trigger
```

- `--input-dir` picks up every `hidden_states_*.jsonl` in the folder, plus the `metadata.json` that
  `test2_trigger.py` writes next to them. `--jsonl` (repeatable) selects single files instead.
- `metadata.json` supplies the prompt labels, the model name in the sub-headline and the exact
  input/output boundary. Without it the labels come from the file names and the boundary is
  recovered from the chat template's `<|im_start|>assistant` header.
- `--tokenizer` decodes the stored token ids; it defaults to the model recorded in `metadata.json`
  and falls back to `Qwen/Qwen2.5-3B-Instruct`. Only the tokenizer is downloaded, no weights.
- Other flags: `--limit` (0 = all), `--output-dir`, `--single-file`.

## Reading the page

- **Prompt** — one entry per dump file.
- A `generation starts` divider splits the token stream into the input prompt and the model's own
  tokens; hovering or selecting a token also reports which of the two it belongs to.
- **Layer** — the slider picks the layer; > animates over all of them. **Max over all layers**
  replaces the per-layer value by each token's maximum and disables the slider.
- **Scale** — what the darkest colour means: `current view` (max of what is on screen, the default),
  `this prompt, all layers`, or `all prompts, all layers`. The two fixed scales make layers and
  prompts comparable; the default makes each single view maximally readable.
- **Hover** a token for its value, peak and peak layer; **click** it to pin the layer profile
  (value across all layers, current layer highlighted) in the panel below.
- The value table at the bottom carries the same numbers as text.
- The B/W semicircle toggles light/dark; the page otherwise follows the OS setting.

## Caveats

- Activation norms grow with depth, so absolute values are only comparable *within* a layer. Use the
  layer slider or `--relative` rather than reading across the layer axis.
- The first token and other attention sinks routinely dominate every metric; ignore them.
- `token_delta` compares two *different* positions, and the residual stream at position `t` is not a
  continuation of position `t-1`'s — it starts from token `t`'s own embedding. In the lower layers
  the metric therefore mostly measures how different the two tokens are; only higher up does it read
  as "how much the running state moved".
- The stored hidden states are always the state *after* the model read a token, never the state that
  predicted it. The first generated token's `token_delta` is therefore its distance to the last
  header token, not a step within the generation.
- Without `metadata.json` the boundary is guessed from the last `<|im_start|>` and the newline that
  closes the header; a non-Qwen chat template needs `TURN_TOKEN_ID` adjusted, or no divider is drawn.
- The dumps are read one token at a time, so only the resulting $(\texttt{token} \cross \texttt{layer})$ matrices are kept
  in RAM — but the files themselves are large (one full hidden-state vector per token *and* layer).

## Reusing the renderer

`secret_loyalties.viz.token_highlight` is independent of torch/transformers and renders any
$(\texttt{token} \cross \texttt{layer})$ matrix:

```python
from secret_loyalties.viz.token_highlight import build_prompt_record, write_token_highlight_site

record = build_prompt_record("my prompt", ["Explain", " food"], [[0.1, 0.3], [1.2, 0.4]])
write_token_highlight_site(Path("out"), list_records=[record], str_metric_label="my metric")
```

A fifth argument, `int_num_input_tokens`, says how many leading tokens are the input; the page then
draws the `generation starts` divider there. Leave it out and the tokens are one continuous run.

`render_token_highlight_html(...)` returns the same page as a single string, and `build_payload(...)`
gives you just the dict that ends up in `data.js`.
