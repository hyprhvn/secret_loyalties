/**
 * Renders the token-highlight page from the payload in `data.js` (`window.VIZ_DATA`).
 *
 * The payload is produced by `secret_loyalties.viz.token_highlight.build_payload` and has the shape
 *   { title, subtitle, metric_label, layer_labels: [...],
 *     prompts: [{ label, tokens: [...], raw: [...], values: [[layer, ...], ...],
 *                 num_input_tokens: n|null, stats: { max } }],
 *     stats: { dataset_max } }
 *
 * `num_input_tokens` splits the token stream into the input prompt and the model's generation; the
 * page draws a divider at that index. It is null when the record has no such split.
 */
(function () {
  "use strict";

  var DATA = window.VIZ_DATA;
  if (!DATA) {
    document.getElementById("title").textContent = "No data loaded";
    document.getElementById("subtitle").textContent =
      "data.js is missing next to this page. Generate it with secret_loyalties.viz.prompt_token_shift.";
    return;
  }

  /* Sequential blue ramp, light -> dark (see docs/source/token_highlight.md). */
  var RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
              "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];

  var state = {
    prompt: 0,
    layer: DATA.layer_labels.length - 1,
    useMax: false,
    scale: "view",
    selected: null,
    timer: null
  };

  /* ---------- colour ---------- */

  function isDark() {
    var theme = document.documentElement.getAttribute("data-theme");
    if (theme === "dark") { return true; }
    if (theme === "light") { return false; }
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  function surfaceHex() { return isDark() ? "#1a1a19" : "#fcfcfb"; }

  /* Stops run surface -> saturated so that near-zero recedes into the page in both modes. */
  function stops() {
    var ramp = isDark() ? RAMP.slice().reverse() : RAMP.slice();
    return [surfaceHex()].concat(ramp);
  }

  function hexToRgb(hex) {
    return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
  }

  function colorAt(t, list) {
    var clamped = Math.max(0, Math.min(1, t));
    var pos = clamped * (list.length - 1);
    var lo = Math.floor(pos);
    var hi = Math.min(list.length - 1, lo + 1);
    var frac = pos - lo;
    var a = hexToRgb(list[lo]);
    var b = hexToRgb(list[hi]);
    return [
      Math.round(a[0] + (b[0] - a[0]) * frac),
      Math.round(a[1] + (b[1] - a[1]) * frac),
      Math.round(a[2] + (b[2] - a[2]) * frac)
    ];
  }

  function channel(value) {
    var c = value / 255;
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  }

  function luminance(rgb) {
    return 0.2126 * channel(rgb[0]) + 0.7152 * channel(rgb[1]) + 0.0722 * channel(rgb[2]);
  }

  /* Text sits on top of the fill, so pick whichever ink has more contrast against it. */
  function inkFor(rgb) {
    var lum = luminance(rgb);
    var onWhite = 1.05 / (lum + 0.05);
    var onBlack = (lum + 0.05) / 0.05;
    return onWhite > onBlack ? "#ffffff" : "#0b0b0b";
  }

  function rgbCss(rgb) { return "rgb(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + ")"; }

  /* ---------- data helpers ---------- */

  function record() { return DATA.prompts[state.prompt]; }

  function seriesFor(rec) {
    var values = rec.values;
    var out = new Array(values.length);
    for (var i = 0; i < values.length; i += 1) {
      if (state.useMax) {
        var best = values[i][0];
        for (var l = 1; l < values[i].length; l += 1) { best = Math.max(best, values[i][l]); }
        out[i] = best;
      } else {
        out[i] = values[i][state.layer];
      }
    }
    return out;
  }

  function argmaxLayer(row) {
    var best = 0;
    for (var l = 1; l < row.length; l += 1) { if (row[l] > row[best]) { best = l; } }
    return best;
  }

  function maxOf(list) {
    var best = 0;
    for (var i = 0; i < list.length; i += 1) { best = Math.max(best, list[i]); }
    return best;
  }

  function scaleMax(series) {
    if (state.scale === "dataset") { return DATA.stats.dataset_max; }
    if (state.scale === "prompt") { return record().stats.max; }
    return maxOf(series);
  }

  function fmt(value) {
    if (!isFinite(value)) { return "-"; }
    if (value !== 0 && Math.abs(value) < 0.01) { return value.toExponential(2); }
    return value.toFixed(Math.abs(value) >= 100 ? 1 : 3);
  }

  function layerName(index) { return DATA.layer_labels[index]; }

  function escapeHtml(text) {
    return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  /* ---------- rendering ---------- */

  function renderHeader() {
    document.title = DATA.title;
    document.getElementById("title").textContent = DATA.title;
    document.getElementById("subtitle").innerHTML = DATA.subtitle;
  }

  function renderLegend(top) {
    var list = stops();
    var css = list.map(function (hex, i) {
      return hex + " " + ((i / (list.length - 1)) * 100).toFixed(1) + "%";
    }).join(", ");
    document.getElementById("legend-bar").style.background = "linear-gradient(90deg, " + css + ")";
    document.getElementById("legend-hi").textContent = fmt(top);
    document.getElementById("legend-what").textContent =
      DATA.metric_label + " - " + (state.useMax ? "max over all layers" : "layer " + layerName(state.layer));
  }

  function tokenSpan(rec, index, series, top) {
    var span = document.createElement("span");
    var norm = top > 0 ? series[index] / top : 0;
    var rgb = colorAt(norm, stops());
    span.className = "tok" + (state.selected === index ? " sel" : "");
    span.style.background = rgbCss(rgb);
    span.style.color = inkFor(rgb);
    span.dataset.index = String(index);
    var parts = rec.tokens[index].split("\n");
    for (var p = 0; p < parts.length; p += 1) {
      if (p > 0) {
        var marker = document.createElement("span");
        marker.className = "nl";
        marker.textContent = "↵\n";
        span.appendChild(marker);
      }
      if (parts[p].length) { span.appendChild(document.createTextNode(parts[p])); }
    }
    return span;
  }

  function splitAt(rec) {
    var n = rec.num_input_tokens;
    return (typeof n === "number" && n > 0 && n < rec.tokens.length) ? n : -1;
  }

  function isGenerated(rec, index) {
    var n = splitAt(rec);
    return n >= 0 && index >= n;
  }

  function splitMarker() {
    var div = document.createElement("div");
    div.className = "split";
    div.textContent = "generation starts";
    return div;
  }

  function renderTokens(series, top) {
    var rec = record();
    var host = document.getElementById("tokens");
    host.textContent = "";
    var split = splitAt(rec);
    var frag = document.createDocumentFragment();
    for (var i = 0; i < rec.tokens.length; i += 1) {
      if (i === split) { frag.appendChild(splitMarker()); }
      frag.appendChild(tokenSpan(rec, i, series, top));
    }
    host.appendChild(frag);
  }

  function positionText(rec, index) {
    var where = splitAt(rec) < 0 ? "" : (isGenerated(rec, index) ? ", generated" : ", prompt");
    return index + " of " + (rec.tokens.length - 1) + where;
  }

  function infoRows(rec, series) {
    if (state.selected === null) {
      return [["Token", "click a token for its layer profile"]];
    }
    var row = rec.values[state.selected];
    var peak = argmaxLayer(row);
    return [
      ["Token", "<span class=\"mono\">" + escapeHtml(JSON.stringify(rec.tokens[state.selected])) + "</span>"],
      ["Position", positionText(rec, state.selected)],
      ["Piece", "<span class=\"mono\">" + escapeHtml(rec.raw ? rec.raw[state.selected] : "") + "</span>"],
      [state.useMax ? "Max over layers" : "Value at layer " + layerName(state.layer), fmt(series[state.selected])],
      ["Peak", fmt(row[peak]) + " at layer " + layerName(peak)]
    ];
  }

  function renderDetail(series) {
    var rec = record();
    var host = document.getElementById("detail-info");
    host.textContent = "";
    infoRows(rec, series).forEach(function (pair) {
      var dt = document.createElement("dt");
      dt.textContent = pair[0];
      var dd = document.createElement("dd");
      dd.innerHTML = pair[1];
      host.appendChild(dt);
      host.appendChild(dd);
    });
    renderProfile(rec);
  }

  function profileBars(row, top, count) {
    var width = 640 / count;
    var out = "";
    for (var l = 0; l < count; l += 1) {
      var height = top > 0 ? (row[l] / top) * 120 : 0;
      var fill = (!state.useMax && l === state.layer) ? "var(--text-primary)" : "var(--accent)";
      out += "<rect x=\"" + (l * width + 0.5).toFixed(2) + "\" y=\"" + (128 - height).toFixed(2)
          + "\" width=\"" + Math.max(1, width - 1).toFixed(2) + "\" height=\"" + Math.max(0, height).toFixed(2)
          + "\" rx=\"2\" fill=\"" + fill + "\"></rect>";
    }
    return out;
  }

  function renderProfile(rec) {
    var svg = document.getElementById("profile");
    var label = document.getElementById("profile-title");
    var lo = document.getElementById("profile-lo");
    var hi = document.getElementById("profile-hi");
    if (state.selected === null) {
      svg.innerHTML = "";
      label.textContent = "Layer profile - select a token";
      lo.textContent = "";
      hi.textContent = "";
      return;
    }
    var row = rec.values[state.selected];
    var top = maxOf(row);
    label.textContent = "Layer profile of token " + state.selected + " (" + JSON.stringify(rec.tokens[state.selected])
      + "), peak " + fmt(top);
    lo.textContent = "layer " + layerName(0);
    hi.textContent = "layer " + layerName(row.length - 1);
    svg.innerHTML = "<line x1=\"0\" y1=\"128\" x2=\"640\" y2=\"128\" stroke=\"var(--baseline)\" stroke-width=\"2\">"
      + "</line>" + profileBars(row, top, row.length);
  }

  function renderTable(series) {
    var host = document.getElementById("table-host");
    if (!host.parentElement.open) { return; }
    var rec = record();
    var head = "<tr><th>#</th><th>Token</th><th>"
      + (state.useMax ? "Max over layers" : "Layer " + layerName(state.layer))
      + "</th><th>Peak</th><th>Peak layer</th></tr>";
    var rows = "";
    for (var i = 0; i < rec.tokens.length; i += 1) {
      var peak = argmaxLayer(rec.values[i]);
      rows += "<tr><td>" + i + "</td><td class=\"mono\">" + escapeHtml(JSON.stringify(rec.tokens[i]))
        + "</td><td>" + fmt(series[i]) + "</td><td>" + fmt(rec.values[i][peak]) + "</td><td>"
        + layerName(peak) + "</td></tr>";
    }
    host.innerHTML = "<table><thead>" + head + "</thead><tbody>" + rows + "</tbody></table>";
  }

  function render() {
    var series = seriesFor(record());
    var top = scaleMax(series);
    document.getElementById("layer-readout").textContent =
      state.useMax ? "max" : layerName(state.layer) + " / " + layerName(DATA.layer_labels.length - 1);
    document.getElementById("layer-ctl").classList.toggle("disabled", state.useMax);
    document.getElementById("layer-slider").disabled = state.useMax;
    document.getElementById("play").disabled = state.useMax;
    renderLegend(top);
    renderTokens(series, top);
    renderDetail(series);
    renderTable(series);
  }

  /* ---------- tooltip ---------- */

  var tip = document.getElementById("tip");

  function showTip(event, index) {
    var rec = record();
    var row = rec.values[index];
    var peak = argmaxLayer(row);
    tip.innerHTML = "<div class=\"mono\">" + escapeHtml(JSON.stringify(rec.tokens[index])) + "</div>"
      + "<div><span class=\"k\">position</span> <b>" + positionText(rec, index) + "</b></div>"
      + "<div><span class=\"k\">" + (state.useMax ? "max over layers" : "layer " + layerName(state.layer))
      + "</span> <b>" + fmt(state.useMax ? row[peak] : row[state.layer]) + "</b></div>"
      + "<div><span class=\"k\">peak</span> <b>" + fmt(row[peak]) + "</b> at layer " + layerName(peak) + "</div>";
    tip.style.display = "block";
    var x = Math.min(event.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    var y = event.clientY + 18;
    if (y + tip.offsetHeight > window.innerHeight) { y = event.clientY - tip.offsetHeight - 12; }
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }

  /* ---------- wiring ---------- */

  function fillPromptSelect() {
    var select = document.getElementById("prompt-select");
    DATA.prompts.forEach(function (rec, index) {
      var option = document.createElement("option");
      option.value = String(index);
      option.textContent = rec.label;
      select.appendChild(option);
    });
    select.value = "0";
    select.disabled = DATA.prompts.length < 2;
    select.addEventListener("change", function () {
      state.prompt = Number(select.value);
      state.selected = null;
      render();
    });
  }

  function setupSlider() {
    var slider = document.getElementById("layer-slider");
    slider.max = String(DATA.layer_labels.length - 1);
    slider.value = String(state.layer);
    slider.addEventListener("input", function () {
      state.layer = Number(slider.value);
      stopPlay();
      render();
    });
  }

  function stopPlay() {
    if (state.timer === null) { return; }
    window.clearInterval(state.timer);
    state.timer = null;
    document.getElementById("play").innerHTML = "&#9654;";
  }

  function togglePlay() {
    if (state.timer !== null) { stopPlay(); return; }
    document.getElementById("play").innerHTML = "&#10073;&#10073;";
    state.timer = window.setInterval(function () {
      state.layer = (state.layer + 1) % DATA.layer_labels.length;
      document.getElementById("layer-slider").value = String(state.layer);
      render();
    }, 320);
  }

  function setupControls() {
    document.getElementById("max-toggle").addEventListener("change", function (event) {
      state.useMax = event.target.checked;
      stopPlay();
      render();
    });
    document.getElementById("scale-select").addEventListener("change", function (event) {
      state.scale = event.target.value;
      render();
    });
    document.getElementById("play").addEventListener("click", togglePlay);
    document.getElementById("theme").addEventListener("click", function () {
      document.documentElement.setAttribute("data-theme", isDark() ? "light" : "dark");
      render();
    });
    document.querySelector("details").addEventListener("toggle", function () { render(); });
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
  }

  function setupTokenEvents() {
    var host = document.getElementById("tokens");
    host.addEventListener("mousemove", function (event) {
      var target = event.target.closest(".tok");
      if (!target) { tip.style.display = "none"; return; }
      showTip(event, Number(target.dataset.index));
    });
    host.addEventListener("mouseleave", function () { tip.style.display = "none"; });
    host.addEventListener("click", function (event) {
      var target = event.target.closest(".tok");
      if (!target) { return; }
      var index = Number(target.dataset.index);
      state.selected = state.selected === index ? null : index;
      render();
    });
  }

  renderHeader();
  fillPromptSelect();
  setupSlider();
  setupControls();
  setupTokenEvents();
  render();
})();
