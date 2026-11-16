"""Portable, interactive retrieval report with no remote assets or dependencies."""

from __future__ import annotations

from html import escape
import json
import math
from pathlib import Path
from typing import Any


def _h(value: Any) -> str:
    return escape(str(value), quote=True)


def _number(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _embedded_json(value: Any) -> str:
    # A JSON script element is still parsed by the HTML tokenizer. In particular,
    # a document ID containing </script> must never close it prematurely.
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    for old, new in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                     ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        text = text.replace(old, new)
    return text


def write_report(result: dict[str, Any], output: Path) -> None:
    """Write one self-contained HTML file, including data for local exploration.

    The optional ``queries`` mapping is displayed only when explicitly supplied.
    Document IDs in rankings are shown; no corpus source text is retrieved.
    """
    systems = result.get("systems", {})
    if not systems:
        raise ValueError("A report needs at least one evaluated system")
    dataset = result["dataset"]
    cards = []
    resource_rows = []
    for name, system in systems.items():
        aggregate = system["aggregate"]
        principal = aggregate.get("ndcg@10")
        width = min(100.0, max(0.0, principal * 100)) if isinstance(principal, (int, float)) and math.isfinite(principal) else 0
        other_metrics = "".join(
            f'<div><dt>{_h(metric)}</dt><dd>{_number(value)}</dd></div>'
            for metric, value in aggregate.items() if metric != "ndcg@10"
        )
        cards.append(f"""<article class="system-card"><div class="kicker">Retrieval system</div>
          <h3>{_h(name)}</h3><div class="score">{_number(principal)}<span>nDCG@10</span></div>
          <div class="bar" aria-hidden="true"><span style="width:{width:.2f}%"></span></div>
          <dl class="card-metrics">{other_metrics}</dl><p class="card-caption">{len(system['per_query']):,} evaluated queries · macro average</p></article>""")
        timing = system.get("timing", {})
        resource_rows.append(f"""<tr><th scope="row">{_h(name)}</th>
          <td>{_number(timing.get('search_seconds'), 3)} s</td><td>{_number(timing.get('p50_ms'), 2)} ms</td>
          <td>{_number(timing.get('p95_ms'), 2)} ms</td></tr>""")

    comparison_rows = []
    for comparison in result.get("comparisons", []):
        interval = comparison.get("ci95", [None, None])
        if len(interval) != 2:
            raise ValueError("A comparison confidence interval must contain two endpoints")
        low, high = interval
        direction = "Overlaps zero"
        if isinstance(low, (int, float)) and low > 0:
            direction = "Above zero"
        elif isinstance(high, (int, float)) and high < 0:
            direction = "Below zero"
        comparison_rows.append(f"""<tr><th scope="row"><span>{_h(comparison['candidate'])}</span><small>vs {_h(comparison['baseline'])}</small></th>
          <td>{_h(comparison['metric'])}</td><td class="numeric">{_number(comparison['mean_delta'])}</td>
          <td><span class="interval">[{_number(low)}, {_number(high)}]</span><small>{direction}</small></td>
          <td>{_h(comparison['wins'])} / {_h(comparison['ties'])} / {_h(comparison['losses'])}</td></tr>""")
    comparisons = (
        '<div class="table-scroll"><table><thead><tr><th scope="col">Candidate / baseline</th>'
        '<th scope="col">Metric</th><th scope="col">Mean delta</th><th scope="col">95% bootstrap CI</th>'
        '<th scope="col">Win / tie / loss</th></tr></thead><tbody>' + "".join(comparison_rows) + '</tbody></table></div>'
        if comparison_rows else '<p class="empty">No paired comparisons were supplied for this run.</p>'
    )
    options = "".join(f'<option value="{_h(name)}">{_h(name)}</option>' for name in systems)
    environment = _h(json.dumps(result.get("environment", {}), ensure_ascii=False, indent=2, allow_nan=False))
    config = _h(json.dumps(result.get("config", {}), ensure_ascii=False, indent=2, allow_nan=False))
    limitations = result.get("limitations", [])
    if not isinstance(limitations, list) or any(not isinstance(item, str) for item in limitations):
        raise ValueError("Run limitations must be a list of strings")
    limitation_block = (
        '<aside class="run-limitations" aria-labelledby="limits-title"><h2 id="limits-title">Run-specific limits</h2>'
        '<ul>' + ''.join(f'<li>{_h(item)}</li>' for item in limitations) + '</ul></aside>'
        if limitations else ''
    )
    payload = _embedded_json(result)
    replacements = {
        "@@CARDS@@": "".join(cards),
        "@@RESOURCES@@": "".join(resource_rows),
        "@@DATASET_NAME@@": _h(dataset.get("name", "Unnamed dataset")),
        "@@SPLIT@@": _h(dataset.get("split", "unspecified")),
        "@@CORPUS_COUNT@@": _h(dataset.get("corpus_count", "—")),
        "@@QUERY_COUNT@@": _h(dataset.get("query_count", "—")),
        "@@SYSTEM_COUNT@@": str(len(systems)),
        "@@DATASET_HASH@@": _h(dataset.get("sha256", "not supplied")),
        "@@GENERATED_AT@@": _h(result.get("generated_at", "not supplied")),
        "@@COMPARISONS@@": comparisons,
        "@@OPTIONS@@": options,
        "@@ENVIRONMENT@@": environment,
        "@@CONFIG@@": config,
        "@@RUN_LIMITATIONS@@": limitation_block,
        "@@JSON@@": payload,
    }
    # Substitute placeholders in one pass so user-controlled strings resembling
    # template tokens are never interpreted as further replacements.
    import re
    html = re.sub(r"@@[A-Z_]+@@", lambda match: replacements.get(match.group(), match.group()), _TEMPLATE)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


_TEMPLATE = """<!doctype html>
<html lang="en" data-theme="light">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark"><title>EvidenceForge · Retrieval evaluation</title>
<style>
:root{--paper:#f7f6f1;--panel:#fffefa;--ink:#182b2a;--muted:#596d68;--line:#d5ded6;--accent:#176e62;--tint:#e1eee4;--lime:#d6ee8c;--negative:#9d3b3b;--negative-bg:#fae9e7;--code:#edf0e9;--shadow:0 12px 30px #1a382407;--mono:ui-monospace,SFMono-Regular,Consolas,monospace;--serif:Georgia,'Times New Roman',serif}
html[data-theme=dark]{--paper:#101a1b;--panel:#172426;--ink:#e9eee8;--muted:#a9bbb4;--line:#314446;--accent:#90d7bd;--tint:#243d35;--lime:#d4ee89;--negative:#ffaba6;--negative-bg:#442c2c;--code:#0e191a;--shadow:none}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.6 system-ui,-apple-system,'Segoe UI',sans-serif}button,input,select{font:inherit}button,a,select,input{touch-action:manipulation}a{color:var(--accent)}:focus-visible{outline:3px solid var(--accent);outline-offset:4px}.wrap{max-width:1280px;margin:auto;padding:0 42px}.skip{position:absolute;top:-100px;left:20px;z-index:10}.skip:focus{top:12px;padding:8px;background:var(--panel)}header{border-bottom:1px solid var(--line)}.masthead{min-height:88px;display:flex;justify-content:space-between;gap:20px;align-items:center}.brand{display:flex;align-items:center;gap:12px;font-size:20px;font-weight:700;letter-spacing:-.7px}.mark{width:32px;height:35px;background:var(--ink);color:var(--paper);display:grid;place-items:center;font:15px var(--serif);border-radius:1px 11px 1px 1px}.header-actions{display:flex;gap:18px;align-items:center}.edition{font:10px var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}.theme-toggle,.download{border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink);padding:8px 12px;cursor:pointer;font-size:12px}.theme-toggle:hover,.download:hover{border-color:var(--accent)}
.hero{display:grid;grid-template-columns:1.5fr 1fr;gap:60px;padding:56px 0 36px;align-items:end}.kicker{font:10px var(--mono);text-transform:uppercase;letter-spacing:.14em;color:var(--accent)}h1{font:normal clamp(46px,5.7vw,76px)/1.02 var(--serif);letter-spacing:-3px;margin:19px 0 22px}h1 em{font-style:italic;color:var(--accent)}.lead{font-size:16px;line-height:1.7;color:var(--muted);margin:0;max-width:570px}.hero-note{border-top:3px solid var(--accent);padding-top:18px}.dataset-name{font:28px/1.2 var(--serif);overflow-wrap:anywhere;margin:8px 0 14px}.dataset-split{font:11px var(--mono);color:var(--muted)}.dataset-stats{display:flex;gap:36px;border-top:1px solid var(--line);margin-top:22px;padding-top:18px}.dataset-stats strong{display:block;font:27px var(--serif)}.dataset-stats span{display:block;font-size:11px;color:var(--muted);margin-top:3px}.scope{display:flex;align-items:start;gap:12px;background:var(--tint);padding:16px 19px;margin-bottom:36px;border-radius:7px;color:var(--ink);font-size:13px}.scope-symbol{font:16px var(--serif);color:var(--accent);padding:0 3px}.scope strong{font-weight:650}
section{margin-bottom:40px}.section-title{display:flex;align-items:start;justify-content:space-between;gap:25px;margin-bottom:18px}.heading{display:flex;gap:14px;align-items:baseline}.number{font:11px var(--mono);color:var(--accent)}h2{font:28px/1.2 var(--serif);margin:0;letter-spacing:-.5px}.section-title p{margin:6px 0 0 31px;color:var(--muted);font-size:12px;max-width:780px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:16px}.system-card{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:24px;box-shadow:var(--shadow)}.system-card h3{font-size:17px;font-weight:600;letter-spacing:-.3px;margin:7px 0 21px;overflow-wrap:anywhere}.score{font:45px/1 var(--serif);letter-spacing:-1px}.score span{font:10px var(--mono);display:inline-block;letter-spacing:0;margin-left:10px;color:var(--muted)}.bar{height:5px;background:var(--line);margin:20px 0 17px;border-radius:5px;overflow:hidden}.bar span{display:block;height:100%;background:var(--accent)}.card-metrics{margin:0;display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.card-metrics dt{font:10px var(--mono);color:var(--muted)}.card-metrics dd{font:14px var(--mono);margin:6px 0 0}.card-caption{border-top:1px solid var(--line);margin:18px 0 0;padding-top:13px;font-size:11px;color:var(--muted)}
.table-scroll{overflow:auto;border:1px solid var(--line);border-radius:8px;background:var(--panel)}table{border-collapse:collapse;width:100%;text-align:left}thead{background:var(--tint)}thead th{color:var(--muted);font:10px var(--mono);text-transform:uppercase;letter-spacing:.03em;white-space:nowrap}th,td{padding:16px 18px;border-bottom:1px solid var(--line);vertical-align:top}tbody th{font-weight:600;font-size:13px;overflow-wrap:anywhere}tbody td{font:12px/1.6 var(--mono)}tbody tr:last-child>*{border-bottom:0}td small,th small{display:block;color:var(--muted);font:10px/1.5 system-ui;margin-top:4px}.interval{white-space:nowrap}.footnote{font-size:11px;color:var(--muted);margin:10px 0 0}.positive{color:var(--accent)}.negative{color:var(--negative)}.neutral{color:var(--muted)}
.query-panel{border:1px solid var(--line);border-radius:9px;background:var(--panel);padding:20px}.controls{display:grid;grid-template-columns:1fr 1fr .8fr 1.3fr;gap:14px}.controls label{color:var(--muted);font-size:11px;display:block}select,input[type=search]{display:block;width:100%;background:var(--paper);border:1px solid var(--line);border-radius:5px;color:var(--ink);padding:10px;margin-top:5px;font-size:12px;min-width:0}select{cursor:pointer}.query-toolbar{display:flex;align-items:center;justify-content:space-between;gap:18px;margin:19px 0;color:var(--muted);font-size:12px}input[type=checkbox]{accent-color:var(--accent);vertical-align:middle;width:14px;height:14px;margin:0 7px 0 0}.query-toolbar label{cursor:pointer}.query-panel .table-scroll{border-radius:5px}.query-table th:first-child{width:53%}.query-table td{white-space:nowrap}.query-table td:first-child{white-space:normal}.query-id{font:12px var(--mono);font-weight:600;overflow-wrap:anywhere;cursor:pointer;list-style:none}.query-id::before{content:'+';display:inline-block;width:20px;color:var(--accent)}details[open]>.query-id::before{content:'−'}.query-id::-webkit-details-marker{display:none}.question{font:12px/1.6 system-ui;color:var(--muted);margin:10px 0;overflow-wrap:anywhere}.ranking-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:12px}.ranking-grid h4{font:11px var(--mono);color:var(--accent);margin:0 0 8px;overflow-wrap:anywhere}.ranking-grid ol{margin:0;padding-left:20px;max-height:210px;overflow:auto;font:10px/1.7 var(--mono);color:var(--muted)}.ranking-grid li{overflow-wrap:anywhere;padding-left:2px}.pager{display:flex;gap:12px;justify-content:flex-end;align-items:center;margin-top:15px;font-size:11px;color:var(--muted)}.pager button{padding:6px 12px;border:1px solid var(--line);background:var(--paper);color:var(--ink);border-radius:5px;cursor:pointer}.pager button:disabled{opacity:.4;cursor:default}.empty{text-align:center;padding:28px;color:var(--muted);font-size:13px}.query-meta{font-size:11px;color:var(--muted);margin-top:12px}
.bottom-grid{display:grid;grid-template-columns:1fr 1fr;gap:28px}.bottom-grid section{min-width:0}.lineage{background:var(--panel);border:1px solid var(--line);padding:20px;border-radius:8px;margin:0}.lineage dt{font:10px var(--mono);text-transform:uppercase;color:var(--muted);margin-top:15px}.lineage dt:first-child{margin-top:0}.lineage dd{font:11px/1.6 var(--mono);margin:5px 0 0;overflow-wrap:anywhere}.technical{margin-top:12px}.technical summary{font-size:12px;color:var(--accent);cursor:pointer;padding:7px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:11px/1.6 var(--mono);background:var(--code);padding:16px;border-radius:5px;max-height:360px;overflow:auto}.run-limitations{margin:24px 0;background:var(--tint);border-radius:8px;padding:21px 24px}.run-limitations h2{font-size:23px}.run-limitations ul{margin:12px 0 0;padding-left:18px;color:var(--muted);font-size:12px}.run-limitations li{margin:6px 0}.closing-note{border-top:1px solid var(--line);padding:22px 0 30px;color:var(--muted);font-size:12px;max-width:900px}footer{border-top:1px solid var(--line);padding:22px 0;font:10px var(--mono);color:var(--muted)}.footer-row{display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap}[hidden]{display:none!important}
@media(max-width:850px){.wrap{padding:0 22px}.hero{grid-template-columns:1fr;gap:30px;padding-top:36px}.hero-note{max-width:550px}.hero h1{font-size:56px}.controls{grid-template-columns:1fr 1fr}.bottom-grid{grid-template-columns:1fr}.edition{display:none}.section-title{flex-wrap:wrap;gap:12px}.section-title p{margin-left:0}.cards{grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}.card-metrics{gap:4px}.query-panel{padding:14px}.query-table th:first-child{min-width:230px}th,td{padding:12px}.ranking-grid{grid-template-columns:1fr}.masthead{min-height:76px}}@media(max-width:450px){.hero h1{font-size:47px;letter-spacing:-2px}.controls{grid-template-columns:1fr}.dataset-stats{gap:24px}.query-toolbar{align-items:start;flex-direction:column;gap:10px}.theme-toggle{padding:7px}.brand{font-size:18px}}@media print{html[data-theme]{--paper:white;--panel:white;--ink:#111;--muted:#555;--line:#ccc;--accent:#176e62;--tint:#f2f5f1;--code:#f5f5f5}.header-actions,.controls,.query-toolbar,.pager,.download,.skip{display:none}.hero{padding-top:25px}h1{font-size:48px}.system-card,.lineage{break-inside:avoid;box-shadow:none}.wrap{padding:0 16px}.query-panel{border:0;padding:0}}
</style></head><body>
<a class="skip" href="#main">Skip to evaluation results</a>
<header><div class="wrap masthead"><div class="brand"><span class="mark" aria-hidden="true">ef</span>EvidenceForge</div><div class="header-actions"><span class="edition">Retrieval research / measured runs</span><button id="theme-toggle" class="theme-toggle" type="button" aria-pressed="false">Dark mode</button></div></div></header>
<main class="wrap" id="main">
<div class="hero"><div><div class="kicker">Evaluation report · research artifact</div><h1>Evidence,<br><em>before answers.</em></h1><p class="lead">Compare retrieval systems on the same relevance judgments. Look beyond the average, inspect the regressions, and keep every claim attached to a measured run.</p></div><aside class="hero-note"><div class="kicker">Benchmark collection</div><div class="dataset-name">@@DATASET_NAME@@</div><div class="dataset-split">Split: @@SPLIT@@</div><div class="dataset-stats"><div><strong>@@CORPUS_COUNT@@</strong><span>documents</span></div><div><strong>@@QUERY_COUNT@@</strong><span>queries</span></div><div><strong>@@SYSTEM_COUNT@@</strong><span>systems</span></div></div></aside></div>
<aside class="scope"><span class="scope-symbol" aria-hidden="true">↳</span><div><strong>Measured here: retrieval quality and search timing.</strong> Answer generation, LLM reasoning, and citation faithfulness are not evaluated in this report.</div></aside>
<section aria-labelledby="systems-title"><div class="section-title"><div><div class="heading"><span class="number">01</span><h2 id="systems-title">System scorecard</h2></div><p>Higher retrieval scores are better. Each evaluated query receives equal weight.</p></div><button class="download" id="download" type="button">Download run JSON ↓</button></div><div class="cards">@@CARDS@@</div></section>
<section aria-labelledby="comparison-title"><div class="section-title"><div><div class="heading"><span class="number">02</span><h2 id="comparison-title">Paired comparisons</h2></div><p>Candidate minus baseline, resampling matched queries. Inspect the interval alongside the mean.</p></div></div>@@COMPARISONS@@<p class="footnote">Percentile bootstrap intervals describe query-sampling variability conditional on this run. They do not cover judgment errors, dataset shift, or model variability. No multiple-comparison adjustment.</p></section>
<section aria-labelledby="query-title"><div class="section-title"><div><div class="heading"><span class="number">03</span><h2 id="query-title">Where does it improve?</h2></div><p>Choose two systems. Filter regressions. Expand a query to inspect ranked document IDs.</p></div></div><div class="query-panel">
<div class="controls"><label for="baseline">Baseline<select id="baseline">@@OPTIONS@@</select></label><label for="candidate">Candidate<select id="candidate">@@OPTIONS@@</select></label><label for="metric">Metric<select id="metric"></select></label><label for="search">Find query ID or supplied query text<input id="search" type="search" placeholder="Search queries…" autocomplete="off"></label></div>
<div class="query-toolbar"><label><input id="negative-only" type="checkbox">Show negative deltas only</label><span id="query-count" aria-live="polite"></span></div>
<div class="table-scroll"><table class="query-table"><thead><tr><th scope="col">Query / ranked document IDs</th><th scope="col">Baseline</th><th scope="col">Candidate</th><th scope="col">Delta</th></tr></thead><tbody id="query-rows"></tbody></table></div><p id="empty" class="empty" hidden>No queries match these filters.</p>
<div class="pager"><button id="previous" type="button">Previous</button><span id="page-label"></span><button id="next" type="button">Next</button></div><p id="query-meta" class="query-meta"></p><noscript><p>Enable JavaScript for per-query filtering. Aggregate results and provenance remain visible above and below.</p></noscript></div></section>
<div class="bottom-grid"><section aria-labelledby="timing-title"><div class="section-title"><div><div class="heading"><span class="number">04</span><h2 id="timing-title">Search timing</h2></div><p>Recorded by this execution; compare within its environment.</p></div></div><div class="table-scroll"><table><thead><tr><th scope="col">System</th><th scope="col">Total</th><th scope="col">p50 / query</th><th scope="col">p95 / query</th></tr></thead><tbody>@@RESOURCES@@</tbody></table></div><p class="footnote">Index construction and other setup may be outside these search timings. See the run configuration for implementation details.</p></section>
<section aria-labelledby="lineage-title"><div class="section-title"><div><div class="heading"><span class="number">05</span><h2 id="lineage-title">Run provenance</h2></div><p>Configuration and environment travel with this artifact.</p></div></div><dl class="lineage"><dt>Dataset SHA-256</dt><dd>@@DATASET_HASH@@</dd><dt>Generated at</dt><dd>@@GENERATED_AT@@</dd><dt>Reporting boundary</dt><dd>Supplied query text and document IDs only. Corpus passages are not fetched by this report.</dd></dl></section></div>
<div class="bottom-grid"><details class="technical"><summary>Inspect evaluation configuration</summary><pre>@@CONFIG@@</pre></details><details class="technical"><summary>Inspect execution environment</summary><pre>@@ENVIRONMENT@@</pre></details></div>
@@RUN_LIMITATIONS@@
<p class="closing-note">Relevance judgments are the benchmark, not a complete description of usefulness. Queries without positive judgments are excluded by the evaluator; missing rankings for eligible queries count as zero. A better aggregate can still hide important failures.</p>
</main><footer><div class="wrap footer-row"><span>EvidenceForge / reproducible retrieval engineering</span><span>Self-contained HTML · no remote assets or tracking</span></div></footer>
<script id="report-data" type="application/json">@@JSON@@</script>
<script>
'use strict';
const data = JSON.parse(document.getElementById('report-data').textContent);
const systems = data.systems;
const names = Object.keys(systems);
const baseline = document.getElementById('baseline');
const candidate = document.getElementById('candidate');
const metric = document.getElementById('metric');
const search = document.getElementById('search');
const negative = document.getElementById('negative-only');
const rows = document.getElementById('query-rows');
const pageSize = 30;
let page = 0;
candidate.value = names[names.length - 1];
function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}
function number(value) { return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(4) : '—'; }
function refreshMetrics() {
  const previous = metric.value;
  metric.replaceChildren();
  const common = Object.keys(systems[baseline.value].aggregate).filter(key => Object.hasOwn(systems[candidate.value].aggregate, key));
  for (const key of common) {
    const option = element('option', key);
    option.value = key;
    metric.append(option);
  }
  metric.value = common.includes(previous) ? previous : common.includes('ndcg@10') ? 'ndcg@10' : common[0] || '';
  page = 0;
  render();
}
function rankingList(name, qid) {
  const block = element('div');
  block.append(element('h4', name));
  const ranking = systems[name].rankings?.[qid] || [];
  if (!ranking.length) block.append(element('p', 'No ranked documents', 'question'));
  else {
    const list = element('ol');
    for (const id of ranking) list.append(element('li', id));
    block.append(list);
  }
  return block;
}
function render() {
  const a = systems[baseline.value].per_query;
  const b = systems[candidate.value].per_query;
  const allIds = [...new Set([...Object.keys(a), ...Object.keys(b)])].sort();
  const term = search.value.trim().toLocaleLowerCase();
  let unpaired = 0;
  const matches = [];
  for (const qid of allIds) {
    const av = a[qid]?.[metric.value];
    const bv = b[qid]?.[metric.value];
    const delta = typeof av === 'number' && typeof bv === 'number' ? bv - av : null;
    if (delta === null) unpaired++;
    const text = String(data.queries?.[qid] || '');
    if (term && !(qid + ' ' + text).toLocaleLowerCase().includes(term)) continue;
    if (negative.checked && !(delta !== null && delta < -1e-12)) continue;
    matches.push({qid, av, bv, delta, text});
  }
  matches.sort((x, y) => (x.delta === null ? Infinity : x.delta) - (y.delta === null ? Infinity : y.delta) || x.qid.localeCompare(y.qid));
  const pageCount = Math.max(1, Math.ceil(matches.length / pageSize));
  page = Math.min(page, pageCount - 1);
  rows.replaceChildren();
  for (const item of matches.slice(page * pageSize, (page + 1) * pageSize)) {
    const tr = element('tr');
    const first = element('td');
    const details = element('details');
    details.append(element('summary', item.qid, 'query-id'));
    if (item.text) details.append(element('p', item.text, 'question'));
    const rankings = element('div', undefined, 'ranking-grid');
    rankings.append(rankingList(baseline.value, item.qid), rankingList(candidate.value, item.qid));
    details.append(rankings);
    first.append(details);
    const deltaClass = item.delta === null || Math.abs(item.delta) <= 1e-12 ? 'neutral' : item.delta > 0 ? 'positive' : 'negative';
    const deltaText = item.delta === null ? 'unpaired' : (item.delta > 0 ? '+' : '') + number(item.delta);
    tr.append(first, element('td', number(item.av)), element('td', number(item.bv)), element('td', deltaText, deltaClass));
    rows.append(tr);
  }
  document.getElementById('query-count').textContent = matches.length + ' of ' + allIds.length + ' queries · lowest delta first';
  document.getElementById('empty').hidden = matches.length !== 0;
  document.getElementById('page-label').textContent = 'Page ' + (page + 1) + ' / ' + pageCount;
  document.getElementById('previous').disabled = page === 0;
  document.getElementById('next').disabled = page + 1 >= pageCount;
  document.getElementById('query-meta').textContent = unpaired ? unpaired + ' queries lack paired metric values; shown as unpaired, never assigned an invented score.' : 'Every displayed query has a paired score. Delta = candidate − baseline. Rankings show document IDs, not retrieved source text.';
}
baseline.addEventListener('change', refreshMetrics);
candidate.addEventListener('change', refreshMetrics);
for (const control of [metric, search, negative]) control.addEventListener(control === search ? 'input' : 'change', () => { page = 0; render(); });
document.getElementById('previous').addEventListener('click', () => { page--; render(); });
document.getElementById('next').addEventListener('click', () => { page++; render(); });
document.getElementById('theme-toggle').addEventListener('click', () => {
  const dark = document.documentElement.dataset.theme !== 'dark';
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  const button = document.getElementById('theme-toggle');
  button.textContent = dark ? 'Light mode' : 'Dark mode';
  button.setAttribute('aria-pressed', String(dark));
});
document.getElementById('download').addEventListener('click', () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'}));
  const link = element('a');
  link.href = url;
  link.download = 'evidenceforge-results.json';
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
refreshMetrics();
</script></body></html>
"""
