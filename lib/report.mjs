function reportUI() {
  const data = JSON.parse(document.getElementById('report-data').textContent);
  const $ = id => document.getElementById(id);
  const fmt = n => n == null ? '—' : Number(n).toFixed(4);
  const svg = (id, values, color, threshold) => {
    const el = $(id), min = threshold == null ? Math.min(...values) : 0, max = threshold == null ? Math.max(...values) : 1, span = max - min || 1;
    el.replaceChildren();
    const make = (tag, attributes) => { const node = document.createElementNS('http://www.w3.org/2000/svg', tag); Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value)); el.append(node); return node; };
    const y = value => 175 - (value - min) / span * 150;
    make('path', { d: values.map((v, i) => `${i ? 'L' : 'M'}${25 + i / Math.max(1, values.length - 1) * 750},${y(v)}`).join(' '), fill: 'none', stroke: color, 'stroke-width': 2 });
    if (threshold != null) make('path', { d: `M25,${y(threshold)}H775`, stroke: '#e9b76b', 'stroke-dasharray': '6 5' });
    make('text', { x: 25, y: 15, fill: '#a8b6c8', 'font-size': 12 }).textContent = `range ${fmt(min)} … ${fmt(max)}`;
  };
  $('source').textContent = data.report.source;
  $('identity').textContent = `${data.report.model.parameters.toLocaleString()} parameters · ${data.rows.length} samples · ${data.report.windows.length} windows · CPU inference`;
  $('fingerprint').textContent = data.report.model.weightsSHA256;
  data.report.featureSummary.forEach((s, index) => { const option = document.createElement('option'); option.value = index; option.textContent = s.name; $('channel').append(option); });
  const drawSignal = () => { const c = Number($('channel').value); svg('signal', data.rows.map(r => r[c]), '#7ddacb'); $('channel-stats').textContent = `Mean ${fmt(data.report.featureSummary[c].mean)} · standard deviation ${fmt(data.report.featureSummary[c].standardDeviation)} · original numeric units (not documented)`; };
  $('channel').addEventListener('change', drawSignal);
  const drawScores = () => {
    const threshold = Number($('threshold').value);
    $('threshold-value').textContent = threshold.toFixed(2);
    svg('scores', data.report.windows.map(w => w.score), '#a9baff', threshold);
    const flagged = data.report.windows.filter(w => w.score > threshold).length;
    $('flagged').textContent = `${flagged} / ${data.report.windows.length}`;
    const body = $('windows'); body.replaceChildren();
    for (const w of data.report.windows.slice(0, 500)) {
      const tr = document.createElement('tr');
      [w.start, w.endExclusive - 1, fmt(w.score), w.score > threshold ? 'Above threshold' : 'Below / equal', w.label ?? 'Not provided'].forEach(value => { const td = document.createElement('td'); td.textContent = value; tr.append(td); });
      body.append(tr);
    }
    $('window-note').textContent = data.report.windows.length > 500 ? 'Showing the first 500 windows. The JSON report contains every window.' : 'Window endpoints are inclusive sample indices. Overlapping windows are correlated.';
  };
  $('threshold').value = data.report.threshold;
  $('threshold').addEventListener('input', drawScores);
  $('split').textContent = `Boundary: sample ${data.report.splitPlan.boundary}. Training-side windows: ${data.report.splitPlan.train.length}; evaluation-side windows: ${data.report.splitPlan.evaluation.length}; shared raw samples: 0. This is a split plan, not a claim that the saved model has never seen these rows.`;
  drawSignal(); drawScores();
}
export function renderReport(report, rows) {
  const payload = JSON.stringify({ report, rows }).replace(/</g, '\\u003c');
  return `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Road Signal · Model report</title><style>
  :root{font-family:system-ui,sans-serif;color:#eef3f8;background:#101824;line-height:1.6}*{box-sizing:border-box}body{max-width:1140px;margin:auto;padding:40px 22px}header{padding-bottom:25px;border-bottom:1px solid #334255}.eyebrow{color:#7ddacb;font-size:12px;letter-spacing:.15em}h1{font-size:clamp(30px,5vw,52px);line-height:1.1;margin:14px 0}h2{font-size:18px}p{color:#b5c3d4}section{background:#192536;border:1px solid #334255;border-radius:14px;padding:24px;margin-top:20px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.grid section{min-width:0}label{display:block;font-size:14px;margin:12px 0}select{background:#101824;color:#eef3f8;border:1px solid #65758a;padding:8px;border-radius:6px}input{width:100%;accent-color:#7ddacb}svg{width:100%;height:auto;display:block}table{border-collapse:collapse;width:100%;font-size:13px;text-align:left}td,th{padding:10px;border-bottom:1px solid #334255}.scroll{overflow:auto;max-height:430px}code{overflow-wrap:anywhere;color:#7ddacb;font-size:12px}.number{font-size:34px;color:#7ddacb}summary{cursor:pointer}a{color:#7ddacb}button:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #e9b76b;outline-offset:3px}@media(max-width:700px){.grid{grid-template-columns:1fr}section{padding:18px}body{padding:25px 14px}}
  </style><header><div class="eyebrow">ROAD SIGNAL / MODEL LAB</div><h1>A signal, examined.</h1><p id="source"></p><p id="identity"></p></header>
  <section><h2>Understand the score</h2><p>This is an uncalibrated output from the repository's saved inertial-sensor model. It is not a road-safety probability. Sensor units, orientation and sampling rate were not preserved; synthetic traces demonstrate computation only.</p></section>
  <div class="grid"><section><h2>Six channels, one sequence</h2><label for="channel">Feature channel</label><select id="channel"></select><svg id="signal" viewBox="0 0 800 200" role="img" aria-label="Selected sensor channel across sample indices"></svg><p id="channel-stats"></p></section><section><h2>Window scores</h2><div class="number" id="flagged"></div><p>Windows above the selected threshold</p><label for="threshold">Decision threshold <output id="threshold-value"></output></label><input id="threshold" type="range" min="0" max="1" step=".01"><svg id="scores" viewBox="0 0 800 200" role="img" aria-label="Model score by window; dashed line is the selected threshold"></svg></section></div>
  <section><h2>Inspect every decision</h2><p id="window-note"></p><div class="scroll"><table><thead><tr><th>Start</th><th>End</th><th>Raw score</th><th>Decision</th><th>Supplied label</th></tr></thead><tbody id="windows"></tbody></table></div></section>
  <section><h2>Separate raw samples before windows</h2><p id="split"></p><p>The original notebook shuffled overlapping windows before dividing them. Its historical evaluation is not independent evidence of performance. This report does not retrain the model or claim to repair those historical metrics.</p><details><summary>Model artifact identity</summary><p>SHA-256 of the original weight shard:</p><code id="fingerprint"></code></details></section>
  <footer><p>Generated locally by <a href="https://github.com/Ashwin-Iyer1/Pothole-detector">Pothole-detector</a>. This report includes its input rows; share only data you intend to disclose. No network requests are needed to use it.</p></footer><script type="application/json" id="report-data">${payload}</script><script>(${reportUI.toString()})();</script></html>`;
}
