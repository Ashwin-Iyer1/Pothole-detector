import { readFile, writeFile } from 'node:fs/promises';
import { loadModel, predictRows } from './lib/model.mjs';
import { parseCSV, toCSV, summarize, summarizeScores, splitPlan, syntheticTrace } from './lib/signals.mjs';
import { renderReport } from './lib/report.mjs';
const args = process.argv.slice(2);
const options = {};
try {
  for (let i = 0; i < args.length; i++) {
    const key = args[i];
    if (key === '--demo' || key === '--help') options[key] = true;
    else if (['--input', '--output', '--html', '--threshold', '--stride', '--sample-output'].includes(key) && args[i + 1] && !args[i + 1].startsWith('--')) options[key] = args[++i];
    else throw new Error(`Unknown option or missing value: ${key}`);
  }
  if (options['--help'] || !args.length) {
    console.log('node cli.mjs --demo | --input trace.csv [--output report.json] [--html report.html] [--threshold 0.5] [--stride 25] [--sample-output sample.csv]');
  } else {
    if (Boolean(options['--demo']) === Boolean(options['--input'])) throw new Error('Choose exactly one of --demo or --input.');
    const { rows, labels } = options['--demo'] ? { rows: syntheticTrace(), labels: null } : parseCSV(await readFile(options['--input'], 'utf8'));
    const threshold = Number(options['--threshold'] ?? .5), stride = Number(options['--stride'] ?? 25);
    summarizeScores([], threshold); // Validate before model initialization.
    const split = splitPlan(rows.length, .8, stride);
    const { model, metadata } = await loadModel();
    let predictions;
    try { predictions = await predictRows(model, rows, stride); } finally { model.dispose(); }
    const report = { source: options['--demo'] ? 'synthetic numerical demonstration; no road labels' : 'user-supplied local CSV', model: metadata, samples: rows.length, featureSummary: summarize(rows), ...summarizeScores(predictions, threshold, labels), splitPlan: split, limitations: ['Training dataset and sensor units/sample rate were not preserved in this repository.', 'Scores are uncalibrated model outputs, not road safety probabilities.', 'The original notebook shuffled overlapping windows before splitting; its historical evaluation is not a reliable independent test.'] };
    if (options['--output']) await writeFile(options['--output'], JSON.stringify(report, null, 2) + '\n');
    if (options['--html']) await writeFile(options['--html'], renderReport(report, rows));
    if (options['--sample-output']) await writeFile(options['--sample-output'], toCSV(rows));
    console.log(JSON.stringify(report, null, 2));
  }
} catch (error) { console.error(error.message); process.exitCode = 1; }
