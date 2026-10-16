import test from 'node:test';
import assert from 'node:assert/strict';
import * as tf from '@tensorflow/tfjs-core';
import { readFile } from 'node:fs/promises';
import { FEATURES, parseCSV, toCSV, validateRows, windowStarts, splitPlan, summarize, summarizeScores, syntheticTrace } from '../lib/signals.mjs';
import { loadModel, predictRows } from '../lib/model.mjs';
import { renderReport } from '../lib/report.mjs';
test('CSV round trips six features and supports column reordering', () => {
  const rows = syntheticTrace();
  assert.deepEqual(parseCSV(toCSV(rows)).rows, rows);
  const reverse = [...FEATURES].reverse();
  const csv = reverse.join(',') + '\n' + rows.map(row => [...row].reverse().join(',')).join('\n');
  assert.deepEqual(parseCSV(csv).rows, rows);
});
test('malformed, incomplete, infinite, duplicate-column or short inputs fail before inference', () => {
  for (const csv of ['', toCSV(syntheticTrace()).replace('9.81', 'NaN'), 'Acc X,Acc X\n1,2', toCSV(syntheticTrace()).replace(/\n[^\n]+/, '\n1,2')]) assert.throws(() => parseCSV(csv));
  assert.throws(() => validateRows(Array.from({length: 99}, () => [0,0,0,0,0,0])));
  assert.throws(() => validateRows(Array.from({length: 100}, () => [0,0,0,0,0,Infinity])));
});
test('windows contain exactly 100 consecutive rows and never pad or repeat samples', () => {
  assert.deepEqual(windowStarts(100), [0]);
  assert.deepEqual(windowStarts(151, 25), [0,25,50]);
  assert.throws(() => windowStarts(300, 0));
  assert.throws(() => windowStarts(300, 101));
});
test('chronological partition creates disjoint raw-sample support', () => {
  const p = splitPlan(1000);
  const train = new Set(p.train.flatMap(start => Array.from({length:100}, (_, i) => start + i)));
  assert.equal(p.boundary, 800);
  assert.ok(p.evaluation.every(start => Array.from({length:100}, (_,i)=>start+i).every(i=>!train.has(i))));
});
test('window labels preserve the original five-positive threshold and strict score threshold', () => {
  const labels = Array(100).fill(0); labels.fill(1, 0, 5);
  const report = summarizeScores([{start:0,score:.5}], .5, labels);
  assert.equal(report.windows[0].label, 1); assert.equal(report.flagged, 0); assert.equal(report.metrics.fn, 1);
  assert.equal(report.metrics.precision, null);
  assert.throws(() => summarizeScores([], NaN));
});
test('summary detects constant signals and uses population variance', () => {
  const rows = Array.from({length:100}, () => [1,2,3,4,5,6]);
  assert.ok(summarize(rows).every(s => s.constant && s.standardDeviation === 0));
  assert.equal(summarize(rows)[5].mean, 6);
});
test('real saved weights load and inference is finite, deterministic, shape-correct and leak-free', async () => {
  const baseline = tf.memory().numTensors;
  const {model,metadata} = await loadModel();
  assert.equal(metadata.parameters, 68097);
  assert.equal(metadata.weightsSHA256, '3c2e40c6054c6359ca445d33894a4e374891630119f881f4b93644429396c76f');
  const rows = syntheticTrace('impulse',100);
  const during = tf.memory().numTensors;
  const a = await predictRows(model, rows), b = await predictRows(model, rows);
  assert.deepEqual(a,b); assert.equal(a.length,1); assert.ok(a[0].score >= 0 && a[0].score <= 1);
  assert.equal(tf.memory().numTensors,during);
  model.dispose(); assert.equal(tf.memory().numTensors,baseline);
});
test('report safely embeds data and records explanatory constraints', () => {
  const html = renderReport({source:'</script><script>alert(1)</script>'}, []);
  assert.ok(!html.includes('</script><script>alert(1)'));
  assert.ok(html.includes('not a road-safety probability'));
});
