import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import * as tf from '@tensorflow/tfjs-core';
import { loadLayersModel } from '@tensorflow/tfjs-layers';
import '@tensorflow/tfjs-backend-cpu';
import { validateRows, windowStarts, WINDOW } from './signals.mjs';
export async function loadModel(root = new URL('../', import.meta.url)) {
  const jsonBytes = await readFile(new URL('model.json', root));
  const spec = JSON.parse(jsonBytes);
  const chunks = [];
  const weightSpecs = [];
  for (const group of spec.weightsManifest) {
    weightSpecs.push(...group.weights);
    for (const path of group.paths) {
      if (!/^[\w.-]+\.bin$/.test(path)) throw new Error('Unexpected weight path.');
      chunks.push(await readFile(new URL(path, root)));
    }
  }
  const buffer = Buffer.concat(chunks);
  const expected = weightSpecs.reduce((n, w) => n + w.shape.reduce((a, b) => a * b, 1) * 4, 0);
  if (weightSpecs.some(w => w.dtype !== 'float32') || buffer.length !== expected) throw new Error('Model weight manifest and bytes do not agree.');
  const topology = structuredClone(spec.modelTopology);
  delete topology.training_config; // Inference does not need Keras optimizer/metric serializers.
  await tf.setBackend('cpu');
  await tf.ready();
  const model = await loadLayersModel(tf.io.fromMemory({ modelTopology: topology, weightSpecs, weightData: buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength) }));
  if (JSON.stringify(model.inputs[0].shape.slice(1)) !== '[100,6]' || model.outputs[0].shape.at(-1) !== 1) { model.dispose(); throw new Error('Expected [batch, 100, 6] → [batch, 1].'); }
  return { model, metadata: { features: 6, window: WINDOW, parameters: model.countParams(), jsonSHA256: createHash('sha256').update(jsonBytes).digest('hex'), weightsSHA256: createHash('sha256').update(buffer).digest('hex'), backend: tf.getBackend() } };
}
export async function predictRows(model, rows, stride = 25) {
  validateRows(rows);
  const predictions = [];
  for (const start of windowStarts(rows.length, stride)) {
    const input = tf.tensor3d([rows.slice(start, start + WINDOW)], [1, WINDOW, 6]);
    let output;
    try {
      output = model.predict(input);
      const score = (await output.data())[0];
      if (!Number.isFinite(score)) throw new Error('Model produced a non-finite score.');
      predictions.push({ start, score });
    } finally { input.dispose(); output?.dispose(); }
  }
  return predictions;
}
