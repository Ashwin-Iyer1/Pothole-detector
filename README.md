# Road Signal Model Lab

Run and inspect the original six-channel inertial-sensor model with reproducible CPU inference. Despite the repository name, this is **not an image detector**. It consumes 100 consecutive accelerometer/gyroscope samples and emits one sigmoid score.

## Run locally

Requires Node 22+. No camera, account, GPU, or training dataset is needed for the synthetic demonstration.

```sh
npm ci
npm test
node cli.mjs --demo --output report.json --html report.html --sample-output sample.csv
# Open report.html in your browser. It is a standalone, offline interactive report.
node cli.mjs --input sample.csv --stride 25 --threshold 0.5 --output report.json --html report.html
```

The report plots all six channels, window scores, an adjustable decision threshold, exact sample spans and model fingerprints. It embeds the input rows, so treat reports made from your own data accordingly. JSON remains the complete machine-readable output.

## Input contract

CSV columns are `Acc X,Acc Y,Acc Z,gyro_x,gyro_y,gyro_z`, in any header order. Optional `label` must contain only 0/1. Supply 100–10,000 consecutive rows; blanks, nonnumeric cells, duplicate headers, infinities and incomplete windows are rejected. Values are passed through unchanged, without undocumented normalization. Stride is 1–100 samples. The model never pads a window by repeating one sample.

The historical notebook labels a window positive when **at least five** of its 100 row labels are positive. The decision rule is strictly `score > threshold`. Descriptive metrics are included only when labels are provided; undefined precision/recall is `null`, not zero.

## What the artifact establishes—and what it does not

- The existing `model.json` and `group1-shard1of1.bin` load into TensorFlow.js with 68,097 parameters, input `[batch,100,6]`, output `[batch,1]`. Original weights and architecture are unchanged. Training-only Keras optimizer/metric metadata is omitted in memory when loading for inference.
- The original `SDS_File.ipynb` records the feature names, three recurrent layers (two bidirectional), dense layers, and the label rule. The dataset, sensor units, orientation, collection protocol and sample rate are absent. We cannot establish deployment compatibility from those artifacts alone.
- The notebook shuffles overlapping windows **before** the train/evaluation split. Neighboring windows can share 99/100 rows; its displayed evaluation figures are not credible independent generalization evidence. No accuracy claim is made here.
- `splitPlan` partitions raw rows before constructing windows, demonstrating disjoint support. It does not retrain the saved model or prove that the artifact has not already seen supplied data. A future trustworthy study should split by independently collected trip/device/person before preprocessing.
- Synthetic traces are deterministic numerical exercises, not labeled pothole events. Scores are uncalibrated model outputs, not road-safety probabilities or driving advice.

## Project history and verification

The original notebook, H5/SavedModel and TF.js artifacts are preserved. Existing Git history includes Ashwin-Iyer and Turtlecap1324; this work adds a maintainable inference/analysis layer without assigning unsupported individual training ownership. The historical `studentsdrivesafe` gitlink has no `.gitmodules`; it is not needed to run this lab. The companion application is [studentsdrivesafe](https://github.com/Ashwin-Iyer1/studentsdrivesafe).

Tests exercise real weight loading, finite deterministic predictions, tensor disposal, exact window bounds, malformed inputs, label semantics, disjoint split support, and report-data escaping. Model loading follows the [TensorFlow.js model IO documentation](https://www.tensorflow.org/js/guide/save_load). Installation is the only network-dependent step; inference and reports run locally afterward.
