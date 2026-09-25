# Step 3: Build datasets and an object-storage mirror

This chapter connects numerical experiments to model training. Its outputs are an immutable Burgers dataset with reproducible splits, and a tested S3-compatible mirror for experiment artifacts. The cloud tests use a local emulator; they do not deploy a bucket, spend cloud money, or establish production reliability.

Read the implementations alongside this chapter:

- [datasets.py](../../src/flowstate/datasets.py): export, import, split assignment, normalization, verification.
- [object_store.py](../../src/flowstate/object_store.py): conditional uploads, resumable retries, checked downloads.
- [test_datasets.py](../../tests/test_datasets.py) and [test_object_store.py](../../tests/test_object_store.py): examples of successful and deliberately broken inputs.

## 1. Understand the data contract before writing a model

A numerical experiment stores a field with dimensions [time, x]. A training dataset adds an outer trajectory dimension:

~~~text
dataset/
  zarr.json
  fields/u/              float32 [trajectory, time, x], original physical values
  time/                  float64 [time], original saved physical times
  x/                     float64 [x], original periodic sample locations
  splits/train/          int64 trajectory indices
  splits/validation/
  splits/test/
  metadata.json          configurations, metrics, sources, families, split assignments
  manifest.json          SHA256 for every other file
~~~

The dataset directory itself is the Zarr group. There is no additional dataset.zarr subdirectory. In Python:

~~~python
from pathlib import Path
import zarr

path = Path("data/datasets/burgers")
group = zarr.open_group(str(path), mode="r")
one_trajectory = group["fields/u"][0]
physical_time = group["time"][:]
training_indices = group["splits/train"][:]
~~~

The colon in a slice means "all entries along this axis." Reading one trajectory is bounded by that trajectory's size. Reading the entire fields/u array would allocate the whole dataset. Shape is part of the scientific contract: swapping the time and x axes can produce plausible-looking arrays that describe the wrong problem.

All exported trajectories must share their exact physical grid, saved times, and effective viscosity. Coordinates must be finite, increasing, and uniformly spaced; the uniformity check allows relative tolerance 0.0001 and absolute tolerance 1e-10 for coordinate values originally stored as float32. Coordinates are preserved, not resampled. There must be at least two saved frames and four spatial points. A shortened final save interval is rejected. Native periodic grids must omit the repeated endpoint.

## 2. Trace extract, transform, and load

The export_dataset function follows an extract-transform-load sequence:

1. **Extract:** select completed Burgers runs and verify every source artifact hash. Reject runs whose metrics say needs_review. A completed program is not automatically an acceptable training reference.
2. **Transform:** check physical compatibility, identify related initial conditions, assign splits, cast field values to float32, and calculate normalization from training trajectories.
3. **Load:** write chunked Zarr arrays and metadata into a staging directory, write the checksum manifest, then publish with one filesystem rename.

The field data remains in physical units. Normalization stores a mean and standard deviation for the loader to apply:

~~~python
normalized_u = (physical_u - training_mean) / training_std
physical_prediction = normalized_prediction * training_std + training_mean
~~~

Normalization parameters are not fitted to validation or test fields. Constant training data receives standard deviation 1 and an explicit constant_training_data flag, preventing division by zero without claiming meaningful variation.

Exported metadata preserves each source ID, configuration, diagnostic metrics, manifest hash, and artifact hashes. This connects a training value to the simulation that produced it. A source hash identifies bytes; it does not establish that the solver was mathematically correct or sufficiently resolved.

## 3. Learn the functions, types, and loops

The public signature is:

~~~python
export_dataset(lake_root, output, ids=None, seed=0)
~~~

A function packages one operation behind named inputs and a return value. lake_root and output accept either a string or a Path. ids is either a list of experiment identifiers or None; None selects completed Burgers experiments. seed controls the reproducible split, independently of the initial-condition seeds that generated the simulations. The returned dictionary contains the metadata also written to disk.

Internally, the important loop has this structure:

~~~python
for index in range(number_of_trajectories):
    values = load_one_trajectory(index)
    check_shape_and_finiteness(values)
    output_array[index] = values
    if index in training_indices:
        update_training_statistics(values)
~~~

range produces indices one at a time. Each iteration loads and writes one trajectory. There is no list accumulating all field arrays. A small list of source handles and metadata is retained, so memory still grows with the number of experiments, but field memory scales with one trajectory rather than the entire dataset. Float conversion and statistics use temporary arrays of that same trajectory size.

Training statistics use a mergeable mean and sum of squared deviations. This is more stable than subtracting two large, nearly equal sums. count records how many field values contributed, making the normalization auditable.

The storage type float32 uses four bytes per value. The computation of summary statistics uses float64. This deliberate distinction saves dataset space while reducing rounding error in the mean and variance. Inputs outside float32's finite range are rejected.

## 4. Prevent leakage before creating temporal windows

Adjacent frames are strongly related. Randomly splitting windows from the same simulation allows the test set to reveal parts of trajectories the model already encountered.

Flowstate assigns complete initial-condition families to splits first. Native random families use initial-condition type and seed; viscosity, grid resolution, amplitude, and solver changes do not create independent families. Deterministic sine initial conditions ignore their unused seed and share a family. Exact duplicate initial fields also unite families, including duplicate imported samples.

This conservative grouping sometimes leaves fewer useful independent examples than the number of experiment directories suggests. Export requires at least three distinct families so train, validation, and test are all nonempty. Approximately one fifth of families goes to each holdout split, with a minimum of one. These are small-prototype splits, not a claim that three families are enough for a scientific benchmark.

Every trajectory records its family ID. Frozen split indices and a split_sha256 digest are stored in metadata; model training must use them unchanged. Temporal windows can then be built entirely within the selected trajectories.

To run a small export after generating random-initial-condition experiments:

~~~powershell
uv run --no-sync flowstate --lake data/lake dataset export data/datasets/burgers --seed 7
uv run --no-sync flowstate dataset verify data/datasets/burgers
~~~

Reusing an existing output directory is rejected. Choose a new output name for a changed selection or split seed.

## 5. Import a small PDEBench file deliberately

The adapter supports the periodic one-dimensional Burgers HDF5 convention: tensor[sample,time,x], x-coordinate, and t-coordinate. It reads one sample at a time. A source containing an extra terminal time coordinate is accepted only when its length is exactly the number of field frames plus one; that unused terminal value is recorded in provenance.

The source URL, version, license name, and uniform effective viscosity are required. The import stores the original file's SHA256 and each source sample index. It does not download a large benchmark or infer licensing permission. Tests construct a tiny synthetic HDF5 fixture; they demonstrate format handling, not PDEBench benchmark performance. The maintained [PDEBench repository](https://github.com/pdebench/PDEBench) documents its data and generation workflows.

**Viscosity convention matters.** Flowstate uses the coefficient multiplying u_xx in u_t + u*u_x = viscosity*u_xx. The official [PDEBench Burgers generator](https://github.com/pdebench/PDEBench/blob/main/pdebench/data_gen/data_gen_NLE/BurgersEq/burgers_multi_solution_Hydra.py) divides its epsilon/filename Nu parameter by pi. For a file generated with Nu0.01, pass 0.01/pi as effective viscosity, after checking that this is the convention used by the specific source version. The adapter does not guess from filenames. Shared viscosity is an explicit caller assertion; mixed per-trajectory viscosity arrays are rejected.

~~~python
import math
from flowstate.datasets import import_pdebench

metadata = import_pdebench(
    "downloads/1D_Burgers_Sols_Nu0.01.hdf5",
    "data/datasets/pdebench-burgers",
    source_url="https://example.org/replace-with-the-actual-source",
    source_version="replace-with-the-actual-source-version",
    license_name="replace-with-the-applicable-data-license",
    viscosity=0.01 / math.pi,
    seed=7,
)
~~~

The example provenance strings are placeholders to replace, not assertions about an external dataset. Stored x, time, and velocity values keep the source's units; the importer does not invent an SI conversion or validate the supplied source attribution.

## 6. Choose chunks around access patterns

Canonical field chunks are [1, min(16, time), min(256, x)]. One chunk never mixes trajectories or dataset splits. Up to sixteen consecutive frames are grouped for temporal training access. Spatial chunks cap individual transfers for larger grids.

These choices are a starting point. Long rollout evaluation, sparse point sampling, and full-trajectory training have different access costs. Measure bytes read, request count, decompression time, and peak memory before changing chunks. More chunks can reduce unnecessary reads while increasing filesystem or object-request overhead.

The experiment lake and training dataset can use different chunks because their access patterns differ. Export is an explicit derived artifact with its own manifest, not an in-place rewrite of numerical evidence.

## 7. Publish an object-store commit last

A filesystem can rename a completed directory atomically. S3 exposes individual objects; it cannot rename an entire prefix atomically. The mirror therefore uses:

~~~text
<prefix>/artifacts/<experiment-id>/<manifest-sha256>/<relative-artifact-path>
<prefix>/experiments/<experiment-id>/manifest.json
~~~

The first prefix contains immutable content. The final manifest key is the commit marker. Upload sends artifacts first and the marker last. All PUT requests use If-None-Match: *, so an existing object cannot be overwritten through this adapter. [AWS conditional-write documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html) describes this request behavior.

If an upload stops halfway through, there is no marker for readers to treat as complete. Retrying checks existing object bytes, reuses identical ones, uploads missing ones, and finally commits the marker. An existing object with conflicting bytes raises an error. Partial uploads can leave unreferenced content prefixes; automatic garbage collection is intentionally absent.

Download starts from the committed manifest, validates paths, streams each object into local staging, verifies SHA256, checks the experiment ID, and publishes locally only after all checks pass. A missing or corrupt remote chunk leaves no final local experiment. Repeating a completed download reuses the verified identical local run.

~~~python
from flowstate.object_store import upload_experiment, download_experiment

upload_experiment("data/lake", experiment_id, bucket, prefix="study-one")
download_experiment("data/restored", experiment_id, bucket, prefix="study-one")
~~~

These calls require an existing bucket and an authorized SDK credential setup. boto3 uses its standard credential chain; Flowstate never writes credentials into provenance. endpoint_url can point to an S3-compatible local service. The adapter uses single-object PUTs, limited to 5 GiB per artifact; it does not implement multipart uploads, bucket creation, distributed transactions, scheduling, or production retention policies.

An unsigned manifest detects corruption against recorded hashes. Someone who can replace both a manifest and all matching data can defeat that check. Immutability here is application behavior, not a replacement for provider permissions, versioning, or retention controls.

## 8. Run the evidence and investigate failures

~~~powershell
uv run --no-sync pytest tests/test_datasets.py tests/test_object_store.py -q
~~~

Dataset tests cover physical compatibility, family isolation, training-only normalization, repeated exports, tampered sources, flagged runs, bounded HDF5 sample reads, and non-finite values. Object tests use Moto's local S3 emulator with the real boto3 request interface. They exercise byte-for-byte round trips, conditional writes, interrupted uploads, commit-marker ordering, retries, missing chunks, corruption, path traversal, and overwrite refusal.

Suggested learning exercises:

1. Add an experiment with the same initial-condition seed but a different amplitude. Confirm it remains in the same split.
2. Change one dataset metadata byte. Observe the integrity failure before model training.
3. Inspect the interrupted-upload test and identify the exact request after which the simulated failure occurs.
4. Compare chunk counts for sixteen-frame versus one-frame temporal chunks on a small temporary dataset. Record timings; do not silently alter existing artifacts.

Passing these checks makes the pipeline inspectable and reproducible at local prototype scale. A future deployment still needs provider-specific integration tests, realistic storage/load measurements, and independently validated numerical references.
