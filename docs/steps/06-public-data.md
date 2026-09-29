# Real public data: acquisition, ETL, and a fixed learning study

This workflow takes numerical trajectories published by PDEBench, downloads a
small declared selection, stores an attributed HDF5 subset, converts it to the
same Zarr dataset used by Flowstate's own experiments, and compares learned
predictions with those trajectories. It does not generate new reference solutions.

```text
DaRUS public HDF5 (8.23 GB)
  → bounded HTTPS ranges (64 MiB default transfer cap)
  → 24 complete selected trajectories + acquisition receipt
  → canonical Zarr + frozen train/validation/test split
  → three fixed FNO seeds + one per-instance PINN
  → persistence comparisons + immutable model/evaluation bundles + study report
```

## Run it

With the ML extra installed, use a fresh directory:

```sh
uv run --no-sync flowstate public-study outputs/public-study-01
```

The command writes `plan.json` before contacting the server. It fixes sample seed
20260926, split seed 17, FNO seeds 0/1/2, ten FNO epochs per seed, and 200 PINN
epochs. Each of 24 trajectories retains all 201 saved frames and all 1,024 spatial
points. No test result determines these choices. The first test trajectory is the
PINN's initial-value problem; its interior reference values are not training labels.

Inspect `events.jsonl` for progress, `report.json` for completed results, and
`failure.json` if a stage fails. Completed stage artifacts remain available after
a later failure. The top-level study does not automatically resume; individual
verified datasets and model checkpoints can be reused through the existing CLI.

Individual acquisition and import commands:

```sh
uv run --no-sync flowstate dataset acquire-pdebench outputs/public-raw --samples 24
uv run --no-sync flowstate dataset verify-acquisition outputs/public-raw
uv run --no-sync flowstate dataset import-acquired outputs/public-raw outputs/public-zarr --seed 17
uv run --no-sync flowstate dataset verify outputs/public-zarr
```

For an already downloaded local HDF5, `dataset import-pdebench` additionally accepts
`--sample-indices 3 20 100`, `--time-stop 201`, and `--max-values 16000000` alongside
its required provenance/viscosity flags. `time-stop` is an exclusive prefix from
the initial frame. Source row numbers and caller ordering survive the import;
spatial downsampling is intentionally not part of this operation.

## Source and physical meaning

Source: Takamoto et al., [PDEBench Datasets, DaRUS V8.0](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986&version=8.0),
released 2024-02-13, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The pinned file is `1D_Burgers_Sols_Nu0.01.hdf5`, file ID 281363, with
8,232,968,312 bytes and publisher-advertised MD5
`e6d9a4f62baf9a29121a816b919e2770`. Selecting trajectories and writing a derived
HDF5/Zarr representation are recorded modifications. Attribution, license, DOI,
version, selection, and source URL travel with the data.

The filename stores epsilon. The [pinned official generator](https://github.com/pdebench/PDEBench/blob/4ff3e3a4aa1561721b5571fa3a048a0a463e0568/pdebench/data_gen/data_gen_NLE/BurgersEq/burgers_multi_solution_Hydra.py)
uses diffusion coefficient epsilon/pi. Flowstate therefore records effective
viscosity `0.01/pi`, approximately `0.00318309886`. Source x coordinates are cell
centers on a periodic length-2 domain. The 201 fields span time 0 through 2; the
source's extra time coordinate at 2.01 has no field and is recorded as discarded.

## Implementation and data processing

1. `HTTPRangeReader` in `http_ranges.py` implements the `seek`, `tell`, `read`, and
   `readinto` operations h5py needs. A loop translates requested byte positions
   into 1 MiB blocks; a four-block cache avoids repeated transfers. Every response
   must be HTTP 206 with the exact byte interval, total size, content length, and
   a consistent strong ETag. Later requests send `If-Match`. Missing range support,
   changed objects, truncation, excess requests, and excess bytes stop acquisition.
   TLS certificate verification uses certifi; redirected signed URLs stay in memory.
2. `acquire_pdebench` chooses source rows without replacement before reading their
   values. It opens the remote HDF5 through that reader and loops over one selected
   trajectory at a time. Each finite float32 array is copied into local HDF5.
   It retains the original physical coordinates and records the selection receipt.
3. Publication stages the local file and JSON receipt, hashes both, then renames
   the directory atomically. Incomplete staging is removed on failure. Existing
   acquisitions are never overwritten. `verify-acquisition` detects changed,
   missing, unexpected, or symbolic-link artifacts.
4. `import_acquired_pdebench` verifies that receipt and binds it to the actual local
   file. The canonical importer checks shape and the 16-million-value cap before
   loading fields. It records both local subset row numbers and public source row
   numbers, groups identical initial fields, freezes whole-trajectory splits, then
   writes chunked Zarr one trajectory at a time. Running mean/variance use only
   training rows; splitting happens before generating learning windows.
5. `run_public_study` iterates over all three predeclared FNO seeds, verifies model
   bundles, and saves held-out one-step and autoregressive rollout predictions.
   Persistence predicts an unchanged field on the same targets. FNO checkpoint
   selection uses validation error, including the epoch-zero persistence model.
   PINN uses initial data, the PDE residual, and periodic boundary constraints.

Range hashes verify the bytes actually received; the derived-file SHA-256 verifies
the complete local subset. **Neither verifies the publisher's full-file MD5**,
because most remote bytes were never downloaded. ETag consistency is a transport
identity check, not that full checksum. All manifests are unsigned integrity records.

This is a small workflow demonstration at one viscosity and one frozen split.
Three FNO seeds measure training randomness conditional on that split; they do not
estimate population uncertainty. The per-instance PINN and pretrained FNO have
different learning tasks. Every accuracy metric compares numerical references,
not an exact PDE solution, and the study is not a full PDEBench reproduction.

## Executed evidence

The [version 0.4 report](../reports/public-burgers-0.4.json) comes from clean commit
`29377f3`. Acquisition transferred 49,881,216 bytes in 49 requests. All three FNO
seeds reduced field RMSE versus persistence, but preserved mean velocity less
accurately. One PINN case also reduced field RMSE while retaining a substantial
physics residual. The [stepbook](../../STEPBOOK.md#measured-public-data-result)
explains those results and why lower prediction error alone is insufficient.
