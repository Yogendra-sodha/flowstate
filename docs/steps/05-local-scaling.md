# Step 5: Measure local worker and storage scaling

This chapter measures a specific question: on this computer, for this declared sweep, how do worker count and field-storage mode affect elapsed time and observed process memory? The result is a local experiment with retained evidence. It does not choose settings automatically or establish distributed, cloud, or production throughput.

Read [scaling.py](../../src/flowstate/scaling.py) beside the [experiment engine](../../src/flowstate/engine.py). The engine performs the work; the benchmark controls repetition, ordering, measurement, and comparison. Read the [dataset and storage chapter](03-data-pipelines.md) for the artifact layout and the difference between field arrays and metadata.

## 1. Define the experiment before running it

The public function accepts a sweep specification, a new output directory, and the comparison settings:

~~~python
benchmark_sweep(
    spec,
    output,
    workers=(1, 2, 4),
    modes=("buffered", "streamed"),
    repeats=3,
    seed=0,
)
~~~

The command-line equivalent reads a JSON sweep:

~~~sh
uv run --no-sync flowstate sweep-benchmark examples/scaling_sweep.json outputs/local-scaling-01 --workers 1 2 4 --modes buffered streamed --repeats 3 --seed 0
~~~

Run this from the repository root after the environment setup in the [README](../../README.md). The output path must be new. Raw fields and trial artifacts remain under the ignored outputs directory; commit code, declared configurations, and deliberately selected compact reports.

The sweep has a base configuration and lists of parameter values. Expansion forms their Cartesian product, fills defaults, validates each configuration, and deduplicates equivalent requests before measured work starts. Invalid parameters and benchmark budgets must fail before a large set of worker processes is launched.

A worker count is the number of independent experiments permitted to run concurrently. It does not change a solver's grid or timestep. Buffered mode retains saved fields in memory before writing them; streamed mode writes saved fields incrementally. Both must solve the same numerical problem.

For three worker counts, two storage modes, and three repeats, there are eighteen fresh trials. Each trial runs the entire expanded sweep and then separately measures immediate reuse. Thus a small configuration file can still request substantial total computation. Count configurations and trials, estimate output sizes, and inspect the benchmark's bounds before increasing the workload.

The implementation requires one to thirty-two distinct configurations, unique worker
counts including one and no higher than the smaller of eight or the configuration
count, and one to seven repeats. A study is bounded to forty-eight trials, 256 fresh
runs, two million integration steps, and 2 GiB of estimated saved fields. Grids are
capped at 512 points per dimension. These limits apply to this benchmark, not every
engine command; they do not guarantee a wall-clock limit or physical disk capacity.

## 2. Separate workload identity from trial conditions

Two nested structures organize the experiment:

~~~text
scientific workload
  canonical configuration A
  canonical configuration B
  ...

trial condition
  storage mode
  worker count
  repetition number
~~~

The canonical configurations are the scientific input. Mode and worker count describe how that workload is executed. Repetition supplies multiple observations under the same condition.

A simplified control loop is:

~~~python
conditions = []
for mode in modes:
    for worker_count in workers:
        for repetition in range(repeats):
            conditions.append((mode, worker_count, repetition))

shuffle_with_recorded_seed(conditions)

for condition in conditions:
    run_one_isolated_trial(condition)
    retain_trial_evidence()
    verify_numerical_equivalence()
~~~

The for loops enumerate conditions; they do not average or alter PDE parameters. The shuffled order reduces the direct association between a condition and its position in the run. Recording the shuffle seed makes that order reproducible. Shuffling does not remove operating-system activity, thermal changes, filesystem caching, or all order effects.

## 3. Use a fresh process and lake for every trial

Every trial uses a fresh subprocess and its own empty experiment lake. Explicit spawn-based process pools start workers in a controlled way across Windows and other supported platforms.

A fresh lake prevents previous experiment artifacts from turning a supposed compute trial into a reuse trial. A fresh process reduces carryover from Python objects, allocator state, and worker pools left by an earlier condition. This matters when comparing a one-worker call with a process pool: starting the pool is real work and should not disappear merely because a previous measurement already created it.

Fresh processes and fresh paths do not create a cold machine. The operating system may retain source pages, imported libraries, filesystem metadata, and disk blocks in caches. Storage synchronization, antivirus scanning, other applications, and CPU power management remain outside the benchmark's control. Treat the result as a reproducible local procedure, not an isolated hardware laboratory.

The repeated trials retain separate artifacts even when their experiment identifiers match. An identifier describes canonical scientific configuration and recorded code/environment identity; a lake path describes one stored execution of that request. The benchmark needs both concepts to compare equivalent work without accidentally reusing it.

## 4. State exactly what the wall timer includes

The fresh-run measurement covers the complete sweep execution: worker startup, numerical integration, field storage, artifact checksums, and atomic publication. It therefore measures a useful unit of work: producing finalized experiment artifacts that the rest of Flowstate can consume.

It is not a pure solver-kernel measurement. A shorter integration loop can still produce a slower complete run if writing chunks or starting workers dominates. Conversely, a compute-heavy workload may benefit from additional workers even when individual writes become slower.

After the timed run, the benchmark independently verifies artifact hashes and compares physical fields, coordinates, and diagnostics. Those verification and comparison operations are outside the fresh-run wall timer. Do not confuse the two checksum operations:

- Creating the experiment's manifest is part of publishing the result and is timed.
- Independently verifying the published manifest is a benchmark correctness check and is not included in the fresh-run duration.

The same trial also times a separate immediate request for the identical sweep. This is the reuse measurement: existing immutable runs must be verified and returned rather than integrated again. Reuse can still read many files and compute checksums. Its duration is neither zero-cost lookup nor a second independent solver run.

Immediate reuse is especially sensitive to warm caches. Compare fresh-run durations with other fresh-run durations and reuse durations with other reuse durations; do not describe their ratio as parallel solver speedup.

## 5. Interpret summed RSS as a sampled observation

The resource sampler uses [psutil's process API](https://psutil.io/) to observe the trial process and its descendant processes with a requested wait of twenty milliseconds between samples. It sums their resident set sizes, then retains the largest observed sum. Sampling itself adds time, so twenty milliseconds is not a guaranteed sampling period.

RSS is memory resident in a process's address space. Summing RSS across processes includes their Python interpreters, imported libraries, working arrays, and other resident allocations. It can count shared pages more than once. It is not proportional set size, unique physical memory consumption, or a whole-machine memory reading.

Sampling also has limits. A short allocation spike, a short-lived child, or work between samples can escape observation. Process discovery and operating-system counters have their own timing. The reported maximum is the largest sampled sum, not a guaranteed true peak.

The sampler's process tree excludes unrelated programs. For example, a separate filesystem synchronization process may consume memory or I/O while affecting wall time, without appearing in the trial's RSS total. This distinction is relevant for repositories or output directories located inside synchronized folders.

This measurement differs from the earlier single-process storage benchmark's Python allocation tracing. Tracemalloc records traced Python allocations; RSS observes resident process memory. Their numbers have different meanings and should not be placed in one comparison table without explicit labels.

## 6. Verify scientific equivalence before interpreting performance

A faster result is useful only if it still represents the requested calculation. Each trial must finish successfully, pass artifact verification, and agree with the reference trial's physical values.

Comparisons cover the saved field arrays, spatial coordinates, time coordinates, and scalar diagnostic arrays. Chunked or framewise inspection avoids rebuilding every trial's complete field history just to compare it. Exact agreement is an expectation for these executions of the same deterministic numerical implementation in the same environment; it is not a promise of bitwise equivalence across arbitrary machines or future library releases.

Artifact identity, physical equality, and file equality answer different questions:

| Check | What it establishes |
| --- | --- |
| Canonical configuration and experiment identity | The same declared numerical request was executed |
| Manifest verification within each trial | Files agree with that trial's recorded checksums |
| Physical-array comparison across trials | Execution/storage choices preserved the calculated values |
| Whole-file byte comparison | Encoding and metadata bytes are identical as well |

Timestamps and runtime metadata can vary between valid executions. Parquet metadata and manifest hashes can therefore differ while fields remain identical. Storage representation can also affect bytes without changing decoded numbers. The benchmark uses the relevant physical checks rather than requiring every metadata file to have the same bytes.

A failed simulation, failed verification, or unequal result invalidates a performance comparison. Events and trial evidence are retained so the cause can be investigated. The benchmark does not publish a success summary that quietly omits the failed condition.

## 7. Follow the benchmark's own ETL pipeline

The benchmark is another extract-transform-load system:

1. **Extract:** read the sweep specification, canonical configurations, trial outcomes, elapsed durations, sampled RSS, and artifact sizes.
2. **Transform:** associate observations with mode/worker conditions, check completion and equivalence, then aggregate repetitions.
3. **Load:** retain per-trial reports and an append-only event log, followed by the successful aggregate report.

The event log records progress and failures. Trial reports are the evidence behind aggregate numbers. Retaining both makes it possible to distinguish an individual slow trial from a consistent trend.

An output-size estimate helps reject unreasonable work before execution. Measured artifact bytes describe what was actually written, including the consequences of compression and metadata. These are different quantities. An estimate is not a measurement, and a small compressed output does not mean the solver needed little working memory.

Likewise, recorded worker count is a requested execution setting. It does not imply every worker was busy for the entire run. The benchmark caps workers by the number of configurations, but one slow experiment may outlast the rest.

## 8. Read medians, ranges, and speedup carefully

The aggregate report groups trials by storage mode and worker count. Wall time has a median, minimum, and maximum. Throughput, reuse time, and stored bytes have medians; sampled aggregate RSS has a maximum across repetitions. Individual trial reports retain all observations. The median reduces the influence of one extreme observation; the wall-time range keeps variability visible.

With a one-worker result available for the same storage mode, the usual wall-time speedup is:

~~~text
speedup(mode, workers) =
    median fresh-run wall time for (mode, 1 worker)
    ------------------------------------------------
    median fresh-run wall time for (mode, workers)
~~~

Use the matching mode's one-worker baseline. Comparing streamed multiworker execution against buffered serial execution changes two conditions at once. The benchmark requires a one-worker condition so each mode has its own measured baseline.

A speedup below one means the measured condition was slower than its serial baseline. More workers may increase process startup, concurrent disk requests, memory pressure, and scheduling overhead. Three repetitions reveal some variation but do not establish a confidence interval or a broad scaling law.

Inspect absolute duration alongside the ratio. A large ratio on a tiny workload may be less useful than a modest improvement on a representative study. Include correctness, memory, and actual storage costs in the decision.

## 9. Choose settings from evidence, without automatic tuning

Use this sequence when selecting a worker count:

1. Choose a workload representative of the planned study's equation, grids, output frequency, and parameter diversity.
2. Check that every trial completed and all physical comparisons passed.
3. Compare median fresh-run duration within each storage mode and inspect the ranges.
4. Check sampled process memory and artifact bytes against the machine's practical capacity.
5. Prefer a setting whose improvement remains meaningful across repeats; measure again if variability makes the choice unclear.

The benchmark does not automatically rewrite solver configuration, choose a production scheduler, or promise that the best tested setting remains best for another workload. Changing output frequency, storage location, equation, or machine can change the result.

Learning exercises:

- Hold the sweep fixed and compare one, two, and four workers. Identify whether startup or integration appears to dominate, using separate evidence rather than assuming the cause.
- Increase saved-frame frequency while preserving integration parameters. Observe how the storage workload changes.
- Compare buffered and streamed mode at the same worker count. Discuss elapsed time and sampled RSS separately.
- Read the event sequence for a deliberately failing small trial and locate the point where successful aggregation is prevented.

No measured results are invented in this chapter. A measured entry should record the exact command, code provenance, canonical workload, trial order, repetitions, observed values, verification outcomes, and limitations. Subsequent steps can then investigate an actual configured object store or distributed workers using their own explicit measurement contracts.
