# Copy verified experiments to Google Cloud Storage

The experiment engine already produces files on the local computer. This step
connects those files to the project's Google Cloud Storage bucket. A researcher
can run an experiment, upload its saved evidence, and restore that evidence into
a different local lake for inspection. The restored experiment keeps its original
identity, parameters, results, and execution provenance.

The path through the program is:

```text
JSON settings -> solver -> local experiment lake
                                |
                        verify every saved file
                                |
                      upload artifact objects
                                |
                    publish completion manifest
                                |
                         GCS bucket
                                |
                   download into temporary space
                                |
                 verify bytes and experiment identity
                                |
                      publish local experiment
                                |
                       list / show / query
```

The new work is an artifact mirror. Calculations still happen where the command
runs. Uploading files does not create a cloud computer, run a solver in GCS, or
turn the local SQLite queue into a distributed scheduler.

## Starting point and implementation choices

The existing S3 adapter already needed several behaviors that also apply to GCS:
verify a local experiment, upload files without replacing different evidence,
publish a completion marker last, and restore through a temporary directory.
Those shared steps now live in `src/flowstate/artifact_mirror.py`. The S3 and GCS
adapters supply the provider-specific read, create, and download operations.

This keeps the publication rules in one place. The new
`src/flowstate/gcs_store.py` uses Google's storage library directly. It obtains
credentials from the standard local or cloud environment; it does not put keys
or access tokens into configuration files or experiment provenance.

| File / function | Responsibility |
| --- | --- |
| `cli.py`, `main()` and `_extended_command()` | Read terminal arguments, choose upload or download, print structured output. |
| `gcs_store.py`, `_client()` | Create a Google storage client using Application Default Credentials. |
| `gcs_store.py`, `upload_experiment()` / `download_experiment()` | Connect the shared mirror workflow to the GCS adapter. |
| `artifact_mirror.py`, `upload()` | Verify local evidence, copy artifacts, publish the completion manifest. |
| `artifact_mirror.py`, `download()` | Read the manifest, restore into staging, verify, publish locally. |
| `artifact_mirror.py`, `_manifest()` | Check the manifest's structure, required files, safe paths, and SHA-256 values. |
| `gcs_store.py`, `_GCSStore.put_immutable()` | Create an object only if absent, or verify an existing identical object. |
| `gcs_store.py`, `_read_chunks()` | Read a fixed generation of an object in blocks and compute its SHA-256. |
| `lake.py`, `Lake.verify()` | Check local artifact bytes against the experiment manifest. |

## Files and data layout

One completed local experiment looks like this:

```text
data/lake/experiments/<experiment-id>/
  record.json
  metadata.parquet
  fields.zarr/
    ... array metadata and chunk files ...
  manifest.json
```

`record.json` holds the detailed record: settings, status, measurements, parent
identity where present, and execution provenance. `metadata.parquet` holds a
queryable row derived from that record. `fields.zarr/` contains chunked numerical
arrays, including fields, coordinates, diagnostics, and saved times. Failed runs
can have a record and metadata without a field store.

`manifest.json` lists each artifact's relative path and SHA-256 hash. A hash is a
fingerprint of the file's bytes. Recomputing it lets the program detect a file
that differs from the recorded copy. The manifest does not list its own hash;
the upload workflow calculates that separately to name the remote content path.

For bucket `flowstate-codex` and prefix `flowstate/`, the remote layout is:

```text
gs://flowstate-codex/flowstate/
  artifacts/<experiment-id>/<manifest-sha256>/
    record.json
    metadata.parquet
    fields.zarr/...
  experiments/<experiment-id>/
    manifest.json
```

The objects under `artifacts/` are the actual evidence. The manifest under
`experiments/` is the completion marker. This separation matters because an
interrupted upload may leave some objects behind; readers only restore a run
when its completion marker is present. GCS object names contain slash-separated
prefixes; these are not local filesystem directories.

## Extraction, transformation, and loading

**Extract:** `upload()` opens an existing local lake, verifies the selected
experiment, and reads its manifest. It extracts the list of files to transfer
from that manifest. It does not ask the solver to regenerate data.

**Transform:** The mirror maps local relative paths to remote object names,
normalizes the configured prefix, and computes the manifest hash. Numerical
arrays and Parquet files retain their original bytes. This transfer step does
not recompute metrics, reshape arrays, or merge experiments into a dataset.

The earlier lake-writing step is where numerical results become Zarr arrays and
the detailed record becomes a flattened Parquet row. Keeping these stages
separate means the GCS copy represents the same evidence as the local run.

**Load:** The GCS adapter creates one remote object per artifact and then creates
the manifest object. On restoration, the program loads the files into a
temporary local experiment, verifies them, and publishes the final directory.
The restored Parquet row is immediately available to the local catalog: DuckDB
reads the committed experiments' Parquet files when a query runs. There is no
separate catalog database to upload or update.

## What the loops actually do

The outer upload loop in `artifact_mirror.upload()` goes through the manifest's
files in sorted order. For each file, it makes a private temporary copy under the
lake and checks that copy's SHA-256 before passing it to `put_immutable()`. This
closes the gap between the initial verification and the later network transfer:
an edit to the original file cannot silently create a cloud object containing
bytes that differ from the manifest. Only the checked copy is sent. Temporary
upload copies are automatically cleaned up during normal exception handling.

The loop increments either `uploaded_objects` or `reused_objects` after the
operation. These counts concern artifacts; the separately published manifest is
not included in either count. Processing one temporary copy at a time needs
additional local disk space up to the largest artifact, plus bounded memory for
copying and hashing blocks, rather than a second complete experiment in memory.

`put_immutable()` calls Google's upload operation with
`if_generation_match=0`. This asks GCS to create the object only if no live
object exists at that name. A competing write cannot silently replace its bytes.
The SDK's conditional retry policy uses this condition to make retrying an
eligible failed request safe.

If GCS reports that an object already exists, the adapter downloads and hashes
the existing bytes. Matching bytes are reused. Different bytes stop the upload
with a conflict. The `sha256` object metadata is useful information, but the
program does not trust that metadata as proof of a match. It checks the bytes.
Uploads also request the SDK's CRC32C transport checksum.

The download loop goes through the same artifact list. It creates each needed
parent directory in staging, downloads the file, and compares its SHA-256 with
the expected value. Any mismatch stops restoration before publication.

Inside `_read_chunks()`, a smaller loop reads blocks of 1 MiB until the stream
returns no more bytes. Each block updates the running hash and, on restoration,
is written to disk. A field's full time history does not need to fit in Python
memory just to transfer its chunk files.

Before a streamed read, `blob.reload()` obtains the object's generation. The
read is pinned to that generation and uses a matching-generation precondition.
This prevents a multi-request read from mixing bytes from different object
versions. The small completion manifest is fetched as a single byte result;
the streamed generation-pinned path handles artifact and conflict-check reads.

## Publication, interrupted transfers, and repeated commands

The upload checks whether a remote completion manifest already exists. If it
exists with different bytes, the command refuses to replace it. Otherwise it
checks or creates every artifact, then creates the completion manifest last.
Readers therefore do not treat an unfinished first upload as complete.

Running the same upload again is safe: matching objects are reused. An upload
interrupted after five files can resume by checking those five and creating the
remaining files. This is resumption at the object level, not a persistent local
checkpoint of an upload stream. The returned `resumed` flag means the completion
manifest already existed; a recovered partial upload may have reused objects
while returning `resumed: false` because it publishes that marker for the first
time.

A download writes under a hidden `.staging-download-...` directory inside the
destination lake's experiment directory. It checks every file, verifies the
complete staged experiment, and confirms that `record.json` contains the
requested experiment ID. Only then does it rename the completed experiment to
its final path. Since staging and final storage are on the same filesystem,
this directory publication is atomic: ordinary catalog readers see either no
experiment or the completed directory.

An existing valid local experiment with the same manifest is reused. A different
or corrupt local experiment is not overwritten. Concurrent restoration of the
same content also checks the winning directory before accepting it as a match.
Temporary files are cleaned up in a `finally` block during normal exception
handling. A process killed without cleanup can leave hidden staging files; those
are not committed experiments. Retrying an interrupted download starts a fresh
staging transfer rather than resuming its partially downloaded files.

## Credentials and permissions

This project's identifiers are:

| Setting | Value |
| --- | --- |
| Google Cloud project | `flowstate-510320` |
| Bucket | `flowstate-codex` |
| Object prefix | `flowstate/` |
| Service account | `flowstate@flowstate-510320.iam.gserviceaccount.com` |
| Bucket region | `us-east1` |

The service account needs **Storage Object Creator** and **Storage Object
Viewer** on this bucket to create and read the mirror objects. The local user's
Google account needs **Service Account Token Creator** on this particular
service account to impersonate it. These are different grants: one controls
access to bucket objects; the other allows the user to act as the service
account. The IAM Service Account Credentials API must be enabled for this
impersonation workflow.

Application Default Credentials, or ADC, is the standard place the Python
Google libraries look for credentials. Local setup uses a browser login and
short-lived impersonated access tokens, without downloading a service-account
private key. Do not repeat this login if the local setup is already working.

```powershell
gcloud auth login
gcloud auth application-default login --impersonate-service-account=flowstate@flowstate-510320.iam.gserviceaccount.com
```

`--project flowstate-510320` selects the client project; it does not grant
permissions. On a future Google Cloud worker with an attached service account,
ADC can use that worker identity instead of the local impersonation login.

Official reference: [Service-account impersonation](https://docs.cloud.google.com/docs/authentication/use-service-account-impersonation).

## Run the workflow from PowerShell

Open the project folder, then install the locked dependencies, including the
GCS package. `uv` manages this project's Python environment.

```powershell
cd "C:\Users\yuvis\OneDrive\Documents\ChatGPT\Flowstate"
uv sync --locked --all-extras --group dev
```

Run one small Burgers experiment locally. This example simulates a changing
one-dimensional speed pattern; it generates data without using sensors.

```powershell
uv run --no-sync flowstate --lake data/lake run examples/burgers.json --stream
uv run --no-sync flowstate --lake data/lake list
```

Copy the experiment ID from the output and replace `EXPERIMENT_ID` in the
following commands. `--lake` is a global option and goes **before** `gcs`,
`verify`, or the other command name. The bucket argument is a name without
`gs://`.

```powershell
uv run --no-sync flowstate --lake data/lake verify EXPERIMENT_ID
uv run --no-sync flowstate --lake data/lake gcs upload EXPERIMENT_ID flowstate-codex --prefix flowstate/ --project flowstate-510320
```

The upload output reports the destination, manifest fingerprint, new and reused
artifact counts, and whether the experiment is committed remotely. Repeat the
same upload command to exercise reuse; it should not create a new experiment.

Restore into a different local lake so the test actually reads the cloud copy:

```powershell
uv run --no-sync flowstate --lake data/gcs-restored gcs download EXPERIMENT_ID flowstate-codex --prefix flowstate/ --project flowstate-510320
uv run --no-sync flowstate --lake data/gcs-restored verify EXPERIMENT_ID
uv run --no-sync flowstate --lake data/gcs-restored show EXPERIMENT_ID
uv run --no-sync flowstate --lake data/gcs-restored query "SELECT id, equation, status FROM experiments"
```

`download` restores the evidence, `verify` checks its bytes, `show` displays the
original detailed record, and `query` reads the restored Parquet metadata through
DuckDB. These commands demonstrate a complete local-to-cloud-to-local path.

## Limits and verification record

The mirror currently handles one experiment per command. Separate training
datasets, model checkpoints, research-graph files, and queue databases are not
automatically included. Parent experiment IDs remain in the record, but uploading
one child does not recursively upload its parents. Cloud-wide search, automatic
upload after every run, and distributed workers remain separate features.

The create-only protocol prevents this client from overwriting existing objects.
It is not a bucket retention policy or a guarantee against administrators
deleting or changing data outside the protocol. Unsigned SHA-256 manifests detect
changed files relative to recorded hashes; someone able to replace both the
evidence and its manifest can replace those hashes too. Required retention or
tamper-resistance needs separate IAM and storage-policy decisions. Hash checks
also do not prove that a simulation is physically correct.

An interrupted upload can leave uncommitted artifact objects. This adapter does
not garbage-collect them or delete objects, and the worker's Creator/Viewer roles
do not provide deletion. Reuse checks read existing object bytes, so repeated
uploads still incur reads and transfer work. Data operations may incur cloud
charges according to the bucket's location and billing configuration.

On 4 October 2026, 39 GCS tests passed using an in-memory fake. They exercise
round trips, repeated uploads, conflicting content, edits to source files during
upload, missing or malformed manifests, corrupt downloads, generation-pinned
reads, and terminal command failures. The combined GCS/S3 checks passed 48 tests;
the full project suite passed 427 tests with four Windows symbolic-link permission
skips. Ruff and the source/wheel builds passed.

The native SDK also completed a live round trip from clean commit `9e6fee0` using
the account and bucket above. The standard Burgers example produced experiment
`186ee23716212de6260e3be2838b668a`: 38 artifacts plus the completion manifest,
36,507 bytes total. The first upload created all 38 artifacts; repeating it
created zero and reused 38. A fresh download matched all 39 original files byte
for byte, passed `verify`, appeared in a SQL query, and produced a viewer export.
The [measured report](../reports/gcs-20261004.json) retains the commands, timings,
source fingerprint, and scope. This small transfer validates the application path;
it does not measure large workloads, cloud compute, or real network fault recovery.
