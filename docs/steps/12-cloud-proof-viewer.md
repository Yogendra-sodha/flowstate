# Approved GCS recovery proof and static viewer package

Current evidence: the [prepared Windows plan](../reports/gcs-proof-plan-20261009.json)
awaits user approval. The [viewer validation](../reports/viewer-container-validation-20261009.json)
records a successful Linux container build/run and passing Windows/Linux tests,
including the earlier failed registry attempts. No live execution or deployment
is implied by these preparation results.

Milestone 3 prepares a small disposable local experiment, uploads its immutable
artifacts, verifies the remote bytes, removes that generated local experiment,
downloads it into a fresh lake, and verifies both the restored files and a metadata
query. Human approval is required before the cloud/removal sequence. The static
viewer container and future deployment proposal are separate deliverables.

## Prepare a concrete plan locally

Run from clean committed source, using a new plain local directory:

```sh
uv run --no-sync python -m flowstate.cloud_proof prepare outputs/gcs-proof-new
```

This creates no cloud client. `prepare_proof` runs a fixed small Burgers problem
through the normal engine, verifies the result, inventories each file including
the completion manifest, exports `viewer.html`, and writes an ownership marker
and immutable `plan.json`. The printed SHA-256 binds approval to those exact plan
bytes. The plan records the cloud destination, generated local removal target,
fresh restoration path, provenance, SDK versions, byte/file counts, and bounds.

The default destination is project `flowstate-510320`, bucket `flowstate-codex`,
under a unique `flowstate/proofs/<proof-id>/` prefix. The protocol preserves the
existing mirror layout under that prefix. Nothing is deployed, no bucket is
created, and existing cloud experiments are not changed.

On Windows, prefer a plain directory under LocalAppData if OneDrive uses reparse
points. The implementation rejects symlinks, junctions, and reparse points in the
proof path. The plan is bound to its original absolute path; do not copy it to
another machine and approve it there. Prepare a fresh plan on the Mac instead.

## Approval and execution

Review `plan.json` and obtain approval for credential use, potential charges, and
removal of its exact generated experiment directory. These flags record operator
intent; their presence does not replace human authorization:

```sh
uv run --no-sync python -m flowstate.cloud_proof execute outputs/gcs-proof-new --plan-sha256 PLAN_SHA256 --allow-cloud --allow-local-delete
```

An altered plan, changed source/runtime, existing execution marker or output,
changed local inventory, or missing approval stops before credential use. A
documentation-only commit may advance while approval is pending: package bytes,
runtime and SDK versions must still match. Execution records the current commit;
restored experiment provenance keeps the original generation commit.

The actual program path is:

1. Verify the plan hash, ownership, expected files and allowed paths.
2. Create an exclusive execution marker so a second process cannot run this plan.
3. Upload through the existing native GCS mirror.
4. Repeat the upload: existing remote objects are downloaded and hashed before
   reuse is accepted. Require every expected artifact, no new uploads on that
   pass, and the same completion manifest hash.
5. Recheck local inventory and path bounds. Remove only its enumerated files,
   then its directories from deepest to shallowest. No generic recursive delete
   is applied to a user-selected directory.
6. Require the source experiment to be absent. Download into the new restored lake.
7. Compare complete file inventories and hashes, verify the lake manifest, query
   the restored Parquet metadata, and export `restored-viewer.html`.

`receipt.json` retains success or the failing stage, both upload results, removal
progress, download result, verification and query evidence. `before-removal.json`
and `after-removal.json` survive outside the removed experiment directory. A
partial removal records the exact file list completed before the exception.
Only the generated numerical experiment is removed; the plan, hashes, viewer,
receipts, and any restored result remain. Remote objects are never deleted.

Path checks assume a trusted local directory. They are not a guarantee against
an adversarial process replacing paths between checks. File size/count bounds
are checked before hashing, and hashing streams blocks rather than allocating a
whole untrusted file. Failed numerical experiments, changed inventories, and
unsafe paths are not acceptable inputs to the deletion proof.

## Failure recovery

Execution is deliberately one-shot. Keep the marker and failed receipt; do not
remove them to retry. Before removal, failures preserve the original local run.
After removal, the plan retains the experiment ID, prefix, bucket and original
hash inventory needed for recovery. An independently approved manual restoration
can use a new directory:

```sh
uv run --no-sync flowstate --lake outputs/manual-restore-new gcs download EXPERIMENT_ID flowstate-codex --prefix EXACT_PREFIX_FROM_PLAN --project flowstate-510320
uv run --no-sync flowstate --lake outputs/manual-restore-new verify EXPERIMENT_ID
```

Use the existing source commit's native GCS transfer commands to restore bytes;
do not regenerate data and present that as a successful download. Failed evidence
remains visible. The in-memory SDK tests cover interrupted uploads, corrupt remote
bytes, source changes, failed downloads after removal, partial local deletion,
unsafe paths, approval gates, conflicting output files, and duplicate execution.
Those tests do not count as the real GCS acceptance receipt.

## Viewer container

`containers/viewer/` packages the existing offline HTML export. The stdlib server
loads one reviewed snapshot at startup, serves only the snapshot and readiness
routes, and has no lake, storage, credential, or execution API. The rootless
container accepts a read-only snapshot mount for local use. See its
[build/run instructions](../../containers/viewer/README.md).

Linux CI resolves the base image to a digest, builds the image, runs it with a
read-only filesystem and dropped capabilities, and compares served HTML with an
actual Flowstate export. It checks the runtime user and file-route rejection,
and logs image/base IDs and the snapshot hash. Because exports are private files
on POSIX, the smoke check makes a readable copy inside its own private temporary
directory and mounts only that file. Original bytes and permissions stay intact.
It then removes only its temporary test container, copy, and empty directory.
That validates the package without publishing an image or service.

## Proposed GCP deployment — not executed

The recommended delivery remains a local/offline viewer. If a private hosted
snapshot is later approved, use the following bounded Cloud Run proposal:

| Item | Proposed configuration |
| --- | --- |
| Region | `us-east1`, after confirming the intended resource location. |
| Artifact | Reviewed HTML baked into a derived image with a pinned parent digest; `COPY --chown=65532:65532 --chmod=0400` makes the snapshot readable by its runtime user. |
| Registry | A dedicated Artifact Registry Docker repository; retain the pushed digest. |
| Service | A private Cloud Run service; require authenticated invoker access. |
| Runtime identity | Dedicated viewer service account with no bucket permissions or application API roles. |
| Container | Unprivileged user, snapshot preloaded, `PORT` honored on `0.0.0.0`. |
| Initial limits | Request-based billing, minimum instances 0, maximum instances 1, concurrency 8, 1 CPU, 256 MiB memory. |
| Build | Local or existing CI image build; any Cloud Build use requires separate approval. |
| Data access | Snapshot only; no bucket mount, experiment API, queue, or model execution. |
| Publication | Review snapshot content and exact image digest before a separate deployment approval. |

These are proposed settings, not measurements or a deployment result. The
[Cloud Run container contract](https://docs.cloud.google.com/run/docs/container-contract)
requires the ingress process to listen on the supplied port and interface.
Use [Cloud Run authentication](https://docs.cloud.google.com/run/docs/authenticating/overview)
to restrict access; the container has no login system. Configure limits and assess
[Cloud Run pricing](https://cloud.google.com/run/pricing) before approval.

The GCS proof uses object operations only. Its file and byte bounds constrain the
workload; they do not enforce an invoice cap. Cloud Storage charges can include
stored bytes, requests and data transfer according to the bucket configuration.
See [official storage pricing](https://cloud.google.com/storage/pricing). The
generated objects remain as proof evidence until separately approved cleanup.
Account-wide free credits and free-tier eligibility must not be inferred as a
guarantee that this operation costs nothing. A
[billing budget](https://docs.cloud.google.com/billing/docs/how-to/budgets) supplies
alerts, not an automatic spending stop.

Milestone 3 becomes complete only after a real approved upload/removal/restore
receipt is retained, the viewer package validation is recorded, and this proposal
is available for review. Deployment itself is not part of the acceptance gate.
