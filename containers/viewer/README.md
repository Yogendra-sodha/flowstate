# Static viewer container

This package serves an existing `flowstate dashboard` HTML export. The server
loads that file once at startup and never reads a lake, runs experiments, lists
directories, accepts uploads, or uses cloud credentials. Updating the snapshot
requires restarting the container with a newly reviewed export.

Build from the repository root with this narrowly scoped context:

```sh
docker build --tag flowstate-viewer:local containers/viewer
```

Only `server.py` and the Dockerfile enter the build context. The base image tag
must be resolved and its digest retained when an image is actually built; this
package alone is not evidence that an image was built or deployed.

Linux CI pulls the Python base from Docker's
[official ECR Public mirror](https://www.docker.com/press-release/docker-official-images-available-amazon-elastic-container-registry/)
to avoid Docker Hub's shared-runner pull limit. It uses no AWS account or
credentials. CI resolves the base tag to its repository digest, builds with
`--build-arg PYTHON_BASE=<resolved-digest>`, and starts the resulting container
against an actual generated Flowstate export. `smoke.py` checks readiness,
snapshot byte equality, the unprivileged UID, and the closed file routes, then
removes only its own temporary container and staged snapshot. The smoke check
copies the reviewed bytes into a private temporary directory and makes only
that copy readable by the container user. The original export and its permissions
remain unchanged. It prints the built image ID, resolved
base, snapshot hash, and runtime constraints into the job log. CI does not
publish the image to a registry or deploy a service.

Export a snapshot to a new file first:

```sh
uv run --no-sync flowstate --lake data/lake dashboard outputs/research.html
```

On macOS/Linux, exports are private files (mode `0600`). Stage a reviewed copy
with mode `0644` inside a private temporary directory so the container's user can
read its single-file mount. The subshell cleans up only that copy and directory
when the container stops; the original export keeps its permissions:

```sh
(
  viewer_stage=$(mktemp -d "${TMPDIR:-/tmp}/flowstate-viewer.XXXXXX") || exit 1
  trap 'rm -f "$viewer_stage/index.html"; rmdir "$viewer_stage"' EXIT
  install -m 0644 outputs/research.html "$viewer_stage/index.html" || exit 1
  docker run --rm --name flowstate-viewer --read-only --cap-drop ALL --security-opt no-new-privileges --publish 127.0.0.1:8080:8080 --mount "type=bind,source=$viewer_stage/index.html,target=/snapshot/index.html,readonly" flowstate-viewer:local
)
```

PowerShell uses its own current-directory syntax:

```powershell
docker run --rm --name flowstate-viewer --read-only --cap-drop ALL --security-opt no-new-privileges --publish 127.0.0.1:8080:8080 --mount "type=bind,source=$($PWD.Path)/outputs/research.html,target=/snapshot/index.html,readonly" flowstate-viewer:local
```

Open `http://localhost:8080/`. `/index.html` serves the same snapshot and
`/healthz` returns readiness after successful loading. GET and HEAD are supported;
other methods and paths cannot modify or expose local files. The service binds
to `0.0.0.0` inside the container and honors `PORT` (default `8080`). The local
publish command limits host access to loopback. `FLOWSTATE_VIEWER_HTML` can select
another mounted file. Invalid configuration exits before listening.

The HTML retains the exporter's inline scripts and styles. The response policy
permits those and blocks outbound connections, external resources, frames, forms,
and object embeds. This does not make arbitrary HTML trustworthy: serve only a
reviewed Flowstate export. Snapshot contents include experiment provenance and
local source paths; inspect them before sharing.

For a future private Cloud Run deployment, prepare a separate, minimal build
directory containing a reviewed `index.html` and a derived Dockerfile that copies
it into this image at `/snapshot/index.html`. Pin the parent by its image digest
and give the container user ownership explicitly, including when the source
export has mode `0600`:

```dockerfile
FROM <pinned-viewer-image-digest>
COPY --chown=65532:65532 --chmod=0400 index.html /snapshot/index.html
```

The resulting image needs no bucket mount, cloud storage roles, or credentials at
runtime. Cloud Run ingress/IAM and HTTPS must provide access control; this small
stdlib server has no authentication, TLS, or multi-user features. Use bounded
instances/concurrency behind that platform, not a directly exposed internet
server. Image publication and deployment require separate approval. No cloud
deployment is performed by this package.
