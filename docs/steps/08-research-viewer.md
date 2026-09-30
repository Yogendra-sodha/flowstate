# Inspect the experiment lake in a browser

The engine already saves evidence; the viewer makes that evidence easier to read.
It takes an existing local lake, checks its files, extracts small plot arrays, and
writes one self-contained HTML file. Opening that file requires no server, account,
API key, or network connection. The browser never runs a simulation or changes a
saved result. Model-training evaluations and research-graph entities are not yet
part of this viewer.

```sh
uv run --no-sync flowstate run examples/burgers.json --stream
uv run --no-sync flowstate run examples/navier_stokes.json --stream
uv run --no-sync flowstate run examples/darcy.json
uv run --no-sync flowstate --lake data/lake dashboard outputs/research.html
```

Open `outputs/research.html` in a browser. Search a parameter, solver, ID, or commit;
filter by equation, status, or recorded review flag; select a run to inspect its
plots. Expand Parameters, Diagnostic metrics, or Provenance to see the underlying
record. A failed run can be inspected even when it has no saved field. Darcy is
steady, so its zero time coordinate is a storage marker rather than elapsed time.

## Extraction, transformation, and loading

1. **Extract identities.** `export_dashboard()` requires an existing lake. It lists
   committed experiment directories in ascending ID order and selects at most 50
   by default; `--max-experiments` accepts 1 through 200. This is not a recent-runs
   ranking. Counts clearly separate exported records from the lake total.
2. **Verify evidence.** For each selected directory, `Lake.verify()` compares every
   file with its SHA-256 manifest. Any selected corrupt experiment stops the whole
   export. Verification reads all selected compressed files; it is not constant
   time or a signature proving scientific truth.
3. **Extract plots.** `_arrays()` opens Zarr read-only. It selects at most 201 energy
   samples, including both endpoints, then reads the final saved scalar frame:
   velocity for Burgers, vorticity for Navier–Stokes, pressure for Darcy. It never
   decodes the complete time history of fields.
4. **Transform for display.** A 1D preview contains at most 256 spatial points;
   a 2D preview at most 64 by 64. The stride is `ceil(size / limit)`, bounded below
   by one. The same slices select coordinates and values, preserving alignment.
   Nonfinite field/diagnostic values become JSON null and appear as gaps or gray
   cells. Stride sampling can miss features and does not apply an anti-alias filter.
5. **Load the snapshot.** Records, hashes, and previews become embedded JSON capped
   at 32 MiB. Script-sensitive characters are escaped; record strings enter the
   page through `textContent`. Inline JavaScript filters records and draws SVG
   plots. The page has no external assets or network calls.
6. **Publish once.** The exporter writes and flushes a temporary sibling file,
   then creates an exclusive hard link at the requested destination. An existing
   file, symlink, or racing export cannot be overwritten. The destination must be
   outside the entire lake and on a filesystem supporting hard links. A new
   export uses a new filename; old snapshots retain their original records.

The Python loop iterates over selected experiments. Within a run it iterates over
spatial axes to slice matching coordinates. In the browser, `render()` filters the
embedded records, `show()` assembles the selected record, `lineChart()` draws finite
segments, and the nested `j`/`i` loop in `preview()` draws each heatmap cell with y
increasing upwards. Each heatmap uses its own value range; compare its numeric
legend before comparing colors between runs.

## What the plots can establish

These are views of saved evidence, not new numerical validation. The final scalar
field does not show the whole trajectory; a sampled energy curve may miss a brief
event. A completed solver can still be inaccurate, and an absent review flag does
not prove correctness. Inspect the original Zarr arrays and convergence reports
before drawing scientific conclusions. Unsigned hashes detect changed files but
cannot protect against someone replacing both data and manifests.

The HTML includes source paths, parameters, and execution provenance. Inspect it
before sharing it outside your research group. A hosted application with access
control, live updates, graph exploration, and model comparisons remains later work.
