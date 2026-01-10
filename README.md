## Pipeline shortcuts

This repo exposes the full people-extraction pipeline through `make` targets.
They rely on [`uv`](https://github.com/astral-sh/uv) (already configured for the
project) and invoke the Typer CLI in `src/cli.py`.

### Full dataset

```bash
make run_full                     # reuse caches when possible
make run_full OVERWRITE=1         # force-rescan of people before downstream steps
make run_full WORKERS=4           # control async scan concurrency
make run_full MATCH_WORKERS=4     # control LLM pairwise matching concurrency
```

The command runs the following stages (with `--workers` and `--match-workers`
wired to the `WORKERS` / `MATCH_WORKERS` variables) in order:

1. `scan-people` — extracts and normalizes people for every note.
2. `cluster` — builds/updates `cache/persons_clusters.json`.
3. `gen-person-notes` — writes Obsidian person notes to `data/obsidian/persons`.
4. `link-persons` — writes linked note copies to `data/obsidian/items`.

### Sample dataset

```bash
make run_sample WORKERS=4 MATCH_WORKERS=2
```

This target always performs a clean run over `data/pages_sample` (with
`SGR_LOGGING=DEBUG` to capture detailed traces):

- temporary caches go to `data/cache_sample`;
- Obsidian output is written to `data/obsidian_sample/{persons,items}`;
- preview images are copied to `data/obsidian_sample/images` from
  `data/obsidian/images` when a matching `<note_id>.jpg` exists.

You can inspect the result end-to-end without touching the main dataset or
cache.
