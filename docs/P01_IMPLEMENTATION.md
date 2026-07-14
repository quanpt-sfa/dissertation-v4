# P01 — Raw-source audit

P01 performs one audit per registered source or source partition.

It does not:

- resolve firms;
- construct firm-year panels;
- assign evidence to reporting periods;
- create labels;
- open outer outcomes;
- read known cases.

Those responsibilities begin in P02 or later.

## Before the final P00 lock

Set the machine-specific raw root:

```powershell
$env:DISSERTATION_RAW_ROOT = "D:\Works\dissertation\raw"
```

Create a source profile outside `config/`, then register and hash the source:

```powershell
uv run python tools/p01_register_source.py `
  --data-sources-config config/methodology/data_sources.yaml `
  --source-id financial_panel `
  --source-root $env:DISSERTATION_RAW_ROOT `
  --relative-path "financial/panel.parquet" `
  --profile docs/P01_SOURCE_PROFILE_TEMPLATE.yaml
```

Review the resulting source entry. Then regenerate P0, commit, and create a new clean protocol lock.

## Run P01

```powershell
uv run python scripts/p01_audit_raw.py `
  --registry "D:\path\to\<run_id>\P00\registry.lock.json" `
  --run-id "<run_id>" `
  --source-id "financial_panel"
```

The output is written only through ArtifactStore:

```text
<run_root>/P01/raw_audit/<source_id>.json
<run_root>/P01/raw_audit/<source_id>.json.manifest.json
```

A failed audit is still recorded, then the process exits with code 2.

## Fatal controls

- file hash differs from the locked hash;
- required columns are absent;
- unregistered fields are present under the locked fail-closed policy;
- registered unique keys contain duplicates;
- required date fields contain invalid nonmissing values;
- row count is below the registered minimum.

Fixing a changed raw file requires a new protocol run because its hash is part of the locked source registry.
