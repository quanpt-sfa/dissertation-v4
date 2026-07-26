GENERATED FILE — DO NOT EDIT
Source: config/pipeline.yaml

# Agent contract
- Read docs/AGENT_REPO_MAP.md and docs/PIPELINE_P00_P17.md before changing a pipeline stage.
- Setup with a Python 3.11+ environment and install PyYAML plus development tools.
- Run python -m pytest -q and python scripts/bootstrap_repository.py --config config/pipeline.yaml --check.
- Run python scripts/bootstrap_repository.py --config config/pipeline.yaml --check before committing configuration changes.
- Configuration has one owner per semantic setting; do not read module YAML after P0.
- Do not use raw physical column strings or direct artifact paths.
- Do not use direct pandas artifact I/O outside the core I/O layer.
- Missing source is not evidence zero; immature follow-up is not negative.
- {'S3 is a next-calendar-year regulatory-event target': 'map sanction year y to fiscal year y-1, and do not use prediction-date or horizon filters to redefine it.'}
- Content predictors cannot enter label models; hierarchical-pi is sensitivity only.
- Outer outcomes and K1-K4 content remain sealed until their configured opening steps.
- Partitioned stages (P01, P08, P10, P11, P12) run units in parallel via --workers N subprocess workers; each worker is single-threaded and writes to its own coordinate path.
- Notebooks do not write artifacts; run required tests before committing.
