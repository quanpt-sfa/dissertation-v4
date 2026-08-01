# Chapter 3 production pipeline

Repo này triển khai pipeline luận án từ P00 đến P17 theo protocol khóa trước,
artifact bất biến và access firewall.

## Bắt đầu ở đây

1. Đọc [`AGENTS.md`](AGENTS.md) để biết hợp đồng bắt buộc.
2. Đọc [`docs/AGENT_REPO_MAP.md`](docs/AGENT_REPO_MAP.md) để định vị code và nguồn cấu hình.
3. Đọc [`docs/PIPELINE_P00_P17.md`](docs/PIPELINE_P00_P17.md) để hiểu từng P.
4. Đặt dữ liệu theo [`docs/data_input_layout.md`](docs/data_input_layout.md).

Các tài liệu trong `docs/generated/` được sinh từ `config/pipeline.yaml`; không sửa trực tiếp.

## Lệnh vận hành chuẩn

```powershell
uv run python scripts/run_pipeline.py `
  --run-id dissertation-2015-2026-start-v1 `
  --raw-root "D:\Works\dissertation\dissertation-v4" `
  --output-root "D:\Works\dissertation\dissertation-v4\artifacts\runs" `
  --through P17
```

Pipeline chỉ quét các đường dẫn được khai báo trong source catalog. Nó không tìm dữ
liệu ở repo khác và không tự chọn file thay thế.

## Quality gates

```powershell
uv run python scripts/bootstrap_repository.py --config config/pipeline.yaml --check
uv run pytest -q
uv run ruff check .
uv run pyright
uv run pre-commit run --all-files
```

## Temporal estimands

The analysis unit is firm fiscal year `(i, t)`. S1 and S2 are annual measurements at
the annual prediction anchor. S3 classifies a regulatory-event endpoint in calendar
year `t+1` for firm fiscal year `t`, denoted `B_reg(i,t+1,c)`. S3 target assignment
uses sanction year (then decision year, then publish year) and maps
`sanction_year - 1`; decision and publication dates are provenance/sensitivity
metadata only. S3 is not a post-prediction 12-month outcome, and its maturity
requires a complete source year `t+1` rather than an anchor-plus-horizon calculation.

## Measurement-process contract

Observed evidence is generated through the conceptual sequence
`Y* -> O -> V -> D -> R -> S`: latent fraud, observation opportunity, verification,
determination, recording/publication, and the observed source label. Legal or
administrative provenance is not converted into a high-specificity assumption.
When intermediate layers are unavailable, the pipeline retains a reduced-form
source model and reports the result as a sensitivity region unless restrictions have
independent justification.
