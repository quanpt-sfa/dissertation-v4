# Bản đồ repo cho AI agent

## S3 temporal estimand

S3 is a next-calendar-year regulatory-event target. For firm fiscal year `t`, the
target is an eligible S3 endpoint decision in calendar year `t+1`. The engine maps
`sanction_year` to `target_fiscal_year = sanction_year - 1`; decision and publish
dates are provenance metadata, not target-year filters. S3 maturity requires source
year `t+1` completeness. Only `delayed_verification` sources use date horizons.

Tài liệu này là điểm định hướng cho agent trước khi đọc hoặc sửa code. Nó mô tả
vị trí, quyền sở hữu cấu hình, luồng artifact và cách mở rộng repo mà không phá
protocol.

## 1. Thứ tự đọc bắt buộc

1. `AGENTS.md`: hợp đồng bắt buộc, được sinh tự động.
2. `docs/AGENT_REPO_MAP.md`: bản đồ và quy trình thay đổi.
3. `docs/PIPELINE_P00_P17.md`: đặc tả vận hành từng stage.
4. `docs/generated/STEP_CARDS.md`: reads/writes chính xác từ registry.
5. `docs/generated/ACCESS_MATRIX.md`: quyền đọc/ghi và seal.
6. `docs/generated/ARTIFACT_CATALOG.md`: đường dẫn và schema artifact.
7. `docs/data_input_layout.md`: vị trí dữ liệu thật trong repo.

Không sửa file trong `docs/generated/` hoặc `AGENTS.md`. Hãy sửa module YAML có
quyền sở hữu tương ứng rồi chạy bootstrap.

## 2. Bản đồ cấp cao

```text
dissertation-v4/
|-- AGENTS.md                 # generated agent contract; không sửa trực tiếp
|-- README.md                 # điểm vào cho người và agent
|-- config/                   # nguồn sự thật protocol trước P00
|   |-- pipeline.yaml         # manifest duy nhất của mọi module cấu hình
|   |-- foundation/           # artifact, step, column, access, vocabulary
|   |-- methodology/          # quyết định phương pháp luận
|   |-- execution/            # learner, simulation, weighting, reproducibility
|   |-- assurance/            # Appendix B, decision/test traceability, agent rules
|   `-- schemas/              # dataframe/JSON/receipt contracts
|-- data/                     # dữ liệu do người dùng đặt vào, không commit
|   |-- raw/
|   |-- processed/
|   `-- standardized/
|-- scripts/                  # CLI mỏng và orchestration P00-P17
|-- src/                      # logic thuần, runtime và core I/O
|-- tests/                    # oracle/invariant/access/provenance tests
|-- docs/
|   |-- generated/            # generated; không sửa trực tiếp
|   |-- AGENT_REPO_MAP.md
|   |-- PIPELINE_P00_P17.md
|   `-- data_input_layout.md
|-- artifacts/runs/           # run artifacts bất biến theo run-id
|-- outputs/                  # đầu ra phụ, không phải artifact catalog chính
|-- work/                     # scratch có kiểm soát; không phải nguồn sự thật
|-- pyproject.toml            # dependency và cấu hình tool
`-- uv.lock                   # môi trường Python khóa
```

## 3. Nguồn sự thật và file dẫn xuất

| Nội dung | Nguồn được phép sửa | Dẫn xuất không sửa |
| --- | --- | --- |
| Danh mục stage | `config/foundation/steps.yaml` | `docs/generated/STEP_CARDS.md` |
| Danh mục artifact | `config/foundation/artifacts.yaml` | `docs/generated/ARTIFACT_CATALOG.md` |
| Quyền đọc/ghi | `steps.yaml` + `access_control.yaml` | `docs/generated/ACCESS_MATRIX.md` |
| Schema | `config/schemas/*.yaml` | `docs/generated/SCHEMA_CATALOG.md` |
| Hợp đồng agent | `config/assurance/agent_contract.yaml` | `AGENTS.md` |
| Quyết định D01–D45 | `appendix_b.yaml` + `decisions.yaml` | `D01_D45_TRACEABILITY.md` |
| Test mapping | `config/assurance/tests.yaml` | `docs/generated/TEST_CATALOG.md` |
| Registry runtime | toàn bộ `config/` tại P00 | `P00/registry.lock.json` |

Sau P00, script không đọc lại module YAML riêng lẻ. Tất cả stage phải đọc registry
đã khóa thông qua `core.pipeline.load_run`.

## 4. Cấu trúc `config/`

### `config/foundation/`

- `metadata.yaml`: metadata protocol và phiên bản.
- `vocabulary.yaml`: trạng thái, capability và reason code hợp lệ.
- `columns.yaml`: owner duy nhất của logical-to-physical column binding.
- `artifacts.yaml`: producer, schema, format, coordinates và path template.
- `steps.yaml`: reads, writes, receipts, state và unit coordinates của mỗi P.
- `access_control.yaml`: chính sách seal và sensitivity class.
- `capabilities.yaml`: seed capability trước khi dữ liệu được audit.

### `config/methodology/`

- `study.yaml`: prediction time và horizon.
- `source_catalog.yaml`: vị trí file chính xác dưới `data/`; không tìm ngoài catalog.
- `data_sources.yaml`: channel, source eligibility và anchor policy.
- `entity_resolution.yaml`: firm identity, alias và lịch báo cáo.
- `evidence.yaml`: missingness, lag, exit-event rules và label-model blocks.
- `risksets.yaml`: maturity, complete follow-up, cutoff và IPCW role.
- `folds.yaml`: initial, fully nested và prospective years.
- `measurement.yaml`: vai trò L0–L3 và Gate 1 selection.
- `s3_taxonomy.yaml`: taxonomy endpoint, sanction-year mapping và source-year completeness của S3.
- `features.yaml`: feature registry, role và preprocessing plan.
- `calibration.yaml`: cross-fitted calibration policy.
- `evaluation.yaml`: metrics, review budgets và Gate 2/Gate 3 criteria.
- `inference.yaml`: bootstrap, multiple-testing families và interaction library.
- `utility.yaml`: operational utility scenarios.
- `domains.yaml`: primary/sensitivity transfer domains.
- `known_cases.yaml`: K1–K4 seal và soft-veto rule.

### `config/execution/`

- `learners.yaml`: learner roster và hyperparameter budget.
- `simulation.yaml`: DGP, scenario list, methods và adaptive MCSE controls.
- `weighting.yaml`: IPW/overlap/IPCW roles.
- `reproducibility.yaml`: clean-tree, RNG, environment và recovery policy.

### `config/assurance/`

Đây là lớp truy vết protocol. Thay đổi quyết định phương pháp phải cập nhật đúng
owner và vẫn giữ liên kết Appendix B → decision → test → output evidence.

## 5. Cấu trúc `src/`

| Package | Trách nhiệm | Không được làm |
| --- | --- | --- |
| `core/` | registry compile, schema/access, ArtifactStore, RNG, semantic keys | chứa logic phương pháp đặc thù stage |
| `snapshot/` | inspect file thật, hash/schema/header, build snapshot | tự chọn file ngoài source catalog |
| `p01/` | reader và audit nguồn | ghi artifact bằng pandas trực tiếp |
| `p02/` | entity resolution và as-of panel | mở outer outcomes |
| `evidence/` | P03 ledger, dedup, availability, lag | mã hóa missing thành zero |
| `risksets/` | P04 maturity, prospective, censoring | gán immature thành negative |
| `labels/` | hàm production L1/L2/L3 fixed-pi | dùng content predictors trong label model |
| `measurement/` | P05 matrices/capabilities/fold aggregates | lộ row-level outer labels |
| `observability/` | P06 descriptive verification registry | tạo analytical weight dùng downstream |
| `features/` | P07 feature metadata và leakage audit | fit preprocessing trước fold |
| `simulation/` | P08 DGP, production-label reuse, MCSE | đọc outer outcomes/K1–K4 |
| `splits/` | P09 rolling split và weight diagnostics | fit propensity trên outer rows |
| `selection/` | P10 development-only measurement selection | dùng held-out channel hoặc outer outcome |
| `modeling/` | P11 fold-contained model fit và OOF predictions | đọc sealed outcome store |
| `evaluation/` | P12 calibration, metrics, bootstrap, utility | refit model sau outer open |
| `sensitivity/` | P13 source/domain/censoring summaries | thay đổi selection đã đóng băng |
| `gates/` | P14/P16 locked gate evaluation | tune threshold theo outer result |
| `known_cases/` | P15 case ranks và soft veto | nâng cấp gate thất bại |
| `reporting/` | P17 ledger/report/SVG/table render | import training/label code |

Tên cột logical dùng từ `src/core/semantic_keys.py`. Physical name chỉ được giải
qua registry; không chép literal physical column vào stage code.

## 6. Cấu trúc `scripts/`

- `run_pipeline.py`: orchestration duy nhất cho luồng định kỳ.
- `create_data_snapshot.py`: snapshot các file đúng catalog trước P00.
- `p00_lock_protocol.py`: compile và khóa protocol.
- `p01_*`, `p02_*`: CLI stage và compatibility wrappers.
- `p03_*` đến `p17_*`: CLI mỏng; nghiệp vụ nằm trong `src/<package>/service.py`.
- P08 có coordinator, worker theo batch và collector MCSE riêng.

Script được phép parse CLI và gọi service/core I/O. Không đặt thuật toán lớn,
đường dẫn artifact trực tiếp hoặc `pandas.read_*`/`to_*` artifact I/O trong script.

## 7. Luồng dữ liệu và seal

```mermaid
flowchart LR
    D["data/ theo source catalog"] --> S["Snapshot SHA-256/schema"]
    S --> P00["P00 protocol lock"]
    P00 --> P01["P01 source audits"]
    P01 --> P02["P02 firm-year panel"]
    P02 --> P03["P03 evidence ledger"]
    P03 --> P04["P04 maturity/risk sets"]
    P04 --> P05["P05 labels + sealed outcomes"]
    P05 --> P06["P06 observability"]
    P06 --> P07["P07 features"]
    P07 --> P08["P08 simulation"]
    P08 --> P09["P09 splits/weights"]
    P09 --> P10["P10 measurement selection"]
    P10 --> P11["P11 models + freeze"]
    P11 --> P12["P12 outer open/evaluation"]
    P12 --> P13["P13 sensitivity"]
    P13 --> P14["P14 Gate 2"]
    P14 --> P15["P15 K1-K4 open"]
    P15 --> P16["P16 Gate 3"]
    P16 --> P17["P17 report only"]
```

Seal quan trọng:

- `sealed_outcome_store` được tạo P05 nhưng P10/P11 không được đọc.
- P12 chỉ mở outer outcomes sau `model_freeze_receipt` hợp lệ.
- K1–K4 chỉ mở ở P15 sau Gate 2, freeze và outer-open receipt.
- P17 chỉ đọc artifact hoàn tất; không mở dữ liệu mới.

## 8. Bản đồ artifact runtime

Mỗi run nằm ở `artifacts/runs/<run-id>/`. Mỗi artifact có file dữ liệu và file
`*.manifest.json` chứa protocol hash, producer, schema, coordinates, content hash
và danh sách dependency artifact đã xác minh. Chỉ `ArtifactStore` được đọc/ghi
artifact chính thức.

Coordinates chính:

- P01: `source_id`.
- P08 batch: `scenario_id`, `method_id`, `batch_id`.
- P09 weight: `fold_id`.
- P10–P12 model/evaluation: `outer_fold`.
- Các output còn lại là singleton trong run.

Xem đường dẫn chính xác ở `docs/generated/ARTIFACT_CATALOG.md`.

## 9. Quy trình sửa code an toàn

### Sửa logic một stage

1. Xác định stage và package trong bảng `src/`.
2. Kiểm tra reads/writes ở `config/foundation/steps.yaml`.
3. Sửa service thuần trong `src/`.
4. Giữ CLI trong `scripts/` mỏng.
5. Thêm test invariant/oracle tương ứng.
6. Chạy Ruff, Pyright, focused test, full test và bootstrap check.

### Thêm artifact

1. Khai báo artifact trong `config/foundation/artifacts.yaml`.
2. Chọn hoặc thêm schema trong `config/schemas/`.
3. Thêm vào `writes` của đúng producer và `reads` của consumer.
4. Chỉ ghi/đọc qua `RunContext`/`ArtifactStore`.
5. Chạy bootstrap `--write`, sau đó `--check`.

### Thêm feature

1. Thêm binding đã khóa trong `features.registry`; dùng `intended_registry` khi công thức,
   mapping hoặc denominator còn cần quyết định nghiên cứu.
2. Khai báo feature registry đầy đủ, lineage, availability, role và target/source flags.
3. Content feature bắt buộc có `allowed_in_label_model: false`; unresolved feature không
   được là confirmatory.
4. Không hard-code physical column trong Python; P07 không đọc outer outcomes/K1–K4 và
   không fit preprocessing.

### Thêm hoặc đổi nguồn

1. Chỉ sửa `config/methodology/source_catalog.yaml`.
2. Đặt file đúng `data/`.
3. Tạo run mới để snapshot và protocol hash mới phản ánh thay đổi.
4. Không dùng script đăng ký nguồn thủ công trong luồng định kỳ.

## 10. Kiểm tra trước khi bàn giao

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest -q
uv run python scripts/bootstrap_repository.py --config config/pipeline.yaml --check
uv run pre-commit run --all-files
git diff --check
```

Nếu có thay đổi config, phải chạy:

```powershell
uv run python scripts/bootstrap_repository.py --config config/pipeline.yaml --write
uv run python scripts/bootstrap_repository.py --config config/pipeline.yaml --check
```

## 11. Trạng thái fail-closed cần phân biệt

- `SKIPPED`: capability hoặc input tùy chọn chưa có; không phải PASS giả.
- `INSUFFICIENT_EVIDENCE`: bằng chứng không đủ để gán verdict.
- `FAIL`: tiêu chí đã đủ dữ liệu nhưng không đạt.
- `PASS`: chỉ dùng khi contract và tiêu chí thực sự đạt.
- File/semantic field bắt buộc thiếu: dừng pipeline, không chuyển thành `SKIPPED`.

Các giá trị còn cần người dùng khóa trước run thật thường gồm
`risksets.data_cutoff`, `features.registry`, L2 scoring/coverage, fixed-π L3
grid/priors, learner search spaces, operational simulation/utility scenarios và
hai threshold cùng Gate 3 operational bindings.

## 12. Bản đồ audit/completion

- `IMPLEMENTATION_GAP_MATRIX.md`: baseline D01–D45 và trạng thái closure có bằng chứng.
- `CHAPTER3_REQUIREMENT_TRACEABILITY.md`: requirement → owner → stage → artifact → test.
- `P00_P17_COMPLETION_REPORT.md`: quality gates, fixture evidence, empirical blockers
  và phần vẫn chưa complete. Không suy diễn “complete” từ generated catalog.
