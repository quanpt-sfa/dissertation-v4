GENERATED FILE — DO NOT EDIT
Source: config/pipeline.yaml

# D01–D45 traceability
| ID | Canonical decision | Main specification | Test | Evidence |
|---|---|---|---|---|
| D01 | Ngày dự báo τit | Ngày muộn nhất mà toàn bộ predictor lõi hợp lệ đã khả dụng; mặc định gắn với ngày BCTC kiểm toán công bố. | T001 | availability_registry, evidence_ledger |
| D02 | Chân trời chính | Horizon 12 tháng chỉ áp dụng cho endpoint date-resolved/delayed-verification; S3 không dùng horizon này. | T002 | risk_sets, evaluation_metrics |
| D03 | Tập nguồn J | Chỉ kênh có ngày khả dụng rõ và liên kết firm-year đủ tin cậy. | T003 | raw_audit, channel_measurement_selection |
| D04 | Nguồn neo | Nguồn chính thức hoặc tư pháp có độ xác nhận cao theo tiêu chí. | T004 | raw_audit, l3_pilot_capability |
| D05 | Outcome chính | L1: BJ_i,t,h đã trưởng thành và follow-up đầy đủ. | T005 | l0_l1_inputs, evaluation_metrics |
| D06 | Miền prior/accuracy | Các kịch bản fixed-π và prior Beta theo profile nguồn được đăng ký trước trong l3_scenarios, đưa vào protocol hash và khóa tại P0; neutral_pi_03 là kịch bản chính. | T006 | l3_pilot_capability, channel_measurement_selection |
| D07 | Ngân sách rà soát | Top 5% tập rủi ro. | T007 | utility_scenarios |
| D08 | Learner roster | Elastic-net logistic, random forest, một boosting chính; nhánh Anchor-PU. | T008 | model_artifacts |
| D09 | Tuning budget | Không quá 50 cấu hình hợp lệ mỗi learner/inner fold hoặc ngân sách tương đương. | T009 | model_freeze_receipt |
| D10 | Outer folds | 2020 là initial; 2021–2024 là fully nested nếu maturity cho phép. | T010 | temporal_split_registry |
| D11 | Cổng 2 | Simultaneous 95% interval của ΔAP nằm trên 0; ΔAP ≥ max{0,01; 0,10×APref}; cùng hướng 3/4 fold; yield@5% không giảm quá 5%. | T011 | evaluation_metrics, gate2_verdict |
| D12 | Cổng 3 | Breakpoint trong ±0,10 SD, lệch giữa miền ≤ 0,20 SD; support ≥ 80% mỗi miền và ≥ 10% quan sát mỗi phía; tái lập 3/4 fold và ít nhất hai miền. | T012 | threshold_interaction_results, gate3_verdict |
| D13 | Miền chuyển giao | Sàn, tầng kiểm toán viên và độ bao phủ. | T013 | domain_transfer_outputs |
| D14 | Vai trò L0–L3 | L0 benchmark, L1 chính, L2 tổng hợp thay thế, L3 measurement sensitivity. | T014 | l0_l1_inputs, measurement_selection_registry |
| D15 | Measurement × learner | So sánh measurement giữ learner; so sánh learner giữ endpoint đánh giá. | T015 | measurement_candidate_results, model_artifacts |
| D16 | Objective L2/L3 | Soft cross-entropy; posterior draws cho L3. | T016 | simulation_scenario_registry, measurement_candidate_results |
| D17 | Mô hình L3 | Fixed-π latent class, source-specific Se/Sp, channel random effect, MCMC. | T017 | l3_pilot_capability, hierarchical_pi_sensitivity |
| D18 | Xác minh chọn lọc | Diagnostic luôn chạy; stabilized IPW khi V quan sát và support đủ. | T018 | observability_registry, fold_aware_weights |
| D19 | Firewall K1–K4 | Niêm phong; không dùng đặt prior, gate, nguồn hoặc tuning. | T019 | known_cases_seal, known_case_results |
| D20 | Biến trong label model | Chỉ S, M, T, Q, ZM; không dùng content predictors. | T020 | feature_registry |
| D21 | Maturity | Maturity theo target: S1/S2 dùng annual anchor; S3 mature khi source year fiscal_year+1 hoàn chỉnh; τ+h≤Tcut chỉ áp dụng cho delayed-verification. | T021 | maturity_audit, risk_sets |
| D22 | Horizon theo nguồn | Tách annual anchor, next-calendar-year cohort và delayed-verification; S3 dự báo regulatory event trong năm dương lịch t+1 và horizon không áp dụng. | T022 | maturity_audit, evaluation_metrics |
| D23 | Tiêu chí cổng | δNI = 5% AP tương đối; MMI = max{0,01; 0,10×APref}; robustness ≥ 80%; FWER α = 0,05; power mục tiêu ≥ 0,80. | T023 | mcse_report, gate2_verdict, gate3_verdict |
| D24 | Interaction library | Hai threshold claims và pressure × monitoring block. | T024 | threshold_interaction_results |
| D25 | Exit/competing events | Không tự động gán âm khi delisting hoặc merger. | T025 | evidence_ledger, risk_sets |
| D26 | Truyền CHNC1 | Tuyến A luôn chạy; Tuyến B dùng M*f hoặc M*f,c theo thiết kế. | T026 | measurement_selection_registry, model_artifacts |
| D27 | Giữ lại nguồn | Nested channel-within-time; chọn M*f,c sau khi loại kênh c khỏi toàn bộ selection. | T027 | channel_time_split_registry, channel_measurement_selection, model_freeze_receipt |
| D28 | Tập dương PU | Anchor-PU chỉ dùng nguồn neo. | T028 | model_artifacts |
| D29 | Đơn vị bất định | Bootstrap theo doanh nghiệp và phân tầng outer year. | T029 | bootstrap_batches |
| D30 | Soft veto known case | Nếu đủ 4 case và 3/4 dưới median trong ≥ 75% scenario, hạ cấp; cả 4 dưới P25 là phản chứng mạnh. | T030 | known_case_results |
| D31 | Metric Tuyến B | Soft loss/concordance cho fit; AP/top-k cho endpoint nhị phân giữ lại. | T031 | evaluation_metrics |
| D32 | Chọn measurement specification | Dùng M*f cho time-only; dùng M*f,c cho strict channel, đều chọn trong development. | T032 | measurement_selection_registry, channel_measurement_selection |
| D33 | Ổn định đo lường | Chỉ tổng hợp hình dạng khi cùng đặc tả được chọn ở ít nhất 3/4 fold. | T033 | measurement_selection_registry, ablation_results |
| D34 | Missingness L2 | Chuẩn hóa trên kênh quan sát và yêu cầu coverage tối thiểu. | T034 | l0_l1_inputs, measurement_candidate_results |
| D35 | Hiệu chuẩn cuối | Calibrator từ pooled cross-fitted development predictions. | T035 | model_freeze_receipt, calibration_outputs |
| D36 | Nhiều phép thử | Hai family CHNC2 và CHNC3; simultaneous intervals/max-T hoặc Holm. | T036 | evaluation_metrics, gate3_verdict |
| D37 | Censoring | Main estimand dùng complete mature follow-up. | T037 | risk_sets, ablation_results |
| D38 | Mục tiêu mô phỏng | Đánh giá L1–L3, selective verification, ba cổng và utility dưới ground truth đã biết. | T038 | simulation_scenario_registry, mcse_report |
| D39 | Hai tầng mô phỏng | Tổng hợp hoàn toàn và bán tổng hợp từ development covariates. | T039 | simulation_scenario_registry |
| D40 | Cơ chế sinh dữ liệu | Sinh Y*, V/O, nguồn Sj, phụ thuộc kênh, độ trễ và dịch chuyển. | T040 | simulation_scenario_registry |
| D41 | Không gian kịch bản | Prevalence, Se/Sp, dependence, verification, delay, shift, cỡ mẫu và signal structure. | T041 | simulation_scenario_registry |
| D42 | Phương pháp mô phỏng | L0–L3, logistic, boosting, observability/content/full và Anchor-PU; oracle chỉ làm chuẩn trên. | T042 | simulation_scenario_registry |
| D43 | Chỉ số mô phỏng | Fixed-π: ranking recovery, misspecification regret và ổn định M*; hierarchical-π: bias/RMSE/coverage của π. Bias Se/Sp chỉ khi được ước lượng. | T043 | mcse_report |
| D44 | Số lần lặp | Kịch bản lõi/phương pháp nhẹ: R ≥ 2.500 và MCSE pass/fail ≤ 0,01. L3: bắt đầu R = 1.000, tăng tuần tự tới MCSE ≤ 0,02. | T044 | mcse_report |
| D45 | Vai trò protocol | Mô phỏng hoàn tất trước outer test; chỉ cho phép hạ địa vị phương pháp hoặc sửa lỗi phương pháp có log. | T045 | registry_lock, mcse_report, outer_open_receipt |
