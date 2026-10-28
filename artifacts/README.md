# 결과 원자료

## 평가와 판정

| 경로 | 내용 |
|---|---|
| `e0_baseline/results.json` | E0 두 모델의 성공, Wilson 95% CI, 평균 ± 표준편차(ddof 1), yaw 부분집합 집계 |
| `e0_baseline/rollouts/seed{101,111}/trial_*.npz` | E0 평가 200회의 매 step EEF·그리퍼·트럭·상자·행동·성공 기록 |
| `e0_baseline/stage_v31.json` | E0 rollout별 접근·파지·운반·배치·성공 분류 |
| `e0_baseline/close_precision.{json,txt}` | 닫는 순간 손목 방향·위치와 파지 자세 모드 |
| `e0_baseline/symmetry_nn_prediction.{json,txt}` | Mimic 원본 선택의 yaw 차이: stock·대칭·접기·같은 자세만 비교 |
| `e0_baseline/source40_convention.txt` | 원본 40개의 잡는 자세 규칙 준수율 |
| `e0_baseline/manual_validation/` | 단계 분류기 표본과 두 번의 사람 라벨 검증 |
| `experiments/{e1,e2,e4}/seed{101,111}/` | 각 조건의 100회 rollout NPZ, trial JSONL, 결과 JSON |
| `experiments/registered_results.{json,md}` | 결과 확인 전에 고정한 분석 코드의 전체 집계·검정·판정 |
| `experiments/verdicts_recomputed.{json,txt}` | 공개 자료만으로 다시 계산한 판정; 등록 결과와 일치 |
| `experiments/final_selection.json` | H1″ inconclusive, H3 rejected, 최종 E2 선택과 원본 가중치 SHA |
| `experiments/generation/{e1,e2}_manifest.json` | chunk seed·목표·성공·실패·정착 재시도·생성 수율 |
| `final_eval/summary.json` | 최종 E2 두 원본 체크포인트의 stock 영상 평가 결과·소요 시간·observer vector 일치 |
| `final_eval/seed{101,111}_stock_trials.txt` | stock `play.py`의 100개 Trial 줄과 `Trial Results` |

## 수집·기제 분석

| 경로 | 내용 |
|---|---|
| `h1pp_sources/collection_summary.json` | T20·N20 round별 수집 시도, yaw 구간별 성공률, 운영 상한 변경 |
| `h1pp_sources/e1_targeted60.manifest.json` | E1 source60 선택과 원본 40개·출력 지문 |
| `h1pp_sources/e2_natural60.manifest.json` | E2 source60 선택과 원본 40개·출력 지문 |
| `h1pp_sources/e1_pilot_gate*.json` | E1 생성 파일럿 결과와 기록된 진행 결정 |
| `h3_corrections/rollouts_policy_seed{101,111}.jsonl` | 실패 감지, 건너뜀, 전문가 결과를 포함한 교정 수집 로그 |
| `mechanism/recovery_metrics.{json,txt}` | 첫 파지 실패 뒤 재시도·회복 지표 |
| `mechanism/transport_slip.{json,txt}` | 첫 확보 뒤 미끄러짐과 닫는 높이의 사후 분석 |
| `mechanism/pregrasp_labels*.json`, `mechanism/pregrasp_labels.txt` | 사전 파지 영역의 열린 그리퍼 상승 행동 분석 |
| `mechanism/close_precision_{E0,E1,E2,E4}.txt` | 조건별 닫는 순간 방향·위치 요약 |
| `h1_fold_pilot/` | 생성 단계에서 기각한 180° 대칭 접기 파일럿 |

`transport_slip`과 `pregrasp_labels`는 사전 등록 판정에 사용하지 않은 사후 분석입니다. 특히 E4 성능 저하의 원인은 이 자료만으로 특정하지 않았습니다.

## 그림·영상·무결성

| 경로 | 내용 |
|---|---|
| `plots/` | 기준선 진단과 E0/E1/E2/E4 비교 그래프 |
| `media/` | 과정별 MP4·GIF와 원본 구간을 기록한 manifest |
| `SHA256SUMS` | 공개 파일의 SHA-256 목록 |

분석 재계산 명령은 [`report/REPRODUCE.md`](../report/REPRODUCE.md#7-공개-자료로-분석-재계산)에 있습니다. 매 step rollout은 용량이 크므로 저장소 전체 무결성 확인에 시간이 걸릴 수 있습니다.

E0에서는 stock `play.py`와 관측 평가기의 rollout별 성공 vector가 두 학습 seed 모두 100/100 일치했습니다. 최종 E2도 `EVAL_COMMANDS.txt`와 같은 강의 형식으로 원본 체크포인트를 평가했고, seed 101은 18/100, seed 111은 22/100으로 각각 관측 vector와 100/100 일치했습니다. 각 seed에서 영상 100개를 생성했습니다.

최종 모델은 공개 전에 체크포인트 config의 머신 경로 두 개만 저장소 상대 경로로 바꿨습니다. `experiments/final_selection.json`은 평가한 원본 체크포인트 SHA를 유지하고, 공개 derivative SHA와 tensor 동일성 증명은 [`deliverables/toy2box_bc/metadata_sanitization.json`](../deliverables/toy2box_bc/metadata_sanitization.json)에 분리해 기록합니다.
