# 결과 원자료

| 경로 | 내용 |
|---|---|
| `e0_baseline/results.json` | E0 두 모델의 rollout별 성공, Wilson 95% CI, 평균 ± 표준편차(ddof 1), \|yaw\| ≤ 90° / > 90° 합산 |
| `e0_baseline/rollouts/seed{101,111}/trial_*.npz` | E0 평가 200회의 매 step 기록(관측 평가기 출력). EEF, 그리퍼, 트럭·상자 자세, 행동, 성공 판정 값 |
| `e0_baseline/stage_v31.json` | rollout별 실패 단계(v3.1 분류기) |
| `e0_baseline/close_precision.{json,txt}` | 닫는 순간 손목 방향 오차, 트럭 기준 위치, 잡는 자세(~0° / ~180° / 그 사이) |
| `e0_baseline/symmetry_nn_prediction.{json,txt}` | Mimic 원본 선택의 yaw 차이 계산(stock, 대칭, 접기, 같은 자세만) |
| `e0_baseline/source40_convention.txt` | 원본 40개의 잡는 자세 규칙 준수율 |
| `e0_baseline/manual_validation/` | 단계 분류기 수동 검증. 표본 키와 사람 라벨(두 번) |
| `h1_fold_pilot/` | 기각된 H1(트럭 프레임 접기)의 생성 파일럿 통과 조건 결과와 생성 데이터 규칙 준수율 |
| `plots/` | README 그래프 |
| `media/` | 과정별 영상(MP4)과 README 미리보기(GIF), `media_manifest.json`(원본 영상과 구간) |
| `SHA256SUMS` | 이 저장소에서 추가한 파일의 SHA-256 |

E0 평가는 stock `play.py`와 관측 평가기의 rollout별 성공 vector가 두 모델 모두 100/100 같았습니다.
