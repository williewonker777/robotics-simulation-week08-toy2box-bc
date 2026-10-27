# 원본 시범

| 파일 | 내용 |
|---|---|
| `toy2box_agibot_codex_smooth.hdf5` | 원본 시범 40개. 행동, 관측, 시뮬레이션 상태, 초기 상태 |
| `toy2box_agibot_codex_smooth_annotated.hdf5` | 같은 40개에 Mimic 자동 annotation(subtask 경계, `datagen_info`)을 더한 파일. Mimic 생성 입력 |

- 사람 키보드 시범 대신 스크립트 수집 제어기(`scripts/tools/record_toy2box_scripted_demos.py`)가 모은 시범입니다. 수집과 선택 방법은 [재현 문서](../report/REPRODUCE.md#1-원본-시범-수집-에이전트)에 있습니다.
- 20 Hz, 시범 길이 511~721 step입니다. 트럭 초기 yaw 분포: |yaw| > 90° 35개, [−45°, 0°) 2개, [0°, 45°) 3개, ±(45~90)° 0개.
- Mimic 생성 데이터(학습용 500개, 245 MB)는 크기 때문에 올리지 않았습니다. 재생성 명령은 [재현 문서](../report/REPRODUCE.md#2-mimic-데이터-생성)에 있습니다.
- Isaac Lab의 `.gitignore`가 `datasets/`를 제외하므로 이 파일들은 개별로 추가했습니다.
