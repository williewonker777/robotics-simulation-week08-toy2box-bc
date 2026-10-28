# 공개 데이터셋

| 파일 | episode | 용도 |
|---|---:|---|
| `toy2box_agibot_codex_smooth.hdf5` | 40 | 스크립트 제어기로 모은 원본 시범 |
| `toy2box_agibot_codex_smooth_annotated.hdf5` | 40 | 원본 40개 + Mimic `obs/datagen_info` |
| `toy2box_agibot_sources_e1_targeted60.hdf5` | 60 | 원본 40 + yaw 네 구간별 5개인 T20 |
| `toy2box_agibot_sources_e1_targeted60_annotated.hdf5` | 60 | E1 Mimic 생성 입력 |
| `toy2box_agibot_sources_e2_natural60.hdf5` | 60 | 원본 40 + yaw 필터 없는 N20 |
| `toy2box_agibot_sources_e2_natural60_annotated.hdf5` | 60 | E2 Mimic 생성 입력 |
| `toy2box_agibot_corrections40.hdf5` | 40 | E0 두 정책에서 각각 20개씩 모은 DAgger식 교정 |

모든 시범은 20 Hz이며 `actions`, `obs`, `states`, `initial_state`를 포함합니다. annotated 파일은 해당 raw 파일과 episode가 대응하고, `datagen_info`는 각 episode의 `obs/` 아래에 있습니다.

원본 40개의 트럭 초기 yaw는 |yaw| > 90°가 35개이고 ±(45~90)°가 0개입니다. E1의 추가 20개는 `[−90,−45)`, `[−45,0)`, `[0,45)`, `[45,90]`에 각각 5개입니다. E2의 추가 20개는 자연 분포입니다.

학습용 Mimic 데이터 E0/E1/E2 500개와 E4 540개는 용량 때문에 포함하지 않았습니다. 생성·병합 명령과 E2의 seed 12 보충 규칙은 [재현 문서](../report/REPRODUCE.md)에 있습니다. source60 선택과 파일 지문은 `artifacts/h1pp_sources/`, 교정 수집 로그는 `artifacts/h3_corrections/`에서 확인할 수 있습니다.

Isaac Lab의 `.gitignore`가 `datasets/`를 제외하므로 공개 HDF5는 명시적으로 추가했습니다. 무결성은 저장소 루트에서 `sha256sum -c artifacts/SHA256SUMS`로 확인합니다.
