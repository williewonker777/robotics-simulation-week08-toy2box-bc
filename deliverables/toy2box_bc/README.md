# 최종 Toy2Box BC 모델

초기 조건의 200회 합산 성공은 E0 39, E1 31, **E2 40**, E4 14회였습니다. 추가 단일 요인 실험 H4/E6은 27/200(11, 16), H5/E7은 40/200(15, 25)으로 둘 다 결과 전에 고정한 성능 기준을 통과하지 못했습니다. H4·H5는 모두 기각됐고 추가 실험은 실행하지 않았습니다. 통계적으로 확인된 성공률 개선은 없으며 사전 등록 규칙에 따라 최종 조건은 E2입니다. E2와 E0의 차이도 단측 두 비율 검정에서 유의하지 않았습니다(`p = 0.450`).

| 학습 seed | 관측 평가(seed 42, 100회) | 공개 파일 SHA-256 |
|---:|---:|---|
| 101 | 18/100 | `f0369c5c46f1cb557b564be1bd59c7224c82e467cc3f64607baa3d05771cfe1c` |
| 111 | 22/100 | `6103aa6b8b95099078319a40bd3c0254c6c6ec9ce319f162dd0c74aa111eb4eb` |

경로:

- `seed101/model_epoch_2000.pth`
- `seed111/model_epoch_2000.pth`

공개 파일은 원본 체크포인트의 metadata-sanitized derivative입니다. config의 `train.data`와 `train.output_dir`만 저장소 상대 경로로 바꿨고, tensor storage bytes 및 나머지 decoded 구조는 원본과 같습니다. `torch.load`와 robomimic policy load도 통과했습니다. 원본과 공개 파일의 전체 SHA는 다릅니다. 원본 SHA, 변경 key, tensor member hash는 [`metadata_sanitization.json`](metadata_sanitization.json)에 기록했습니다.

metadata sanitization 전 원본 체크포인트의 stock 영상 평가는 seed 101이 18/100, seed 111이 22/100이었고 둘 다 관측 평가의 rollout vector와 100/100 일치했습니다. 최종 검증은 위 표의 공개 derivative와 **정확히 같은 SHA-256 바이트**를 직접 사용했습니다. 결과는 다시 18/100·22/100, seed마다 영상 100개였으며 총 200회 성공 vector가 관측 평가와 모두 일치했습니다. stock `play.py`와 BC config 지문도 이전 stock 검증과 같았습니다. 결과는 [`artifacts/final_eval/summary.json`](../../artifacts/final_eval/summary.json)에 있습니다. 이 확인은 체크포인트 바이트·성공 수·성공 vector에 한정되며 clean install이나 step별 궤적 동등성을 증명하지는 않습니다.

강의 형식의 stock `play.py` 명령은 [`EVAL_COMMANDS.txt`](../../EVAL_COMMANDS.txt)에 있습니다. H4·H5 판정 원자료는 `artifacts/experiments/{h4,h5}/`, 전체 실험 원자료는 [`artifacts/experiments/`](../../artifacts/experiments/), 재현 절차는 [`report/REPRODUCE.md`](../../report/REPRODUCE.md)를 참조하십시오.
