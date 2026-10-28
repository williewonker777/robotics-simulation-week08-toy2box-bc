# 최종 Toy2Box BC 모델

사전 등록한 최종 선택 규칙은 지지된 가설이 없을 때 E0·E1·E2·E4 중 200회 합산 성공이 가장 높은 조건을 고릅니다. 결과는 E0 39, E1 31, **E2 40**, E4 14회여서 최종 조건은 E2입니다. E2와 E0의 차이는 단측 두 비율 검정에서 유의하지 않았습니다(`p = 0.450`).

| 학습 seed | 관측 평가(seed 42, 100회) | 공개 파일 SHA-256 |
|---:|---:|---|
| 101 | 18/100 | `f0369c5c46f1cb557b564be1bd59c7224c82e467cc3f64607baa3d05771cfe1c` |
| 111 | 22/100 | `6103aa6b8b95099078319a40bd3c0254c6c6ec9ce319f162dd0c74aa111eb4eb` |

경로:

- `seed101/model_epoch_2000.pth`
- `seed111/model_epoch_2000.pth`

공개 파일은 원본 체크포인트의 metadata-sanitized derivative입니다. config의 `train.data`와 `train.output_dir`만 저장소 상대 경로로 바꿨고, tensor storage bytes 및 나머지 decoded 구조는 원본과 같습니다. `torch.load`와 robomimic policy load도 통과했습니다. 원본과 공개 파일의 전체 SHA는 다릅니다. 원본 SHA, 변경 key, tensor member hash는 [`metadata_sanitization.json`](metadata_sanitization.json)에 기록했습니다.

stock 영상 평가는 metadata sanitization 전 원본 체크포인트로 수행했습니다. seed 101은 18/100, seed 111은 22/100이며 둘 다 관측 평가의 rollout vector와 100/100 정확히 일치했습니다. seed마다 영상 100개를 만들었고, 공개 trial 줄과 요약은 [`artifacts/final_eval/`](../../artifacts/final_eval/)에 있습니다. 공개 derivative는 load 검증을 통과했지만, 공개 파일 자체로 100회 전체 rollout을 별도로 반복하지는 않았습니다.

강의 형식의 stock `play.py` 명령은 [`EVAL_COMMANDS.txt`](../../EVAL_COMMANDS.txt)에 있습니다. 실험 판정 원자료는 [`artifacts/experiments/`](../../artifacts/experiments/), 재현 절차는 [`report/REPRODUCE.md`](../../report/REPRODUCE.md)를 참조하십시오.
