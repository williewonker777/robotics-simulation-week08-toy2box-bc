# 재현 방법

모든 명령은 저장소 루트에서 실행합니다. 환경은 Isaac Sim 5.1.0, Isaac Lab 2.3.0(이 저장소), robomimic 0.4.0입니다.
Isaac Sim은 한 번에 하나만 띄웠고, 모든 실행은 GeForce RTX 5080(16 GB) 1개에서 했습니다.

```bash
export PYTHONUNBUFFERED=1
```

## 1. 원본 시범 수집 (에이전트)

사람 키보드 시범 대신 스크립트 제어기로 모았습니다. 제어기는 트럭 원점 위에서 수직으로 내려와 잡고(손목 방향 = 트럭 yaw를 (−90°, 90°]로 접은 값), 상자 위로 옮겨 놓습니다.

```bash
# 본 수집
./isaaclab.sh -p scripts/tools/record_toy2box_scripted_demos.py --control_mode smooth \
  --dataset_file ./datasets/<새 파일>.hdf5 --num_demos 40 --max_attempts 350 \
  --max_steps 1600 --max_final_xy 0.055 --seed 42 --headless
# 보충 수집 (난수열을 건너뛰어 다른 배치)
./isaaclab.sh -p scripts/tools/record_toy2box_scripted_demos.py --control_mode smooth \
  --dataset_file ./datasets/<보충 파일>.hdf5 --num_demos 12 --max_attempts 70 \
  --max_steps 1600 --max_final_xy 0.055 --seed 42 --rng_skip_draws 1000000 --headless
```

- 수집기는 성공한 시도만 저장합니다. 그 뒤 품질 기준(`scripts/tools/toy2box_quality.py`: 속도·가속도·jerk·각속도 상한, 즉시 반전 0회, 눈에 띄는 반전 횟수 상한)을 통과한 시범에서 40개를 골랐습니다.
- H1″ 수집(E1)은 같은 명령에 `--yaw_bin_quotas` 필터와 seed 3611을 씁니다. 이 옵션은 정착 후 트럭 yaw가 쿼터가 남은 구간에 들 때만 시도하고, 아니면 다시 reset합니다. 시도마다 초기 yaw·x·y를 `--attempt_log`에 남깁니다.

Mimic annotation (자동):

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0 --auto --headless \
  --input_file ./datasets/toy2box_agibot_codex_smooth.hdf5 \
  --output_file ./datasets/toy2box_agibot_codex_smooth_annotated.hdf5
```

## 2. Mimic 데이터 생성

강의의 `generate_dataset.py`는 이 시범들에서 성공 0개였습니다. 생성 시작 때 물체가 아직 흔들리고, RMPflow 상태가 env 사이에 공유되고, 행동 noise와 상자 subtask 원본 선택이 맞지 않았기 때문입니다.
그래서 같은 생성 루프를 쓰되 런타임 보정만 더한 래퍼 `scripts/tools/generate_toy2box_settled.py`를 썼습니다. 환경, 성공 판정, 종료 조건, 관측과 행동은 바꾸지 않습니다.

```bash
# chunk 하나 (env 100개, 성공 50개가 모이면 멈춤)
./isaaclab.sh -p scripts/tools/generate_toy2box_settled.py \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0 --headless \
  --num_envs 100 --generation_num_trials 50 \
  --input_file ./datasets/toy2box_agibot_codex_smooth_annotated.hdf5 \
  --output_file ./datasets/gen/chunk_seed001.hdf5 \
  --settling_log ./datasets/gen/chunk_seed001.settling.jsonl \
  --settle_reset_retries 2 --box_robot_distance --enable_box_reselection --zero_action_noise \
  --box_interpolation_steps 30 --settle_arm_motion --settle_arm_joint_velocity 0.15 \
  --settle_max_steps 20 --grasp_offset_range 50 60 --per_env_rmp_reset --chunk_seed 1

# chunk 10개(seed 1, 3, 4, ..., 11)를 합쳐 정확히 500개
python scripts/tools/merge_hdf5_datasets.py --input_files ./datasets/gen/chunk_seed0{01,03,04,05,06,07,08,09,10,11}.hdf5 \
  --output_file ./datasets/toy2box_agibot_mimic_v1_500.hdf5
```

| 옵션 | 뜻 |
|---|---|
| `--settle_max_steps 20`, `--settle_arm_motion`, `--settle_arm_joint_velocity 0.15`, `--settle_reset_retries 2` | 생성 시작 전에 물체와 팔이 멈출 때까지 기다립니다. 상한에 걸린 env는 최대 2번 다시 reset합니다. |
| `--per_env_rmp_reset` | env마다 RMPflow 제어기 상태를 따로 초기화합니다. |
| `--zero_action_noise` | 생성 행동 noise를 0으로 둡니다. |
| `--box_robot_distance`, `--enable_box_reselection` | 두 번째 subtask(상자)의 원본을 로봇과의 거리로 다시 고릅니다. |
| `--grasp_offset_range 50 60` | 첫 subtask 끝(들어 올림) offset을 50~60 step으로 둡니다. |
| `--box_interpolation_steps 30` | 상자 구간 시작 보간을 30 step으로 둡니다. |

- E0: 933회 시도에서 성공 500개(수율 54%). chunk 1과 3은 `--settle_reset_retries` 옵션을 넣기 전에 만들었습니다.
- E1/E2/E3도 같은 옵션과 같은 chunk seed 순서를 씁니다. E3의 추가 500개는 chunk seed 101부터 씁니다.
- H1(기각)의 접기 옵션은 `--truck_yaw_fold`입니다. 파일럿 결과는 [`artifacts/h1_fold_pilot/`](../artifacts/h1_fold_pilot/)에 있습니다.

## 3. BC 학습

학습 설정 `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/place/config/agibot/agents/robomimic/bc_rnn_low_dim.json`에서 `"seed"`만 101 또는 111로 바꿉니다(BC-RNN-GMM, 2000 epoch, batch 100, 시퀀스 10, epoch당 100 step, lr 1e-3).

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 --algo bc \
  --dataset ./datasets/toy2box_agibot_mimic_v1_500.hdf5
# 체크포인트: logs/robomimic/Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0/bc_rnn_low_dim_agibot_toy2box/<시각>/models/model_epoch_2000.pth
```

GPU 1개에서 2000 epoch에 약 30분 걸렸습니다.

## 4. 평가

강의 명령(stock `play.py`):

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 --checkpoint <model_epoch_2000.pth> \
  --num_rollouts 100 --horizon 800 --video --video_length 1000 \
  --video_folder ./videos/<새 폴더> --seed 42 --headless
```

매 step 상태 기록(관측 평가기). `--layout_log`, `--trajectory_dir`을 뺀 인자는 stock `play.py`에 그대로 넘어갑니다. 기존 출력은 덮어쓰지 않습니다.

```bash
./isaaclab.sh -p scripts/tools/evaluate_toy2box_bc.py \
  --layout_log ./logs/eval/seed101.jsonl --trajectory_dir ./logs/eval/seed101_npz \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 --checkpoint <model_epoch_2000.pth> \
  --num_rollouts 100 --horizon 800 --seed 42 --headless
```

## 5. 분석 (시뮬레이터 없이 실행)

E0 rollout 기록은 [`artifacts/e0_baseline/rollouts/`](../artifacts/e0_baseline/rollouts/)에 있습니다.

```bash
A=scripts/tools/toy2box_analysis; R=artifacts/e0_baseline
python $A/summarize_rollouts.py --label E0 seed101=$R/rollouts/seed101 seed111=$R/rollouts/seed111 --json $R/results.json
python $A/stage_v3.py $R/rollouts/seed101 $R/rollouts/seed111 > $R/stage_v31.json
python $A/close_precision.py $R/rollouts/seed101 $R/rollouts/seed111 --json $R/close_precision.json
python $A/symmetry_nn_prediction.py --json $R/symmetry_nn_prediction.json $R/rollouts/seed101 $R/rollouts/seed111
python $A/dataset_convention.py ./datasets/toy2box_agibot_mimic_v1_500.hdf5   # 생성 데이터의 잡는 자세 규칙 준수율
python $A/plot_e0.py $R/stage_v31.json $R/close_precision.json datasets/toy2box_agibot_codex_smooth.hdf5 artifacts/plots
```

도구 테스트(시뮬레이터 없이):

```bash
python -m pytest -q scripts/tools/tests/test_evaluate_toy2box_bc.py \
  scripts/tools/tests/test_classify_toy2box_failures.py scripts/tools/tests/test_generate_toy2box_settled.py
```

같은 seed라도 GPU 물리 시뮬레이션 특성상 생성과 학습 결과가 완전히 같게 나오지 않을 수 있습니다.
