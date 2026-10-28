# 재현 방법

모든 명령은 저장소 루트에서 실행합니다. 기록 환경은 Isaac Sim 5.1.0, Isaac Lab 2.3.0, robomimic 0.4.0, GeForce RTX 5080(16 GB) 1개입니다. Isaac Sim(Kit)을 사용하는 명령은 한 번에 하나만 실행하십시오.

```bash
export PYTHONUNBUFFERED=1
export TERM=xterm
mkdir -p work/{targeted,natural,corrections,generation,analysis,videos}
```

공개 저장소에는 원본·확장 시범, 교정 데이터, 평가 rollout, 최종 가중치가 들어 있습니다. 용량이 큰 Mimic 생성 데이터 E0/E1/E2/E4는 포함하지 않았습니다.

## 1. 원본 40개 수집

사람 키보드 시범 대신 스크립트 제어기를 사용했습니다. 제어기는 트럭 위에서 수직 접근하고, 트럭 yaw를 (−90°, 90°]로 접은 방향으로 손목을 맞춘 뒤 집어 상자에 놓습니다.

```bash
./isaaclab.sh -p scripts/tools/record_toy2box_scripted_demos.py \
  --control_mode smooth --top_down --headless \
  --dataset_file ./work/source40_candidates.hdf5 \
  --stats_file ./work/source40_candidates.stats.json \
  --attempt_log ./work/source40_attempts.jsonl \
  --num_demos 40 --max_attempts 350 --max_steps 1600 \
  --max_final_xy 0.055 --seed 42
```

수집기는 환경 성공 판정과 `scripts/tools/toy2box_quality.py`의 속도·가속도·jerk·각속도·반전 제한을 모두 통과한 시범만 내보냅니다. 실제 최종 40개는 본 수집과 별도 난수 구간의 보충 수집에서 같은 품질 계약으로 골랐습니다. 공개본은 `datasets/toy2box_agibot_codex_smooth.hdf5`입니다.

Mimic 자동 annotation:

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0 \
  --auto --headless \
  --input_file ./datasets/toy2box_agibot_codex_smooth.hdf5 \
  --output_file ./work/source40_annotated.hdf5
```

## 2. T20·N20 추가 시범과 source60 구성

### T20: yaw 구간별 5개

T20은 (−90°, 90°]를 `[−90,−45)`, `[−45,0)`, `[0,45)`, `[45,90]`로 나누고 각 구간에 최종 5개를 배정합니다. 첫 수집 round는 다음과 같습니다.

```bash
./isaaclab.sh -p scripts/tools/record_toy2box_scripted_demos.py \
  --control_mode smooth --top_down --headless \
  --dataset_file ./work/targeted/round00.hdf5 \
  --stats_file ./work/targeted/round00.stats.json \
  --attempt_log ./work/targeted/round00.attempts.jsonl \
  --num_demos 20 --max_attempts 400 --max_steps 1600 \
  --max_final_xy 0.055 --seed 3611 --rng_skip_draws 0 \
  --yaw_bin_quotas 5 5 5 5
```

annotation 뒤 남은 quota만 다음 round에 요청했습니다. 사용한 `rng_skip_draws` 순서는 `0, 250000, 500000, 750000, 1000000, 125000, 375000, 625000, 875000`입니다. 각 추가 round에서 `--num_demos`는 남은 quota 합, `--yaw_bin_quotas`는 남은 네 구간 값으로 둡니다.

### N20: yaw 필터 없는 자연 분포 20개

```bash
./isaaclab.sh -p scripts/tools/record_toy2box_scripted_demos.py \
  --control_mode smooth --top_down --headless \
  --dataset_file ./work/natural/round00.hdf5 \
  --stats_file ./work/natural/round00.stats.json \
  --attempt_log ./work/natural/round00.attempts.jsonl \
  --num_demos 20 --max_attempts 400 --max_steps 1600 \
  --max_final_xy 0.055 --seed 3612 --rng_skip_draws 0
```

N20은 `rng_skip_draws` 0, 250000, 500000의 세 round에서 자연 분포 표본을 모았습니다. 각 round의 후보를 아래 형식으로 annotation했습니다.

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0 \
  --auto --headless \
  --input_file ./work/targeted/round00.hdf5 \
  --output_file ./work/targeted/round00_annotated.hdf5
```

최종 선별은 다음 순서였습니다.

1. 수집기의 환경 성공·품질 계약을 통과합니다.
2. `annotate_demos.py --auto` 재생에서 성공한 raw/annotated 쌍만 남깁니다.
3. 시범 지문 중복을 제거합니다.
4. T20은 네 yaw 구간마다 5개, N20은 자연 순서에서 20개를 고릅니다.
5. 원본 40개 뒤에 추가 20개를 붙이고 raw/annotated의 episode 대응과 원본 40개 지문을 확인합니다.

이 재진입형 선별·병합 orchestration은 실행 머신의 RAM guard와 중단 복구를 포함한 **로컬 전용 코드**라 공개 저장소에 넣지 않았습니다. 공개된 `artifacts/h1pp_sources/collection_summary.json`은 round별 시도·성공 수를, `e1_targeted60.manifest.json`과 `e2_natural60.manifest.json`은 선택·출력 지문을 기록합니다. 최종 파일은 다음과 같습니다.

- E1: `datasets/toy2box_agibot_sources_e1_targeted60{,_annotated}.hdf5`
- E2: `datasets/toy2box_agibot_sources_e2_natural60{,_annotated}.hdf5`

T20은 실제 시도 561회에서 수집기 성공 48회였고, 최종 20개는 구간당 5개입니다. N20은 108회에서 수집기 성공 28회였습니다.

## 3. DAgger식 교정 40개

교정 수집기는 E0 정책을 stock `play.py`와 같은 경로로 실행하다가 사전 등록한 파지 실패·낙하·접근 정체를 감지하면 smooth 전문가로 전환합니다. E0 체크포인트는 용량을 줄이기 위해 공개 저장소에 포함하지 않았습니다. 이미 생성된 공개 결과를 재사용하려면 `datasets/toy2box_agibot_corrections40.hdf5`를 사용하십시오.

```bash
# E0 학습 seed 101 정책에서 20개
./isaaclab.sh -p scripts/tools/collect_toy2box_corrections.py \
  --checkpoint <E0_SEED101_CHECKPOINT> --seed 4361 \
  --num_corrections 20 --max_rollouts 1000 \
  --dataset_file ./work/corrections/seed101_20.hdf5 \
  --log_file ./work/corrections/seed101_rollouts.jsonl --headless

# E0 학습 seed 111 정책에서 20개
./isaaclab.sh -p scripts/tools/collect_toy2box_corrections.py \
  --checkpoint <E0_SEED111_CHECKPOINT> --seed 4362 \
  --num_corrections 20 --max_rollouts 1000 \
  --dataset_file ./work/corrections/seed111_20.hdf5 \
  --log_file ./work/corrections/seed111_rollouts.jsonl --headless

python scripts/tools/merge_hdf5_datasets.py \
  --input_files ./work/corrections/seed101_20.hdf5 ./work/corrections/seed111_20.hdf5 \
  --output_file ./work/corrections/corrections40.hdf5
```

실측은 seed 101 정책 123 rollout에서 20개, seed 111 정책 68 rollout에서 20개였습니다. 감지/성공은 파지 실패 66/32, 접근 정체 49/5, 낙하 45/3이었습니다. 건너뜀은 넘어짐 43, XY 범위 밖 5, 다른 물체 위 걸림 1회였습니다. 자세한 rollout 로그는 `artifacts/h3_corrections/`에 있습니다.

## 4. Mimic E1·E2 생성

생성은 같은 D3 설정을 모든 조건에 적용했습니다. 아래 함수의 첫 번째 인자는 annotated source60, 두 번째는 chunk seed, 세 번째는 목표 성공 수, 네 번째는 출력 stem입니다.

```bash
gen_chunk () {
  input=$1; seed=$2; goal=$3; stem=$4
  ./isaaclab.sh -p scripts/tools/generate_toy2box_settled.py \
    --settle_reset_retries 2 --settling_log "${stem}.settling.jsonl" \
    --box_robot_distance --enable_box_reselection --zero_action_noise \
    --box_interpolation_steps 30 --settle_arm_motion \
    --settle_arm_joint_velocity 0.15 --settle_max_steps 20 \
    --grasp_offset_range 50 60 --per_env_rmp_reset --chunk_seed "$seed" \
    --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0 \
    --headless --num_envs 100 --generation_num_trials "$goal" \
    --input_file "$input" --output_file "${stem}.hdf5"
}
```

E1은 seed `1, 3, 4, 5, 6, 7, 8, 9, 10`에 목표 50, seed 11에 목표 49를 사용했습니다. 병렬 완료 때문에 seed 10이 51개를 내보내 합계가 정확히 500개가 됐습니다.

```bash
for seed in 1 3 4 5 6 7 8 9 10; do
  gen_chunk ./datasets/toy2box_agibot_sources_e1_targeted60_annotated.hdf5 \
    "$seed" 50 "./work/generation/e1_seed$(printf '%03d' "$seed")"
done
gen_chunk ./datasets/toy2box_agibot_sources_e1_targeted60_annotated.hdf5 \
  11 49 ./work/generation/e1_seed011

python scripts/tools/merge_hdf5_datasets.py \
  --input_files ./work/generation/e1_seed{001,003,004,005,006,007,008,009,010,011}.hdf5 \
  --output_file ./work/toy2box_agibot_mimic_e1_targeted_500.hdf5

python - <<'PY'
import h5py

path = "work/toy2box_agibot_mimic_e1_targeted_500.hdf5"
with h5py.File(path, "r+") as h5_file:
    data = h5_file["data"]
    data.attrs["total"] = sum(int(data[name].attrs["num_samples"]) for name in data)
    assert len(data) == 500
PY
```

E2는 같은 seed/goal 계획에서 499개만 생성됐습니다. 운영 보충으로 seed 12·목표 50 chunk를 만들었고, 51개 중 결정적 병합 순서의 첫 1개만 최종 500개에 포함했습니다.

```bash
for seed in 1 3 4 5 6 7 8 9 10; do
  gen_chunk ./datasets/toy2box_agibot_sources_e2_natural60_annotated.hdf5 \
    "$seed" 50 "./work/generation/e2_seed$(printf '%03d' "$seed")"
done
gen_chunk ./datasets/toy2box_agibot_sources_e2_natural60_annotated.hdf5 \
  11 49 ./work/generation/e2_seed011
gen_chunk ./datasets/toy2box_agibot_sources_e2_natural60_annotated.hdf5 \
  12 50 ./work/generation/e2_seed012

python scripts/tools/merge_hdf5_datasets.py \
  --input_files ./work/generation/e2_seed{001,003,004,005,006,007,008,009,010,011,012}.hdf5 \
  --output_file ./work/e2_merged_550.hdf5
```

공개 저장소에는 생성 당시의 실패 HDF5와 정착 로그가 없으므로, 로컬 finalizer의 전체 fingerprint·실패 데이터 감사는 그대로 재실행할 수 없습니다. 아래 단계는 stock merger가 만든 성공 파일의 앞 500개를 같은 순서로 남기고 `total`을 복원합니다. 기존 출력을 덮어쓰지 않습니다.

```bash
python - <<'PY'
from pathlib import Path
import h5py

src = Path("work/e2_merged_550.hdf5")
dst = Path("work/toy2box_agibot_mimic_e2_natural_500.hdf5")
if dst.exists():
    raise FileExistsError(dst)
with h5py.File(src, "r") as source, h5py.File(dst, "x") as target:
    source.copy("data", target)
    data = target["data"]
    for index in range(500, len(data)):
        del data[f"demo_{index}"]
    data.attrs["total"] = sum(int(data[name].attrs["num_samples"]) for name in data)
with h5py.File(dst, "r") as check:
    assert len(check["data"]) == 500
PY
```

실제 생성 수율과 chunk 계획은 `artifacts/experiments/generation/{e1,e2}_manifest.json`에 있습니다. E1은 500/910, E2는 보충 chunk 전체를 포함해 550/1073입니다.

## 5. E4 데이터와 BC 학습

E4는 E0 Mimic 500개 뒤에 교정 40개를 한 번씩 붙인 540개입니다.

```bash
python scripts/tools/merge_hdf5_datasets.py \
  --input_files ./work/toy2box_agibot_mimic_v1_500.hdf5 \
                ./datasets/toy2box_agibot_corrections40.hdf5 \
  --output_file ./work/toy2box_agibot_e4_mimic500_corr40.hdf5
```

학습 설정은 `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/place/config/agibot/agents/robomimic/bc_rnn_low_dim.json`에서 `seed`만 101 또는 111로 바꿨습니다. 다른 설정은 BC-RNN-GMM, 2000 epoch, batch 100, sequence length 10, epoch당 100 step, learning rate 1e-3으로 고정했습니다.

```bash
# E1/E2/E4 중 하나를 DATASET으로 지정하고, 설정 JSON의 seed만 101 또는 111로 변경합니다.
DATASET=./work/toy2box_agibot_mimic_e2_natural_500.hdf5
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
  --algo bc --dataset "$DATASET"
```

각 학습 뒤 JSON을 원래 seed 101 바이트로 복원했습니다. 최종 선택은 E2이며 공개 가중치는 `deliverables/toy2box_bc/`에 있습니다.

공개 체크포인트는 학습 머신 경로를 제거한 **metadata-sanitized derivative**입니다. 원본에서 `train.data`와 `train.output_dir` 두 config 값만 저장소 상대 경로로 바꿨습니다. tensor storage bytes와 그 밖의 decoded 구조는 원본과 같고, `torch.load`와 robomimic policy load를 확인했습니다. 공개 파일 자체의 SHA-256은 seed 101이 `f0369c5c46f1cb557b564be1bd59c7224c82e467cc3f64607baa3d05771cfe1c`, seed 111이 `6103aa6b8b95099078319a40bd3c0254c6c6ec9ce319f162dd0c74aa111eb4eb`입니다. 원본·공개 SHA와 tensor member 증명은 [`metadata_sanitization.json`](../deliverables/toy2box_bc/metadata_sanitization.json)에 있습니다.

## 6. 평가

강의 형식의 stock `play.py` 명령은 [`EVAL_COMMANDS.txt`](../EVAL_COMMANDS.txt)에 있습니다. 매 step 관측을 저장하는 평가기는 다음과 같이 실행합니다.

```bash
./isaaclab.sh -p scripts/tools/evaluate_toy2box_bc.py \
  --layout_log ./work/analysis/e2_seed101.jsonl \
  --trajectory_dir ./work/analysis/e2_seed101_npz \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
  --checkpoint ./deliverables/toy2box_bc/seed101/model_epoch_2000.pth \
  --num_rollouts 100 --horizon 800 --seed 42 --headless
```

관측 평가 결과는 E2 seed 101이 18/100, seed 111이 22/100입니다. 원본 체크포인트의 stock 영상 평가도 각각 18/100(3682.807초), 22/100(3544.403초)이며 두 seed 모두 관측 평가와 rollout vector가 100/100 정확히 일치했습니다. seed마다 영상 100개를 만들었습니다. 공개 trial 줄과 요약은 [`artifacts/final_eval/`](../artifacts/final_eval/)에 있습니다. 이 stock 실행은 metadata sanitization 전 원본을 사용했습니다. 공개 derivative는 tensor가 같고 robomimic load가 통과했지만, 공개 파일 자체로 100회 전체 rollout을 별도로 반복하지는 않았습니다. 조건 간 정책은 GMM sampling과 reset이 같은 난수열을 공유해 배치가 달라질 수 있으므로 짝지어 비교하지 않았습니다.

## 7. 공개 자료로 분석 재계산

아래 명령은 Isaac Sim을 띄우지 않습니다.

```bash
A=scripts/tools/toy2box_analysis

# 공개 rollout과 등록 결과에서 사전 등록 판정을 독립 재계산
python "$A/verdicts.py" --root . --json ./work/analysis/verdicts.json \
  > ./work/analysis/verdicts.txt
python "$A/summarize_results.py" artifacts/experiments/registered_results.json \
  > ./work/analysis/registered_summary.txt

# 회복과 운반 중 미끄러짐
python "$A/recovery_metrics.py" \
  artifacts/e0_baseline/rollouts/seed101 artifacts/e0_baseline/rollouts/seed111 \
  artifacts/experiments/e1/seed101/rollouts artifacts/experiments/e1/seed111/rollouts \
  artifacts/experiments/e2/seed101/rollouts artifacts/experiments/e2/seed111/rollouts \
  artifacts/experiments/e4/seed101/rollouts artifacts/experiments/e4/seed111/rollouts \
  --json ./work/analysis/recovery_metrics.json > ./work/analysis/recovery_metrics.txt

python "$A/transport_slip.py" \
  E0=artifacts/e0_baseline/rollouts/seed101,artifacts/e0_baseline/rollouts/seed111 \
  E1=artifacts/experiments/e1/seed101/rollouts,artifacts/experiments/e1/seed111/rollouts \
  E2=artifacts/experiments/e2/seed101/rollouts,artifacts/experiments/e2/seed111/rollouts \
  E4=artifacts/experiments/e4/seed101/rollouts,artifacts/experiments/e4/seed111/rollouts \
  --json ./work/analysis/transport_slip.json > ./work/analysis/transport_slip.txt

# 교정 데이터와 정책 rollout의 사전 파지 행동
python "$A/pregrasp_labels.py" \
  --hdf5 corrections40=datasets/toy2box_agibot_corrections40.hdf5 \
  --hdf5 source40=datasets/toy2box_agibot_codex_smooth.hdf5 \
  --npz E2=artifacts/experiments/e2/seed101/rollouts,artifacts/experiments/e2/seed111/rollouts \
  --json ./work/analysis/pregrasp_labels.json > ./work/analysis/pregrasp_labels.txt

python "$A/plot_v2.py" artifacts/experiments/registered_results.json \
  artifacts/mechanism/transport_slip.json ./work/analysis/plots
```

`scripts/tools/toy2box_analysis/registered/`의 분석기는 결과 확인 전에 고정한 원본 판정 코드입니다. 생성 데이터와 생성 시도 파일까지 요구하므로 공개 자료만으로는 그대로 실행할 수 없습니다. `verdicts.py`는 공개된 rollout·등록 결과만으로 같은 판정과 수치를 다시 계산합니다.

## 8. 무결성·focused test

```bash
sha256sum -c artifacts/SHA256SUMS

env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  scripts/tools/tests/test_record_toy2box_scripted_demos.py \
  scripts/tools/tests/test_toy2box_quality.py \
  scripts/tools/tests/test_generate_toy2box_settled.py \
  scripts/tools/tests/test_collect_toy2box_corrections.py \
  scripts/tools/tests/test_toy2box_correction_logic.py \
  scripts/tools/tests/test_evaluate_toy2box_bc.py \
  scripts/tools/tests/test_classify_toy2box_failures.py
```

GPU 물리 시뮬레이션과 GMM sampling 때문에 같은 seed라도 생성·학습 결과가 비트 단위로 같지 않을 수 있습니다. 원자료와 최종 파일의 무결성 기준은 `artifacts/SHA256SUMS`입니다.
