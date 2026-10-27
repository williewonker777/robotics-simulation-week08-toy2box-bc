import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "evaluate_toy2box_bc.py"
SPEC = importlib.util.spec_from_file_location("evaluate_toy2box_bc", SCRIPT)
EVALUATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATE)


class FakeValue:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=float)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def __getitem__(self, item):
        return FakeValue(self.value[item])


class FakeObject:
    def __init__(self, position, quaternion):
        self.data = type(
            "Data",
            (),
            {"root_pos_w": FakeValue([position]), "root_quat_w": FakeValue([quaternion])},
        )()


class FakeScene(dict):
    def __init__(self):
        super().__init__(
            toy_truck=FakeObject([10.1, 19.8, 30.5], [1, 0, 0, 0]),
            box=FakeObject([10.3, 20.05, 30.4], [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]),
        )
        self.env_origins = FakeValue([[10, 20, 30]])


class FakeEnv:
    def __init__(self):
        self.unwrapped = self
        self.scene = FakeScene()
        self.reset_calls = []

    def reset(self, *args, **kwargs):
        self.reset_calls.append((args, kwargs))
        return {"policy": "unchanged"}, {"metadata": 7}


class FakeGym:
    def __init__(self, env):
        self.env = env
        self.make_calls = []

    def make(self, *args, **kwargs):
        self.make_calls.append((args, kwargs))
        return self.env


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_execute_observes_original_make_reset_and_prints_then_restores(tmp_path, capsys):
    env = FakeEnv()
    gym = FakeGym(env)
    original_make = gym.make
    original_reset_function = env.reset.__func__
    original_argv = sys.argv
    original_stdout = sys.stdout
    output = tmp_path / "layouts.jsonl"
    seen = {}

    def fake_run_path(path, run_name):
        seen["argv"] = list(sys.argv)
        made = gym.make("Toy2Box", cfg="same")
        seen["same_env"] = made is env
        reset_result = made.unwrapped.reset(seed=42)
        seen["reset_result"] = reset_result
        print("[INFO] Starting trial 0")
        print("[INFO] Trial 0: True\n")
        print("Trial Results: [True]\n")

    EVALUATE.execute_play(
        ["--task", "Toy2Box", "--num_rollouts", "1", "--seed", "42"],
        output,
        gym_module=gym,
        play_script="original_play.py",
        run_path=fake_run_path,
    )

    records = read_jsonl(output)
    trial = records[0]
    assert seen["argv"] == [
        "original_play.py",
        "--task",
        "Toy2Box",
        "--num_rollouts",
        "1",
        "--seed",
        "42",
    ]
    assert seen["same_env"] is True
    assert seen["reset_result"] == ({"policy": "unchanged"}, {"metadata": 7})
    assert env.reset_calls == [((), {"seed": 42})]
    assert trial["record_type"] == "trial"
    assert trial["trial"] == 0
    assert trial["success"] is True
    assert trial["layout"]["toy_truck"]["position_env"] == pytest.approx([0.1, -0.2, 0.5])
    assert trial["layout"]["box"]["yaw_deg"] == pytest.approx(90.0)
    assert records[-1] == {"record_type": "observer_status", "status": "complete", "trials": 1}
    assert "[INFO] Trial 0: True" in capsys.readouterr().out
    assert gym.make == original_make
    assert env.reset.__func__ is original_reset_function
    assert sys.argv is original_argv
    assert sys.stdout is original_stdout


def test_reattaches_after_kit_replaces_stdout_and_records_two_trials(tmp_path):
    env = FakeEnv()
    calls = []

    class SequencedGym(FakeGym):
        def make(self, *args, **kwargs):
            calls.append(("make", args, kwargs))
            return super().make(*args, **kwargs)

    gym = SequencedGym(env)
    original_stdout = sys.stdout
    kit_console = io.StringIO()
    output = tmp_path / "kit-reset-layouts.jsonl"

    def fake_run_path(path, run_name):
        # AppLauncher restores sys.__stdout__ after the wrapper installed its
        # initial tee.  A StringIO models that replacement without Kit.
        sys.stdout = kit_console
        made = gym.make("Toy2Box", cfg="same")
        calls.append(("reset", 0, made.reset(seed=42)))
        print("[INFO] Trial 0: True")
        calls.append(("reset", 1, made.reset()))
        print("[INFO] Trial 1: False")
        print("Trial Results: [True, False]")

    EVALUATE.execute_play(
        ["--num_rollouts", "2", "--seed", "42"],
        output,
        gym_module=gym,
        play_script="original_play.py",
        run_path=fake_run_path,
    )

    records = read_jsonl(output)
    assert [(record["trial"], record["success"]) for record in records[:-1]] == [
        (0, True),
        (1, False),
    ]
    assert records[-1] == {"record_type": "observer_status", "status": "complete", "trials": 2}
    assert kit_console.getvalue() == (
        "[INFO] Trial 0: True\n[INFO] Trial 1: False\nTrial Results: [True, False]\n"
    )
    assert calls == [
        ("make", ("Toy2Box",), {"cfg": "same"}),
        ("reset", 0, ({"policy": "unchanged"}, {"metadata": 7})),
        ("reset", 1, ({"policy": "unchanged"}, {"metadata": 7})),
    ]
    assert env.reset_calls == [(((), {"seed": 42})), ((), {})]
    assert sys.stdout is original_stdout


def test_missing_reset_records_error_and_fails_after_original_returns(tmp_path):
    output = tmp_path / "mismatch.jsonl"
    gym = FakeGym(FakeEnv())

    def fake_run_path(path, run_name):
        gym.make("Toy2Box")
        print("[INFO] Trial 0: False")
        print("Trial Results: [False]")

    with pytest.raises(EVALUATE.ObservationError, match="trial/reset mismatch"):
        EVALUATE.execute_play(
            ["--num_rollouts", "1"], output, gym_module=gym, run_path=fake_run_path
        )

    records = read_jsonl(output)
    assert any(record.get("status") == "error" for record in records)
    assert gym.make("after") is gym.env


def test_no_trial_data_is_explicitly_incomplete_and_state_is_restored(tmp_path):
    output = tmp_path / "no-data.jsonl"
    gym = FakeGym(FakeEnv())
    original_make = gym.make

    with pytest.raises(EVALUATE.ObservationError, match="expected 1 trial results"):
        EVALUATE.execute_play([], output, gym_module=gym, run_path=lambda *args, **kwargs: None)

    assert gym.make == original_make
    assert read_jsonl(output)[-1]["status"] == "error"


def test_complete_marker_is_flushed_before_fast_shutdown_system_exit(tmp_path):
    output = tmp_path / "fast-exit.jsonl"
    gym = FakeGym(FakeEnv())

    def fake_run_path(path, run_name):
        gym.make("Toy2Box").reset()
        print("[INFO] Trial 0: False")
        print("Trial Results: [False]")
        assert read_jsonl(output)[-1] == {
            "record_type": "observer_status",
            "status": "complete",
            "trials": 1,
        }
        raise SystemExit(0)

    with pytest.raises(SystemExit) as raised:
        EVALUATE.execute_play(
            ["--num_rollouts", "1"], output, gym_module=gym, run_path=fake_run_path
        )

    assert raised.value.code == 0
    assert read_jsonl(output)[-1]["status"] == "complete"


def test_final_results_mismatch_stays_error_and_never_completes(tmp_path):
    output = tmp_path / "final-mismatch.jsonl"
    gym = FakeGym(FakeEnv())

    def fake_run_path(path, run_name):
        gym.make("Toy2Box").reset()
        print("[INFO] Trial 0: True")
        print("Trial Results: [False]")

    with pytest.raises(EVALUATE.ObservationError, match="does not match"):
        EVALUATE.execute_play(
            ["--num_rollouts", "1"], output, gym_module=gym, run_path=fake_run_path
        )

    records = read_jsonl(output)
    assert any(record.get("status") == "error" for record in records)
    assert not any(record.get("status") == "complete" for record in records)


def test_missing_final_results_line_stays_incomplete(tmp_path):
    output = tmp_path / "missing-final.jsonl"
    gym = FakeGym(FakeEnv())

    def fake_run_path(path, run_name):
        gym.make("Toy2Box").reset()
        print("[INFO] Trial 0: True")

    with pytest.raises(EVALUATE.ObservationError, match="missing final Trial Results"):
        EVALUATE.execute_play(
            ["--num_rollouts", "1"], output, gym_module=gym, run_path=fake_run_path
        )

    records = read_jsonl(output)
    assert records[-1]["status"] == "error"
    assert not any(record.get("status") == "complete" for record in records)


def test_summary_uses_eight_45_degree_bins_for_both_objects(tmp_path):
    output = tmp_path / "layouts.jsonl"
    records = [
        {
            "record_type": "trial",
            "status": "ok",
            "trial": 0,
            "success": True,
            "layout": {"toy_truck": {"yaw_deg": -179}, "box": {"yaw_deg": 10}},
        },
        {
            "record_type": "trial",
            "status": "ok",
            "trial": 1,
            "success": False,
            "layout": {"toy_truck": {"yaw_deg": 181}, "box": {"yaw_deg": 46}},
        },
        {"record_type": "observer_status", "status": "complete", "trials": 2},
    ]
    output.write_text("".join(json.dumps(record) + "\n" for record in records))

    summary = EVALUATE.summarize_layout_log(output)

    assert summary["attempts"] == 2
    assert summary["successes"] == 1
    assert summary["success_rate"] == 0.5
    assert len(summary["toy_truck_yaw_45deg"]) == 8
    assert summary["toy_truck_yaw_45deg"][0]["attempts"] == 2
    assert summary["toy_truck_yaw_45deg"][0]["successes"] == 1
    assert summary["box_yaw_45deg"][4]["attempts"] == 1
    assert summary["box_yaw_45deg"][5]["attempts"] == 1


def test_summary_refuses_incomplete_or_error_log(tmp_path):
    incomplete = tmp_path / "incomplete.jsonl"
    incomplete.write_text("")
    with pytest.raises(EVALUATE.ObservationError, match="no complete"):
        EVALUATE.summarize_layout_log(incomplete)

    failed = tmp_path / "failed.jsonl"
    failed.write_text(json.dumps({"record_type": "observer_status", "status": "error", "message": "bad"}) + "\n")
    with pytest.raises(EVALUATE.ObservationError, match="bad"):
        EVALUATE.summarize_layout_log(failed)


def test_layout_log_refuses_existing_and_all_protected_paths(tmp_path):
    existing = tmp_path / "existing.jsonl"
    existing.write_text("owned")
    with pytest.raises(FileExistsError):
        EVALUATE.validate_layout_log_path(existing)
    assert existing.read_text() == "owned"

    assert len(EVALUATE.PROTECTED_OUTPUT_PATHS) == 4
    for protected in EVALUATE.PROTECTED_OUTPUT_PATHS:
        with pytest.raises(ValueError, match="protected source-demo"):
            EVALUATE.validate_layout_log_path(protected)


def test_video_folder_must_be_fresh_and_passthrough_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    existing = tmp_path / "existing-videos"
    existing.mkdir()
    arguments = ["--video", "--video_folder", str(existing), "--seed", "42"]

    with pytest.raises(FileExistsError, match="video folder already exists"):
        EVALUATE.validate_video_output(arguments)
    assert arguments == ["--video", "--video_folder", str(existing), "--seed", "42"]

    EVALUATE.validate_video_output(["--video", "--video_folder", str(tmp_path / "fresh")])
    EVALUATE.validate_video_output(["--video_folder", str(existing)])
    for protected in EVALUATE.PROTECTED_OUTPUT_PATHS:
        with pytest.raises(ValueError, match="protected source-demo"):
            EVALUATE.validate_video_output(["--video", f"--video_folder={protected}"])


def test_wrapper_arguments_are_removed_without_reordering_passthrough():
    mode, path, trajectory_dir, passthrough = EVALUATE._extract_wrapper_args(
        [
            "--task",
            "Toy2Box",
            "--layout_log=out.jsonl",
            "--trajectory_dir",
            "steps",
            "--headless",
            "--seed",
            "42",
        ]
    )
    assert mode == "play"
    assert path == Path("out.jsonl")
    assert trajectory_dir == Path("steps")
    assert passthrough == ["--task", "Toy2Box", "--headless", "--seed", "42"]

    with pytest.raises(SystemExit, match="required wrapper argument"):
        EVALUATE._extract_wrapper_args(["--task", "Toy2Box"])


def _policy_observation(offset=0.0):
    return {
        "policy": {
            "eef_pos": FakeValue([[1 + offset, 2, 3]]),
            "eef_quat": FakeValue([[1, 0, 0, 0]]),
            "gripper_pos": FakeValue([[0.1, 0.2]]),
            "toy_truck_positions": FakeValue([[4 + offset, 5, 6]]),
            "toy_truck_orientations": FakeValue([[1, 0, 0, 0]]),
            "box_positions": FakeValue([[7, 8, 9]]),
            "box_orientations": FakeValue([[1, 0, 0, 0]]),
        }
    }


def test_step_observer_writes_aligned_compressed_npz_and_terminal_summary(tmp_path):
    class SteppingEnv(FakeEnv):
        def reset(self, *args, **kwargs):
            self.reset_calls.append((args, kwargs))
            return _policy_observation(), {"metadata": 7}

        def step(self, action):
            return _policy_observation(0.5), 0.0, FakeValue([False]), FakeValue([False]), {}

    env = SteppingEnv()
    gym = FakeGym(env)
    output = tmp_path / "summary.jsonl"
    trajectory_dir = tmp_path / "trajectories"

    def fake_run_path(path, run_name):
        success_term = type("SuccessTerm", (), {})()
        success_term.func = lambda current_env, **kwargs: FakeValue([True])
        success_term.params = {}
        made = gym.make("Toy2Box")
        made.reset()
        action = FakeValue([[0, 1, 2, 3, 4, 5, 6]])
        made.step(action)
        assert bool(success_term.func(made, **success_term.params)[0].value)
        print("[INFO] Trial 0: True")
        print("Trial Results: [True]")

    EVALUATE.execute_play(
        ["--num_rollouts", "1", "--horizon", "800", "--seed", "42"],
        output,
        trajectory_dir,
        gym_module=gym,
        play_script="original_play.py",
        run_path=fake_run_path,
    )

    trial = read_jsonl(output)[0]
    assert trial["trial_index"] == 0
    assert trial["reset_seed"] is None
    assert trial["evaluation_seed"] == 42
    assert "env.reset() without a seed" in trial["reset_semantics"]
    assert trial["final_success"] is True
    assert trial["final_success_value"] is True
    assert trial["timeout_known"] is True
    assert trial["timeout"] is False
    assert trial["terminal_info"]["end_reason"] == "success"

    with np.load(trial["npz_path"], allow_pickle=False) as trajectory:
        assert trajectory["step_index"].tolist() == [0]
        assert trajectory["action"].shape == (1, 7)
        assert trajectory["eef_pos"].tolist() == [[1.5, 2.0, 3.0]]
        assert trajectory["reset_initial_eef_pos"].tolist() == [1.0, 2.0, 3.0]
        assert trajectory["truck_pos"].shape == (1, 3)
        assert trajectory["gripper_pos"].shape == (1, 2)
        assert trajectory["success_value"].tolist() == [True]
        assert trajectory["terminated"].tolist() == [False]
        assert "post-step policy observation" in str(trajectory["timestep_semantics"])


def test_trajectory_output_refuses_existing_directory(tmp_path):
    output = tmp_path / "summary.jsonl"
    trajectory_dir = tmp_path / "trajectories"
    trajectory_dir.mkdir()

    with pytest.raises(FileExistsError, match="trajectory directory already exists"):
        EVALUATE.execute_play([], output, trajectory_dir, gym_module=FakeGym(FakeEnv()))
    assert not output.exists()
