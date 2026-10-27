#!/usr/bin/env python3
"""Observe Toy2Box robomimic ``play.py`` without changing its rollout logic.

All arguments except ``--layout_log`` and optional ``--trajectory_dir`` are
passed verbatim to the original ``scripts/imitation_learning/robomimic/play.py``.
The observer records the actual post-reset Toy2Box layout and pairs it with the
original script's exact ``[INFO] Trial n: True/False`` output.  When requested,
it also writes one compressed, step-aligned trajectory per trial without
advancing physics or making any additional stochastic call.

An existing output is never overwritten.  ``--summarize_layout_log`` is a
simulation-free mode that reports success rates in 45 degree yaw bins.
"""

from __future__ import annotations

import importlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import runpy
import sys
import types
from typing import Any, Callable, Iterable, TextIO

import numpy as np


PLAY_SCRIPT = Path(__file__).resolve().parents[1] / "imitation_learning" / "robomimic" / "play.py"
TRIAL_RESULT_RE = re.compile(r"^\[INFO\] Trial (\d+): (True|False)$")
FINAL_RESULTS_RE = re.compile(r"^Trial Results: \[(.*)\]$")
YAW_EDGES_DEG = tuple(range(-180, 181, 45))
TIMESTEP_SEMANTICS = (
    "row i stores action[i] passed to env.step and the post-step policy observation returned by that same call; "
    "success_value[i] is the value returned by stock play.py's success predicate immediately after that step; "
    "terminated[i] and truncated[i] are from that env.step return; "
    "reset_initial_* stores the separate post-reset state"
)
STATE_FIELDS = {
    "eef_pos": ("eef_pos", 3),
    "eef_quat": ("eef_quat", 4),
    "gripper_pos": ("gripper_pos", None),
    "truck_pos": ("toy_truck_positions", 3),
    "truck_quat": ("toy_truck_orientations", 4),
    "box_pos": ("box_positions", 3),
    "box_quat": ("box_orientations", 4),
}

# The source-demo files are immutable inputs.  Refusing them as outputs
# prevents an evaluation log from overwriting a demo file.
_REPO_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_OUTPUT_PATHS = (
    _REPO_ROOT / "datasets" / "toy2box_agibot_codex_smooth.hdf5",
    _REPO_ROOT / "datasets" / "toy2box_agibot_codex_smooth_annotated.hdf5",
    _REPO_ROOT / "datasets" / "toy2box_agibot_codex_smooth.manifest.json",
    _REPO_ROOT / "datasets" / "toy2box_agibot_codex_smooth.stats.json",
)


class ObservationError(RuntimeError):
    """Raised after the original play script closes when observation failed."""


def _resolved(path: os.PathLike[str] | str) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_protected_output(path: os.PathLike[str] | str) -> bool:
    return _resolved(path) in {_resolved(item) for item in PROTECTED_OUTPUT_PATHS}


def validate_layout_log_path(path: os.PathLike[str] | str) -> Path:
    """Return a fresh absolute output path or raise without changing the disk."""
    output = Path(path).expanduser()
    resolved = _resolved(output)
    if _is_protected_output(output):
        raise ValueError(f"refusing protected source-demo path: {output}")
    if os.path.lexists(output):
        raise FileExistsError(f"layout log already exists: {output}")
    if not output.parent.exists():
        raise FileNotFoundError(f"layout log parent does not exist: {output.parent}")
    if not output.parent.is_dir():
        raise NotADirectoryError(f"layout log parent is not a directory: {output.parent}")
    return resolved


def validate_trajectory_dir(path: os.PathLike[str] | str) -> Path:
    """Return a fresh trajectory directory path or raise without changing disk."""
    output = Path(path).expanduser()
    resolved = _resolved(output)
    if _is_protected_output(output):
        raise ValueError(f"refusing protected source-demo path: {output}")
    if os.path.lexists(output):
        raise FileExistsError(f"trajectory directory already exists: {output}")
    if not output.parent.exists():
        raise FileNotFoundError(f"trajectory directory parent does not exist: {output.parent}")
    if not output.parent.is_dir():
        raise NotADirectoryError(f"trajectory directory parent is not a directory: {output.parent}")
    return resolved


def _cpu_array(value: Any) -> np.ndarray:
    """Make an observational CPU copy from a tensor-like or array-like value."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.array(value, dtype=np.float64, copy=True)


def _cpu_native_array(value: Any) -> np.ndarray:
    """Copy a tensor-like value without changing its numeric dtype."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.array(value, copy=True)


def quaternion_yaw_deg(quaternion_wxyz: Iterable[float]) -> float:
    """Return wrapped yaw in degrees for a wxyz quaternion."""
    w, x, y, z = (float(item) for item in quaternion_wxyz)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return math.degrees(yaw)


def capture_layout(env: Any) -> dict[str, Any]:
    """Copy the first environment's actual post-reset object poses."""
    origin = _cpu_array(env.scene.env_origins[0]).reshape(-1)[:3]
    layout: dict[str, Any] = {}
    for name in ("toy_truck", "box"):
        data = env.scene[name].data
        position = _cpu_array(data.root_pos_w[0]).reshape(-1)[:3] - origin
        quaternion = _cpu_array(data.root_quat_w[0]).reshape(-1)[:4]
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ObservationError(f"unexpected {name} root-pose shape")
        if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
            raise ObservationError(f"non-finite {name} root pose")
        layout[name] = {
            "position_env": position.tolist(),
            "quaternion_wxyz": quaternion.tolist(),
            "yaw_deg": quaternion_yaw_deg(quaternion),
        }
    return layout


def _single_env_array(value: Any, name: str, width: int | None = None) -> np.ndarray:
    array = _cpu_native_array(value)
    if array.ndim >= 2 and array.shape[0] == 1:
        array = array[0]
    array = array.reshape(-1)
    if width is not None and array.shape != (width,):
        raise ObservationError(f"unexpected {name} shape {array.shape}, expected ({width},)")
    if not array.size or not np.isfinite(array).all():
        raise ObservationError(f"empty or non-finite {name}")
    return array


def capture_policy_state(observation: Any) -> dict[str, np.ndarray]:
    """Copy the seven low-dimensional Toy2Box policy observations for env zero."""
    if not isinstance(observation, dict) or not isinstance(observation.get("policy"), dict):
        raise ObservationError("environment observation has no policy dictionary")
    policy = observation["policy"]
    state: dict[str, np.ndarray] = {}
    for output_name, (policy_name, width) in STATE_FIELDS.items():
        if policy_name not in policy:
            raise ObservationError(f"policy observation is missing {policy_name}")
        state[output_name] = _single_env_array(policy[policy_name], policy_name, width)
    if state["gripper_pos"].size < 2:
        raise ObservationError("unexpected gripper_pos shape; expected at least two values")
    return state


def _single_bool(value: Any, name: str) -> bool:
    array = _cpu_array(value).reshape(-1)
    if array.size != 1:
        raise ObservationError(f"unexpected {name} shape; expected one environment, got {array.shape}")
    return bool(array[0])


class TrialTrajectory:
    """In-memory observations for one rollout; disk writes happen only at trial completion."""

    def __init__(self, initial_state: dict[str, np.ndarray]):
        self.initial_state = initial_state
        self.rows: list[dict[str, Any]] = []
        self.pending_success_row: int | None = None

    def append_step(self, action: Any, step_result: Any) -> None:
        if not isinstance(step_result, tuple) or len(step_result) != 5:
            raise ObservationError("env.step did not return the expected five values")
        observation, _, terminated, truncated, _ = step_result
        action_array = _single_env_array(action, "action", 7)
        row = capture_policy_state(observation)
        row.update(
            action=action_array,
            terminated=_single_bool(terminated, "terminated"),
            truncated=_single_bool(truncated, "truncated"),
            success_value=None,
        )
        self.rows.append(row)
        self.pending_success_row = len(self.rows) - 1

    def observe_success(self, value: Any) -> None:
        if self.pending_success_row is None:
            raise ObservationError("stock success predicate was called without a pending env.step")
        self.rows[self.pending_success_row]["success_value"] = _single_bool(value, "success_value")
        self.pending_success_row = None

    def arrays(self, final_success: bool) -> dict[str, np.ndarray]:
        if self.pending_success_row is not None:
            raise ObservationError("last env.step has no observed stock success value")
        if not self.rows:
            raise ObservationError("trial has no observed env.step calls")
        arrays: dict[str, np.ndarray] = {
            "schema_version": np.asarray(1, dtype=np.int64),
            "timestep_semantics": np.asarray(TIMESTEP_SEMANTICS),
            "state_frame": np.asarray("robot_base_frame (exact stock policy observations)"),
            "quaternion_convention": np.asarray("wxyz"),
            "step_index": np.arange(len(self.rows), dtype=np.int64),
            "final_success": np.asarray(final_success, dtype=np.bool_),
        }
        for name in STATE_FIELDS:
            arrays[name] = np.stack([row[name] for row in self.rows])
            arrays[f"reset_initial_{name}"] = np.array(self.initial_state[name], copy=True)
        arrays["action"] = np.stack([row["action"] for row in self.rows])
        for name in ("success_value", "terminated", "truncated"):
            arrays[name] = np.asarray([row[name] for row in self.rows], dtype=np.bool_)
        return arrays


class JsonlObserver:
    """Pair captured reset layouts with trial-result lines and write immediately."""

    def __init__(
        self,
        stream: TextIO,
        expected_trials: int,
        trajectory_dir: Path | None = None,
        horizon: int = 800,
        reset_seed: int = 101,
    ):
        self.stream = stream
        self.expected_trials = expected_trials
        self.trajectory_dir = trajectory_dir
        self.horizon = horizon
        self.reset_seed = reset_seed
        self.layouts: list[dict[str, Any]] = []
        self.trajectories: list[TrialTrajectory] = []
        self.results: list[bool] = []
        self.trials = 0
        self.errors: list[str] = []
        self.complete = False
        self.final_results_seen = False
        self.error_status_written = False

    def _write(self, record: dict[str, Any]) -> None:
        self.stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self.stream.flush()

    def observe_reset(self, env: Any, reset_result: Any) -> None:
        try:
            self.layouts.append(capture_layout(env))
            if self.trajectory_dir is not None:
                if not isinstance(reset_result, tuple) or len(reset_result) != 2:
                    raise ObservationError("env.reset did not return the expected observation and info")
                self.trajectories.append(TrialTrajectory(capture_policy_state(reset_result[0])))
        except Exception as exc:
            message = f"reset {len(self.layouts)} layout capture failed: {exc}"
            self.errors.append(message)
            if len(self.layouts) == len(self.trajectories):
                self.layouts.append({})
            self._write_error(message)

    def observe_step(self, action: Any, step_result: Any) -> None:
        if self.trajectory_dir is None:
            return
        if len(self.trajectories) != len(self.layouts) or not self.trajectories:
            self._fail("env.step observed without a successfully captured reset")
            return
        try:
            self.trajectories[-1].append_step(action, step_result)
        except Exception as exc:
            self._fail(f"step {len(self.trajectories[-1].rows)} capture failed: {exc}")

    def observe_success(self, value: Any) -> None:
        if self.trajectory_dir is None:
            return
        try:
            self.trajectories[-1].observe_success(value)
        except Exception as exc:
            self._fail(f"success capture failed: {exc}")

    def observe_line(self, line: str) -> None:
        match = TRIAL_RESULT_RE.fullmatch(line)
        if match is not None:
            self._observe_trial_result(int(match.group(1)), match.group(2) == "True")
            return
        final_match = FINAL_RESULTS_RE.fullmatch(line)
        if final_match is not None:
            self._observe_final_results(final_match.group(1))

    def _observe_trial_result(self, trial: int, success: bool) -> None:
        if trial != self.trials or trial >= len(self.layouts):
            message = (
                f"trial/reset mismatch: result trial={trial}, expected={self.trials}, "
                f"captured_resets={len(self.layouts)}"
            )
            self.errors.append(message)
            self._write_error(
                message,
                {
                    "trial": trial,
                }
            )
            self.trials += 1
            return
        layout = self.layouts[trial]
        status = "ok" if layout else "error"
        record = {
            "record_type": "trial",
            "status": status,
            "trial": trial,
            "success": success,
            "layout": layout,
        }
        if self.trajectory_dir is not None:
            try:
                record.update(self._write_trajectory(trial, success))
            except Exception as exc:
                self._fail(f"trial {trial} trajectory write failed: {exc}")
                status = "error"
                record["status"] = status
        self._write(record)
        self.results.append(success)
        self.trials += 1

    def _write_trajectory(self, trial: int, final_success: bool) -> dict[str, Any]:
        if trial >= len(self.trajectories):
            raise ObservationError(f"missing trajectory for trial {trial}")
        trajectory = self.trajectories[trial]
        arrays = trajectory.arrays(final_success)
        if bool(arrays["success_value"][-1]) != final_success:
            raise ObservationError("final stock success value does not match printed trial result")
        path = self.trajectory_dir / f"trial_{trial:05d}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        final_terminated = bool(arrays["terminated"][-1])
        final_truncated = bool(arrays["truncated"][-1])
        horizon_reached = len(trajectory.rows) == self.horizon
        if final_success:
            end_reason = "success"
        elif final_terminated:
            end_reason = "environment_terminated"
        elif final_truncated:
            end_reason = "environment_truncated"
        elif horizon_reached:
            end_reason = "horizon"
        else:
            end_reason = "unknown"
        record = {
            "npz_path": str(path),
            "trial_index": trial,
            "reset_seed": None,
            "evaluation_seed": self.reset_seed,
            "reset_semantics": (
                "stock play.py calls env.seed(evaluation_seed) once, then env.reset() without a seed per trial"
            ),
            "num_steps": len(trajectory.rows),
            "timestep_semantics": TIMESTEP_SEMANTICS,
            "state_frame": "robot_base_frame (exact stock policy observations)",
            "quaternion_convention": "wxyz",
            "reset_initial_state": {
                name: value.tolist() for name, value in trajectory.initial_state.items()
            },
            "final_success": final_success,
            "final_success_value": bool(arrays["success_value"][-1]),
            "terminal_info": {
                "terminated": final_terminated,
                "truncated": final_truncated,
                "horizon": self.horizon,
                "horizon_reached": horizon_reached,
                "end_reason": end_reason,
                "source": "stock env.step return and stock success predicate",
                "stock_time_out_termination_disabled": True,
            },
            "timeout_known": True,
            "timeout": end_reason == "horizon",
        }
        trajectory.rows.clear()
        trajectory.initial_state.clear()
        return record

    def _observe_final_results(self, contents: str) -> None:
        if self.final_results_seen:
            self._fail("duplicate final Trial Results line")
            return
        self.final_results_seen = True
        tokens = [] if not contents.strip() else [item.strip() for item in contents.split(",")]
        if any(item not in ("True", "False") for item in tokens):
            self._fail("invalid final Trial Results boolean list")
            return
        final_results = [item == "True" for item in tokens]
        validation_errors = list(self.errors)
        if len(final_results) != self.expected_trials:
            validation_errors.append(
                f"final Trial Results has {len(final_results)} entries, expected {self.expected_trials}"
            )
        if self.trials != self.expected_trials:
            validation_errors.append(f"expected {self.expected_trials} trial results, observed {self.trials}")
        if len(self.layouts) != self.trials:
            validation_errors.append(f"captured {len(self.layouts)} resets for {self.trials} trial results")
        if self.trajectory_dir is not None and len(self.trajectories) != self.trials:
            validation_errors.append(f"captured {len(self.trajectories)} trajectories for {self.trials} trial results")
        if final_results != self.results:
            validation_errors.append("final Trial Results does not match per-trial results")
        if validation_errors:
            self._fail("; ".join(dict.fromkeys(validation_errors)))
            return
        self._write({"record_type": "observer_status", "status": "complete", "trials": self.trials})
        self.complete = True

    def _fail(self, message: str) -> None:
        self.errors.append(message)
        self._write_error(message)

    def _write_error(self, message: str, extra: dict[str, Any] | None = None) -> None:
        record = {"record_type": "observer_status", "status": "error", "message": message}
        if extra:
            record.update(extra)
        self._write(record)
        self.error_status_written = True

    def finish(self) -> None:
        """Validate on ordinary return; completion itself is written pre-shutdown."""
        if self.complete:
            return
        if not self.final_results_seen:
            self.errors.append("missing final Trial Results line")
        if self.trials != self.expected_trials:
            self.errors.append(f"expected {self.expected_trials} trial results, observed {self.trials}")
        if len(self.layouts) != self.trials:
            self.errors.append(f"captured {len(self.layouts)} resets for {self.trials} trial results")
        message = "; ".join(dict.fromkeys(self.errors))
        if not self.error_status_written:
            self._write_error(message)
        raise ObservationError(message)


class StdoutTee:
    """Transparent stdout proxy that sends complete lines to an observer."""

    def __init__(self, target: TextIO, on_line: Callable[[str], None]):
        self._target = target
        self._on_line = on_line
        self._pending = ""

    def write(self, text: str) -> int:
        written = self._target.write(text)
        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            self._on_line(line.rstrip("\r"))
        return written

    def flush(self) -> None:
        self._target.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


def _patch_reset(env: Any, observer: JsonlObserver) -> Callable[[], None]:
    unwrapped = env.unwrapped
    had_instance_reset = "reset" in getattr(unwrapped, "__dict__", {})
    previous_instance_reset = getattr(unwrapped, "__dict__", {}).get("reset")
    original_reset = unwrapped.reset

    def observed_reset(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_reset(*args, **kwargs)
        observer.observe_reset(self, result)
        return result

    unwrapped.reset = types.MethodType(observed_reset, unwrapped)

    def restore() -> None:
        if had_instance_reset:
            unwrapped.reset = previous_instance_reset
        else:
            del unwrapped.reset

    return restore


def _patch_step(env: Any, observer: JsonlObserver) -> Callable[[], None]:
    unwrapped = env.unwrapped
    had_instance_step = "step" in getattr(unwrapped, "__dict__", {})
    previous_instance_step = getattr(unwrapped, "__dict__", {}).get("step")
    original_step = unwrapped.step

    def observed_step(self: Any, action: Any) -> Any:
        result = original_step(action)
        observer.observe_step(action, result)
        return result

    unwrapped.step = types.MethodType(observed_step, unwrapped)

    def restore() -> None:
        if had_instance_step:
            unwrapped.step = previous_instance_step
        else:
            del unwrapped.step

    return restore


def _patch_success_term(success_term: Any, observer: JsonlObserver) -> Callable[[], None]:
    """Observe the exact predicate call already made by stock ``play.py``."""
    original_func = success_term.func

    def observed_success(*args: Any, **kwargs: Any) -> Any:
        value = original_func(*args, **kwargs)
        observer.observe_success(value)
        return value

    success_term.func = observed_success

    def restore() -> None:
        success_term.func = original_func

    return restore


def _caller_success_term() -> Any | None:
    """Read stock ``main``'s retained success term at its direct ``gym.make`` call."""
    frame = inspect.currentframe()
    try:
        caller = frame.f_back.f_back if frame is not None and frame.f_back is not None else None
        return None if caller is None else caller.f_locals.get("success_term")
    finally:
        del frame


def _expected_rollouts(argv: list[str]) -> int | None:
    value = 1
    for index, argument in enumerate(argv):
        if argument == "--num_rollouts" and index + 1 < len(argv):
            value = int(argv[index + 1])
        elif argument.startswith("--num_rollouts="):
            value = int(argument.split("=", 1)[1])
    return value


def _horizon(argv: list[str]) -> int:
    value = 800
    for index, argument in enumerate(argv):
        if argument == "--horizon" and index + 1 < len(argv):
            value = int(argv[index + 1])
        elif argument.startswith("--horizon="):
            value = int(argument.split("=", 1)[1])
    return value


def _seed(argv: list[str]) -> int:
    value = 101
    for index, argument in enumerate(argv):
        if argument == "--seed" and index + 1 < len(argv):
            value = int(argv[index + 1])
        elif argument.startswith("--seed="):
            value = int(argument.split("=", 1)[1])
    return value


def validate_video_output(passthrough_argv: list[str]) -> None:
    """Refuse an existing video directory without changing passthrough arguments."""
    if "--video" not in passthrough_argv:
        return
    folder = Path("videos/robomimic_play")
    for index, argument in enumerate(passthrough_argv):
        if argument == "--video_folder" and index + 1 < len(passthrough_argv):
            folder = Path(passthrough_argv[index + 1])
        elif argument.startswith("--video_folder="):
            folder = Path(argument.split("=", 1)[1])
    if _is_protected_output(folder):
        raise ValueError(f"refusing protected source-demo path: {folder}")
    if os.path.lexists(folder.expanduser()):
        raise FileExistsError(f"video folder already exists: {folder}")


def execute_play(
    passthrough_argv: list[str],
    layout_log: os.PathLike[str] | str,
    trajectory_dir: os.PathLike[str] | str | None = None,
    *,
    gym_module: Any | None = None,
    play_script: os.PathLike[str] | str = PLAY_SCRIPT,
    run_path: Callable[..., Any] = runpy.run_path,
) -> None:
    """Execute the original play ``__main__`` under a temporary observer."""
    output = validate_layout_log_path(layout_log)
    trajectory_output = validate_trajectory_dir(trajectory_dir) if trajectory_dir is not None else None
    if trajectory_output == output:
        raise ValueError("layout log and trajectory directory outputs must be distinct")
    validate_video_output(passthrough_argv)
    gym_module = gym_module if gym_module is not None else importlib.import_module("gymnasium")
    original_make = gym_module.make
    original_stdout = sys.stdout
    original_argv = sys.argv
    restore_resets: list[Callable[[], None]] = []
    patched_envs: set[int] = set()
    patched_success_terms: set[int] = set()
    observer_tees: list[StdoutTee] = []

    with output.open("x", encoding="utf-8", buffering=1) as stream:
        expected_rollouts = _expected_rollouts(passthrough_argv)
        assert expected_rollouts is not None
        observer = JsonlObserver(
            stream,
            expected_rollouts,
            trajectory_output,
            _horizon(passthrough_argv),
            _seed(passthrough_argv),
        )
        if trajectory_output is not None:
            trajectory_output.mkdir()

        def attach_stdout_observer() -> None:
            """Attach after Kit may have replaced ``sys.stdout`` during startup."""
            if any(sys.stdout is tee for tee in observer_tees):
                return
            tee = StdoutTee(sys.stdout, observer.observe_line)
            observer_tees.append(tee)
            sys.stdout = tee

        def observed_make(*args: Any, **kwargs: Any) -> Any:
            success_term = _caller_success_term()
            env = original_make(*args, **kwargs)
            attach_stdout_observer()
            unwrapped_id = id(env.unwrapped)
            if unwrapped_id not in patched_envs:
                restore_resets.append(_patch_reset(env, observer))
                if trajectory_output is not None:
                    restore_resets.append(_patch_step(env, observer))
                patched_envs.add(unwrapped_id)
            if (
                trajectory_output is not None
                and success_term is not None
                and id(success_term) not in patched_success_terms
            ):
                restore_resets.append(_patch_success_term(success_term, observer))
                patched_success_terms.add(id(success_term))
            return env

        gym_module.make = observed_make
        attach_stdout_observer()
        sys.argv = [str(play_script), *passthrough_argv]
        script_error: BaseException | None = None
        try:
            run_path(str(play_script), run_name="__main__")
        except BaseException as exc:
            script_error = exc
            if not observer.complete:
                observer._write(
                    {
                        "record_type": "observer_status",
                        "status": "error",
                        "message": f"original play.py raised {type(exc).__name__}: {exc}",
                    }
                )
        finally:
            sys.argv = original_argv
            sys.stdout = original_stdout
            gym_module.make = original_make
            for restore in reversed(restore_resets):
                restore()

        if script_error is not None:
            raise script_error
        observer.finish()


def _yaw_bin_index(yaw_deg: float) -> int:
    wrapped = ((float(yaw_deg) + 180.0) % 360.0) - 180.0
    return min(7, int((wrapped + 180.0) // 45.0))


def summarize_layout_log(path: os.PathLike[str] | str) -> dict[str, Any]:
    """Read a completed JSONL log and aggregate success by 45 degree yaw bin."""
    trials: list[dict[str, Any]] = []
    complete = False
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") == "observer_status":
                if record.get("status") == "error":
                    raise ObservationError(f"observer error on line {line_number}: {record.get('message')}")
                complete = record.get("status") == "complete"
            elif record.get("record_type") == "trial":
                if record.get("status") != "ok":
                    raise ObservationError(f"invalid trial record on line {line_number}")
                trials.append(record)
    if not complete:
        raise ObservationError("layout log has no complete observer status")

    summary: dict[str, Any] = {
        "attempts": len(trials),
        "successes": sum(bool(record["success"]) for record in trials),
        "yaw_bin_width_deg": 45,
    }
    summary["success_rate"] = summary["successes"] / len(trials) if trials else None
    for object_name in ("toy_truck", "box"):
        bins = [
            {"range_deg": [YAW_EDGES_DEG[index], YAW_EDGES_DEG[index + 1]], "successes": 0, "attempts": 0}
            for index in range(8)
        ]
        for record in trials:
            index = _yaw_bin_index(record["layout"][object_name]["yaw_deg"])
            bins[index]["attempts"] += 1
            bins[index]["successes"] += int(bool(record["success"]))
        for item in bins:
            item["success_rate"] = item["successes"] / item["attempts"] if item["attempts"] else None
        summary[f"{object_name}_yaw_45deg"] = bins
    return summary


def _extract_wrapper_args(argv: list[str]) -> tuple[str, Path | None, Path | None, list[str]]:
    layout_log: Path | None = None
    summary_log: Path | None = None
    trajectory_dir: Path | None = None
    passthrough: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument in ("--layout_log", "--summarize_layout_log", "--trajectory_dir"):
            if index + 1 >= len(argv):
                raise SystemExit(f"{argument} requires a path")
            value = Path(argv[index + 1])
            index += 2
        elif argument.startswith("--layout_log="):
            argument, value = "--layout_log", Path(argument.split("=", 1)[1])
            index += 1
        elif argument.startswith("--summarize_layout_log="):
            argument, value = "--summarize_layout_log", Path(argument.split("=", 1)[1])
            index += 1
        elif argument.startswith("--trajectory_dir="):
            argument, value = "--trajectory_dir", Path(argument.split("=", 1)[1])
            index += 1
        else:
            passthrough.append(argument)
            index += 1
            continue
        if argument == "--layout_log":
            if layout_log is not None:
                raise SystemExit("--layout_log may be specified only once")
            layout_log = value
        elif argument == "--summarize_layout_log":
            if summary_log is not None:
                raise SystemExit("--summarize_layout_log may be specified only once")
            summary_log = value
        else:
            if trajectory_dir is not None:
                raise SystemExit("--trajectory_dir may be specified only once")
            trajectory_dir = value
    if summary_log is not None:
        if layout_log is not None or trajectory_dir is not None or passthrough:
            raise SystemExit("--summarize_layout_log cannot be combined with play arguments")
        return "summary", summary_log, None, []
    if layout_log is None:
        raise SystemExit("required wrapper argument: --layout_log PATH")
    return "play", layout_log, trajectory_dir, passthrough


def main(argv: list[str] | None = None) -> None:
    mode, path, trajectory_dir, passthrough = _extract_wrapper_args(list(sys.argv[1:] if argv is None else argv))
    assert path is not None
    if mode == "summary":
        print(json.dumps(summarize_layout_log(path), indent=2, sort_keys=True))
    else:
        execute_play(passthrough, path, trajectory_dir)


if __name__ == "__main__":
    main()
