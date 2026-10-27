# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Sim-free focused tests for targeted Toy2Box collection telemetry and quotas."""

from __future__ import annotations

import builtins
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "record_toy2box_scripted_demos.py"


@pytest.fixture(scope="module")
def collector():
    forbidden = {"gymnasium", "isaaclab", "isaaclab_tasks", "torch"}
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in forbidden:
            raise AssertionError(f"simulator dependency imported: {name}")
        return original_import(name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location("toy2box_yaw_filter_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        builtins.__import__ = guarded_import
        spec.loader.exec_module(module)
    finally:
        builtins.__import__ = original_import
    return module


@pytest.mark.parametrize(
    ("yaw", "expected"),
    [
        (-180.0, None),
        (-90.0, None),
        (-89.999, 0),
        (-45.0, 1),
        (-0.001, 1),
        (0.0, 2),
        (44.999, 2),
        (45.0, 3),
        (90.0, 3),
        (90.001, None),
        (180.0, None),
    ],
)
def test_targeted_yaw_bin_boundaries(collector, yaw, expected):
    assert collector.targeted_yaw_bin(yaw) == expected


def test_quota_counts_only_accepted_demos_and_skips_full_bins(collector):
    quotas = (1, 1, 1, 1)
    accepted = [0, 0, 0, 0]
    attempted = 0
    skipped = 0

    for yaw, succeeds in [(-120, True), (-80, False), (-80, True), (-70, True), (-30, True), (10, True), (60, True)]:
        allowed, yaw_bin, _ = collector.yaw_quota_decision(yaw, quotas, accepted)
        if not allowed:
            skipped += 1
            continue
        attempted += 1
        if succeeds:
            accepted[yaw_bin] += 1

    assert attempted == 5
    assert skipped == 2  # outside target and the already-full first bin
    assert accepted == [1, 1, 1, 1]


def test_filter_off_never_skips_or_changes_bin_reporting(collector):
    counts = [9, 9, 9, 9]
    for yaw in (-180.0, -90.0, -20.0, 20.0, 90.0, 170.0):
        allowed, yaw_bin, reason = collector.yaw_quota_decision(yaw, None, counts)
        assert allowed is True
        assert reason is None
        assert yaw_bin == collector.targeted_yaw_bin(yaw)


def test_failed_warmup_is_not_fabricated_as_settled_or_charged_to_target_quota(collector):
    state = {
        "truck": np.array([9.0, 9.0, 9.0]),
        "truck_quat": np.array([1.0, 0.0, 0.0, 0.0]),
    }
    telemetry, allowed, yaw_bin, reason = collector.collection_layout_decision(
        state, False, "smooth", (5, 5, 5, 5), [0, 0, 0, 0]
    )
    assert allowed is False
    assert yaw_bin is None
    assert reason == "warmup_environment_termination"
    assert telemetry["initial_layout_settled"] is False
    assert telemetry["initial_truck_yaw_degrees"] is None
    assert telemetry["initial_truck_position"] is None

    # Filter-off retains the legacy behavior: the failed warmup is still an actual
    # attempt, but its auto-reset replacement pose is explicitly not reported.
    telemetry, allowed, yaw_bin, reason = collector.collection_layout_decision(
        state, False, "smooth", None, [0, 0, 0, 0]
    )
    assert (allowed, yaw_bin, reason) == (True, None, None)
    assert telemetry["initial_layout_settled"] is False


def test_quota_validation_locks_total_demo_contract(collector):
    assert collector.validate_yaw_bin_quotas(None, 20) is None
    assert collector.validate_yaw_bin_quotas([5, 5, 5, 5], 20) == (5, 5, 5, 5)
    with pytest.raises(ValueError, match="nonnegative"):
        collector.validate_yaw_bin_quotas([5, -1, 6, 10], 20)
    with pytest.raises(ValueError, match="must equal"):
        collector.validate_yaw_bin_quotas([5, 5, 5, 4], 20)


def test_settled_telemetry_and_streaming_jsonl_are_lossless(collector, tmp_path):
    yaw = np.deg2rad(45.0)
    state = {
        "truck": np.array([0.625, -0.225, 0.42]),
        "truck_quat": np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),
    }
    telemetry = collector.settled_truck_telemetry(state)
    assert telemetry == {
        "initial_truck_yaw_degrees": pytest.approx(45.0),
        "initial_truck_x": 0.625,
        "initial_truck_y": -0.225,
        "initial_truck_position": [0.625, -0.225, 0.42],
    }

    path = tmp_path / "attempts.jsonl"
    collector.append_jsonl(path, {"event": "layout_skip", **telemetry})
    collector.append_jsonl(path, {"event": "attempt", "attempt": 1, **telemetry})
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["event"] for record in records] == ["layout_skip", "attempt"]
    assert records[1]["initial_truck_yaw_degrees"] == pytest.approx(45.0)
