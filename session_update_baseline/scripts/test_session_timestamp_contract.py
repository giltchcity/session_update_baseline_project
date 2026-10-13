#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


PLAYER = Path(__file__).with_name("nss_flat_ros2_player.py")
SPEC = importlib.util.spec_from_file_location("session_update_nss_player", PLAYER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SessionTimestampContractTest(unittest.TestCase):
    def test_acquisition_name_is_exact_and_reproducible(self) -> None:
        run = Path("session_a_20260809_204010_201_flat_rgbd_30hz_1080p")
        frames = [("000000", 0), ("000001", 33_300_000)]
        first = MODULE.resolve_session_time_contract(run, frames)
        second = MODULE.resolve_session_time_contract(run, frames)
        self.assertEqual(first, second)
        self.assertEqual(first["session_start_ns"], 1_786_279_210_201_000_000)
        self.assertEqual(first["session_end_ns"], 1_786_279_210_234_300_000)
        self.assertEqual(first["timezone"], "Asia/Shanghai")

    def test_dataset_origin_is_relative_not_assumed_zero(self) -> None:
        run = Path("recording_20260810_030502_620")
        result = MODULE.resolve_session_time_contract(
            run, [("010000", 8_000_000_000), ("010001", 8_033_334_000)]
        )
        self.assertEqual(result["session_start_ns"], 1_786_302_302_620_000_000)
        self.assertEqual(
            result["session_end_ns"] - result["session_start_ns"], 33_334_000
        )

    def test_explicit_start_supports_diagnostic_session_c(self) -> None:
        frames = [("000000", 0), ("000001", 10)]
        result = MODULE.resolve_session_time_contract(
            Path("name_without_acquisition_time"), frames, 1_900_000_000_000_000_000
        )
        self.assertEqual(result["session_start_source"], "explicit_cli")
        self.assertIsNone(result["acquisition_local"])
        self.assertEqual(result["session_end_ns"], 1_900_000_000_000_000_010)

    def test_missing_acquisition_name_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "--session-start-ns"):
            MODULE.resolve_session_time_contract(Path("flat_rgbd"), [("0", 0)])

    def test_ros_signed_seconds_range_is_checked_for_last_frame(self) -> None:
        with self.assertRaisesRegex(ValueError, "signed-int32"):
            MODULE.resolve_session_time_contract(
                Path("unused"),
                [("0", 0), ("1", 2)],
                MODULE.ROS_TIME_MAX_NS,
            )

    def test_negative_explicit_start_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            MODULE.resolve_session_time_contract(Path("unused"), [("0", 0)], -1)

    def test_recurrent_start_must_be_strictly_newer(self) -> None:
        contract = MODULE.resolve_session_time_contract(
            Path("unused"), [("0", 0)], 1000
        )
        MODULE.validate_recurrent_session_start(contract, 999)
        for prior in (1000, 1001):
            with self.assertRaisesRegex(ValueError, "do not start after"):
                MODULE.validate_recurrent_session_start(contract, prior)


if __name__ == "__main__":
    unittest.main()
