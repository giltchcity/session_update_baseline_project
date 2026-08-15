#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parent
PLAYER = SCRIPTS / "nss_flat_ros2_player.py"
RUNNER = SCRIPTS / "run_session.sh"
CATALOG = SCRIPTS.parent / "configs" / "room18_physical_catalog.json"
SPEC = importlib.util.spec_from_file_location("session_update_nss_player", PLAYER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PlayerInputContractTest(unittest.TestCase):
    def test_semantic_and_instance_directory_must_differ(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "must be different"):
                MODULE.preflight_session_inputs(
                    [("000000", 0)], root, root, root, CATALOG
                )

    def test_catalog_enforces_i10_as_semantic_75(self) -> None:
        catalog = MODULE.load_physical_catalog(CATALOG)
        self.assertIsNotNone(catalog)
        mapping = catalog["instance_to_semantic"]
        self.assertEqual(mapping[10], 75)
        labels = np.array([[4, 4], [4, 4]], dtype=np.uint8)
        instances = np.array([[0, 10], [10, 0]], dtype=np.uint16)
        contracted = MODULE.apply_physical_semantic_contract(
            labels, instances, mapping, "000000"
        )
        np.testing.assert_array_equal(contracted, [[4, 75], [75, 4]])

    def test_unknown_physical_id_fails_closed(self) -> None:
        labels = np.zeros((1, 2), dtype=np.uint8)
        instances = np.array([[10, 65535]], dtype=np.uint16)
        with self.assertRaisesRegex(RuntimeError, r"unknown.*65535"):
            MODULE.apply_physical_semantic_contract(
                labels, instances, {10: 75}, "000001"
            )

    def test_empty_depth_is_a_hard_failure(self) -> None:
        invalid = (
            np.zeros((2, 2), dtype=np.float32),
            np.full((2, 2), np.nan, dtype=np.float32),
            np.array([[0.0, -1.0]], dtype=np.float32),
        )
        for depth in invalid:
            with self.subTest(depth=depth):
                with self.assertRaisesRegex(RuntimeError, "no valid measurements"):
                    MODULE.require_valid_depth(depth, "000002")
        MODULE.require_valid_depth(
            np.array([[np.nan, 0.001]], dtype=np.float32), "000003"
        )


class ProductionRunnerTransactionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "baseline"
        (self.root / "scripts").mkdir(parents=True)
        (self.root / "configs").mkdir()
        (self.root / "ports" / "mapping_core").mkdir(parents=True)
        shutil.copy2(RUNNER, self.root / "scripts" / "run_session.sh")

        python = sys.executable
        self._write_executable(
            self.root / "scripts" / "khronos_env.sh",
            f"#!/usr/bin/env bash\nBASE1_PYTHON={python!s}\n"
            f"BASE1_BUILD_DIR={self.root / 'fake_build'!s}\n"
            "export BASE1_PYTHON BASE1_BUILD_DIR\n",
        )
        self._write_executable(
            self.root / "scripts" / "check_canonical_runtime.sh",
            "#!/usr/bin/env bash\nexit 0\n",
        )
        self._write_executable(
            self.root / "scripts" / "run_khronos_session_strict.sh",
            textwrap.dedent(
                """\
                #!/usr/bin/env bash
                set -euo pipefail
                output=""
                recurrent=false
                finalization_timeout=""
                while [[ $# -gt 0 ]]; do
                  if [[ "$1" == --output-dir ]]; then output=$2; shift 2
                  elif [[ "$1" == --input-state ]]; then recurrent=true; shift 2
                  elif [[ "$1" == --finalization-timeout-s ]]; then finalization_timeout=$2; shift 2
                  else shift
                  fi
                done
                mkdir -p "$output" "${output}_control/logs"
                [[ "$recurrent" != true ]] || printf '%s\n' recurrent >"$output/recurrent_stub"
                printf '%s' 'map bytes' >"$output/final.4dmap"
                printf '%s\n' 'Experiment Finished Cleanly' >"$output/experiment_log.txt"
                printf '%s\n' 'node diagnostic' >"${output}_control/logs/khronos.log"
                printf '%s\n' 'node_id,first_absent' >"$output/object_changes.csv"
                printf '%s\n' '0,0,0' >"$output/background_changes.csv"
                printf '{"map":"%s","time_steps":1}\n' "$output/final.4dmap" >"$output/state_summary.json"
                printf '{"finalization_timeout_s":"%s"}\n' "$finalization_timeout" >"${output}_control/strict_args.json"
                printf '%s\n' '{"schema":"session_update_transport/v2","profile_env_variable":"FASTRTPS_DEFAULT_PROFILES_FILE","profile_path":"/test/fastdds_session_update_ack.xml","profile_sha256":"test-digest","transaction_writers":[{"role":"frame_processed_ack","ros_topic":"/session_update/frame_processed","native_dds_topic":"rt/session_update/frame_processed","reliability":"RELIABLE","history":"KEEP_LAST","depth":10,"initial_heartbeat_ns":1000000,"heartbeat_period_ns":10000000,"nack_response_delay_ns":1000000},{"role":"rgb_input","ros_topic":"/nss/rgb/image_raw","native_dds_topic":"rt/nss/rgb/image_raw","reliability":"RELIABLE","history":"KEEP_LAST","depth":10,"initial_heartbeat_ns":1000000,"heartbeat_period_ns":10000000,"nack_response_delay_ns":1000000},{"role":"depth_input","ros_topic":"/nss/depth/image_raw","native_dds_topic":"rt/nss/depth/image_raw","reliability":"RELIABLE","history":"KEEP_LAST","depth":10,"initial_heartbeat_ns":1000000,"heartbeat_period_ns":10000000,"nack_response_delay_ns":1000000},{"role":"packed_semantic_instance_input","ros_topic":"/nss/semantic/image_raw","native_dds_topic":"rt/nss/semantic/image_raw","reliability":"RELIABLE","history":"KEEP_LAST","depth":10,"initial_heartbeat_ns":1000000,"heartbeat_period_ns":10000000,"nack_response_delay_ns":1000000}]}' >"${output}_control/transport_provenance.json"
                printf '%s\n' '{"frames_available":1,"frames_published":1,"frames_encountered":1,"frames_skipped_empty_depth":0,"published_bounds_ns":[1000,1000],"timestamp_provenance":{"policy":"test"},"input_preflight":{"frame_count":1}}' >"${output}_control/playback_manifest.json"
                [[ "${STRICT_STUB_MODE:-fail}" == success ]] || exit 23
                """
            ),
        )
        (self.root / "fake_build").mkdir()
        self._write_executable(
            self.root / "fake_build" / "inspect_session_state",
            textwrap.dedent(
                """\
                #!{python}
                import json
                import pathlib
                import sys
                path = pathlib.Path(sys.argv[1])
                scene = {{
                    "canonical_current_scene_schema": "session_update_current_scene/v1",
                    "canonical_current_scene_bytes": 42,
                    "canonical_current_scene_objects": 1,
                    "canonical_current_scene_fingerprint_fnv1a64": 123456,
                    "global_mesh_vertices": 1,
                    "global_mesh_faces": 1,
                    "current_object_nodes": 1,
                    "current_private_mesh_vertices": 3,
                    "current_private_mesh_faces": 1,
                    "current_physical_ids": [10],
                    "current_physical_id_node_counts": {{"10": 1}},
                    "current_physical_id_semantic_labels": {{"10": [75]}},
                    "duplicate_current_physical_ids": [],
                }}
                if os.environ.get("INSPECT_STUB_MODE") == "empty_mesh":
                    scene["global_mesh_vertices"] = 0
                is_prior = path.parent.name == "prior_state"
                is_recurrent = (path.parent / "recurrent_stub").is_file()
                if is_prior:
                    current = dict(scene, dsg_fingerprint_fnv1a64=111)
                    result = {{"map": str(path), "latest_stamp_ns": 900,
                              "first_stamp_ns": 900, "time_steps": 1,
                              "strictly_increasing_stamps": True,
                              "initial": current, "current": current}}
                elif is_recurrent:
                    initial = dict(scene, dsg_fingerprint_fnv1a64=222)
                    current = dict(scene, dsg_fingerprint_fnv1a64=333)
                    result = {{"map": str(path), "latest_stamp_ns": 1000,
                              "first_stamp_ns": 900, "time_steps": 2,
                              "strictly_increasing_stamps": True,
                              "initial": initial, "current": current}}
                else:
                    current = dict(scene, dsg_fingerprint_fnv1a64=333)
                    result = {{"map": str(path), "latest_stamp_ns": 1000,
                              "first_stamp_ns": 1000, "time_steps": 1,
                              "strictly_increasing_stamps": True,
                              "initial": current, "current": current}}
                print(json.dumps(result))
                """
            ).format(python=python).replace(
                "import json\n", "import json\nimport os\n", 1
            ),
        )
        (self.root / "scripts" / "nss_flat_ros2_player.py").write_text(
            textwrap.dedent(
                """
                import pathlib
                def read_frames(run_dir): return [("000000", 0)]
                def load_intrinsics(run_dir): return (1, 1, 1, 1, 0, 0)
                def load_world_transform(path): return None
                def preflight_session_inputs(frames, run_dir, semantic_dir, instance_dir, catalog):
                    if semantic_dir.resolve() == instance_dir.resolve():
                        raise ValueError("semantic and physical-instance directories must be different")
                    return {"frame_count": len(frames)}
                def resolve_session_time_contract(run_dir, frames, explicit):
                    return {"session_start_ns": 1000, "session_end_ns": 1000}
                def validate_recurrent_session_start(contract, prior): return None
                """
            ),
            encoding="utf-8",
        )
        for name in (
            "room18_instance_5cm.yaml",
            "nss_ade20k_room_label_space.yaml",
            "room18_physical_catalog.json",
        ):
            (self.root / "configs" / name).write_text("{}\n", encoding="utf-8")
        (self.root / "configs" / "nss_flat_input.yaml").write_text(
            "input_separation_s: 0.02\n", encoding="utf-8"
        )
        self.run_dir = Path(self.temp.name) / "session_a_20260809_204010_201"
        self.semantic_dir = Path(self.temp.name) / "semantic"
        self.instance_dir = Path(self.temp.name) / "instance"
        self.run_dir.mkdir()
        self.semantic_dir.mkdir()
        self.instance_dir.mkdir()
        (self.run_dir / "timestamps.csv").write_text(
            "ImageID,TimeStamp\n000000,0\n", encoding="utf-8"
        )
        self.output = Path(self.temp.name) / "accepted_state"
        self.rejected_root = Path(self.temp.name) / "rejected"

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _write_executable(path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)

    def _command(self, *extra: str) -> list[str]:
        return [
            "bash",
            str(self.root / "scripts" / "run_session.sh"),
            "--run-dir",
            str(self.run_dir),
            "--semantic-dir",
            str(self.semantic_dir),
            "--instance-dir",
            str(self.instance_dir),
            "--output-state",
            str(self.output),
            *extra,
        ]

    def _run(
        self,
        *extra: str,
        strict_mode: str = "fail",
        inspect_mode: str = "normal",
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            self._command(*extra),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "STRICT_STUB_MODE": strict_mode,
                "INSPECT_STUB_MODE": inspect_mode,
                "SESSION_UPDATE_REJECTED_ROOT": str(self.rejected_root),
            },
            check=False,
        )

    def test_failed_internal_runner_never_exposes_formal_state(self) -> None:
        result = self._run()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertFalse(self.output.exists(), result.stdout)
        self.assertFalse(Path(f"{self.output}.lock").exists(), result.stdout)
        leftovers = list(self.output.parent.glob(f".{self.output.name}.incomplete.*"))
        self.assertEqual(leftovers, [], result.stdout)
        rejected = list(self.rejected_root.glob(f"{self.output.name}.rejected.*"))
        self.assertEqual(len(rejected), 1, result.stdout)
        manifest = json.loads(
            (rejected[0] / "rejection_manifest.json").read_text(encoding="utf-8")
        )
        self.assertFalse(manifest["formal_state_published"])
        self.assertFalse(manifest["final_4dmap_retained"])
        self.assertFalse((rejected[0] / "state" / "final.4dmap").exists())
        self.assertTrue((rejected[0] / "state" / "object_changes.csv").is_file())
        self.assertTrue((rejected[0] / "state" / "background_changes.csv").is_file())
        self.assertTrue((rejected[0] / "control" / "logs" / "khronos.log").is_file())
        self.assertTrue((rejected[0] / "control" / "playback_manifest.json").is_file())
        summary = json.loads(
            (rejected[0] / "state" / "state_summary.json").read_text(encoding="utf-8")
        )
        self.assertIsNone(summary["map"])
        self.assertFalse(summary["rejected_map_retained"])
        self.assertIn(f"diagnostics={rejected[0]}", result.stdout)
        self.assertFalse(any(".incomplete." in p.name for p in self.output.parent.rglob("*")))

    def test_failed_acceptance_gate_keeps_only_small_readable_diagnostics(self) -> None:
        result = self._run(
            strict_mode="success", inspect_mode="empty_mesh"
        )
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("empty global mesh", result.stdout)
        self.assertFalse(self.output.exists())
        self.assertFalse(Path(f"{self.output}.lock").exists())
        rejected = list(self.rejected_root.glob(f"{self.output.name}.rejected.*"))
        self.assertEqual(len(rejected), 1, result.stdout)
        manifest = json.loads(
            (rejected[0] / "rejection_manifest.json").read_text(encoding="utf-8")
        )
        self.assertFalse(manifest["final_4dmap_retained"])
        summary = json.loads(
            (rejected[0] / "state" / "state_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["current"]["global_mesh_vertices"], 0)
        self.assertIsNone(summary["map"])
        self.assertFalse((rejected[0] / "state" / "final.4dmap").exists())
        self.assertEqual(
            list(self.output.parent.glob(f".{self.output.name}.incomplete.*")), []
        )

    def test_success_commits_once_and_records_formal_manifest_path(self) -> None:
        result = self._run(strict_mode="success")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertTrue((self.output / "final.4dmap").is_file(), result.stdout)
        manifest = json.loads(
            (self.output / "transition_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["output_state"], str(self.output / "final.4dmap")
        )
        self.assertEqual(
            manifest["state_summary"]["map"], str(self.output / "final.4dmap")
        )
        state_summary = json.loads(
            (self.output / "state_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(state_summary["map"], str(self.output / "final.4dmap"))
        self.assertNotIn(".incomplete.", json.dumps(manifest))
        self.assertEqual(manifest["last_acked_frame_stamp_ns"], 1000)
        transport = manifest["transport_provenance"]
        self.assertEqual(transport["schema"], "session_update_transport/v2")
        writers = transport["transaction_writers"]
        self.assertEqual(
            [writer["ros_topic"] for writer in writers],
            [
                "/session_update/frame_processed",
                "/nss/rgb/image_raw",
                "/nss/depth/image_raw",
                "/nss/semantic/image_raw",
            ],
        )
        self.assertTrue(
            all(writer["heartbeat_period_ns"] == 10_000_000 for writer in writers)
        )
        self.assertTrue(
            all(
                writer["native_dds_topic"] == f"rt{writer['ros_topic']}"
                and writer["reliability"] == "RELIABLE"
                and writer["history"] == "KEEP_LAST"
                and writer["depth"] == 10
                and writer["initial_heartbeat_ns"] == 1_000_000
                and writer["nack_response_delay_ns"] == 1_000_000
                for writer in writers
            )
        )
        self.assertEqual(
            transport["profile_env_variable"],
            "FASTRTPS_DEFAULT_PROFILES_FILE",
        )
        self.assertTrue((self.output / "control" / "playback_manifest.json").is_file())
        strict_args = json.loads(
            (self.output / "control" / "strict_args.json").read_text(encoding="utf-8")
        )
        self.assertEqual(strict_args["finalization_timeout_s"], "1800")
        leftovers = list(self.output.parent.glob(f".{self.output.name}.incomplete.*"))
        self.assertEqual(leftovers, [], result.stdout)

    def test_nonfinite_finalization_timeout_is_rejected_before_staging(self) -> None:
        result = self._run("--finalization-timeout-s", "inf")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("finite positive number", result.stdout)
        self.assertFalse(self.output.exists())
        self.assertFalse(Path(f"{self.output}.lock").exists())
        self.assertFalse(self.rejected_root.exists())

    def test_bare_map_is_rejected(self) -> None:
        bare_map = Path(self.temp.name) / "prior.4dmap"
        bare_map.write_bytes(b"map")
        result = self._run("--input-state", str(bare_map))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not a bare map", result.stdout)
        self.assertFalse(self.output.exists())

    def test_input_manifest_checksum_is_verified(self) -> None:
        prior = Path(self.temp.name) / "prior_state"
        prior.mkdir()
        (prior / "final.4dmap").write_bytes(b"real map bytes")
        (prior / "state_summary.json").write_text("{}\n", encoding="utf-8")
        (prior / "transition_manifest.json").write_text(
            json.dumps(
                {
                    "schema": "session_update_transition/v1",
                    "output_state_sha256": "0" * 64,
                }
            ),
            encoding="utf-8",
        )
        result = self._run("--input-state", str(prior))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checksum differs", result.stdout)
        self.assertFalse(self.output.exists())

    def test_recurrent_seed_uses_canonical_not_binary_fingerprint(self) -> None:
        import hashlib

        prior = Path(self.temp.name) / "prior_state"
        prior.mkdir()
        map_bytes = b"accepted prior map"
        (prior / "final.4dmap").write_bytes(map_bytes)
        (prior / "state_summary.json").write_text("{}\n", encoding="utf-8")
        (prior / "transition_manifest.json").write_text(
            json.dumps(
                {
                    "schema": "session_update_transition/v1",
                    "output_state_sha256": hashlib.sha256(map_bytes).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        result = self._run(
            "--input-state", str(prior), strict_mode="success"
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        manifest = json.loads(
            (self.output / "transition_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["input_state_summary"]["current"][
                "dsg_fingerprint_fnv1a64"
            ],
            111,
        )
        self.assertEqual(manifest["state_summary"]["initial"]["dsg_fingerprint_fnv1a64"], 222)
        self.assertEqual(
            manifest["input_state_summary"]["current"][
                "canonical_current_scene_fingerprint_fnv1a64"
            ],
            manifest["state_summary"]["initial"][
                "canonical_current_scene_fingerprint_fnv1a64"
            ],
        )


if __name__ == "__main__":
    unittest.main()
