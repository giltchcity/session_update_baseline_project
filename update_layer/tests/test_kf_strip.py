"""CPU regression checks for keyframe-strip verification; no dataset or GPU is loaded.

Run with the project's PyTorch environment:
    python -m unittest update_layer.tests.test_kf_strip

Only dataset reconstruction is mocked. Checkpoints are saved and loaded by real PyTorch,
and the production tensor/array comparison and atomic replacement paths are exercised.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from update_layer.eval import kf_strip


class KeyframeStripTests(unittest.TestCase):
    def _full(self, side=2):
        return {
            "color": torch.zeros((3, side, side), dtype=torch.float32),
            "depth": torch.ones((side, side), dtype=torch.float32),
            "masks": torch.ones((1, side, side), dtype=torch.bool),
            "pose": torch.eye(4),
            "intrinsics": (1.0, 1.0, 0.0, 0.0),
        }

    def _stripped(self, full):
        return {
            "stripped": True,
            "session": "synthetic_a",
            "index": 0,
            "scale": 1.0,
            "depth_shape": tuple(full["depth"].shape),
            "zeroed": np.packbits(np.zeros(full["depth"].shape, dtype=bool)),
            "masks_shape": tuple(full["masks"].shape),
            "masks": np.packbits(full["masks"].numpy()),
            "pose": full["pose"],
            "intrinsics": full["intrinsics"],
        }

    def _invoke(self, path, candidate, restore):
        def strip_checkpoint(ck, _ds):
            out = dict(ck)
            out["backend"] = dict(ck["backend"])
            out["backend"]["keyframes"] = candidate
            return out, len(candidate), 0

        output = io.StringIO()
        with mock.patch.object(kf_strip, "strip", side_effect=strip_checkpoint), \
                mock.patch.object(kf_strip, "_sources", return_value=({}, {})), \
                mock.patch.object(kf_strip, "restore_one", side_effect=restore), \
                mock.patch.object(kf_strip.sys, "argv", ["kf_strip", "strip", str(path), str(path), "synthetic"]), \
                contextlib.redirect_stdout(output):
            kf_strip.main()
        return output.getvalue()

    def test_new_strip_restores_for_comparison_and_reports_original_size(self):
        # Big enough that the two-decimal GB log distinguishes original and stripped sizes.
        full = self._full(side=768)
        packed = self._stripped(full)
        calls = []

        def restore(_sess, kf):
            calls.append(kf)
            return full

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"backend": {"keyframes": {0: full}}}, path)
            original_size = path.stat().st_size
            log = self._invoke(path, {0: packed}, restore)
            final_size = path.stat().st_size
            self.assertEqual(len(calls), 1)
            self.assertLess(final_size, original_size)
            self.assertNotEqual(f"{original_size / 1e9:.2f}", f"{final_size / 1e9:.2f}")
            self.assertIn(f"{original_size / 1e9:.2f} GB -> {final_size / 1e9:.2f} GB", log)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            self.assertTrue(saved["backend"]["keyframes"][0]["stripped"])

    def test_mixed_full_stripped_and_memory_view_preserve_their_representation(self):
        full = self._full()
        packed = self._stripped(full)
        view = {k: v for k, v in packed.items() if k not in ("masks", "masks_shape")}
        view.update(view=True, valid=np.packbits(np.ones((2, 2), dtype=bool)))
        frames = {0: packed, 1: view, 2: full}
        new_packed = self._stripped(full)
        new_packed["index"] = 2
        candidate = {0: packed, 1: view, 2: new_packed}
        calls = []

        def restore(_sess, kf):
            self.assertEqual(kf["index"], 2, "only the newly stripped full frame may be rebuilt")
            calls.append(kf)
            return full

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"backend": {"keyframes": frames}}, path)
            self._invoke(path, candidate, restore)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(len(calls), 1)
            for kid, expected in candidate.items():
                self.assertTrue(kf_strip._same(expected, saved["backend"]["keyframes"][kid]))

    def test_all_stripped_inplace_is_noop_without_dataset_or_temporary_copy(self):
        packed = self._stripped(self._full())
        view = {k: v for k, v in packed.items() if k not in ("masks", "masks_shape")}
        view.update(view=True, valid=np.packbits(np.ones((2, 2), dtype=bool)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"backend": {"keyframes": {0: packed, 1: view}}}, path)
            original_inode = path.stat().st_ino
            original_hash = hashlib.sha256(path.read_bytes()).digest()
            output = io.StringIO()
            with mock.patch.object(kf_strip, "strip", side_effect=AssertionError("strip must not run")), \
                    mock.patch.object(kf_strip, "_sources", side_effect=AssertionError("dataset must not load")), \
                    mock.patch.object(kf_strip, "restore_one", side_effect=AssertionError("must not restore")), \
                    mock.patch.object(kf_strip.torch, "save", side_effect=AssertionError("must not write")), \
                    mock.patch.object(kf_strip.sys, "argv", ["kf_strip", "strip", str(path), str(path), "synthetic"]), \
                    contextlib.redirect_stdout(output):
                kf_strip.main()
            self.assertEqual(path.stat().st_ino, original_inode)
            self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), original_hash)
            self.assertFalse(Path(str(path) + ".tmp").exists())
            self.assertIn("已压缩，源文件未改", output.getvalue())
            self.assertNotIn("restored bitwise", output.getvalue())

    def test_changed_compressed_metadata_fails_without_replacing_source(self):
        packed = self._stripped(self._full())
        changed = copy.deepcopy(packed)
        changed["index"] = 1

        def no_restore(*_args):
            self.fail("compressed-to-compressed verification must compare the stored metadata")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            full = self._full()
            torch.save({"backend": {"keyframes": {0: packed, 1: full}}}, path)
            original_hash = hashlib.sha256(path.read_bytes()).digest()
            with self.assertRaisesRegex(SystemExit, "verification FAILED"):
                self._invoke(path, {0: changed, 1: full}, no_restore)
            self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), original_hash)
            self.assertFalse(Path(str(path) + ".tmp").exists())

    def test_changed_rebuilt_tensor_still_fails_verification(self):
        full = self._full()
        wrong = copy.deepcopy(full)
        wrong["depth"][0, 0] = 2.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"backend": {"keyframes": {0: full}}}, path)
            original_hash = hashlib.sha256(path.read_bytes()).digest()
            with self.assertRaisesRegex(SystemExit, "verification FAILED"):
                self._invoke(path, {0: self._stripped(full)}, lambda *_args: wrong)
            self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), original_hash)
            self.assertFalse(Path(str(path) + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
