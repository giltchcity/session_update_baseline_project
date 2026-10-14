"""Exactness of the stored keyframe form of evlight ([L1] storage, GPU short, read-only).

Checks
1. game._StoredKeyframe.of(kf, sample) read back = the tensors GameBackend._keyframe stored before (gu.dict2device(kf, "cpu")),
   bitwise: colour, depth, masks, pose, intrinsics; also after a torch.save / torch.load round trip (stored as a plain dict and
   wrapped again by the rule prior_from_state uses).
2. eval/kf_strip.py strips the stored form, and its verification restores it bitwise (restore_one_stored); the float restore
   (restore_one) equals the old stored tensors bitwise.
3. The sampler's coverage cache: TrackedGaME._sample_valid_keyframe with the cache draws the same keyframe sequence as the
   published sampler (86bf77e's code, copied below) from the same random state, on stored keyframes with occlusion masks.

usage: test_stored_keyframe.py CODE_DIR DATASET FRAME [FRAME ...]
"""
import io
import random
import sys
from pathlib import Path

import numpy as np
import torch

CODE, DS = Path(sys.argv[1]), sys.argv[2]
FRAMES = [int(x) for x in sys.argv[3:]]
sys.path.insert(0, str(CODE))
from update_layer.run import dataset_config  # noqa: E402
from update_layer.backends.game import game as G  # noqa: E402
from update_layer.eval import kf_strip  # noqa: E402

cfg, info, specs = dataset_config(DS)
spec_b = next(sp for sp in specs if sp.name.endswith('_b'))
be = G.GameBackend(info, own_update=False)
be.start_session(spec_b, None)
ok_all = True


def same(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    return str(a) == str(b)


stored_plain, old_float, stamps = {}, {}, {}
for i in FRAMES:
    sample, instance, dyn = be._sample(i)
    pose = sample["pose"]
    kf = {"color": G.gu.np2torch(sample["color"], device="cuda").permute(2, 0, 1) / 255.0,
          "depth": G.gu.np2torch(sample["depth"], device="cuda"),
          "masks": sample["masks"].cuda(),
          "pose": G.gu.np2torch(pose, device="cuda"),
          "intrinsics": sample["intrinsics"]}
    old = G.gu.dict2device(kf, "cpu")
    new = G._StoredKeyframe.of(kf, sample)
    checks = {k: same(old[k], new[k]) for k in ("color", "depth", "masks", "pose", "intrinsics")}
    buf = io.BytesIO()
    torch.save({"kf": new}, buf)
    buf.seek(0)
    back = torch.load(buf, weights_only=False)["kf"]
    plain = type(back) is dict and G._StoredKeyframe.stored(back)
    wrapped = G._StoredKeyframe(back)
    checks.update({f"pickled {k}": same(old[k], wrapped[k]) for k in ("color", "depth", "masks", "pose", "intrinsics")})
    checks["pickled as a plain dict"] = plain
    mem_old = sum(v.numel() * v.element_size() for v in old.values() if torch.is_tensor(v))
    mem_new = sum((v.numel() * v.element_size()) if torch.is_tensor(v) else (v.nbytes if isinstance(v, np.ndarray) else 0)
                  for v in dict.values(new))
    print(f'frame {i}: ' + ', '.join(f'{k} {"ok" if v else "DIFFERS"}' for k, v in checks.items())
          + f';  memory {mem_old / 1e6:.2f} MB -> {mem_new / 1e6:.2f} MB', flush=True)
    ok_all &= all(checks.values())
    stored_plain[i] = back
    old_float[i] = old
    stamps[i] = be.session.stamp_ns(i)

# 2. kf_strip on the stored form
ck = {"backend": {"keyframes": dict(stored_plain), "kf_stamp": dict(stamps)}}
new_ck, n_s, n_k = kf_strip.strip(ck, DS)
sess, _ = kf_strip._sources(DS)
for i in FRAMES:
    y = new_ck["backend"]["keyframes"][i]
    r_stored = kf_strip._keyframe_for_verification(stored_plain[i], y, sess)
    r_float = kf_strip.restore_one(sess, y)
    c1 = kf_strip._same(stored_plain[i], r_stored)
    c2 = all(same(old_float[i][k], r_float[k]) for k in ("color", "depth", "masks", "pose", "intrinsics"))
    print(f'kf_strip frame {i}: stripped {bool(y.get("stripped"))}; verification restore == stored form {c1}; '
          f'float restore == old stored tensors {c2}', flush=True)
    ok_all &= bool(y.get("stripped")) and c1 and c2
print(f'kf_strip: {n_s} stripped, {n_k} kept')
ok_all &= n_s == len(FRAMES)


# 3. the sampler: published logic (86bf77e) vs the cached one
def published(self, selected_frames, only_frame_id=None):
    if only_frame_id is not None:
        return only_frame_id
    thresh = self.config["occlusion_ignore_threshold"]
    active = [f for f in selected_frames if f not in self.ignored_frames]
    while active:
        keyframe_id = random.choice(active)
        if keyframe_id not in self.occlusion_masks:
            return keyframe_id
        all_masks = self.keyframes[keyframe_id]["masks"].cuda()
        ignore_mask = self.occlusion_masks[keyframe_id]
        covered = (ignore_mask * all_masks).sum(dim=(1, 2)) / all_masks.sum(dim=(1, 2))
        if (covered > thresh).any():
            self.ignored_frames.add(keyframe_id)
            active.remove(keyframe_id)
        else:
            return keyframe_id
    raise G._NoFrames


class Stub:
    pass


def stub():
    s = Stub()
    s.config = dict(be.config)
    s.keyframes = {i: G._StoredKeyframe(stored_plain[i]) for i in FRAMES}
    s.occlusion_masks = {}
    s.ignored_frames = set()
    s._coverage_passed = set()
    for j, i in enumerate(FRAMES):
        m = s.keyframes[i]["masks"]
        h, w = m.shape[-2:]
        occ = torch.zeros((h, w), dtype=torch.bool)
        if j % 2 == 0 and len(m):            # every other keyframe: an occlusion mask over half of its first mask's rows
            rows = torch.nonzero(m[0].any(1)).squeeze(1)
            occ[rows[: len(rows) // 2 + (j % 4)]] = True
        s.occlusion_masks[i] = occ.cuda()
    return s


a, b = stub(), stub()
random.seed(20261014)
seq_a = [published(a, FRAMES) for _ in range(400)]
random.seed(20261014)
seq_b = [G.TrackedGaME._sample_valid_keyframe(b, FRAMES) for _ in range(400)]
c3 = seq_a == seq_b and a.ignored_frames == b.ignored_frames
print(f'sampler: 400 draws identical {seq_a == seq_b}, ignored frames identical {a.ignored_frames == b.ignored_frames} '
      f'({sorted(a.ignored_frames)}), coverage tests computed once per keyframe: {sorted(b._coverage_passed)}')
ok_all &= c3
print('TEST_STORED_KEYFRAME', 'PASS' if ok_all else 'FAIL')
