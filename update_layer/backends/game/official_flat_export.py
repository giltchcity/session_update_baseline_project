"""Write our frames as GaME's own FlatDataset (src/entities/datasets.py) so that the published GaME code runs
on exactly what our adapter feeds it -- to tell GaME's behaviour apart from ours (2026-10-01).

  python -m update_layer.backends.game.official_flat_export {real|synthetic} STAGE OUT_DIR/run1

Frames: the ones run.py reads (the layer's rate, every 6th of 30 Hz), each produced by GameBackend._sample
(the same crop + subsampling, depth NaN / out of range / people -> 0, the same segment masks), so the
official run and ours see identical pixels. Files as FlatDataset reads them: NNNNNN_color.png (BGR, cv2),
NNNNNN_depth.tiff (float32 metres; FlatDataset multiplies depth and translation by 10, i.e. flat.yaml's
scale 10), NNNNNN_pose.txt (camera-to-world), sam_masks.h5 (group NNNNNN_color, bit-packed masks with
attribute original_shape). Prints the data-config block (intrinsics of the crop) for the official config.
"""
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))       # the project root
from update_layer.backends.game.game import STEP, Crop, GameBackend  # noqa: E402
from update_layer.run import dataset_config  # noqa: E402
from update_layer.frames import FlatSession  # noqa: E402


def main():
    dataset, stage, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    cfg, info, specs = dataset_config(dataset)
    spec = next(sp for sp in specs if sp.name.endswith("_" + stage))
    be = GameBackend(info, own_update=True)            # only its frame preparation is used (no GPU)
    assert be.scale == 1.0, "write unscaled metres (unset GAME_CONFIG): FlatDataset scales by 10 itself"
    be.session = FlatSession(spec, pixel_step=1)
    be.crop = Crop(be.session.K, STEP[info.name])
    step = max(1, int(round(30.0 / cfg.evidence_hz)))
    layer_session = FlatSession(spec, pixel_step=cfg.pixel_step)
    indices = list(range(0, len(layer_session.ids), step))
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(out / "sam_masks.h5", "w") as h5:
        for n, i in enumerate(indices):
            sample, _, _ = be._sample(i)
            cv2.imwrite(str(out / f"{n:06d}_color.png"), cv2.cvtColor(sample["color"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out / f"{n:06d}_depth.tiff"), sample["depth"].astype(np.float32))
            np.savetxt(out / f"{n:06d}_pose.txt", be.session.pose(i))
            grp = h5.create_group(f"{n:06d}_color")
            for k, m in enumerate(sample["masks"].numpy()):
                d = grp.create_dataset(f"{k:04d}", data=np.packbits(m.reshape(-1)))
                d.attrs["original_shape"] = m.shape
            if n % 100 == 0:
                print(f"{n}/{len(indices)}", flush=True)
    K = be.crop.K
    print(f"data: width {be.crop.cols} height {be.crop.rows} fx {K[0, 0]} fy {K[1, 1]} cx {K[0, 2]} cy {K[1, 2]} "
          f"frames {len(indices)}")


if __name__ == "__main__":
    main()
