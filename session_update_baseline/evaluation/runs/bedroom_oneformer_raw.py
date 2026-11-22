"""Write the pretrained OneFormer semantic class of each BedRoom pixel."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor


MODEL = "shi-labs/oneformer_ade20k_swin_large"
REVISION = "4a5bac8e64f82681a12db2e151a4c2f4ce6092b2"
SIZE = (960, 540)
BATCH = 4


def read_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB").resize(SIZE, Image.Resampling.BILINEAR)


def write_mask(dest, pixels):
    # The player expects 1080p PNGs and samples at half resolution with nearest
    # neighbor. Repeating each value produces the exact 960x540 prediction.
    full = pixels.repeat(2, axis=0).repeat(2, axis=1)
    pending = dest.with_suffix(".part")
    Image.fromarray(full).save(pending, format="PNG", compress_level=1)
    pending.replace(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rgbd_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "source.json"
    source = dict(model=MODEL, revision=REVISION, task="semantic", size=list(SIZE),
                  class_ids="ADE20K raw", rgbd=str(args.rgbd_root.resolve()))
    if manifest.is_file():
        if json.loads(manifest.read_text()) != source:
            raise ValueError(f"Different prediction source in {args.output}")
    elif any(args.output.iterdir()):
        raise ValueError(f"Existing predictions have no source manifest in {args.output}")
    else:
        manifest.write_text(json.dumps(source, indent=2) + "\n")

    processor = OneFormerProcessor.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = OneFormerForUniversalSegmentation.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, use_safetensors=False
    ).to("cuda").eval()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False

    with ThreadPoolExecutor(max_workers=4) as io, torch.inference_mode():
        for visit in "abc":
            recording, = args.rgbd_root.glob(f"session_{visit}_*")
            folder = args.output / f"session_{visit}"
            folder.mkdir(exist_ok=True)
            images = sorted(recording.glob("*_color.png"))
            if not images:
                raise ValueError(f"No RGB frames in {recording}")
            pairs = [(p, folder / p.name.replace("_color.png", "_segmentation.png"))
                     for p in images]
            todo = [(p, dest) for p, dest in pairs if not dest.is_file()]
            done = len(images) - len(todo)
            print(f"{visit} {done}/{len(images)}", flush=True)
            writes = []
            for start in range(0, len(todo), BATCH):
                batch = todo[start:start + BATCH]
                rgbs = list(io.map(read_rgb, (p for p, _ in batch)))
                inputs = processor(images=rgbs, task_inputs=["semantic"] * len(batch),
                                   return_tensors="pt").to("cuda")
                output = model(**inputs)
                masks = processor.post_process_semantic_segmentation(
                    output, target_sizes=[SIZE[::-1]] * len(batch)
                )
                for f in writes:
                    f.result()
                writes = []
                for (_, dest), mask in zip(batch, masks, strict=True):
                    pixels = mask.cpu().numpy().astype(np.uint8)
                    writes.append(io.submit(write_mask, dest, pixels))
                done += len(batch)
                if done % 200 < BATCH:
                    print(f"{visit} {done}/{len(images)}", flush=True)
            for f in writes:
                f.result()
            print(f"{visit} {len(images)}/{len(images)} complete", flush=True)


if __name__ == "__main__":
    main()
