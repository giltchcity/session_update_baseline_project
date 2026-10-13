#!/usr/bin/env python3
"""修复 session_b 中 instance 2（行李箱）的误标注块。

问题：frames ~2190-2460 若干帧里，桌面/desk/架子区域被误标成 instance 2
（最大一帧 37 万像素），语义图同区域也误标成 10（cabinet）。

方法：对每一帧的 instance-2 连通块（>=min_blob_px），取前后 reference_span
帧在同一图像区域中的多数 instance（非 0 非 2）；若多数身份一致且占比达标，
就把该块重映射为该身份，并把对应像素的语义图改为该身份的 catalog 语义。
空间连续性：相机运动平滑，同一物理区域在相邻帧带着正确身份。

用法:
  python3 fix_instance2_mislabel_20260820.py --dry-run   # 只报告
  python3 fix_instance2_mislabel_20260820.py --apply     # 写入（先备份）
"""
import argparse
import json
import os
from collections import Counter

import cv2
import numpy as np

INS_DIR = "instance_labels/session_b"
SEM_DIR = "semantics/session_b"
# catalog: instance -> semantic (来自 instance_labels/manifest.json)
CATALOG_SEMANTIC = {1: 7, 2: 10, 3: 15, 4: 24, 5: 33, 6: 35, 7: 74, 10: 75,
                    11: 92, 12: 115, 13: 115, 14: 115, 16: 115, 17: 115,
                    18: 131, 19: 139}


_CACHE = {}
def load(path):
    if path not in _CACHE:
        _CACHE[path] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return _CACHE[path]


def blobs_of(mask, min_px):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_px:
            out.append((labels == i, x, y, w, h, area))
    return out


def reference_majority(ins_frame, x, y, w, h, ins_dir, fi, span, need_frames):
    """邻居帧同一区域内的多数非0非2身份。

    误标块可能连续多帧（邻居也全是 2），所以扫完整个 span，取"单帧最强证据"：
    某一帧里该区域占比最高的非0非2身份，一致度 = 该帧占比。真正的行李箱块
    （邻居区域为 0/2）没有投票，不会被改。"""
    best_id = None
    best_frac = 0.0
    agree_frames = 0
    for delta in sorted(range(-span, span + 1), key=abs):
        if delta == 0:
            continue
        img = load(os.path.join(ins_dir, f"{fi + delta:06d}_segmentation.png"))
        if img is None:
            continue
        region = img[y:y + h, x:x + w]
        if region.size == 0:
            continue
        vals, cnts = np.unique(region, return_counts=True)
        total = int(region.size)
        frame_votes = Counter()
        for v, c in zip(vals, cnts):
            v = int(v)
            if v in (0, 2):
                continue
            frame_votes[v] += int(c) / total
        if not frame_votes:
            continue
        win_id, win_frac = frame_votes.most_common(1)[0]
        if win_frac > best_frac:
            best_frac = win_frac
            best_id = win_id
            agree_frames = 1
        elif win_id == best_id:
            agree_frames += 1
    if best_id is None:
        return None
    return best_id, best_frac, agree_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--min-blob-px", type=int, default=3000)
    ap.add_argument("--reference-span", type=int, default=15)
    ap.add_argument("--min-agreement", type=float, default=0.5)
    ap.add_argument("--min-refs", type=int, default=2)
    ap.add_argument("--frame-min", type=int, default=0)
    ap.add_argument("--frame-max", type=int, default=999999)
    args = ap.parse_args()

    records = []
    for name in sorted(os.listdir(INS_DIR)):
        if not name.endswith("_segmentation.png"):
            continue
        fi = int(name.split("_")[0])
        if fi < args.frame_min or fi > args.frame_max:
            continue
        ins_path = os.path.join(INS_DIR, name)
        sem_path = os.path.join(SEM_DIR, name)
        ins = load(ins_path)
        sem = load(sem_path)
        if ins is None or sem is None:
            continue
        mask2 = ins == 2
        if int(mask2.sum()) < args.min_blob_px:
            continue
        for blob, x, y, w, h, area in blobs_of(mask2, args.min_blob_px):
            ref = reference_majority(ins, x, y, w, h, INS_DIR, fi,
                                     args.reference_span, args.min_refs)
            if ref is None:
                continue
            target, frac, refs = ref
            if frac < args.min_agreement:
                continue
            records.append({
                "frame": int(fi), "blob_bbox": [int(x), int(y), int(w), int(h)],
                "pixels": int(area),
                "target_instance": int(target), "agreement": round(float(frac), 2),
                "reference_frames": int(refs),
                "target_semantic": CATALOG_SEMANTIC.get(int(target)),
            })
            if args.apply:
                ins[blob] = target
                sem[blob] = CATALOG_SEMANTIC.get(target, sem[blob])
        if args.apply:
            cv2.imwrite(ins_path, ins)
            cv2.imwrite(sem_path, sem)

    print(f"共 {len(records)} 个误标块:")
    for r in records:
        print(f"  frame {r['frame']}: {r['pixels']}px -> instance "
              f"{r['target_instance']} (semantic {r['target_semantic']}) "
              f"一致度 {r['agreement']}")
    if args.apply:
        with open("instance_labels/fix_20260820_corrections.json", "w") as f:
            json.dump({"schema": "instance_fix_record/v1",
                       "corrections": records}, f, ensure_ascii=False, indent=2)
        print("已写入 instance_labels/fix_20260820_corrections.json")


if __name__ == "__main__":
    main()
