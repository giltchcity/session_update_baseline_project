"""Render fixed cameras from stored maps next to the photo (GPU, nothing trained): the raw-image check of a mapping result.
原理思想：先看原图再下结论。同一组相机（卫生间 a2700/a2850/a2985、花盆 b3447/b3525 等）在每份地图“它自己的时刻”渲染
（T1：只画该时刻还活着的高斯），与照片并排，肉眼判断建图是否保住或修好。来源：syn_row4f3e5_r11s/visual/render_bathroom_r11s.py
（同一渲染流程：t1.alive_at、GaussianModel 载入、flashsplat_render）。
usage: render_views.py CODE_DIR OUT_DIR VIEWS(comma, e.g. a2700,a2985,b3447) LABEL=CHECKPOINT ...
"""
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

CODE, OUT, VIEWS = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), sys.argv[3].split(',')
MAPS = [a.split('=', 1) for a in sys.argv[4:]]
sys.path.insert(0, str(CODE))
from update_layer.run import dataset_config                                                      # noqa: E402
from update_layer.backends.game.game import GameBackend, GaussianModel, Crop, gu, flashsplat_render, t1   # noqa: E402
from update_layer.frames import FlatSession, synthetic_session                                   # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
_, info, _ = dataset_config('synthetic')
scale = GameBackend(info, own_update=False).scale
cams = {}
for key in VIEWS:
    fs = FlatSession(synthetic_session(key[0]), pixel_step=1)
    frame = fs.load(int(key[1:]), color=True)
    crop = Crop(frame.K, 1)
    photo = crop(frame.color)
    Image.fromarray(photo).save(OUT / f'photo__{key}.png')
    pose = np.linalg.inv(frame.T_world_cam)
    pose[:3, 3] *= scale
    h, w = photo.shape[:2]
    cams[key] = (gu.flashsplat_cam(torch.zeros((3, h, w), device='cuda'), torch.zeros((h, w), device='cuda'), None, crop.K,
                                   torch.as_tensor(pose.astype(np.float32)), None), photo)
pipe = gu.flashsplat_pipe()
background = torch.ones(3, device='cuda')
with torch.inference_mode():
    for label, path in MAPS:
        ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        b, stamp = ck['backend'], int(ck['prev_final'])
        model = b['model']
        alive = t1.alive_at(stamp, b['identity'], b['created'], b['state_birth'], b['death_state'], b['death_evidence'],
                            b.get('bg_birth'), b.get('death_prune'))
        gm = GaussianModel(3, scale=scale)
        gm.active_sh_degree = int(model[0])
        for j, attr in enumerate(('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity'), 1):
            setattr(gm, attr, model[j].detach()[alive].cuda())
        for key, (view, photo) in cams.items():
            rgb = flashsplat_render(view, gm, pipe, background, obj_num=1)['render'].clamp(0, 1)
            pixels = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(pixels).save(OUT / f'{label}__{key}.png')
            print(f'{label} {key}: MAE vs photo {np.abs(pixels.astype(float) - photo).mean():.2f}', flush=True)
        del gm, model, alive, b, ck
        gc.collect()
        torch.cuda.empty_cache()
for key in cams:
    files = [OUT / f'photo__{key}.png'] + [OUT / f'{label}__{key}.png' for label, _ in MAPS]
    labels = ['photo'] + [label for label, _ in MAPS]
    pics = [Image.open(f).convert('RGB') for f in files]
    w, h = pics[0].size
    sheet = Image.new('RGB', (w * len(pics), h + 24), 'white')
    draw = ImageDraw.Draw(sheet)
    for col, (lab, pic) in enumerate(zip(labels, pics)):
        sheet.paste(pic, (col * w, 24))
        draw.text((col * w + 6, 6), lab, fill=(0, 0, 0))
    sheet.save(OUT / f'sheet__{key}.png')
print('RENDER_VIEWS_DONE', flush=True)
