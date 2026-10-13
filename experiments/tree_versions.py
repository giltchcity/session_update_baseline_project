"""The plant (id 109) after every refinement version that still has a render or a checkpoint (CPU; renders + checkpoints only).
1) One image: the plant crop of view b3525 from each version's saved post-refinement render (same crop box), labelled.
2) Rows of id 109 alive at the end in each post-refinement checkpoint: count, opacity, axis ratio.
usage: tree_versions.py OUT_PNG
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from update_layer.frames import FlatSession, synthetic_session  # noqa: E402

R = Path('/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006')
F = Path('/home/jixian/Desktop/FT/runs/focused_update_20261012')
INT64_MAX = 9223372036854775807
renders = [('photo b3525', R / 'syn_row4f3e5_r11p/visual/photo__b3525.png'),
           ('pre-ref (R11p mapping)', R / 'syn_row4f3e5_r11p/visual/R11p_B_pre__b3525.png'),
           ('R6 post', R / 'syn_row4f3e5_r6/visual/R6_B_post__b3525.png'),
           ('R6b post', R / 'syn_row4f3e5_r6b/visual/R6b_B_post__b3525.png'),
           ('R7g post', R / 'syn_row4f3e5_r7g/visual/R7g_B_post__b3525.png'),
           ('R10m post', R / 'syn_row4f3e5_r10m/visual/R10m_B_post__b3525.png'),
           ('R11p post', R / 'syn_row4f3e5_r11p/visual/R11p_B_post__b3525.png'),
           ('R11t post', R / 'syn_row4f3e5_r11t/visual/R11t_B_post__b3525.png'),
           ('R13u post', R / 'syn_row4f3e5_r13u/visual/R13u_B_post__b3525.png'),
           ('R13v post', R / 'syn_row4f3e5_r13v/visual/R13v_B_post__b3525.png'),
           ('r15p 30K', F / 'quick_30k/core_check/r15p__b3525.png'),
           ('R21 30K', F / 'claude_progress_20261014/r21__b3525.png'),
           ('R22 30K', F / 'claude_progress_20261014/r22__b3525.png')]
fs = FlatSession(synthetic_session('b'), pixel_step=1)
mask = (fs.load(3525).instance == 109) | (fs.load(3447).instance == 109)
ys, xs = np.nonzero(mask)
y0, y1, x0, x1 = max(0, ys.min() - 15), min(480, ys.max() + 15), max(0, xs.min() - 40), min(680, xs.max() + 15)
tiles = []
for label, path in renders:
    if not path.exists():
        print('missing', label, path); continue
    im = Image.open(path).convert('RGB')
    if im.size != (680, 480):
        print('size', label, im.size); continue
    tile = im.crop((x0, y0, x1, y1))
    ImageDraw.Draw(tile).text((4, 4), label, fill=(255, 0, 0))
    tiles.append(tile)
w, h = tiles[0].size
cols = 5
sheet = Image.new('RGB', (w * cols, h * ((len(tiles) + cols - 1) // cols)), (255, 255, 255))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % cols) * w, (i // cols) * h))
sheet.save(sys.argv[1])
print('crop', (x0, y0, x1, y1), 'tiles', len(tiles), '->', sys.argv[1])
for label in ('r10m', 'r11p', 'r11s', 'r15p', 'r21', 'r22'):
    path = R / f'syn_row4f3e5_{label}/checkpoint_b_post_ref.pt'
    ck = torch.load(path, mmap=True, map_location='cpu', weights_only=False)
    b, t, m = ck['backend'], int(ck['prev_final']), ck['backend']['model']
    end = np.minimum(np.minimum(b['death_state'].numpy(), b['death_evidence'].numpy()), b['death_prune'].numpy())
    sb = b['state_birth'].numpy()
    born = np.maximum(b['created'].numpy(), np.where(sb < INT64_MAX, sb, -INT64_MAX))
    sel = (born <= t) & (end > t) & (b['identity'].numpy() == 109)
    s = np.exp(m[4].detach().numpy()[sel])
    o = 1 / (1 + np.exp(-m[6].detach().numpy()[sel, 0]))
    print(f'{label:5s} post: id109 rows alive {sel.sum():6d} | opacity median {np.median(o):.2f} (share < 0.5: {np.mean(o < 0.5) * 100:4.1f} %) | '
          f'axis ratio median {np.median(s.max(1) / s.min(1)):.2f}', flush=True)
