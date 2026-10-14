"""Rows of the table-top items without a tracked label (identities 79 80 81 117 170 181) alive at a checkpoint's final stamp,
counted as latest_evidence.py counts them (alive = birth <= t < end; FINAL_20261014/LIMITATIONS.txt item 3: pre-ref 814, r21 6,018).
CPU only, memory-mapped checkpoints.
usage: table_items_count.py NAME=CHECKPOINT [NAME=CHECKPOINT ...]
"""
import sys

import numpy as np
import torch

INT64_MAX = np.iinfo(np.int64).max
ITEMS = (79, 80, 81, 117, 170, 181)
for arg in sys.argv[1:]:
    name, path = arg.split('=', 1)
    ck = torch.load(path, mmap=True, map_location='cpu', weights_only=False)
    b, t = ck['backend'], int(ck['prev_final'])
    end = np.minimum(np.minimum(b['death_state'].numpy(), b['death_evidence'].numpy()), b['death_prune'].numpy())
    sb = b['state_birth'].numpy()
    born = np.maximum(b['created'].numpy(), np.where(sb < INT64_MAX, sb, -INT64_MAX))
    alive = (born <= t) & (end > t)
    ident = b['identity'].numpy()
    items = alive & np.isin(ident, ITEMS)
    print(f'{name:14s} alive rows {int(alive.sum()):9d}  table items {int(items.sum()):6d}  per id',
          {i: int((items & (ident == i)).sum()) for i in ITEMS}, flush=True)
    del ck, b
print('TABLE_ITEMS_DONE')
