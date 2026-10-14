"""原理思想：建图改动先在同一份已通过的 A 上只建 B 的一段，比较建图结果本身（不精修），用同一把卫生间尺子和同样的门（提交、config、两边配置）衡量。来源：FINAL_20261014/tools/smoke/map_smoke.py 的 run_chain 调用；records/probe_bathroom_rows_20261013.py。
Quick mapping test: the first N frames of synthetic B, mapped from the accepted synthetic A with a committed code tree
(FINAL_20261014/tools/smoke/map_smoke.py, same run_chain call), no end-of-run refinement (the mapping result is what is tested).

usage: quick_map.py CODE_DIR OUT_DIR N_FRAMES
Before anything runs it writes OUT_DIR/commit.txt, OUT_DIR/config.json (environment, every UPPER_CASE constant of the loaded code,
GaME yaml, backend settings, the layer's dataset config, the run_chain call) and the same construction for the real dataset,
OUT_DIR/config_other_dataset.json + config_diff.txt. It refuses when the code or this harness is not committed (rule 1) or when
the two configs differ outside the data items (rule 3). Data items: dataset identity, sessions (paths, poses, intrinsics, image
size), sensor depth range, label ids, the pixel step from the camera's native resolution, and the layer's per-dataset TSDF values
copied from each dataset's TSDF mapper YAML (kept unchanged by the user's decision of 10-14 07:56: 'config 暂时不动 建图参数没事').
"""
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch

HARNESS = Path(__file__).resolve().parents[1]
CODE, OUT, N = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), int(sys.argv[3])
R = Path('/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006')
A_INPUT = R / 'syn_row4s0/checkpoint_a.pt'            # the accepted synthetic A (RUN_ALL preflight)
A_SESSION = R / 'syn_row4f3e5_r11p/session_a'         # its session-A outputs, as map_smoke.py links them
CHAIN = dict(backend='game', row=4, sessions='b', max_frames=N, resume=True, core='l2', d1=True, g8_holdout=True)
DATA_ITEMS = ('run.dataset', 'run.input', 'dataset.', 'backend.step', 'backend.min_mask_px', 'layer.')


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


OUT.mkdir(parents=True, exist_ok=True)
commits = {}
for name, repo in (('code', CODE), ('harness', HARNESS)):
    dirty = git(repo, 'status', '--porcelain')
    if dirty:
        sys.exit(f'REFUSED (rule 1): uncommitted changes in {repo}:\n{dirty}')
    commits[name] = dict(repo=str(repo), commit=git(repo, 'rev-parse', 'HEAD'))
(OUT / 'commit.txt').write_text(json.dumps(commits, indent=2) + '\n')

sys.path.insert(0, str(CODE))
from update_layer import run                                           # noqa: E402
from update_layer.backends.game.game import GameBackend                # noqa: E402
from update_layer.eval.kf_strip import restore                         # noqa: E402
assert Path(sys.modules['update_layer.backends.game.game'].__file__).resolve().is_relative_to(CODE)


def plain(v, depth=0):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    if isinstance(v, (np.ndarray, torch.Tensor)):
        return v.tolist() if np.prod(v.shape) <= 64 else f'{type(v).__name__}{tuple(v.shape)}'
    if depth > 6:
        return repr(v)
    if isinstance(v, (set, frozenset)):
        return sorted((plain(x, depth + 1) for x in v), key=repr)
    if isinstance(v, (list, tuple)):
        return [plain(x, depth + 1) for x in v]
    if isinstance(v, dict):
        return {str(k): plain(x, depth + 1) for k, x in v.items()}
    if is_dataclass(v):
        return plain(asdict(v), depth + 1)
    if type(v).__module__.startswith('update_layer') and hasattr(v, '__dict__'):
        return plain(vars(v), depth + 1)
    return None


def configuration(ds):
    cfg, info, specs = run.dataset_config(ds)
    be = GameBackend(info, own_update=False)
    consts = {}
    for name, mod in sorted(sys.modules.items()):
        if mod is not None and name.startswith('update_layer'):
            for k, v in sorted(vars(mod).items()):
                if k.isupper() and not callable(v) and (p := plain(v)) is not None:
                    consts[f'{name}.{k}'] = p
    hidden = ('TOKEN', 'SECRET', 'PASSWORD', 'KEY', 'CREDENTIAL')
    env = {k: ('<credential, not recorded>' if any(h in k.upper() for h in hidden) else v) for k, v in sorted(os.environ.items())}
    return dict(
        run=dict(dataset=ds, input=str(A_INPUT) if ds == 'synthetic' else None, chain=CHAIN, refinement_iters=0),
        environment=env,
        software=dict(python=platform.python_version(), torch=torch.__version__, argv=sys.argv),
        module_constants=consts,
        game_yaml=plain(be.config),
        backend={k: p for k, v in sorted(vars(be).items())
                 if k not in ('info', 'config') and not callable(v) and (p := plain(v)) is not None},
        layer=plain(cfg),
        dataset=dict(name=info.name, dynamic_semantics=plain(info.dynamic_semantics), depth_range=plain(info.depth_range),
                     sessions=[plain(sp) for sp in specs]),
        code=commits)


def flatten(d, prefix=''):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from flatten(v, f'{prefix}.{k}' if prefix else k)
    else:
        yield prefix, json.dumps(d)


config, other = configuration('synthetic'), configuration('real')
a, b = dict(flatten(config)), dict(flatten(other))
diff = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
blocking = [k for k in diff if not k.startswith(DATA_ITEMS) and not k.startswith(('software.', 'code.'))]
(OUT / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
(OUT / 'config_other_dataset.json').write_text(json.dumps(other, indent=2) + '\n')
lines = [f'{"BLOCKING" if k in blocking else "data    "}  {k}:  synthetic={a.get(k, "<absent>")[:120]}  |  real={b.get(k, "<absent>")[:120]}'
         for k in diff if not k.startswith('dataset.sessions')]
(OUT / 'config_diff.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines), flush=True)
if blocking:
    sys.exit(f'REFUSED (rule 3): configs differ outside the data items: {blocking}')

# the run itself: map_smoke.py's call (restore of the compressed accepted A, A's session outputs linked, refinement off)
native_restore = GameBackend.prior_from_state
GameBackend.prior_from_state = lambda self, state: native_restore(self, restore(dict(backend=state), 'synthetic')['backend'])
make = run.make_backend


def make_unrefined(*args, **kwargs):
    backend = make(*args, **kwargs)
    backend.config['refinement_iters'] = 0
    return backend


run.make_backend = make_unrefined
for name, target in (('session_a', A_SESSION), ('checkpoint.pt', A_INPUT), ('checkpoint_a.pt', A_INPUT)):
    p = OUT / name
    if not p.exists() and not p.is_symlink():
        os.symlink(target, p)
run.run_chain(CHAIN['backend'], CHAIN['row'], 'synthetic', OUT, sessions=CHAIN['sessions'], max_frames=N, resume=True,
              core=CHAIN['core'], d1=CHAIN['d1'], g8_holdout=CHAIN['g8_holdout'])
print('QUICK_MAP_DONE', flush=True)
