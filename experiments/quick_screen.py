"""3K same-input refinement check (FINAL_20261014/tools/experiments/quick_screen_rt.py, native optimizer and sampling).

usage: quick_screen.py {short|short4k|config} ARM DATASET OUT_DIR [CODE_DIR]   (short4k: 4000 steps, for timing at SH degree 3)
  CODE_DIR = the committed checkout under test (default: this repository).
Before anything runs it writes OUT_DIR/commit.txt and OUT_DIR/config.json (every effective setting, every environment
variable) and OUT_DIR/config_other_dataset.json + OUT_DIR/config_diff.txt (the same construction for the other dataset).
It refuses to run when the code is not committed (rule 1) or when the two configs differ outside the data items (rule 3).
"""
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HARNESS = Path(__file__).resolve().parents[1]
mode, arm, dataset, OUT = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
CODE = Path(sys.argv[5]).resolve() if len(sys.argv) > 5 else HARNESS
R = Path('/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006')
INPUT = {'synthetic': R / 'syn_row4f3e5_r11p/checkpoint_b.pt'}
OTHER = {'synthetic': 'real', 'real': 'synthetic'}[dataset]
# items that may differ between the datasets (rule 3): the data itself -- which dataset and input file, the sessions
# (frame paths, start times, poses, camera intrinsics and image size, sensor depth range), the label ids of the dataset's
# own label space, and the pixel step that sets the working resolution from the camera's native resolution
DATA_ITEMS = ('run.dataset', 'run.input', 'run.refined_session', 'dataset.', 'backend.step', 'backend.min_mask_px')


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
from update_layer.run import dataset_config                                                    # noqa: E402
from update_layer.backends.game.game import GameBackend, Crop, gu, flashsplat_render, _NoFrames  # noqa: E402
from update_layer.eval.kf_strip import restore                                                 # noqa: E402
from update_layer.frames import FlatSession, synthetic_session, real_session                   # noqa: E402
assert Path(sys.modules['update_layer.backends.game.game'].__file__).resolve().is_relative_to(CODE)


def plain(v, depth=0):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (np.ndarray, torch.Tensor)):
        return v.tolist() if v.size == 0 or np.prod(v.shape) <= 64 else f'{type(v).__name__}{tuple(v.shape)}'
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
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


def constants():
    """Every module-level UPPER_CASE value of every loaded update_layer module (the switches live there)."""
    out = {}
    for name, mod in sorted(sys.modules.items()):
        if mod is None or not name.startswith('update_layer'):
            continue
        for k, v in sorted(vars(mod).items()):
            if k.isupper() and not callable(v) and (p := plain(v)) is not None:
                out[f'{name}.{k}'] = p
    return out


def environment():
    hidden = ('TOKEN', 'SECRET', 'PASSWORD', 'KEY', 'CREDENTIAL')
    return {k: (f'<credential, sha256 {hashlib.sha256(v.encode()).hexdigest()[:16]}>'
                if any(h in k.upper() for h in hidden) else v) for k, v in sorted(os.environ.items())}


def configuration(ds, iterations):
    """What a refinement run on dataset ds actually reads: environment, constants, GaME yaml, backend settings, data."""
    _, info, specs = dataset_config(ds)        # the LayerConfig (first item) is the mapping layer's; refinement never reads it
    be = GameBackend(info, own_update=False)
    be.config['refinement_iters'] = iterations
    sessions = []
    for sp in specs:
        frame = FlatSession(sp, pixel_step=1).load(0, color=True)
        sessions.append(dict(spec=plain(sp), intrinsics=plain(frame.K), image=list(frame.color.shape)))
    backend = {k: p for k, v in sorted(vars(be).items())
               if k not in ('info', 'config') and not callable(v) and (p := plain(v)) is not None}
    return be, dict(
        run=dict(mode=mode, arm=arm, dataset=ds, iterations=iterations, input=str(INPUT.get(ds)),
                 sampling='native _sample_valid_keyframe; python/numpy/torch/cuda RNG restored from the input checkpoint',
                 quick_time_budget=None, refined_session={'synthetic': 'b', 'real': 'c'}[ds]),
        environment=environment(),
        software=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, argv=sys.argv),
        module_constants=constants(),
        game_yaml=plain(be.config),
        backend=backend,
        backend_changes=plain(getattr(be, 'CHANGES', ())),
        dataset=dict(name=info.name, dynamic_semantics=plain(info.dynamic_semantics),
                     depth_range=plain(info.depth_range), sessions=sessions),
        code=commits)


def flatten(d, prefix=''):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from flatten(v, f'{prefix}.{k}' if prefix else k)
    else:
        yield prefix, json.dumps(d)


torch.set_num_threads(2)
iterations = 60 if mode == 'benchmark' else 4000 if mode == 'short4k' else 3000
be, config = configuration(dataset, iterations)
_, other = configuration(OTHER, iterations)
a, b = dict(flatten(config)), dict(flatten(other))
diff = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k) and k != 'code.harness.commit')
blocking = [k for k in diff if not k.startswith(DATA_ITEMS) and not k.startswith(('software.', 'code.'))]
(OUT / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
(OUT / 'config_other_dataset.json').write_text(json.dumps(other, indent=2) + '\n')
lines = [f'{"BLOCKING" if k in blocking else "data    "}  {k}:  {dataset}={a.get(k, "<absent>")[:160]}  |  '
         f'{OTHER}={b.get(k, "<absent>")[:160]}' for k in diff if not k.startswith('dataset.sessions')]
lines.append(f'(dataset.sessions.*: {sum(k.startswith("dataset.sessions") for k in diff)} differing data items -- '
             f'frame paths, start times, poses, intrinsics, image size)')
(OUT / 'config_diff.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines), flush=True)
if blocking:
    sys.exit(f'REFUSED (rule 3): real and synthetic configs differ outside the data items: {blocking}')
if mode == 'config':
    sys.exit(0)

input_path = INPUT[dataset]
ck = torch.load(input_path, mmap=True, map_location='cpu', weights_only=False)
be.game = be.prior_from_state(restore(ck, dataset)['backend'])
g = be.game
g.ignored_frames = set(ck['backend']['ignored_frames'])
be._closed_labels, be._live_state_intervals, be._closed_mask_cache = {}, {}, {}
be._closed_mask_stats = dict(builds=0, masked_draws=0, excluded_pixel_draws=0)
be._closed_sources = None
g.closed_pixel_mask = be._closed_state_mask
session_name = config['run']['refined_session']
be.session = FlatSession(next(sp for sp in dataset_config(dataset)[2] if sp.name == f'{dataset}_{session_name}'),
                         pixel_step=1)
be.stamps, be.scenes = [], []
stamp = int(ck['prev_final'])
random.setstate(ck['rng']['python'])
np.random.set_state(ck['rng']['numpy'])
torch.set_rng_state(ck['rng']['torch'])
torch.cuda.set_rng_state_all(ck['rng']['cuda'])
config['restored_state'] = {k: p for k, v in sorted(vars(g).items())      # the model's own settings (no per-row data)
                            if not callable(v) and (p := plain(v)) is not None and len(json.dumps(p, default=repr)) <= 2000
                            and not (isinstance(p, str) and p.startswith(('Tensor(', 'Parameter(', 'ndarray(')))}
config['run']['input_bytes'] = input_path.stat().st_size
config['run']['input_mtime'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(input_path.stat().st_mtime))
(OUT / 'config.json').write_text(json.dumps(config, indent=2, default=repr) + '\n')
be.config['refinement_iters'] = iterations
draws = []
native_sample = g._sample_valid_keyframe


def sample(*args, **kwargs):
    key = native_sample(*args, **kwargs)
    draws.append(key)
    return key


g._sample_valid_keyframe = sample
times = []
native_optimize = g.optimize_model


def measured(*args, **kwargs):
    torch.cuda.synchronize()
    start = time.monotonic()
    result = native_optimize(*args, **kwargs)
    torch.cuda.synchronize()
    times.append(time.monotonic() - start)
    return result


g.optimize_model = measured
VIEWS = [('a', 2700), ('a', 2850), ('a', 2985), ('b', 1830), ('b', 1845), ('b', 1860), ('b', 2190), ('b', 2625),
         ('b', 345), ('b', 1395), ('b', 2955), ('b', 3135), ('b', 3447), ('b', 3480), ('b', 3525), ('b', 3540)]


@torch.no_grad()
def pictures(label):
    directory = OUT / 'core_check'
    directory.mkdir(exist_ok=True)
    gm = g.gaussian_model
    gm.alive = g.alive_at(stamp)
    results = {}
    for session, index in VIEWS:
        fs = FlatSession((synthetic_session if dataset == 'synthetic' else real_session)(session), pixel_step=1)
        frame = fs.load(index, color=True)
        crop = Crop(frame.K, 1)
        photo, inst = crop(frame.color), crop(frame.instance)
        pose = np.linalg.inv(frame.T_world_cam)
        pose[:3, 3] *= be.scale
        h, w = photo.shape[:2]
        view = gu.flashsplat_cam(torch.zeros((3, h, w), device='cuda'), torch.zeros((h, w), device='cuda'), None,
                                 crop.K, torch.as_tensor(pose.astype(np.float32)), None)
        key = f'{session}{index}'
        roi = ((inst == 109) & (np.indices((h, w))[0] >= 225)) if key == 'b3525' else None
        mask = torch.from_numpy(roi.astype(np.float32)).cuda() if roi is not None else None
        pkg = flashsplat_render(view, gm, gu.flashsplat_pipe(), torch.ones(3, device='cuda'), gt_mask=mask, obj_num=1)
        pixels = (pkg['render'].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(pixels).save(directory / f'{label}__{key}.png')
        Image.fromarray(photo).save(directory / f'photo__{key}.png')
        rec = dict(photo_MAE_255=float(np.abs(pixels.astype(float) - photo).mean()),
                   photo_time_ns=fs.stamp_ns(index), map_time_ns=stamp)
        if roi is not None:
            rec['pot_own_alpha_T'] = float(pkg['used_count'][1, g.identity == 109].sum()) / int(roi.sum())
            rec['pot_photo_MAE_255'] = float(np.abs(pixels[roi].astype(float) - photo[roi]).mean())
        results[key] = rec
        del pkg, view, mask
    alive = g.alive_at(stamp)
    results['alive_object_rows'] = {str(i): int(((g.identity == i) & alive).sum())
                                    for i in (69, 70, 72, 109, 147, 148, 153, 164)}
    gm.alive = None
    return results


record = dict(mode=mode, arm=arm, dataset=dataset, commits=commits, iterations=iterations, input=str(input_path))
record['before'] = pictures('input')
torch.cuda.reset_peak_memory_stats()
start = time.monotonic()
report = be.finish_session(stamp)
record.update(seconds=sum(times), finish_seconds=time.monotonic() - start, report=report,
              peak_allocated_GB=torch.cuda.max_memory_allocated() / 1e9,
              peak_reserved_GB=torch.cuda.max_memory_reserved() / 1e9, sampled_frames=draws,
              sampling_sha256=hashlib.sha256(np.asarray(draws, dtype=np.int64).tobytes()).hexdigest(),
              mask_samples=be._closed_mask_stats)
if hasattr(g, '_measurement_stats'):
    record['measurement_samples'] = g._measurement_stats
assert report['completed_iterations'] == iterations, report
record['after'] = pictures(arm)
record['active_sh_degree_at_end'] = int(g.gaussian_model.active_sh_degree)      # the code path check (SH ramp reached?)
(OUT / f'{mode}_{arm}.json').write_text(json.dumps(record, indent=2, default=repr))
print({k: v for k, v in record.items() if k not in ('sampled_frames', 'before', 'after')}, flush=True)
for key in record['after']:
    if key != 'alive_object_rows':
        x, y = record['before'][key], record['after'][key]
        print(f'{key:6s} photo MAE before {x["photo_MAE_255"]:6.2f} after {y["photo_MAE_255"]:6.2f}'
              + (f'   pot ROI before {x["pot_photo_MAE_255"]:6.2f} after {y["pot_photo_MAE_255"]:6.2f}'
                 if 'pot_photo_MAE_255' in y else ''), flush=True)
