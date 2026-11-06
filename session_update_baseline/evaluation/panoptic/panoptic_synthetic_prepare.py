"""The synthetic A/B input and launcher of the Panoptic rerun (analysis/panoptic_rerun.sh), made by the 09-17 scripts that
made the table's earlier synthetic Panoptic run, unchanged except for paths (their output, datasets/synthetic_ab/
baselines_20260917, was moved to ALL_TABLES/ and its relative links no longer resolve):
  /mnt/d/3Study/ETH/FT/Synthetic_ABC/scripts/prepare_panoptic_synthetic.py: RGB, depth and poses linked, the ID image
    encoded from the dataset's instance labels (things) and semantics (stuff), groundtruth_labels.csv;
  .../prepare_panoptic_runner.py: the launcher and player of the real runs with the synthetic changes (stamp offset
    from FLAT_PLAYER_STAMP_OFFSET, 20 ms before each frame for the mapper's TF listener, 15 m camera range, the input
    mounted at its own path so its relative links resolve), from today's run_flat_save_monotonic.sh and
    flat_data_player_monotonic.py, which also save the maps during the visit.
The label table is checked against the earlier run's (ALL_TABLES/.../panoptic_input/groundtruth_labels.csv).
  usage: panoptic_synthetic_prepare.py OUT_DIR    (writes OUT_DIR/synthetic_input and OUT_DIR/synthetic_launcher)
"""
import filecmp, sys
from pathlib import Path

FT = Path('/home/jixian/Desktop/FT')
SCRIPTS = Path('/mnt/d/3Study/ETH/FT/Synthetic_ABC/scripts')
OLD_LABELS = FT / 'ALL_TABLES/synthetic_ab/dataset/baselines_20260917/panoptic_input/groundtruth_labels.csv'
out = Path(sys.argv[1]).resolve()
rel = out.relative_to(FT)


def patched(path, swaps):
    text = path.read_text()
    for old, new in swaps:
        assert text.count(old) == 1, (path, old)
        text = text.replace(old, new)
    return text


exec(compile(patched(SCRIPTS / 'prepare_panoptic_synthetic.py', [
    ("OUT = DATA / 'baselines_20260917'", f"OUT = Path({str(out)!r})"),
    ("FLAT = OUT / 'panoptic_input'\nFLAT.mkdir(exist_ok=True)", "FLAT = OUT / 'synthetic_input'\nFLAT.mkdir(parents=True, exist_ok=True)"),
]), str(SCRIPTS / 'prepare_panoptic_synthetic.py'), 'exec'), {'__name__': '__main__'})
assert filecmp.cmp(out / 'synthetic_input/groundtruth_labels.csv', OLD_LABELS, shallow=False), 'label table differs'
print('label table identical to the earlier run', flush=True)
exec(compile(patched(SCRIPTS / 'prepare_panoptic_runner.py', [
    ("OUT=DATA/'baselines_20260917'", f"OUT=Path({str(out / 'synthetic_launcher')!r}); OUT.mkdir(exist_ok=True)"),
    ("'cp /ft/datasets/synthetic_ab/baselines_20260917/flat_data_player_monotonic.py'",
     f"'cp /ft/{rel}/synthetic_launcher/flat_data_player_monotonic.py'"),
    ("'/ft/datasets/synthetic_ab/baselines_20260917/panoptic_input'", f"'/ft/{rel}/synthetic_input'"),
]), str(SCRIPTS / 'prepare_panoptic_runner.py'), 'exec'), {'__name__': '__main__'})
(out / 'synthetic_launcher/run_panoptic_ab.sh').unlink()   # the 09-17 driver; analysis/panoptic_rerun.sh runs A and B
