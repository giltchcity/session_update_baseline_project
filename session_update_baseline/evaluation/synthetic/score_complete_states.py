# HARNESS COPY of /mnt/d/3Study/ETH/FT/Synthetic_ABC/scripts/score_complete_states.py
# (sha256 8beec419..., mtime 2026-09-17 04:52). Only changes:
#  - the run directory C (was the hardcoded DATA/'evaluation_complete_20260917') comes from
#    $HARNESS_OUT_ROOT;
#  - group_state_and_report.py is read from $HARNESS_SYN_SCRIPTS (the original script
#    directory) instead of this file's directory, because it is not copied here;
#  - an '--ours-only' method selector was added next to the existing '--khronos-only';
#  - an extra assertion that the exec'd source no longer references the published results
#    folder (the original replace() calls below already redirect PUB to group_scores).
# The same two string replacements are applied to the unmodified group_state_and_report.py,
# exactly as in the original, so the scoring algorithm is unchanged.
# 10-21: the unit list (EVAL_SCOPE_UNITS.csv) comes from $HARNESS_SCOPE_UNITS when set (the 29-unit scope of
# update_layer/eval/synthetic_scope_29: the large sofa and its throw pillows as two units, as the mapping input).
from cvpr_eval_common import *
import sys
import os
C=Path(os.environ['HARNESS_OUT_ROOT']);assert 'evaluation_complete_20260917' not in str(C.resolve()),'refusing to write into the original run directory'
HERE=Path(os.environ['HARNESS_SYN_SCRIPTS']);SOURCE=OUT/'paper_aligned'
for method in (['khronos'] if '--khronos-only' in sys.argv else ['ours'] if '--ours-only' in sys.argv else ['ours','khronos']):
    p=C/method/'state_eval';p.mkdir(exist_ok=True)
    for name in ['observed_state_refs.npz','observed_state_meta.json','EVAL_SCOPE_UNITS.csv']:
        src=Path(os.environ['HARNESS_SCOPE_UNITS']) if name=='EVAL_SCOPE_UNITS.csv' and os.environ.get('HARNESS_SCOPE_UNITS') else SOURCE/name
        if (p/name).is_symlink() and (p/name).resolve()!=src.resolve():(p/name).unlink()
        if not (p/name).exists():(p/name).symlink_to(src)
    for s in 'ab':
        (p/s).mkdir(exist_ok=True)
        if not (p/s/'native').exists():(p/s/'native').symlink_to(C/method/'native'/s,target_is_directory=True)
    (p/'group_scores').mkdir(exist_ok=True)
    source=(HERE/'group_state_and_report.py').read_text()
    source=source.replace("P=OUT/'paper_aligned';D=P/'group_scores';PUB=Path('/mnt/d/3Study/ETH/FT/results/synthetic_ab_cvpr')",f"P=Path({str(p)!r});D=P/'group_scores';PUB=D")
    source=source.replace('    shutil.copy2(D/name,PUB/name)','')
    assert f'P=Path({str(p)!r})' in source
    assert '/mnt/d/3Study/ETH/FT/results' not in source and 'shutil.copy2' not in source,'group_state_and_report.py changed; refusing to exec'
    exec(compile(source,'group_state_and_report.py [new native run]','exec'),{})
    print('NEW_RUN_STATE_COMPLETE',method,flush=True)
