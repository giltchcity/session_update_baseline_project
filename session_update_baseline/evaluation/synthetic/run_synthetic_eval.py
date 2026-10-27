"""Validation orchestration: unchanged synthetic metrics, RAM snapshot input.

Runs geometry, online objects, D2/D3 and background/change/update metrics.
The original online aggregation block is executed verbatim; live D1 tracking
is a separate task and is not synthesized from map snapshots.
The table (FULL_TABLE.md) compares every metric with the previous version's run
(REFERENCE), whose values the same table builder computed the same way.
Other maps (env): SYN_MAPS = a directory with session_{a,b}.4dmap.zpk, SYN_OUT = the output directory,
SYN_REF / SYN_REF_LABEL = the reference run; SYNTHETIC_TOOLS=khronos for official Khronos maps (synthetic_ram.py).
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap

ROOT=Path('/home/jixian/Desktop/FT')
RUNS=ROOT/'wt_takeover_runs'
HARNESS=ROOT/'eval/synthetic_base_v38/harness'
FULL=ROOT/'datasets/synthetic_ab/eval_tools_full_20260922'
SOURCES=Path('/mnt/d/3Study/ETH/FT/Synthetic_ABC/scripts')
PUBLISHED=Path('/mnt/d/3Study/ETH/FT/results/synthetic_ab_cvpr')
REFERENCE=(Path(os.environ.get('SYN_REF',RUNS/'superseded/d1/eval/tk_synthetic')),
           os.environ.get('SYN_REF_LABEL','3a6cddd'))  # default: the committed previous version
MAPS=os.environ.get('SYN_MAPS')
def session_map(session):
    return Path(MAPS)/f'session_{session}.4dmap.zpk' if MAPS else RUNS/f'synthetic_{session}_tk/final.4dmap.zpk'
PY='/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python'


def aggregate(base,out):
    source=(HARNESS/'summarize_ours_online.py').read_text()
    start='        # ---- verbatim from the original (Online Object aggregation) ----\n'
    end='        # ---- end verbatim ----'
    assert source.count(start)==source.count(end)==1
    block=textwrap.dedent(source.split(start)[1].split(end)[0])
    online=[];queries=[]
    values=dict(C=base,json=json,method='ours',online=online,online_queries=queries,
                read=lambda p:list(csv.DictReader(p.open())))
    for session in 'ab':
        values['s']=session
        exec(compile(block,str(HARNESS/'summarize_ours_online.py'),'exec'),values)
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('ONLINE_OBJECT_SUMMARY.csv',online),('ONLINE_OBJECT_QUERIES.csv',queries)]:
        with (out/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    (out/'aggregation_source.json').write_text(json.dumps(dict(source=str(HARNESS/'summarize_ours_online.py'),
        sha256=hashlib.sha256(source.encode()).hexdigest(),selection='verbatim Online Object aggregation block'),indent=2)+'\n')


if len(sys.argv)>1 and sys.argv[1]=='--aggregate-online':
    aggregate(Path(sys.argv[2]),Path(sys.argv[3]))
    raise SystemExit(0)

out=Path(os.environ.get('SYN_OUT',RUNS/'eval/tk_synthetic'))
# The native exports and per-step inputs only feed the scorers: they live in memory (/dev/shm) for this run and are
# dropped at the end; the tables and score files go to OUT.
base=Path(tempfile.mkdtemp(prefix='synthetic-eval-',dir='/dev/shm'))/'base'
base.mkdir()
# SYN_OBJECTS_FROM=EARLIER_OUT: only the object scoring changed (e.g. the object matching distance), so every other
# step's inputs and results are taken from that earlier evaluation of the same maps, and only the online object and
# historical object steps run again (with SYNTHETIC_TOOLS_BIN's evaluator).
OBJECTS_FROM=os.environ.get('SYN_OBJECTS_FROM')
if OBJECTS_FROM:
    old=Path(OBJECTS_FROM)
    shutil.copytree(old/'base',base,dirs_exist_ok=True,ignore=lambda d,names:['online'] if Path(d)==old/'base' else [])
    out.mkdir(parents=True,exist_ok=True)
    for name in ('background','changes','updates'):
        shutil.copytree(old/name,out/name,dirs_exist_ok=True)
    (out/'history').mkdir(parents=True,exist_ok=True)
    for f in (old/'history').glob('*'):
        if f.is_file() and f.name!='HISTORICAL_OBJECT_SUMMARY.csv':
            shutil.copy2(f,out/'history'/f.name)
    for name in ('export_a','export_b','geometry','online_manifest','states','background','changes','updates',
                 'history_manifest'):
        (base/('.done_'+name)).write_text('taken from '+str(old)+'\n')
logs=out/'eval_logs';logs.mkdir(parents=True,exist_ok=True)
env=os.environ.copy()
env.update(PYTHONPATH=str(SOURCES),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
    HARNESS_OUT_ROOT=str(base),HARNESS_SYN_SCRIPTS=str(SOURCES),FULL_LIVE_ROOT=str(base),
    HARNESS_CAUSAL_BANK=str(ROOT/'datasets/synthetic_ab/evaluation_complete_20260917/causal_reference'),
    FULL_SYN_SCRIPTS=str(SOURCES),FULL_NATIVE_ROOT=str(base),FULL_OUT_ROOT=str(out),
    FULL_CAUSAL_BANK=str(ROOT/'datasets/synthetic_ab/evaluation_complete_20260917/causal_reference'))


def step(name,cmd):
    marker=base/('.done_'+name)
    if marker.exists():return
    print('STEP',name,flush=True)
    with (logs/(name+'.log')).open('w') as f:
        subprocess.run(list(map(str,cmd)),cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    marker.write_text('complete\n')


for session in 'ab':
    step('export_'+session,['python',RUNS/'analysis/synthetic_ram.py','export',
        session_map(session),base/('ours/native/'+session)])
step('geometry',[PY,HARNESS/'evaluate_cross_session_geometry.py','--methods','ours'])
step('online_manifest',[PY,HARNESS/'prepare_historical_object_eval.py','--native-only','--ours-only'])
for session in 'ab':
    step('objects_'+session,['python',RUNS/'analysis/synthetic_ram.py','objects',
        session_map(session),base/('history/ours_'+session+'_online_manifest.json'),
        base/('online/ours/'+session)])
step('online_summary',[PY,Path(__file__),'--aggregate-online',base,base/'online'])
step('states',[PY,HARNESS/'score_complete_states.py','--ours-only'])
for name in ('background','changes','updates'):
    step(name,[PY,FULL/'run_original.py',name])
# Historical Object (10 s T,t grid), the history step of evaluate_ours_full.sh: the original manifest, the same
# ObjectEvaluator run snapshot by snapshot (synthetic_ram.py objects), the original summary.
step('history_manifest',[PY,FULL/'run_original.py','history_manifest'])
for session in 'ab':
    step('history_'+session,['python',RUNS/'analysis/synthetic_ram.py','objects',
        session_map(session),out/('history/ours_'+session+'_manifest.json'),
        out/('history/ours/'+session)])
step('history_summary',[PY,FULL/'summarize_history.py'])
# The table is rebuilt every time (cheap), as in evaluate_ours_full.sh.
with (logs/'values.log').open('w') as f:
    subprocess.run(list(map(str,[PY,FULL/'build_full_table.py',base,out,PUBLISHED,RUNS])),cwd=ROOT,env=env,
                   stdout=f,stderr=subprocess.STDOUT,check=True)


def number(v):
    try:return float(v)
    except (TypeError,ValueError):return None


def text(v):
    x=number(v)
    return '—' if v in ('',None) else v if x is None else f'{x:.4f}'


def compare(new_dir,reference):
    """FULL_TABLE.md: this run against the reference run, row by row (group, metric, stage)."""
    ref_dir,label=reference
    key=lambda r:(r['group'],r['metric'],r['stage'])
    ref={key(r):r['new'] for r in csv.DictReader((ref_dir/'FULL_RESULTS_ours.csv').open())}
    lines=['# Full synthetic A/B table, method "ours"','',f'- this run: `{new_dir}`',
           f'- reference: {label}, `{ref_dir}`','','delta = this run − reference (same unit as the row). '
           'Counts are compared as strings. `—` = not available.','',
           f'| metric | stage | this run | {label} | delta |','|---|---|---:|---:|---:|']
    group=None
    for r in csv.DictReader((new_dir/'FULL_RESULTS_ours.csv').open()):
        if r['group']!=group:
            group=r['group'];lines.append(f'| **{group}** | | | | |')
        a,b=r['new'],ref.get(key(r),'')
        x,y=number(a),number(b)
        delta='—' if a in ('',None) or b in ('',None) else f'{x-y:+.4f}' if x is not None and y is not None \
            else 'same' if a==b else 'differs'
        lines.append(f"| {r['metric']} | {r['stage']} | {text(a)} | {text(b)} | {delta} |")
    (new_dir/'FULL_TABLE.md').write_text('\n'.join(lines)+'\n')


compare(out,REFERENCE)
shutil.rmtree(base.parent)
print('Synthetic takeover evaluation complete',flush=True)
