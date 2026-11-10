# HARNESS COPY of /mnt/d/3Study/ETH/FT/Synthetic_ABC/scripts/prepare_historical_object_eval.py
# (sha256 2a2271c7..., mtime 2026-09-17 04:52). Only changes: the run directory C (was the
# hardcoded DATA/'evaluation_complete_20260917') comes from $HARNESS_OUT_ROOT, and an
# '--ours-only' method selector was added next to the existing '--khronos-only'.
# Manifest construction below is byte-identical to the original.
# 10-21: the unit list (EVAL_SCOPE_UNITS.csv) comes from $HARNESS_SCOPE_UNITS when set (the 29-unit scope of
# update_layer/eval/synthetic_scope_29: the large sofa and its throw pillows as two units, as the mapping input).
"""Coarse causal (T,t) manifests for the unchanged official Object evaluator."""
from cvpr_eval_common import *
from scipy.spatial import cKDTree
from functools import lru_cache
import sys
import os
native_only='--native-only' in sys.argv
C=Path(os.environ['HARNESS_OUT_ROOT']);assert 'evaluation_complete_20260917' not in str(C.resolve()),'refusing to write into the original run directory'
P=OUT/'paper_aligned';DEST=C/'history';DEST.mkdir(exist_ok=True)
unit_rows=list(csv.DictReader(Path(os.environ.get('HARNESS_SCOPE_UNITS',P/'EVAL_SCOPE_UNITS.csv')).open()))
selected={int(i) for u in unit_rows for i in u['core_members'].split(',')};labels={CAT[i]['semantic_id'] for i in selected}
@lru_cache(maxsize=32)
def trees_at(s,frame):
    trees={}
    for sem in labels:
        sets=[]
        for inside in [True,False]:
            pp=[]
            for i in CAT:
                if CAT[i]['semantic_id']!=sem or ((i in selected)!=inside) or not GT[s]['render_visible'][frame,IX[s][i]]:continue
                pp.append(sample_surface(*worldmesh(s,i,frame),4096,1900+i))
            sets.append(cKDTree(np.concatenate(pp)) if pp else None)
        trees[sem]=sets
    return trees
for method in (['khronos'] if '--khronos-only' in sys.argv else ['ours'] if '--ours-only' in sys.argv else ['ours','khronos']):
    for s in 'ab':
        root=C/f'{method}/native/{s}';native=[json.loads(l) for l in (root/'meshes.jsonl').read_text().splitlines()]
        original=json.loads((P/s/'gt_manifest.json').read_text());objects=original['objects'];oi={o['id']:j for j,o in enumerate(objects)}
        units=[dict(id=j+1,name=u['unit'],semantic=CAT[int(u['core_members'].split(',')[0])]['semantic_id'],
                    members=[oi[int(i)] for i in u['core_members'].split(',')]) for j,u in enumerate(unit_rows)]
        end=float(GT[s]['time_s'][-1])
        grid=sorted(set(([0.]+[max(0.,(r['stamp_ns']-ORIGIN[s])/1e9) for r in native]+[end]) if native_only else (np.arange(0,end,10).tolist()+[end])));queries=[]
        for T in grid:
            Tstamp=ORIGIN[s]+round(T*1e9);ix=np.searchsorted([r['stamp_ns'] for r in native],Tstamp,side='right')-1
            # B contains a native inherited seed in ours. No invented seed or
            # previous-session information is supplied to independent Khronos.
            row=native[ix] if ix>=0 else None;pred=[]
            if row:
                for o in row['objects']:
                    pts=np.fromfile(root/(o['prefix']+'.f32'),np.float32).reshape(-1,3) if o['vertices'] else np.zeros((0,3),np.float32)
                    pred.append((o,pts[::max(1,len(pts)//4096)]))
            for t in ([T] if native_only else [x for x in grid if x<=T]):
                frame=int(np.clip(round(t*30),0,len(GT[s]['time_s'])-1));trees=trees_at(s,frame);ignore=[]
                for o,pts in pred:
                    if not o['vertices'] or o['semantic'] not in labels:ignore.append(o['id']);continue
                    main,detail=trees[o['semantic']]
                    if detail is None:continue
                    di=detail.query(pts)[0];dm=main.query(pts)[0] if main else np.full(len(pts),np.inf)
                    if ((di<=.1)&(di<dm)).mean()>.5:ignore.append(o['id'])
                queries.append(dict(robot_time_s=T,query_time_s=t,snapshot_index=int(ix),snapshot_stamp_ns=row['stamp_ns'] if row else None,
                    query_stamp_ns=ORIGIN[s]+round(t*1e9),frame=frame,ignore_prediction_nodes=ignore,
                    transforms=[GT[s]['T_world_object'][frame,IX[s][o['id']]].tolist() for o in objects],
                    visible=[bool(GT[s]['render_visible'][frame,IX[s][o['id']]]) for o in objects]))
            print('HISTORY_MANIFEST',method,s,T,flush=True)
        suffix='online_manifest' if native_only else 'manifest'
        (DEST/f'{method}_{s}_{suffix}.json').write_text(json.dumps(dict(objects=objects,units=units,queries=queries)))
(DEST/'PROTOCOL.json').write_text(json.dumps(dict(scope='same fixed28 units and geometry-based excluded-detail filtering as Online Object',
    queries='10-second T,t grid plus terminal; t<=T; latest native snapshot<=T; no next-snapshot backfill',
    native='same original ObjectEvaluator SURFACE association, same-semantics, max_matching_distance2m and compensate_missegmentation',
    interpretation='Object belief at past t using the native DSG available by T, including its presence intervals and retained geometry; not just reusing the t snapshot',
    aggregation='uniform average over distinct sampled historical pairs t<T; online diagonal and terminal separated; not continuous full4D time integral',
    excluded='Do not consume emitted Appeared/Disappeared columns: this GT manifest specifies per-query objects, not full temporal change labels',
    panoptic='D3 process benchmark only; no claim of equivalent retrospective object graph interface'),indent=2))
