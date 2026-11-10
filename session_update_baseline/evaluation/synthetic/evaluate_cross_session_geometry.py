# HARNESS COPY of /mnt/d/3Study/ETH/FT/Synthetic_ABC/scripts/evaluate_cross_session_geometry.py
# (sha256 71adae97..., mtime 2026-09-17 03:16). Only change: the run directory C (was the
# hardcoded DATA/'evaluation_complete_20260917') comes from $HARNESS_OUT_ROOT, and the
# GT-only causal reference bank (was C/'causal_reference') from $HARNESS_CAUSAL_BANK.
# Metric code below is byte-identical to the original.
# 10-21: the unit list (EVAL_SCOPE_UNITS.csv) comes from $HARNESS_SCOPE_UNITS when set (the 29-unit scope of
# update_layer/eval/synthetic_scope_29: the large sofa and its throw pillows as two units, as the mapping input).
"""Matched-resolution current geometry and second-visit accumulation curves.

Exact temporal USD geometry supplies surface distance. Causal observed surface
eligibility supplies recall. This is an adaptation of Panoptic MAD/Coverage,
not a claim to reproduce its original dataset or reference sampling.
"""
from cvpr_eval_common import *
import open3d as o3d
from functools import lru_cache
import argparse
import os

arg=argparse.ArgumentParser();arg.add_argument('--methods',nargs='+',default=['ours','khronos','panoptic']);args=arg.parse_args()
C=Path(os.environ['HARNESS_OUT_ROOT']);assert 'evaluation_complete_20260917' not in str(C.resolve()),'refusing to write into the original run directory'
DEST=C/'geometry';DEST.mkdir(exist_ok=True)
BANK=Path(os.environ['HARNESS_CAUSAL_BANK']);records=json.loads((BANK/'manifest.json').read_text());refmeta={r['key']:r for r in records}
PAPER=OUT/'paper_aligned';units=list(csv.DictReader(Path(os.environ.get('HARNESS_SCOPE_UNITS',PAPER/'EVAL_SCOPE_UNITS.csv')).open()))
major={int(i) for r in units for i in r['core_members'].split(',')}
attached={int(i) for r in units for i in r['all_members'].split(',')}-major
classes=json.loads((DATA/'registry.json').read_text())['classes'];structsem={7,8,10,13,16,17,18,21,34,44,45}
def category(i):
    if i==0 or classes[REG[i]['class']] in structsem:return 'structure'
    if i in major:return 'major_body'
    if i in attached:return 'group_attachments'
    return 'other_detail'
def key(s,i,f):return str(i)+':'+','.join(map(str,np.round(GT[s]['T_world_object'][f,IX[s][i]].ravel(),5)))
@lru_cache(maxsize=256)
def refload(file):
    a=np.load(BANK/file);return a['points'],a['first_stamp_ns']
native={}
for method in args.methods:
    native[method]={}
    for s in 'ab':
        if method=='panoptic':
            root=C/f'panoptic/session_{s}/native';rr=[]
            for path in root.glob('*/meshes.jsonl'):
                r=json.loads(path.read_text());r['directory']=path.parent;rr.append(r)
        else:
            root=C/f'{method}/native/{s}'
            rr=[dict(json.loads(l),directory=root) for l in (root/'meshes.jsonl').read_text().splitlines()]
        native[method][s]=sorted(rr,key=lambda x:x['stamp_ns'])
def snapshot(method,s,stamp):
    rr=native[method][s];at=np.searchsorted([r['stamp_ns'] for r in rr],stamp,side='right')-1
    if at>=0:return rr[at]
    if s=='b' and method!='khronos':return native[method]['a'][-1]
    return None
def combine(pp):
    if not pp:return np.zeros((0,3),np.float32),np.zeros((0,3),np.uint32)
    vv=[];ff=[];offset=0
    for v,f in pp:vv.append(v);ff.append(f+offset);offset+=len(v)
    return np.concatenate(vv),np.concatenate(ff)
def prediction(row):
    if row is None:return combine([])
    pp=[];root=row['directory']
    for o in row['objects']:
        if o['current'] and o['faces']:
            pp.append((np.fromfile(root/(o['prefix']+'.f32'),np.float32).reshape(-1,3),np.fromfile(root/(o['prefix']+'.u32'),np.uint32).reshape(-1,3)))
    bg=row.get('background_prefix')
    if bg and (root/(bg+'.u32')).exists():
        pp.append((np.fromfile(root/(bg+'.f32'),np.float32).reshape(-1,3),np.fromfile(root/(bg+'.u32'),np.uint32).reshape(-1,3)))
    return combine(pp)
def write(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (DEST/name).open('w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
previous=DEST/'CURRENT_GEOMETRY_CURVES.csv'
rows=[r for r in csv.DictReader(previous.open()) if r['method'] not in args.methods] if previous.exists() else []
stable_hits={};stable_ref=None
for s in 'ab':
    frames=[len(GT[s]['time_s'])-1] if s=='a' else sorted(set(range(0,len(GT[s]['time_s']),150))|{len(GT[s]['time_s'])-1})
    for frame in frames:
        stamp=ORIGIN[s]+round(float(GT[s]['time_s'][frame])*1e9)
        gt=o3d.t.geometry.RaycastingScene(nthreads=4);gids={};bg=o3d.t.geometry.RaycastingScene(nthreads=4)
        for i in REG:
            if i not in IX[s] or not GT[s]['render_visible'][frame,IX[s][i]]:continue
            v,f=worldmesh(s,i,frame)
            if not len(f):continue
            mesh=o3d.t.geometry.TriangleMesh(o3d.core.Tensor(v),o3d.core.Tensor(f))
            gid=gt.add_triangles(mesh);gids[gid]=category(i)
            if category(i)=='structure':bg.add_triangles(mesh)
        refs=[];cats=[];stable=[]
        for i in [0]+list(REG):
            if i and (i not in IX[s] or not GT[s]['render_visible'][frame,IX[s][i]]):continue
            k=key(s,i,frame) if i else 'structure';r=refmeta.get(k)
            if r is None:continue
            p,t=refload(r['file']);p=p[t<=stamp]
            if not len(p):continue
            surf=bg if i==0 else gt
            q=surf.compute_closest_points(o3d.core.Tensor(p),nthreads=4)['points'].numpy()
            q=q[np.linalg.norm(q-p,axis=1)<=.05]
            refs.append(q);cats.extend([category(i)]*len(q))
            # Fixed A-observed stable reference: no later viewpoint can raise
            # its denominator, so B growth reflects retention and correction.
            if s=='a':
                is_stable=i==0 or (i in IX['b'] and all(GT[x]['render_visible'][:,IX[x][i]].all() and
                    np.max(np.abs(GT[x]['T_world_object'][:,IX[x][i]]-GT['a']['T_world_object'][0,IX['a'][i]]))<1e-5 for x in 'ab'))
                if is_stable:stable.append(q)
        reference=np.concatenate(refs) if refs else np.zeros((0,3),np.float32);cats=np.asarray(cats)
        if s=='a':stable_ref=np.concatenate(stable) if stable else np.zeros((0,3),np.float32)
        for method in args.methods:
            row=snapshot(method,s,stamp);v,f=prediction(row);pred=scene_for(v,f)
            samples=sample_surface(v,f,150000,20260917);d=distance(gt,samples);rec=distance(pred,reference)
            closest=gt.compute_closest_points(o3d.core.Tensor(samples),nthreads=4) if len(samples) else None
            pcats=np.array([gids[int(i)] for i in closest['geometry_ids'].numpy()]) if closest is not None else np.array([])
            for scope in ['all','structure','major_body','group_attachments','other_detail']:
                da=d if scope=='all' else d[pcats==scope];dr=rec if scope=='all' else rec[cats==scope]
                for th in [.05,.1,.2]:
                    p=float((da<=th).mean()) if len(da) else 0.;r=float((dr<=th).mean()) if len(dr) else None
                    rows.append(dict(method=method,session=s,frame=frame,time_s=float(GT[s]['time_s'][frame]),
                        prediction_stamp_ns=row['stamp_ns'] if row else None,scope=scope,threshold_m=th,
                        Precision=p,Recall=r,F1=(2*p*r/(p+r) if p+r else 0.) if r is not None else None,
                        MAD_m=float(da.mean()) if len(da) else None,prediction_samples=len(da),GT_reference_cells=len(dr),
                        stable_A_reference_coverage=float((distance(pred,stable_ref)<=th).mean()) if scope=='all' and len(stable_ref) else None))
            if frame==len(GT[s]['time_s'])-1:stable_hits[method,s]=distance(pred,stable_ref)
            print('GEOMETRY_CURVE',method,s,frame,flush=True)
        write('CURRENT_GEOMETRY_CURVES.csv',rows)
write('FINAL_GEOMETRY.csv',[r for r in rows if int(r['frame'])==len(GT[r['session']]['time_s'])-1])
previous=DEST/'STABLE_ACCUMULATION.csv'
retention=[r for r in csv.DictReader(previous.open()) if r['method'] not in args.methods] if previous.exists() else []
for method in args.methods:
    for th in [.05,.1,.2]:
        a=stable_hits[method,'a']<=th;b=stable_hits[method,'b']<=th
        retention.append(dict(method=method,threshold_m=th,reference_cells=len(a),coverage_A=float(a.mean()),coverage_B=float(b.mean()),
            retained_fraction_of_A=float(b[a].mean()) if a.any() else None,new_cells=int((b&~a).sum()),lost_cells=int((a&~b).sum())))
write('STABLE_ACCUMULATION.csv',retention)
(DEST/'PROTOCOL.json').write_text(json.dumps(dict(precision='150k triangle-area samples to full exact CURRENT GT',
    recall='Causally eligible observed surface cells projected to exact current GT, including instance-zero structures',
    strata='GT spatial regions: predicted samples assigned by closest current GT entity; not semantic-recognition scores or native-layer quality',
    stable_reference='fixed A-observed surfaces of GT entities static across both full sessions',
    query='latest actually emitted snapshot at or before query; inherited A fallback for ours/Panoptic only',
    thresholds_m=[.05,.1,.2],different_from_legacy='Full predicted surface precision; causal reference; background included. Do not mix with legacy table numbers.'),indent=2))
