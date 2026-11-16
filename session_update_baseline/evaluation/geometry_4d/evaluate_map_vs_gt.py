"""Temporal mesh F1 on Home (synthetic A -> B): every map slice against the Isaac GT mesh of its own moment, 5 cm.

  precision  150k area samples of the map slice (seed 20260917) -> distance to the complete GT mesh of that moment
  recall     150k area samples of the GT reference of that moment (seed 20261115) -> distance to the map slice
  reference  GEO4D_RECALL=room (default, the paper's "scheme C", user 11-17): every GT entity whole at its pose of that
             moment, except walls, floors and ceilings, which are replaced by their room-facing part (room_facing_shell.npz,
             made by room_facing_shell.py: the slab above the suspended ceiling, the slab under the floor and the outer
             faces of the walls, which no camera position reaches, are left out). Furniture backs and undersides count.
             GEO4D_RECALL=complete: the whole GT mesh (the 11-15 numbers; about half of it is shell no camera sees).
  queries    one per emitted map slice of the first --methods entry (the first GT frame at or after the slice stamp)
             plus each session's last frame; GEO4D_QUERY_TIMES=CSV (session,frame) gives the frames instead.
  inputs     HARNESS_OUT_ROOT/<method>/native/{a,b}/meshes.jsonl (synthetic_ram.py export or write_native), Panoptic as
             HARNESS_OUT_ROOT/panoptic/session_{a,b}/native; HARNESS_CAUSAL_BANK and HARNESS_SCOPE_UNITS as the synthetic
             harness (only the unit categories are read); PYTHONPATH with Synthetic_ABC/scripts (cvpr_eval_common).
  output     HARNESS_OUT_ROOT/geometry/MAP_VS_GT.csv, one row per (method, session, frame).
Origin: eval/synthetic_base_v38/harness/evaluate_cross_session_geometry.py (map loading and sampling code unchanged);
evaluation/geometry_4d/README.md has the procedure, the runs and the check values.
usage: evaluate_map_vs_gt.py --methods METHOD [METHOD ...]
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
            if (root/'meshes.jsonl').exists():   # one manifest for the whole run (the main table's Panoptic export)
                rr=[dict(json.loads(l),directory=root) for l in (root/'meshes.jsonl').read_text().splitlines()]
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
RECALL=os.environ.get('GEO4D_RECALL','room');assert RECALL in ('room','complete'),'GEO4D_RECALL must be room or complete'
ROOM=RECALL=='room'
SHELL_PATH=Path(os.environ.get('GEO4D_SHELL','/home/jixian/Desktop/FT/wt_takeover_runs/geometry_4d_20261114/room_facing_shell.npz'))
assert not ROOM or SHELL_PATH.exists(),f'{SHELL_PATH} missing: run room_facing_shell.py first (README.md)'
SHELL=np.load(SHELL_PATH) if ROOM else None
rows=[]
for s in 'ab':
    gstamp=[ORIGIN[s]+round(float(t)*1e9) for t in GT[s]['time_s']]
    frames=sorted({min(int(np.searchsorted(gstamp,r['stamp_ns'],side='left')),len(gstamp)-1) for r in native[args.methods[0]][s]}|{len(gstamp)-1})
    if os.environ.get('GEO4D_QUERY_TIMES'):   # given query frames (session,frame csv) instead of the method's own slices
        frames=[int(r['frame']) for r in csv.DictReader(open(os.environ['GEO4D_QUERY_TIMES'])) if r['session']==s]
    for frame in frames:
        stamp=ORIGIN[s]+round(float(GT[s]['time_s'][frame])*1e9)
        gt=o3d.t.geometry.RaycastingScene(nthreads=4);parts=[(SHELL['vertices'],SHELL['faces'])] if ROOM else []
        for i in REG:
            if i not in IX[s] or not GT[s]['render_visible'][frame,IX[s][i]]:continue
            v,f=worldmesh(s,i,frame)
            if not len(f):continue
            gt.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(v),o3d.core.Tensor(f)))
            if not ROOM or REG[i]['class'] not in ('wall','floor','ceiling'):parts.append((v,f))
        gv,gf=combine(parts);gtsamples=sample_surface(gv,gf,150000,20261115)
        for method in args.methods:
            row=snapshot(method,s,stamp);v,f=prediction(row);pred=scene_for(v,f)
            samples=sample_surface(v,f,150000,20260917);d=distance(gt,samples);rec=distance(pred,gtsamples)
            for th in [.05]:
                p=float((d<=th).mean()) if len(d) else 0.;r=float((rec<=th).mean())
                rows.append(dict(method=method,session=s,frame=frame,time_s=float(GT[s]['time_s'][frame]),
                    prediction_stamp_ns=row['stamp_ns'] if row else None,threshold_m=th,Precision=p,Recall=r,
                    F1=(2*p*r/(p+r) if p+r else 0.),map_samples=len(d),gt_samples=len(rec)))
            print('MAP_VS_GT',method,s,frame,flush=True)
        write('MAP_VS_GT.csv',rows)
