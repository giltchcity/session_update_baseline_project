# 11-01 single-run copy of Synthetic_ABC/scripts/evaluate_live_dynamics.py: unchanged except the live tracks read from
# wt_takeover_runs/eval/single_live_dynamics/{ours,khronos}_live and the tables written to its dynamics/.
"""Evaluate actual native dynamic tracks, never GT-derived predictions.

Centres are observed RGB-D axis-aligned bounding-box centres in the common world
frame. V37 emits these natively; Khronos observer derives these from its native
associated dynamic cluster, retaining its pixel mean as a diagnostic. GT motion comes from temporal
USD transforms. Fixed main-unit scope; detail exclusion is spatial, not ID
assignment to predicted tracks. Full 30 Hz timestamps are evaluated.
"""
from cvpr_eval_common import *
from PIL import Image
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

C=Path('/home/jixian/Desktop/FT/wt_takeover_runs/eval/single_live_dynamics');D=C/'dynamics';D.mkdir(exist_ok=True)
units=list(csv.DictReader((OUT/'paper_aligned/EVAL_SCOPE_UNITS.csv').open()))
core={r['unit']:[int(i) for i in r['core_members'].split(',')] for r in units}
selected={i for ids in core.values() for i in ids};events=json.loads((OUT/'GT_EVENTS.json').read_text())
def write(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (D/name).open('w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
details=[];associations=[];summaries=[];tracking_rows=[];tracking_summary=[]
for s in 'ab':
    n=len(GT[s]['time_s']);live={}
    for method in ['ours','khronos']:
        rr=[json.loads(l) for l in (C/f'{method}_live/{s}/live_tracks.jsonl').read_text().splitlines()]
        assert len(rr)==n,(method,s,len(rr),n)
        assert all(abs(r['stamp_ns']-(ORIGIN[s]+round(float(GT[s]['time_s'][f])*1e9)))<=2 for f,r in enumerate(rr))
        live[method]=rr
    moving={i:np.r_[movement(s,i),False]|np.r_[False,movement(s,i)] for i in selected}
    K=np.asarray(json.loads((DATA/f'session_{s}/VALIDATION.json').read_text())['K']);vv,uu=np.mgrid[:480,:680]
    rayx=(uu+.5-K[0,2])/K[0,0];rayy=(vv+.5-K[1,2])/K[1,1]
    for frame in range(n):
        active={i for i in selected if moving[i][frame] and GT[s]['visible_pixel_count'][frame,IX[s][i]]>=50 and GT[s]['render_visible'][frame,IX[s][i]]}
        any_prediction=any(any(t['dynamic'] for t in live[m][frame]['tracks']) for m in live)
        centroids={};counts={};bounds={}
        if active or any_prediction:
            z=np.asarray(Image.open(DATA/f'session_{s}/rgbd/{frame:06d}_depth.tiff'))
            ids=np.asarray(Image.open(DATA/f'session_{s}/instance_labels/{frame:06d}_segmentation.png'))
            cam=np.stack([rayx*z,rayy*z,z],-1);valid=np.isfinite(z)&(z>=.1)&(np.linalg.norm(cam,axis=-1)<=15)&(ids>0)
            T=np.loadtxt(DATA/f'session_{s}/rgbd/{frame:06d}_pose.txt')
            flat=ids[valid].astype(int);p=cam[valid]@T[:3,:3].T+T[:3,3]
            num=np.bincount(flat)
            for i in np.flatnonzero(num>=50):
                q=p[flat==i];lo=q.min(0);hi=q.max(0)
                bounds[i]=(lo,hi);centroids[i]=(lo+hi)/2;counts[i]=int(num[i])
        targets=[];allmajor=[];visible_targets=[]
        for u,ids in core.items():
            visible=[i for i in ids if i in centroids]
            if not visible:continue
            center=(np.min([bounds[i][0] for i in visible],axis=0)+np.max([bounds[i][1] for i in visible],axis=0))/2
            allmajor.append(center)
            visible_targets.append(dict(unit=u,center=center,semantic=CAT[ids[0]]['semantic_id'],moving=any(i in active for i in visible)))
            if any(i in active for i in visible):targets.append(dict(unit=u,center=center))
        excluded=np.asarray([p for i,p in centroids.items() if i not in selected])
        for method in live:
            # Separate online geometric tracking from the native dynamic flag.
            # A tracker may follow a moving instance while labelling it static.
            # Evaluate its actual observed positions; never synthesize them from GT.
            tracked=live[method][frame]['tracks']
            tc=np.full((len(tracked),len(visible_targets)),1e6)
            for a,p in enumerate(tracked):
                for b,g in enumerate(visible_targets):
                    if p['semantic']<0 or p['semantic']==g['semantic']:
                        tc[a,b]=np.linalg.norm(np.asarray(p['centroid'])-g['center'])
            ai,bi=linear_sum_assignment(np.where(tc<=.5,tc,1e6))
            match={b:a for a,b in zip(ai,bi) if tc[a,b]<=.5}
            for b,g in enumerate(visible_targets):
                if not g['moving']:continue
                a=match.get(b);p=tracked[a] if a is not None else None
                tracking_rows.append(dict(method=method,session=s,frame=frame,time_s=float(GT[s]['time_s'][frame]),unit=g['unit'],
                    matched=p is not None,track_id=p['id'] if p else None,native_dynamic_flag=p['dynamic'] if p else None,
                    center_error_m=float(tc[a,b]) if p else None,predicted_x=p['centroid'][0] if p else None,
                    predicted_y=p['centroid'][1] if p else None,predicted_z=p['centroid'][2] if p else None))
            candidates=[t for t in live[method][frame]['tracks'] if t['dynamic']]
            predictions=[];ignored=0
            for p in candidates:
                center=np.asarray(p['centroid'])
                dm=min((np.linalg.norm(center-q) for q in allmajor),default=np.inf)
                de=np.min(np.linalg.norm(excluded-center,axis=1)) if len(excluded) else np.inf
                if de<=.5 and de<dm:ignored+=1
                else:predictions.append(p)
            cost=np.zeros((len(predictions),len(targets)))
            for a,p in enumerate(predictions):
                for b,g in enumerate(targets):cost[a,b]=np.linalg.norm(np.asarray(p['centroid'])-g['center'])
            aa,bb=linear_sum_assignment(np.where(cost<=.5,cost,1e6));pairs=[(a,b) for a,b in zip(aa,bb) if cost[a,b]<=.5]
            tp=len(pairs);dt=float(GT[s]['time_s'][frame+1]-GT[s]['time_s'][frame]) if frame+1<n else 0.
            details.append(dict(method=method,session=s,frame=frame,time_s=float(GT[s]['time_s'][frame]),duration_s=dt,
                **metric(tp,len(predictions)-tp,len(targets)-tp),ignored_detail_predictions=ignored))
            matched={b:a for a,b in pairs}
            for b,g in enumerate(targets):
                a=matched.get(b)
                associations.append(dict(method=method,session=s,frame=frame,unit=g['unit'],
                    track_id=predictions[a]['id'] if a is not None else None,matched=a is not None,
                    centroid_error_m=float(cost[a,b]) if a is not None else None))
        if frame%900==0:print('DYNAMICS',s,frame,flush=True)
    for method in live:
        tr=[r for r in tracking_rows if r['method']==method and r['session']==s];ok=[r for r in tr if r['matched']]
        switches=0;last={}
        for r in ok:
            old=last.get(r['unit'])
            if old and old['frame']+1==r['frame'] and old['track_id']!=r['track_id']:switches+=1
            last[r['unit']]=r
        tracking_summary.append(dict(method=method,session=s,GT_visible_motion_unit_frames=len(tr),matched_unit_frames=len(ok),
            observed_motion_track_coverage=len(ok)/len(tr) if tr else None,
            matched_observed_AABB_center_RMSE_m=float(np.sqrt(np.mean([r['center_error_m']**2 for r in ok]))) if ok else None,
            native_dynamic_flagged_matches=sum(r['native_dynamic_flag'] for r in ok),consecutive_matched_identity_switches=switches))
        rr=[r for r in details if r['method']==method and r['session']==s]
        weighted=metric(*[sum(r[k]*r['duration_s'] for r in rr) for k in ['TP','FP','FN']])
        matches=[r for r in associations if r['method']==method and r['session']==s]
        errors=[r['centroid_error_m'] for r in matches if r['matched']]
        switches=0;last={}
        for r in matches:
            if not r['matched']:continue
            old=last.get(r['unit'])
            if old and r['frame']==old['frame']+1 and r['track_id']!=old['track_id']:switches+=1
            last[r['unit']]=r
        summaries.append(dict(method=method,session=s,**{'time_'+k:v for k,v in weighted.items()},
            GT_visible_motion_unit_frames=len(matches),matched_unit_frames=len(errors),
            trajectory_coverage=len(errors)/len(matches) if matches else None,
            matched_observed_AABB_center_RMSE_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            consecutive_matched_identity_switches=switches))
write('LIVE_DYNAMICS_FRAMES.csv',details);write('LIVE_DYNAMICS_ASSOCIATIONS.csv',associations);write('LIVE_DYNAMICS_SUMMARY.csv',summaries)
write('VISIBLE_MOTION_TRACKING.csv',tracking_rows);write('VISIBLE_MOTION_TRACKING_SUMMARY.csv',tracking_summary)
(D/'PROTOCOL.json').write_text(json.dumps(dict(predictions='native is_dynamic tracks at their actual last_seen input timestamp',
    GT='temporal USD motion; >=50 visible valid-depth pixels; observed surface AABB centre, not prim origin',
    scope='fixed28 main units; physical core members of same unit pooled for GT observed AABB',
    prediction_centres='V37 native last_centroid is observed AABB centre; Khronos observer exports AABB of its actual associated dynamic cluster, preserving native pixel mean separately',
    tracking='Separate VISIBLE_MOTION_TRACKING uses ALL actual observed native tracks, one-to-one with all visible major units by semantic/AABB-centre distance<=0.5m; reports moving-target coverage and error, not motion-detection P/R/F1 or native stored4D trajectory success',
    association='one-to-one nearest centroid within 0.5m; no GT identity supplied to predicted tracks; no interpolation',
    detail_exclusion='dynamic prediction nearer an observed excluded detail centroid within0.5m than any main-body centroid is ignored',
    aggregation='TP/FP/FN integrated over input time; static intervals included for motion FP',
    limitation='Active-window live Dynamics, not retrospective trajectory repair or original full4D aggregate'),indent=2))
print('DYNAMICS_SUMMARY',json.dumps(summaries),flush=True)
