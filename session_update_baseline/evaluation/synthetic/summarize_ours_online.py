# HARNESS SUBSET of /mnt/d/3Study/ETH/FT/Synthetic_ABC/scripts/validate_and_summarize_complete.py
# (sha256 964961b0..., mtime 2026-09-17 03:35), restricted to method "ours".
# The original hardcodes C=.../evaluation_complete_20260917 for both the mapper outputs
# (session dirs, *_live) and the evaluation outputs (native, history, online). Here:
#   IN = $HARNESS_IN_ROOT  : ours/session_{a,b}/{control,final.4dmap}, ours_live/{a,b}
#   C  = $HARNESS_OUT_ROOT : ours/native/{a,b}, history/ours_*_online_manifest.json,
#                            online/ours/{a,b}/object.csv, geometry/CURRENT_GEOMETRY_CURVES.csv
# Kept verbatim for ours: run-completeness assertions, RESOURCES row, and the Online Object
# duration-weighted aggregation that produces online/ONLINE_OBJECT_SUMMARY.csv (the source
# of Online_Object_full_duration_F1_pct = F1 and Online_Object_compat_nonempty_F1_pct =
# compatibility_nonempty_output_F1 in MAIN_RESULTS.csv).
# Dropped (not inputs of the four metrics): khronos/panoptic blocks, the 10-s historical
# Object block (history/<m>_<s>_manifest.json + history/<m>/<s>/object.csv), and the
# existence checks for changes/background/dynamics/updates tables. The geometry row-count
# assertion is kept, scaled from 3 methods to 1.
# $HARNESS_SKIP_RUN_CHECKS=1 skips only the mapper-run completeness assertions.
from pathlib import Path
import csv,json,re,math,os
IN=Path(os.environ['HARNESS_IN_ROOT']);C=Path(os.environ['HARNESS_OUT_ROOT'])
assert 'evaluation_complete_20260917' not in str(C.resolve()),'refusing to write into the original run directory'
RUN_CHECKS=os.environ.get('HARNESS_SKIP_RUN_CHECKS','0')!='1'
def read(path):return list(csv.DictReader(path.open()))
def write(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
checks=[];resources=[];online=[];online_queries=[]
for method in ['ours']:
    for s,n in [('a',4186),('b',3555)]:
        d=IN/f'{method}/session_{s}';control=d/'control';state=d if method=='ours' else d/'state'
        live=[json.loads(l) for l in (IN/f'{method}_live/{s}/live_tracks.jsonl').read_text().splitlines()]
        if RUN_CHECKS:
            playback=json.loads((control/'playback_manifest.json').read_text())
            assert playback['frames_available']==playback['frames_published']==n
            assert playback['flow_control']=='ack' and playback['image_scale']==1 and playback['frames_skipped_empty_depth']==0
            assert (control/'khronos.exit_code').read_text().strip()=='0'
            assert len(live)==n and len({r['stamp_ns'] for r in live})==n
            processed=json.loads((control/'processing_manifest.json').read_text())
            assert processed['frames_processed_by_active_window']==n
        assert (state/'final.4dmap').stat().st_size>0
        native=[json.loads(l) for l in (C/f'{method}/native/{s}/meshes.jsonl').read_text().splitlines()]
        assert all(native[j]['stamp_ns']<native[j+1]['stamp_ns'] for j in range(len(native)-1))
        peak=json.loads((IN/f'{method}_live/{s}/terminal_runtime.json').read_text())
        elapsed=live[-1]['wall_since_first_frame_s']
        resources.append(dict(method=method,session=s,frames=n,online_span_seconds=elapsed,online_ack_span_fps=(n-1)/elapsed,
            peak_mapper_rss_mib=peak['peak_mapper_rss_kib']/1024,user_cpu_s=peak['user_cpu_s'],system_cpu_s=peak['system_cpu_s'],
            serialized_map_bytes=(state/'final.4dmap').stat().st_size,native_map_versions=len(native),
            artifact_scope='full native4D archive; not a single terminal mesh'))
        checks.append(dict(method=method,session=s,processed_frames=n,complete=RUN_CHECKS,per_frame_ACK=RUN_CHECKS,native_snapshots=len(native)))
        # ---- verbatim from the original (Online Object aggregation) ----
        om=json.loads((C/f'history/{method}_{s}_online_manifest.json').read_text());oo=read(C/f'online/{method}/{s}/object.csv')
        assert len(oo)==len(om['queries']) and len({(r['Name'],r['Query']) for r in oo})==len(oo)
        orows=[]
        for j,(r,q) in enumerate(zip(oo,om['queries'])):
            assert int(r['Name'])==j and int(r['Query'])==q['query_stamp_ns']
            tp=int(r['NumObjDetected']);fp=int(r['NumObjHallucinated']);fn=int(r['NumObjMissed'])
            t=q['query_time_s'];dt=om['queries'][j+1]['query_time_s']-t if j+1<len(oo) else 0
            assert dt>=0
            orows.append(dict(method=method,session=s,time_s=t,duration_s=dt,TP=tp,FP=fp,FN=fn,
                Precision=tp/(tp+fp) if tp+fp else 0,Recall=tp/(tp+fn) if tp+fn else 0,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0))
        duration=sum(r['duration_s'] for r in orows);assert abs(duration-om['queries'][-1]['query_time_s'])<1e-7
        nonempty=[r for r in orows if r['TP']+r['FP']>0];ne_duration=sum(r['duration_s'] for r in nonempty)
        online.append(dict(method=method,session=s,duration_s=duration,queries=len(orows),
            **{k:sum(r[k]*r['duration_s'] for r in orows)/duration for k in ['Precision','Recall','F1']},
            nonempty_output_duration_s=ne_duration,compatibility_nonempty_output_F1=sum(r['F1']*r['duration_s'] for r in nonempty)/ne_duration,
            **{'final_'+k:orows[-1][k] for k in ['TP','FP','FN','Precision','Recall','F1']}))
        online_queries.extend(orows)
        # ---- end verbatim ----
geometry=read(C/'geometry/CURRENT_GEOMETRY_CURVES.csv')
assert len(geometry)==1*26*5*3
assert len({(r['method'],r['session'],r['frame'],r['scope'],r['threshold_m']) for r in geometry})==len(geometry)
for row in geometry:
    if row['prediction_stamp_ns']:
        origin=1000000000 if row['session']=='a' else 141000000000
        assert int(row['prediction_stamp_ns'])<=origin+round(float(row['time_s'])*1e9)
    for k in ['Precision','Recall','F1']:
        if row[k]:assert 0<=float(row[k])<=1
(C/'online').mkdir(exist_ok=True)
write(C/'RESOURCES.csv',resources)
write(C/'online/ONLINE_OBJECT_SUMMARY.csv',online);write(C/'online/ONLINE_OBJECT_QUERIES.csv',online_queries)
(C/'VALIDATION.json').write_text(json.dumps(dict(passed=True,scope='HARNESS: method ours only; run checks '+('on' if RUN_CHECKS else 'SKIPPED'),
    input_root=str(IN),output_root=str(C),checks=checks,geometry_queries=len(geometry)),indent=2))
print('VALIDATION_PASSED',json.dumps(checks),flush=True)
print('ONLINE',json.dumps(online),flush=True)
