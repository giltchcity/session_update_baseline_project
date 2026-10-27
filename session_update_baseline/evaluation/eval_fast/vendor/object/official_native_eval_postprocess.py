#!/usr/bin/env python3
"""Call official plotting functions; preserve their NaN and equal-sample means.

No scoring or averaging formula is reimplemented. Repeated raw rows are first
verified identical and retained in the source file. The unique online view is
an explanation of the native output, not another object/change evaluator.
"""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OFFICIAL=ROOT/'official/src/khronos_eval'

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('raw_csv',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    rows=list(csv.DictReader(args.raw_csv.open()))
    assert rows
    unique={}; repetitions={}
    for r in rows:
        key=(r['Name'],r['Query'])
        assert key not in unique or unique[key]==r,('Repeated raw row changed',key)
        unique[key]=r;repetitions[key]=repetitions.get(key,0)+1
    data={name:{} for name in rows[0] if name not in ['Name','Query']}
    for (name,query),r in unique.items():
        for column in data:data[column].setdefault(name,{})[query]=float(r[column])
    module_path=OFFICIAL/'plotting/utils.py'
    spec=importlib.util.spec_from_file_location('unchanged_official_plotting_utils',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.augment_object_metrics(data)
    metrics=['ObjectPrecision','ObjectRecall','ObjectF1']
    metrics += [prefix+metric+suffix for prefix in ['Appeared','Disappeared','Change']
                for metric in ['Precision','Recall','F1'] for suffix in ['', 'Incl']]
    # Upstream names without Incl are the matched-association change metric.
    names=sorted({k[0] for k in unique},key=int)
    queries=sorted({k[1] for k in unique},key=int)
    assert len(names)==len(queries),'Original pipeline output is not a single diagonal query per map'
    for name,query in zip(names,queries):assert (name,query) in unique
    args.output.mkdir(parents=True,exist_ok=False)
    output=[]
    for name,query in zip(names,queries):
        row=dict(unique[name,query],raw_repetitions=repetitions[name,query])
        for metric in metrics:
            val=data[metric][name][query]
            row[metric]=float(val) if np.isfinite(val) else 'NaN'
        output.append(row)
    with (args.output/'native_unique_online_rows.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    summary={}
    for metric in metrics:
        values=module.get_4d_data_slice(data[metric],names,queries,'Online')
        defined=np.isfinite(values)
        summary[metric]=dict(mean=float(np.nanmean(values)) if defined.any() else None,
            defined_samples=int(defined.sum()),total_samples=len(values),nan_samples=int((~defined).sum()))
    result=dict(schema='native-official-postprocessing-v1',raw_rows=len(rows),unique_online_rows=len(unique),
        repetitions={f'{name}/{query}':n for (name,query),n in repetitions.items()},
        duplicate_rows_identical=True,raw_fields_in_unique_rows_unchanged=True,
        temporal_mode='Online diagonal in official get_4d_data_slice',
        aggregation='unweighted np.nanmean of native query samples, as official tables.py; not duration integration',
        zero_tp_f1='Official compute_metrics_single may return NaN when TP=0. Preserved, not changed to zero.',
        matched_change='ChangePrecision/Recall/F1',inclusive_change='ChangePrecisionIncl/RecallIncl/F1Incl',
        raw_sha256=hashlib.sha256(args.raw_csv.read_bytes()).hexdigest(),
        plotting_source=str(module_path),plotting_source_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),
        summary=summary)
    (args.output/'NATIVE_ONLINE_SUMMARY.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'raw_rows':len(rows),'unique_online_rows':len(unique),'summary':summary}))

if __name__=='__main__':main()
