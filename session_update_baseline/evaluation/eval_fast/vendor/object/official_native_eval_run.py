#!/usr/bin/env python3
"""Run the unchanged official pipeline in a new evaluation directory."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess

ROOT=Path(__file__).resolve().parents[2]
OFFICIAL=ROOT/'official'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--map',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--only-final-argument',choices=['true','false'],default='false',
                    help='Passed verbatim. Upstream currently ignores this configuration field.')
    args=ap.parse_args()
    source=args.map.resolve(strict=True);config=args.config.resolve(strict=True)
    binary=OFFICIAL/'install/lib/khronos_eval/exp_pipeline'
    assert binary.is_file()
    args.output.mkdir(parents=True,exist_ok=False)
    output=args.output.resolve()
    (output/'final.4dmap').symlink_to(source)
    (output/'experiment_log.txt').write_text('[INFO] Evaluation input directory only. Source mapper completion must be checked separately.\n')
    command=[str(binary),str(config),str(output),'false','true',args.only_final_argument]
    # The fixed environment script selects clean official libraries; positional
    # arguments are passed as argv, never interpolated into shell source.
    shell=['/bin/bash','-c','source "$1"; shift; exec "$@"','official-native-eval',
           str(OFFICIAL/'environment.sh'),*command]
    provenance=dict(schema='unchanged-official-pipeline-run-v1',command=command,
        source_map=str(source),source_map_sha256=sha(source),config=str(config),config_sha256=sha(config),
        executable=str(binary),executable_sha256=sha(binary),
        source_object_evaluator_sha256=sha(OFFICIAL/'src/khronos_eval/src/object_evaluator.cpp'),
        source_pipeline_evaluator_sha256=sha(OFFICIAL/'src/khronos_eval/src/pipeline_evaluator.cpp'),
        requested_only_final_argument=args.only_final_argument,scoring_source_modified=False,
        aggregation_changed=False,native_raw_duplicate_rows_preserved=True)
    (output/'NATIVE_RUN_PROVENANCE.json').write_text(json.dumps(provenance,indent=2))
    with (output/'native_pipeline.log').open('w') as log:
        result=subprocess.run(shell,stdout=log,stderr=subprocess.STDOUT)
    provenance['returncode']=result.returncode
    raw=output/'results/static_objects.csv'
    provenance['native_static_objects_exists']=raw.is_file()
    if raw.is_file():provenance['native_static_objects_sha256']=sha(raw)
    (output/'NATIVE_RUN_PROVENANCE.json').write_text(json.dumps(provenance,indent=2))
    if result.returncode or not raw.is_file():raise RuntimeError(f'Native evaluation failed; inspect {output}/native_pipeline.log')
    print(raw)

if __name__=='__main__':main()
