#!/usr/bin/env bash
# Rules 1 and 2 for every analysis run: refuse uncommitted code; write commit.txt and config.json (argv, every environment
# variable except credentials, commit, software) into OUT_DIR before the command runs; stdout/stderr go to OUT_DIR/run.log.
#   experiments/run_logged.sh OUT_DIR COMMAND [ARGS...]
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd); OUT=$1; shift
if [ -n "$(git -C "$REPO" status --porcelain)" ]; then echo "REFUSED (rule 1): uncommitted changes in $REPO" >&2; git -C "$REPO" status --porcelain >&2; exit 3; fi
mkdir -p "$OUT"
git -C "$REPO" rev-parse HEAD > "$OUT/commit.txt"
python3 - "$OUT/config.json" "$REPO" "$@" <<'PY'
import json, os, sys, platform, subprocess
out, repo, argv = sys.argv[1], sys.argv[2], sys.argv[3:]
hidden = ('TOKEN', 'SECRET', 'PASSWORD', 'KEY', 'CREDENTIAL')
env = {k: ('<credential, not recorded>' if any(h in k.upper() for h in hidden) else v) for k, v in sorted(os.environ.items())}
commit = subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
json.dump(dict(command=argv, commit=commit, repo=repo, cwd=os.getcwd(), environment=env, python=platform.python_version(),
               note='analysis run: its settings are the command line above and the constants in the committed script'),
          open(out, 'w'), indent=2)
PY
"$@" > "$OUT/run.log" 2>&1
rc=$?; echo "rc=$rc $(date '+%F %T')" > "$OUT/status.txt"; exit $rc
