# Workspace Authority

Updated: 2026-08-13

## Canonical repositories

- Code, configurations, runtime adapters, and compact experiment evidence:
  `https://github.com/giltchcity/session_update_baseline_project`
- Paper and manuscript sources:
  `https://github.com/giltchcity/myncv---rpsl---daicma`

The paper repository is intentionally separate. The baseline code must not be
maintained in multiple editable local directories.

## Local layout

The Git worktree is the authority. On the current workstation,
`/home/jixian/Desktop/FT/session_update_baseline` is a convenience link to the
`session_update_baseline/` directory inside this worktree. Edits made through
either path therefore modify the same files and appear immediately in
`git status`.

Generated builds, installed overlays, logs, datasets, run directories, maps,
meshes, and external research repositories remain outside version control.

## Snapshot status

Branch `codex/unified-baseline-20260813` captures the latest local source even
where correctness is still under review. In particular, committing this
snapshot does not certify the end-of-session flush, end-to-end ACK semantics,
cross-session instance identity, or final-map serialization behavior.
