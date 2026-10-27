# Goals and acceptance (read before every step)

## The goal
One complete results table, every dataset x every baseline, every cell the legitimate latest result.
- Object matching 0.5 m everywhere (also tell the 3DGS session).
- Evaluation in memory (/dev/shm), no unnecessary disk writes, compressed maps stay compressed.
- TESSE: the bug-fixed official evaluation (each map queried at every map time up to its own, eq. 20; the official
  loop's bug was querying only the map's own time), each method on its own map times. One protocol, no
  re-gridding variants. Ours: the 4D map derived from the run's own snapshots by eval/chain_fill, the same
  definition as 3RScan.
- "Ours" = 4898f5b (290e110 + D1) + f03d47f (memory reading attribution, Flat moved-picture fix), plus any further
  fix only after it passes acceptance.

## Acceptance for every code change (all of it, not only the point just raised)
No metric may drop by more than 0.1 pp against the current ours on any dataset the change can affect:
real A/B/C (object F1, G1, D2, D3), synthetic A/B, Flat A/B (G1), TESSE office/apartment (geometry, object,
dynamic, change), 3RScan (object, change; only if real B/C or synthetic B moved beyond 0.1 pp).
Also look at the maps from several viewpoints, not only the numbers. The implicit requirements stay: floor holes
filled online (no post-processing), person tracking (D1), no object residue meshes, 4D must not affect mapping.

## Rules of work
- Root causes, one definition per thing; no new constants/parameters/switches; no add-only; no two coexisting
  definitions; no withdrawn-then-re-added.
- Check the record (this file, notes/, git log, the session transcript) before calling something new or unknown.
- One heavy task at a time (runs/HEAVY.lock); no self-stop guards; resources shared with other sessions.
- Commit/push only when asked; never delete shared data without asking.

## Status (update after each step)
- [x] Flat moved-picture residue: f03d47f (branch object-memory-resupport, local; push needs the user).
      Flat B G1 90.20 -> 90.11, picture residue 0.51 -> 0 m2; real B unchanged; real C unchanged on shared snapshots,
      G1 +0.34.
- [x] TESSE ours 4D via chain_fill (eval only): office object F1 65.2 -> 65.9, change F1 64.1 -> 77.5.
- [x] Snapshot cadence: change detection requests coalesced when the worker was busy (office 90 of 97 requests ran,
      real C 95 of 113). Fix 5c7e5ec: each request clones its state and is queued, all run in order, no blocking.
      Accepted (with f03d47f): TESSE office 78.7/65.9/98.7/77.5 -> 81.7/69.5/98.7/79.0 (98 snapshots, 2.25 s
      throughout); real B unchanged (G1 95.99); real C object F1 97.46 unchanged, G1 95.36 -> 95.78; synthetic B
      unchanged (mesh F1 -0.003); Flat B G1 90.11. Pushed: origin/object-memory-resupport (f03d47f, 5c7e5ec).
- [ ] Then the fuse-once surface, in two steps, each accepted on real A/B/C, synthetic, Flat, TESSE office:
      (1) votes from stored per-frame readings (same verdicts as now, no re-decoding); (2) identity pixels not fused
      into the background TSDF, objects from their own per-state reconstruction with the official frame buffer
      (every frame, 300 frames), no re-integration. Earlier TSDF-prior attempts failed by putting A into B's TSDF
      as voxels (GPT 09-25, DS 10-20: B precision 89); keep A as mesh memory decided by B's votes.
- [x] (uncommitted) Rare "mapper deadlock": not the mapper. Reproduced twice in real A under load (wavemap jobs); gdb
      (conda gdb + a PR_SET_PTRACER preload, yama scope 1) showed every mapper thread idle waiting for input and the
      player blocked in rcl_wait without timeout: its waits compute min(step, deadline - now) after checking the
      deadline, and under load that comes out negative, which rclpy takes as "wait forever". Fix: one time_left()
      clamped at 0 in the three waits of scripts/nss_flat_ros2_player.py (the copies analysis/ and
      evaluation/runs/nss_flat_ros2_player_resend.py carry the same lines).
- [ ] Speed: votes and sigma re-read every kept frame each round (round 0.6 s -> 7-11 s over real C). Only with
      results held within 0.1 pp.
      - [x] (uncommitted) frames no longer wait for a round: the active window called PresentSurface::integrate
        under the round's lock (held up to 10 s). Frames are packed outside it into an arrival list the round takes.
        TESSE office: all 98 rounds identical; ack 143 -> 62 ms/frame; mapping 16.4 -> 11.5 min; compose 419 -> 307 s.
      - Phase totals before it: office votes 149 / sigma 115 / assign 73 / states 11 s of 420; real C states 211 /
        votes 165 / assign 84 / sigma 45 s of 597. Votes/sigma recount every frame each round by definition (README:
        "with the current surface, noise and all frames"); states re-fold old frames when a state's extent grows.
      - History: 10-22 speed3 (votes accumulated per frame, fill continued by position within h, plus no look-back on
        surface growth) real A G1 -2.04, reverted; 10-23 fuse-once step 1 (faces by reading id) B D3 96.77 -> 85.71,
        reverted. Exact speedups (290e110) are in.
      - [ ] (uncommitted) read once: every frame votes once (fresh frames on all elements; a new element, keyed by its
        vertices / shown index, reads the earlier frames once); a frame whose domain moves replaces its votes on the
        changed pixels; sigma from each frame's residuals on the previous round's present (counting a frame against a
        present that already holds it biased sigma low: real B 13-30 mm vs 15-111 mm).
        In-run comparison with the full recount (no decision change): office fill 0.025% verdicts differ at the end
        (moved-frame replacement removed a 16k spike); real B memory ~10% (mostly duplicates of the present within cm,
        whose per-reading attribution follows the converged layer), fill 0.4%.
        Accepted on real A -> B -> C remapped with it (user: an optimization needs 1-2 runs within 0.1 pp, not the
        full chain; synthetic/Flat/TESSE runs stopped): object F1, D2, D3 identical; G1 A 94.33 -> 94.55,
        B 95.99 -> 96.18, C 95.78 -> 96.16 (B area 372 -> 347 m2: duplicate earlier-session layers dropped);
        views of B/C from five viewpoints: no new holes or residue. Mapping A 8.1 -> 7.0 min, B 8.5 -> 7.2,
        C 27 -> 19.6; surface time C 597 -> 407 s (sigma 45 -> 2, votes 165 -> 50), B 122 -> 97 s.
      - [x] (uncommitted) per-round answers reused where nothing they read changed: face identities, rivals,
        memory-copy test, closed-state ownership. Real A: all 63 rounds identical to real_a_once; assign 9.2 -> 6.9 s.
        (Four-voxel state cells tried for the limit casts: rounds identical, no faster; reverted.)
      - [x] (uncommitted) a state limits the frames before it once, when it starts or stops (State::operator== is
        current + t_l; changedPart/markChanged deleted). Measured on 10-22 as B G1 -0.01. Real A: object 93.04, G1
        94.55 -> 94.56, D2 same; real C (from real_b_once): object 97.46, G1 96.16 -> 96.14, D2/D3 same.
        Real C surface time 597 -> 219 s (states 211 -> 38, votes 165 -> 41, sigma 45 -> 3, assign 84 -> 45).
        Mapping vs official Khronos per visit: A 7.0 vs 5.2 min, B 7.2 vs 9.7, C 18.0 vs 14.5.
      - [x] synthetic B with it (from synthetic_a_once; user: after real C at most one synthetic run): object and
        state rows identical, every other row within 0.006 of read-once. Mapping 14.2 -> 16.2 min (load).
- [x] Evaluation tools into the repository: session_update_baseline/evaluation (17ae40d, pushed): eval_fast,
      TESSE 4D, chain_fill + timeline tools, synthetic, Flat, 0.5 m configs, the fixed pipeline_evaluator.cpp,
      run scripts, GOALS.md. The working copies stay under /home/jixian/Desktop/FT; re-copy after changing them.
- [ ] Final table with all cells.
- [ ] Last: the 4D history stored apart from the snapshots (never read by mapping).
