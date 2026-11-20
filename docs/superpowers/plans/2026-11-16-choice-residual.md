# Choice residual extraction implementation plan

**Goal:** Start the authorized M197 subject-side 64-dimensional extraction and arrange M181 third-round evaluation, reporting Taipei times.

**Architecture:** Preserve the M197 pristine humanv0 projection and occupancy-masked policy convention. Store residual64 plus policy scalars3 as calibrated int8, indexed by game key and original ply below 256. Keep both sides for existing validation queries and official test games. Write small atomic parts with explicit provenance and fail on incomplete coverage.

**Tech Stack:** Existing Python, NumPy, PyTorch, KataGo replay, Slurm.

**Spec:** reports/architecture_v2_2026-11-02.md, M197 and M181.

## Constraints and review focus

- No change to running scripts, production ensemble, fold selection, or restored external data.
- Original M197 masks occupied points only: this is not a complete Go legality mask. Preserve it for comparable features and name the convention explicitly.
- Actual colours after handicap correction, original ply indices, and pass (zero local trunk) must remain aligned.
- Fit quantization on training games only; independently validate error on held-out calibration games. Stop if standardized RMSE exceeds 0.035 or clipping exceeds 0.001.
- Resume must reject changed inputs, calibration, model, code, and malformed parts. Replay failures are explicit failures, never silently omitted.
- Require quota-visible free space with 3 GiB reserve; never delete to make space automatically.

## Task 1: extraction and verification

Files: src/aicup/choice_residual.py (math and storage), scripts/choice_residual_extract.py (replay, calibration, extraction, audit), tests/test_choice_residual.py.

- [x] Write and run failing tests for hand-derived residual/pass/policy mask, quantization, original-ply alignment, and provenance/partial-part rejection.
- [x] Implement math/storage and selective replay, preserving existing replay inputs before ply 256.
- [x] Run focused tests and actual-GPU calibration/smoke via Slurm; compare replay with trunk_extract.replay on ordinary and handicap games.
- [x] Review changes and fix material findings before full launch.
- [x] Launch 8-GPU/96-CPU extraction only after calibration and space forecast pass; follow with exact key/ply coverage audit (audit scheduled; completion still pending).

## Task 2: M181 and timing report

- [x] Queue registered third-round TTA and comparison after successful job 782825; no submission/upload.
- [x] Record job IDs, measured throughput, Taipei ETA and queue uncertainty in takeover status/report.

## Execution ledger

2026-11-16 14:34: Existing user-approved design is the implementation authority. New files only in current pr40 research checkout: running and untracked scripts depend on this cwd, so a separate checkout would omit their inputs. No broad staging or cleanup.
Calibration is a storage transform, not model selection. Training extraction may contain existing dataset folds, but does not authorize fold 0 scoring or use in the final ensemble.

14:38: Initial tests: Slurm 785612 RED (four missing-implementation failures), 785618 GREEN (4 passed). First actual calibration 785622 (2,000 games); project suite 785623. M181 evaluation 785608 afterok:782825 and ledger 785644 afterok:785608 queued. Existing scripts are frozen; added ledger wrapper only.

14:44: Independent read-only reviewer found incomplete source provenance, mutable shard layout and ETA denominator including serial forecast. Fixes will land after the running calibration exits, then calibration reruns under final code. Layout RED confirmed 785638 (1 failed, 4 passed). Add 255/256 boundary and missing-owner-ply coverage tests. Current 2,000-game extraction took 284.9 seconds for 428,272 replayed plies; do not use forecast-inclusive elapsed as throughput.

Review scope: unrelated dirty changes, downstream model training and upload behavior were excluded from code review; existing registered evaluation scripts are reused. Concurrent reserve checks can overshoot by up to eight small parts; keep 3 GiB reserve rather than treating it as an exact filesystem limit. Report per-dimension quantization diagnostics without inventing a new outcome-dependent gate.

14:52: Material review fixes implemented; relevant suite 785656: 18 passed. Full project suite 785623: 386 passed, 4 unrelated existing-test failures, 7 skipped; failure names and reasons in report section 10. Final calibration 785657 → full 785659 → audit 785662 submitted, dependent on success. M181 jobs resubmitted with explicit OMP/MKL/OpenBLAS=2 to prevent five TTA children oversubscribing CPUs: evaluation 785681 → ledger 785682; pending superseded 785608 and 785644 canceled. Scheduler refused changing the old ledger dependency, so a fresh dependent job was submitted and verified instead. No active training job was canceled.

Format for the later model reader: each `part_s*of8_p*.npz` has per-game `key`, original truncated `LEN`, actual owner `SUBJ` (B=0,W=1), `BOTH`, and cumulative `OFF`. Flat `T`, `MOV`, `POS`, `XQ` index selected plies; XQ columns 0:64 are residuals, 64:67 are surprisal/entropy/log1p rank. Decode using stats.npz `X = XQ * sd / scale + mu`. Missing opponent plies in owner-only games are absent observations, not numeric zero residuals. Use the stored original T and side mask when joining existing per-ply inputs. Full completion requires audit.json status complete_verified.

14:54: Final calibration passed at 14:53:25; full job 785659 started immediately on 8 GPUs/96 CPUs and first parts have passed write/readback and independent SGF alignment checks. Calibration 281.59s, 1520.92 replay plies/s, both-side-adjusted eight-GPU estimate 5.51h; conservative full ETA 20:30–22:30 Taipei. Audit 785662 pending successful full run. Extraction and third-round training remain IN PROGRESS; this launch plan is executed, not a claim of completed scientific results.
