# QUAL-021-ABSGS-SCREEN-PRUNE-FIX

Status: `completed_early_gate_failed_full_not_run` on 2026-09-22.

QUAL018 showed that AbsGS append erased the complete screen-radius history, so
this matched experiment tested the smallest possible fix. RGB161 kept native
behavior; RGB162 kept old histories and zero-initialized only new children. Both
used the same frozen source, 353 training cameras, 30 confirmation cameras, 17
buffer cameras, 78,045-point initialization, seed and thresholds. No labels or
semantics entered training.

The smoke established causality: native append retained oversized histories in
0/2 eligible observations, while the candidate retained them in 4/4. It also
failed the early topology/recovery review. The candidate's first active screen
prune removed 291,067 points and left 68,599 versus 306,405 control points. A
read-only 33-view bridge review measured candidate-control mean deltas of
-2.731111 dB PSNR and -0.037185 SSIM; 390 alone was -9.424304 dB and visibly
lost large parts of the scene. Therefore the complete 30k pair was stopped
before spending the additional GPU time.

This result explains the code behavior but does not show that all large
Gaussians are bad. The fixed 20-pixel history accumulates across many views and
marks most parents by step 3000; restoring it wholesale converts a dormant rule
into mass deletion. A future reconstruction candidate must use a new mechanism
that separates harmful broad coverage from legitimate bridge/deck/cable support.
It must not scan the same screen threshold or run matched semantics on RGB162.

- Smoke gate SHA256: `2827e99f7494bdc24ef05080856fa637d14edfd07fc7ba08fe9282f9d337db67`.
- Bridge review SHA256: `a358292771f555d0f8709640a6c77aa0c9ce4aa202c8f7bdaf49040f67c1b7a3`.
- Review artifact: `outputs/quality/QUAL-021/r1/smoke-r2-bridge-review/result.json`.

RGB140+SEM378 remains the full-data quality candidate; Standard remains the
efficiency fallback. No production or frozen checkpoint changed.
