# SEM-161-DECK-BINARY-FINAL — frozen deck recovery developer audit

## Status

Completed on 2026-09-04. The only final checkpoint and the frozen 0.99
one-way recovery passed every registered developer and engineering gate.
SEM-161 is the strongest local developer candidate and is recommended for
official hidden evaluation, with SEM-151 retained as its direct rollback.

Training took 488.93 seconds (0.13581 GPU-hours) and peaked at 5,570 MiB.
Loss changed from 0.753586 to 0.004830; the last-100 mean was 0.007480.
The result SHA-256 is
`abd6136f38c4641f7f1b7fa0f5ead08fa5bfd8bea7fe0cd363e88c589aa04213`
and the checkpoint SHA-256 is
`8e50ffca587412f2a015e01bc5ffc80ae3503e6c302cc5291d55c183d664628f`.

## Protocol

Train one final-3000 SegFormer-B2 checkpoint on all 219 frozen rendered RGBs
using the exact SEM-160 non-deck/deck recipe. Freeze and reload its checkpoint
before registering the audit hashes or opening the reused 74-view labels.

At fusion time, a SEM-151 background pixel may become deck only if SEM-141 was
also background there. Existing deck, `stay_cable`, tower and foundation stay
exact, and pixels rejected by the foundation gate cannot be relabelled.

Relative to SEM-151, require at least +0.2 pp deck IoU, +0.5 pp deck recall, no
global mIoU loss, at least 94% deck precision, no more than 1.25x
background-to-deck false positives, no loss on the 158--161 oblique group, no
more than 0.5 pp loss on global worst-10% or 140--146, exact protected pixels,
strict checkpoint reload, RGB identity and the official external-camera output
contract.

## Boundary

The 74-view set is explicitly reused development data. Its result can only
accept or reject this frozen candidate for hidden evaluation; it cannot change
the model, threshold, checkpoint or fusion rule. SEM-151 remains the immediate
rollback.

## Completed evidence

All 74 predictions and their probability maps were materialised before the
developer labels were opened. Relative to SEM-151, aggregate mIoU with
background rose from 0.830934 to 0.833870 (+0.294 pp), and mIoU without
background rose from 0.792639 to 0.796056 (+0.342 pp). Deck IoU changed from
0.900999 to 0.914667 (+1.367 pp), recall from 0.938629 to 0.960525 (+2.190 pp),
and precision remained 0.950392 after a 0.701 pp reduction.

The branch added 128,641 deck pixels: 93,060 were ground-truth deck, 18,646
background and 16,935 `stay_cable`; none were tower or foundation. The
background-to-deck count rose from 74,985 to 93,631 (1.24866x), just below the
frozen 1.25x ceiling. Of 74 views, deck IoU improved on 63, was unchanged on
three and regressed on eight. The 158--161 oblique mean improved from 0.769695
to 0.796598 (+2.690 pp); all four individual oblique views improved. The
standard worst-10% non-background mIoU rose by 0.053 pp and the 140--146 mean
rose by 0.963 pp. The deck-specific worst-10% mean rose by 1.610 pp.

Every non-eligible prediction, every existing deck pixel, and every
`stay_cable`, tower and foundation prediction remained exact. Strict checkpoint
reload, frozen RGB byte identity and the official 1320x989 external-camera RGB,
five-logit and semantic-PNG contracts passed. The external camera gained 1,028
deck pixels without changing protected predictions.

The developer result SHA-256 is
`fbf3eb9999407fec83ce3c74577603ae06e96360004d51b555327c5dfb8ec5f2`;
the prediction manifest SHA-256 is
`c23ab7029463fe81d65a0f8ed624740c3e3bb411170270e85cd19f28d76e2c0e`.

## Visual review and retained risk

The registered 081/159/160/161 panels show that most recovered pixels fill
small holes and broken deck edges. The 158--161 sequence visibly gains deck
coverage with sparse errors. View 081 exposes the main failure mode: part of a
near-deck `stay_cable` envelope is recovered as deck. This agrees with the
16,935 cable-target additions and the eight regressed views, led by 080
(-5.313 pp deck IoU), 081 (-4.959 pp), 079 (-3.965 pp) and 082 (-3.543 pp).
No post-hoc threshold or checkpoint change is allowed; hidden evaluation should
compare SEM-161 directly with the preserved SEM-151 package.
