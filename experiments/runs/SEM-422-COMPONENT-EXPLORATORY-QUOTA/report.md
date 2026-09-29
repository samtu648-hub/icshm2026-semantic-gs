# SEM-422-COMPONENT-EXPLORATORY-QUOTA

Completed2026-09-21; first-seed positive only. Same fixed16-point/5% small-component quota as
originally proposed, tested exploratorily after SEM416's negative prevalence
gate. Matched with SEM421, RGB154/v2/seed20260805/6000.
[Protocol](../../../docs/experiments/component_exploratory_protocol.md).

6000 successful updates in1562.390 seconds,4 AMP retries, reserved8564MiB.
Training/confirmation foreground mIoU94.399250%/87.676210%; confirmation
five-class90.009365%, cable IoU84.043589%, boundary F175.425527%.
Versus SEM421 foreground/cable IoU/boundary+0.261825/+0.585769/+0.576574pp;
precision/recall+0.317626/+0.377915pp. Deck IoU-0.011855pp, tower+0.037975pp,
foundation+0.435410pp. The original first-seed retention checks pass.

Quota replaced2850310 point draws across101080 mask calls (includes AMP retries
and auxiliary losses), without changing the fixed total point budget. Small
cable TP rises7641→9443/15257 pixels; small foundation TP21136→21747/23992.
Small tower TP declines2658→2634/3366 and one component becomes fully missed.
Thus small-region gains have costs even in the positive seed.

Initial state/sampling/RGB/379 predictions and285 confusion/boundary recounts
passed; all650 checkpoint tensors finite. Checkpoint SHA256
`c817675b01927539fdbc81bebca5e7abe69ea2e9068a7de32e787ad19e95cd60`;
inference SHA256 `d5b837aa7a3fcab27458fa32b82a056549fd025ac11b24af8c07479c11891631`.
Subsequent SEM424 confirmation failed, so retain this as an exploratory positive
result without migration or choosing the better seed.
