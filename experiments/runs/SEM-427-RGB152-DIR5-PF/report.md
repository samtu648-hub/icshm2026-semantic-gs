# SEM-427-RGB152-DIR5-PF

Status: completed engineering preflight; all 28 independent audit checks passed.

Protocol: [sem427_directional5_preflight_protocol.md](../../../docs/experiments/sem427_directional5_preflight_protocol.md)

The only representation change from SEM426 is a first-order registered-view
direction basis per Gaussian. Eight matched updates trained 15,785,620
coefficients on 789,281 frozen Gaussians. Bias and directional gradients were
finite and non-zero in every update. Probability conservation error was
`2.3841858e-7`; peak reserved memory was 1,954 MiB; checkpoint state/render and
frozen RGB state/fresh render replayed exactly.

For registered training cameras `001.png` and `300.png`, 84.4423% of
Gaussian-class probabilities changed by more than `1e-6` after the eight
updates (mean absolute change 0.018191, maximum 0.324876). This demonstrates
that the directional branch is active, not that quality improves. Confirmation
and buffer accesses remained zero. The passed preflight authorized SEM428 only.
