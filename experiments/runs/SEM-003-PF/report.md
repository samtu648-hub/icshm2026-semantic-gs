# SEM-003-PF — restricted bg/stay Dirichlet regularizer preflight

- Status: **completed engineering preflight; not a quality baseline**.
- Actual run HEAD / code freeze: d84e5c93c40a6d58cca836582a0ec23bfa340cd4 / b4cdc4e3e7721b8180ff409b5da7836cd3b01adb.
- Evidence artifact SHA-256: 84b60b867e07e2a1a1e17cd47d053e1b00c8c19a820efdb0dfd53bdd35b8d3b0; eligible anchors: 7,681.

The candidate retained ordinary five-class NLL for every semantic anchor. Its only additional term was the pre-registered concentration regularizer on the detached train-side bg/stay mask; anchors with material deck, tower or foundation evidence were excluded. No global contribution gate was enabled.

Eight iterations completed in 2.858 s with peak PyTorch allocation 743 MiB. Conflict loss decreased from 190.8683 to 181.3520 without NaN/Inf/OOM. Gradients were restricted to dirichlet_class_logits and dirichlet_log_evidence_strength_on_diagnostic_mask; no frozen RGB parameter had a gradient.

The final checkpoint reloaded bit-identically (4,540,781 bytes; SHA-256 af4793c64cb4c90458144c7ff8f8033c63638294f96a80edef3a94d7b2e6ada9). Frozen artifact/PNG hashes were unchanged, RGB maximum pixel difference was 0, and the official external camera produced 1320×989 RGB, five logits and uint8 IDs in 0–4. Result JSON SHA-256: 44f2d32ab3a76edb19326bd69759db0927e5795484c79a6d9e82fc0e06f98090.

Conclusion: all engineering/isolation gates passed, authorizing the registered matched 10k candidate. Smoke metrics are not a quality conclusion.
