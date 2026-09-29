# SEM-003-CTRL-PF — Dirichlet representation control preflight

- Status: **completed engineering preflight; not a quality baseline**.
- Actual run HEAD / code freeze: d84e5c93c40a6d58cca836582a0ec23bfa340cd4 / b4cdc4e3e7721b8180ff409b5da7836cd3b01adb.
- Evidence artifact SHA-256: 84b60b867e07e2a1a1e17cd47d053e1b00c8c19a820efdb0dfd53bdd35b8d3b0.

## Representation equivalence

The frozen final SEM-001 checkpoint was converted to the registered Dirichlet parameterization with all new regularizers disabled. On official camera 001.png, the maximum probability difference was 4.1127205e-6, below the 1e-5 tolerance; class IDs were bit-identical, RGB maximum pixel difference was 0, and no frozen RGB parameter was trainable. Result JSON SHA-256: 7bf834cff7fac59af91b8573c6ffe2fd7cc2a47f04de6253e831501cce58d6a1.

## Eight-iteration CTRL smoke

The control trained only dirichlet_class_logits; concentration regularization and global visibility gating were disabled. Eight iterations completed in 2.648 s with peak PyTorch allocation 741 MiB and no NaN/Inf/OOM. The final checkpoint was reloaded bit-identically (4,540,781 bytes; SHA-256 0cecf3cb1075c557a256d0d19996e80ae3131015766ddcd53a7cb3a8c1e0c8d9).

Frozen RGB artifacts and three RGB PNG hashes were unchanged; RGB maximum pixel difference was 0. The external official CAMERA_FROM_WORLD camera produced 1320×989 RGB, five 989×1320 logits, and single-channel uint8 IDs in 0–4. Result JSON SHA-256: 557bfe22b8b78c9b3ac6c4ee97c7e2472a646850b4adc24f7d802766cdba43c8.

Conclusion: representation and engineering gates passed, authorizing the matched 10k CTRL. The three-view smoke metrics are not used as a quality conclusion.
