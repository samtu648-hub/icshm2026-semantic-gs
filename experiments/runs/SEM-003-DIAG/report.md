# SEM-003-DIAG — training-side bg/stay_cable conflict evidence

- Status: **completed training-side-only diagnostic**.
- Parent: SEM-001, SEM-002.
- Actual run HEAD: d84e5c93c40a6d58cca836582a0ec23bfa340cd4.
- Code freeze: b4cdc4e3e7721b8180ff409b5da7836cd3b01adb.
- Run-start tracked-diff SHA-256: b8cbe809dc5e1d3381440a525bc10911fb9e241a84f0893fe23295accc7442a8; pre-existing user modifications were preserved.

## Runtime recovery and retained failures

The earlier CUDA failure was not a 580.173.02 kernel/userspace mismatch. The restricted command namespace mounted an empty per-user tmpfs over /dev, hiding /dev/nvidia*; the host execution context saw the RTX 5070 normally. Host nvidia-smi reported 12,227 MiB and the frozen R1 Python reported torch.cuda.is_available() == True, device NVIDIA GeForce RTX 5070. GPU commands for this run were therefore launched in the host context; no driver package, kernel module or frozen Python package was changed.

The two original no-GPU attempts remain under the driver-unavailable directories. Two later startup-only failures are also retained: one command omitted the frozen environment bin from PATH (failure JSON SHA-256 9dad66f6d83fb6d4b38c254e585fe4cbfc46f67142bf55b65aaa8f96fe742719), and one omitted the already frozen local CUDA 12.8 CUDA_HOME (5698dfe6d2eb96ff7443da8a9fcb519041bb88d8ada7aeee7d43c06c268a6029). Neither reached diagnostic view 1, wrote evidence or started training. No result was overwritten.

## Frozen mechanism and data boundary

The completed diagnostic used only the 219 semantic_train_labeled cameras and released labels. It accumulated exact frozen-RGB all-contributor marginal contribution r(g,p) times the train-side one-hot label. Cumulative threshold 1.0 extracted every contributor; it was not a supervision gate and did not filter or reweight training.

Eligibility was frozen before CTRL/candidate training: at least three training views, at least 10% evidence each for background and stay_cable, at most 5% protected-class evidence, then the training-only 90th percentile of C_bc = 2 min(E_background, E_stay_cable) / (E_background + E_stay_cable + epsilon).

Validation RGB/labels, guard-band images, the 100 unlabeled RGB images and hidden test information were excluded.

## Result

| Item | Value |
|---|---:|
| Training views | 219 |
| Total anchors | 189,073 |
| Base-eligible anchors | 76,806 |
| Frozen conflict threshold | 0.8615600148 |
| Final eligible anchors | 7,681 (4.06245%) |
| Support views, mean / median / max | 60.6009 / 62 / 138 |
| Elapsed time | 56.167 s |

External artifact: $R1_OUTPUT_ROOT/r2/sem_003/diagnostic-attempt5-host-context/diagnostic_evidence.pth, 15,507,885 bytes, SHA-256 84b60b867e07e2a1a1e17cd47d053e1b00c8c19a820efdb0dfd53bdd35b8d3b0. Result JSON SHA-256: 6f53a81a3155232c007fe16c21773608273ae49d7a35aed2143dee4eaeceb588.

Conclusion: the train-side conflict mask exists and is reproducible, so it authorized the registered CTRL/candidate preflights. It is evidence of label conflict, not evidence that the proposed regularizer improves validation quality.
