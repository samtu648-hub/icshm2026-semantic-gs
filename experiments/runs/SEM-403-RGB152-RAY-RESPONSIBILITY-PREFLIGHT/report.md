# SEM-403-RGB152-RAY-RESPONSIBILITY-PREFLIGHT

- Status: completed; all engineering gates passed; label diagnostic authorized.
- Parent: SEM-402.
- Variable: none; unlabeled native-buffer replay feasibility preflight.
- Label access: forbidden.

The fixed protocol is `docs/experiments/sem403_ray_responsibility_preflight_protocol.md`. The run used only the three sealed training cameras and read zero semantic labels or protected cameras. It decoded the unmodified native rasterizer buffers and replayed 576 fixed-grid pixels. RGB152 rendering of `001.png` remained pixel-exact before and after the run.

All five gates passed. Maximum absolute replay errors were `1.236847e-6` for diagnostic color, `4.202821e-8` for final transmittance, and `1.064412e-6` for inverse depth, each below `1e-5`. All 576 traces were nonempty and accepted-contributor depth order had zero violations.

The label-free ray structure is materially distributed: accepted contributor count quantiles were `35 / 82 / 97 / 119 / 221`; effective contributor count was `2.214 / 12.304 / 20.100 / 28.937 / 60.434`; largest-contributor share was `0.0418 / 0.0889 / 0.1318 / 0.2009 / 0.6592`. Reaching 80% of non-background contribution required `8 / 33 / 45 / 56 / 91` front-to-back contributors. Weighted depth standard deviation had median `1.5338` scene units. These are feasibility and descriptive statistics only; without labels they do not show that semantic errors are caused by ray mixing.

`label_diagnostic_authorized=true`. A separate experiment must freeze the pixel strata, front-surface responsibility formula and training-authorization gates before reading training labels. SEM403 itself does not authorize training. Complete result: `outputs/quality/SEM-403/preflight-r1/result.json`; evidence SHA256: `e0dd327cc7a06f17bb5117bce6a31175f8a75c8dee006dc6d6ae41b1230a4828`.
