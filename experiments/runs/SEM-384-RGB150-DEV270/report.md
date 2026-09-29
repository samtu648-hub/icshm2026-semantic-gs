# SEM-384-RGB150-DEV270

- Status: completed on 2026-09-17; fixed local-composition development baseline.
- Parent recipe: `SEM-378-RGB140-ABSGRAD-ALL300`.
- Split: `dev30_local_composition_v1`, 270 semantic train views and 30 development holdout views.
- Seed: `20260805`; endpoint: 6000 successful updates; 768-pixel native-grid crops.
- Initialization: fresh Swin-L ADE pretrained state; no SEM378 fitted checkpoint reuse.
- Access rule: holdout RGB and labels remain inaccessible during fitting and may be opened only after the 6000-step checkpoint is sealed.

## Results

- Training role: five-class mIoU `0.953252`; foreground mIoU `0.942996`.
- Holdout: five-class mIoU `0.936986`; foreground mIoU `0.922768`.
- Holdout IoU by class: background `0.993860`, deck `0.971594`, stay_cable `0.914223`, tower `0.934854`, foundation `0.870399`.
- Holdout `stay_cable`: FP `67,057`, FN `121,543`, precision `0.967718`, recall `0.942983`.
- Holdout 2px `stay_cable` boundary: precision `0.799563`, recall `0.822747`, F1 `0.810989`.
- Train-to-holdout foreground mIoU gap: `-2.022789 pp`; boundary F1 gap: `-7.349009 pp`.

Group foreground mIoU / stay_cable IoU / boundary F1:

- Regular-local: `0.937664 / 0.938243 / 0.775707`.
- Oblique-local: `0.896449 / 0.852086 / 0.766447`.
- Distant-local: `0.896427 / 0.929397 / 0.929363`; tower and foundation IoU are `0.865786` and `0.828431`.

The oblique block is the weakest for `stay_cable`; the distant block instead concentrates tower/foundation loss. The fixed holdout quantifies this single split and seed only.

## Access And Reproducibility

- Preparation copied only the 370 RGB-training-role renders. The 30 holdout image bytes and labels were not read during sampling or fitting.
- The frozen 6000-step plan SHA256 is `b17b813ae84d6523f4fc6b322643348e44663e47788cf02c2e9ffa521f0b7dc0`; every training view was sampled 9 to 38 times.
- The actual 6000 trace exactly matched the frozen plan. The ADE initial model state and resolved recipe match SEM378; no fitted SEM378 checkpoint was loaded.
- Training completed 6000 successful updates with 5 AMP retries in `1578.918 s`; peak allocated/reserved memory was `8097.570/8590.000 MiB`.
- The checkpoint was sealed before holdout access and remained SHA-identical afterward: `d9373e3330e5fc06f6c914638c1cc948b07faf5514d8b4677322dd7310a5d8a0`.
- Inference result SHA256: `a61ba1471be8e3399935aa20c1330de6e7c648015823278990b8ac53a9c39f78`.

## Coverage Diagnostic

- A native 768 crop covers `45.181%` of a 1320x989 frame and cannot retain full width or height.
- Across the actual plan, per-view unseen-pixel fraction ranges from `2.608%` to `17.399%`, median `8.153%`.
- Never-sampled pixel fractions are background `9.156%`, deck `5.022%`, stay_cable `3.201%`, tower `1.603%`, foundation `5.674%`.
- Views 311/350/356 are unlabeled probes with zero supervised draws. They show complete-tower, distant partial-tower, and no-complete-tower contexts respectively; no accuracy is claimed for them.
- SEM379 raised stay_cable exposure by `9.029%` while lowering foundation exposure by `4.433%`, yet did not improve cable recall. More local cable exposure alone is therefore not an established remedy.

Review page: `outputs/quality/SEM-384/r1/review-r2/index.html`. The first `review` build failure caused by treating an existing colorized SEM378 asset as class IDs is preserved separately.

Conclusion: retain SEM-384 as the fixed dev30 baseline, not as an automatic replacement for the all-data RGB140+SEM378 candidate. At most two future matched tests are justified: first a fixed full-frame context-mix arm; only afterward, if needed, one RGB fine-detail preservation arm.

Final verification: 11 task-specific tests passed, 240 JSON-compatible configs validated, and `repo_health` passed 568 tests with 76 optional skips.
