# SEM-386-RGB150-DEV270-CONTEXT50-MEMORY

## Completed background run

2026-09-17 SEM386训练与评价完成，待人工视觉复核。固定Dev30 270/30、seed20260805、6000步，保留1025x768完整画面并启用Swin激活重计算。留出五类mIoU 93.741796%（较SEM384 +0.043190pp），前景mIoU 92.329989%（+0.053221pp）。结果为单划分单seed开发证据；未自动替换基线，未启动其他训练。下一步：人工检查对照页；据留出及局部表现判断是否保留。

Training: 2012.83 seconds; peak reserved 6116.0 MiB.

Independent stored-prediction confusion checks: 600. Frozen RGB equality: True.

Comparison: `outputs/quality/SEM-386/r2/comparison/index.html`. Visual review remains pending; no promotion claim.

Group and class changes:
```json
{
  "train": {
    "baseline_five_class_miou": 0.9532522423986072,
    "candidate_five_class_miou": 0.953854932913799,
    "five_class_delta_pp": 0.060269051519179495,
    "baseline_foreground_miou": 0.9429955675056274,
    "candidate_foreground_miou": 0.9437209857218124,
    "foreground_delta_pp": 0.0725418216184992,
    "per_class_delta_pp": {
      "background": 0.011177971122067198,
      "deck": 0.07837955756753168,
      "stay_cable": 0.04368120943034626,
      "tower": -0.025539096546811635,
      "foundation": 0.1936456160228639
    },
    "cable_fp_delta": -53555,
    "cable_fn_delta": 39782,
    "boundary_f1_delta_pp": 0.4051032634087126
  },
  "holdout": {
    "baseline_five_class_miou": 0.9369860558090111,
    "candidate_five_class_miou": 0.9374179598056255,
    "five_class_delta_pp": 0.04319039966144134,
    "baseline_foreground_miou": 0.9227676769399956,
    "candidate_foreground_miou": 0.9232998857631813,
    "foreground_delta_pp": 0.05322088231857203,
    "per_class_delta_pp": {
      "background": 0.0030684690330073927,
      "deck": 0.10308936961535009,
      "stay_cable": 0.012328635400649102,
      "tower": -0.05529830615335074,
      "foundation": 0.15276383041162855
    },
    "cable_fp_delta": -5428,
    "cable_fn_delta": 4692,
    "boundary_f1_delta_pp": 0.5383306300119073
  },
  "regular-local": {
    "baseline_five_class_miou": 0.9484108926149976,
    "candidate_five_class_miou": 0.9472170339375274,
    "five_class_delta_pp": -0.11938586774701276,
    "baseline_foreground_miou": 0.9376641718716342,
    "candidate_foreground_miou": 0.9362739110518751,
    "foreground_delta_pp": -0.1390260819759126,
    "per_class_delta_pp": {
      "background": -0.040825010831258,
      "deck": 0.15301958974925922,
      "stay_cable": -0.3503297443083975,
      "tower": -0.14068657285616348,
      "foundation": -0.21810760048835975
    },
    "cable_fp_delta": -1691,
    "cable_fn_delta": 6208,
    "boundary_f1_delta_pp": -0.5011457757852611
  },
  "oblique-local": {
    "baseline_five_class_miou": 0.9155656187839873,
    "candidate_five_class_miou": 0.9212144031289619,
    "five_class_delta_pp": 0.5648784344974511,
    "baseline_foreground_miou": 0.896448875924019,
    "candidate_foreground_miou": 0.9033944172787476,
    "foreground_delta_pp": 0.6945541354728579,
    "per_class_delta_pp": {
      "background": 0.04617563059582386,
      "deck": -0.07072229116950046,
      "stay_cable": 0.8448300311046375,
      "tower": -0.016001625047756818,
      "foundation": 2.0201104270040626
    },
    "cable_fp_delta": -2649,
    "cable_fn_delta": -2643,
    "boundary_f1_delta_pp": 1.323381712228766
  },
  "distant-local": {
    "baseline_five_class_miou": 0.9167012675675323,
    "candidate_five_class_miou": 0.9181642183961344,
    "five_class_delta_pp": 0.14629508286020387,
    "baseline_foreground_miou": 0.8964268456135474,
    "candidate_foreground_miou": 0.8982522504665116,
    "foreground_delta_pp": 0.18254048529642297,
    "per_class_delta_pp": {
      "background": 0.001313473115316377,
      "deck": 0.2504615788096354,
      "stay_cable": -0.039376795627354966,
      "tower": 0.07230574762664421,
      "foundation": 0.44677141037678947
    },
    "cable_fp_delta": -1088,
    "cable_fn_delta": 1127,
    "boundary_f1_delta_pp": 0.6488735233774467
  }
}
```

## Preparation history

Status: authorized, memory adaptation in progress; no training or quality results.

Parent SEM384; SEM385 OOM retained. Same Dev30 270/30, seed20260805, RGB150, 6000 steps, 3000 native768 crops + 3000 full-frame1025x768 inputs. Native Swin activation recomputation is enabled on all four stages at load time and recorded in memory-policy.json; base resolved model YAML is retained and alone does not describe this execution policy. Inference remains unchanged.

User authorized routine memory fixes and complete training/evaluation without repeated approval. First try native recomputation with unchanged images; if insufficient, preserve failure and adapt before selecting a final training configuration using training-side resource evidence only. No hidden-test claims or checkpoint choice by holdout.

Preparation r1 stopped after the first optimizer update because an added diagnostic incorrectly demanded a nonzero gradient on the first parameter of every first block. Swin DropPath can legitimately zero an individual block; this was not an OOM result. r1 retained. r2 checks finite/present gradients across every stage and nonzero aggregate participation over all eight updates, without requiring every stochastic block active on every sample.

r2 preflight passed: eight successful updates (4 native768, 4 context1025x768), all four backbone stages finite and active over the probe; initial ADE state and base recipe match SEM384, RGB hashes unchanged, no holdout access. Peak allocated5665.76MiB/reserved6116MiB, 12.60s, two normal AMP overflow retries. Proceeding with unchanged images and model to fresh 6000-step training.

Additional CPU numerical test of the pinned native Swin BasicLayer (shifted windows, dropout and DropPath) passed: forward outputs and final RNG state identical; input/parameter gradients match at rtol1e-6/atol1e-7. Test harness first lacked the runtime dependency path; failed log retained, second invocation passed. This component test is not a claim of bitwise-identical full mixed-shape training.


## Inference numerical recovery

2026-09-17 SEM386已完成6000训练（2012.827秒），冻结RGB一致，六次AMP重试；首次推理完成376/400后在377.png触发134像素FP16零质量下溢。单图同logits FP32诊断已恢复。原失败保留inference-failed-r1，正在按ADR127只恢复零质量像素并重跑完整推理，不重训；随后后台生成对照与报告。尚无最终完整评分。
The same logits produce positive FP32 mass (minimum4.76898e-9) where half precision zeros134 pixels. Only those zero-mass pixels are recovered; valid half scores remain exact. No labels consulted and no epsilon/background filling. Previously completed376 PNGs will be hash-checked after recovery.

## Post-completion independent audit and visual review

Completed on 2026-09-19. This section supersedes the earlier `visual review remains pending` note; it does not change the sealed checkpoint, predictions, or reported metrics.

- An independent pass reloaded all 600 stored labeled predictions (300 per arm), recomputed each 5x5 confusion matrix directly from the PNG and ground-truth mask, and exactly matched every stored per-view matrix. Aggregating those matrices reproduced the train and holdout five-class/foreground mIoU values above without reading `comparison/result.json` as metric input.
- Both 6000-step checkpoints retain their recorded SHA256 values (`d9373e3330e5fc06f6c914638c1cc948b07faf5514d8b4677322dd7310a5d8a0` and `a0de96c779459a336014204e24be6e8dc96d497651aaa5a40d6a1ebf35615796`). Each checkpoint contains 650 tensors; an independent CPU load found zero non-finite floating or complex tensors.
- The two exports contain the same 370 image names and identical per-image RGB hashes. Their source RGB result and evaluation hashes also match. The candidate sampling plan independently verifies 6000 entries with the same parent view and flip sequence, 3000 odd native `768x768` updates, 3000 even full-frame `1025x768` updates, and unchanged parent crop coordinates on every native update.
- All 30 holdout views were visually reviewed as RGB / ground truth / SEM384 / SEM386 / change panels. No new large-area semantic collapse, remote isolated color block, or whole-deck/tower failure was observed. Most changes are thin structural boundaries and one-to-few-pixel cable decisions.
- Regular-local is the visible risk: views 110, 112, 113, 114, and 115 show shortened or interrupted thin `stay_cable` segments, with 114 and 115 the clearest regressions. This agrees with the subgroup `stay_cable` IoU change of `-0.350330 pp`, cable FN change of `+6208`, and boundary F1 change of `-0.501146 pp`.
- Oblique-local contains both the largest visible recoveries and local failures. Views 146, 149, and 151 recover cable or tower/foundation regions; views 147 and 148 lose local cable/tower detail. Distant-local changes are smaller and mostly boundary-localized; view 182 degrades a small tower region, while 187 and 188 improve local foundation/deck structure.
- The confirmation gate distinguishes overall per-class holdout changes from per-class subgroup diagnostics. On this seed, every overall holdout class changes by more than `-0.20 pp` (worst: tower `-0.055298 pp`), every subgroup five-class mIoU changes by more than `-0.20 pp` (worst: regular-local `-0.119386 pp`), holdout five-class and foreground mIoU improve, and holdout 2px `stay_cable` boundary F1 improves by `+0.538331 pp`. The larger regular-local per-class declines remain a recorded risk but are not silently substituted for the preregistered gate definition.

Audit conclusion: SEM386 remains a positive but small single-seed result, not a promotion. Proceed only with the preregistered second-seed matched control/candidate confirmation; do not migrate the recipe to RGB140 until the two-seed gate and repeated visual review pass.
