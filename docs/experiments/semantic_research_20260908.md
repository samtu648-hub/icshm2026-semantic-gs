# Research continuation authorized 2026-09-08

## 当前执行摘要：已完成至SEM329

更新2026-09-08。阶段总览以[研究总结](research_summary_20260908.md)为入口；具体定义见[METRICS](../../METRICS.md)，日常推进看[PROJECT_STATUS](../../PROJECT_STATUS.md)。

SEM310–314的DINO来源、完整替换、辅助复核及最终组合实验已有独立记录。SEM310预训练质量受独立HF申请阻塞；公开DINOv2在匹配控制上表现更好，SEM313的OOF辅助复核有效，但SEM314最终组合召回损失较大。RoMa完整官方权重已在SEM320实际运行，不再处于下面旧计划的“尚未安装/验证”阶段。

SEM320–324完成匹配、几何、pose-only选源与类别诊断；SEM325实际原生OOF复核有效，SEM326将其固定到全219源库+SEM161/74开发组合。SEM327源边界保护暴露较大纠错代价；SEM328第二源否决在OOF微增IoU并恢复召回，SEM329已完成最终三臂及4次fresh实际外相机验证。

最新同74开发图mIoU：SEM161为83.387017%、SEM326为83.682675%、SEM329为83.705003%。SEM329相对SEM326基础IoU/recall再提高0.111552/0.441838pp，precision下降0.231028pp；保留为优先研究候选，正式SEM161包保持原状。数值来源、seed20260805、Git/配置/数据划分及局部风险见[SEM329报告](../../experiments/runs/SEM-329-SECOND-SOURCE-VETO-FINAL/report.md)。

当前下一步是另登记训练侧剩余误删诊断；方法选定后安排全量拟合与打包，论文/视频/PPT继续后置。本次仅整理文档，尚未启动下一实验。旧状态页与路线已分别[存档](../history/project_status_pre_summary_20260908.md)和[存档](../history/roadmap_pre_summary_20260908.md)。

## 历史执行与初始计划

以下保留当时的执行更新和原计划。文内“Latest”“not yet run”“next”等均代表对应阶段，当前状态以上方SEM329摘要及独立实验报告为准。

Latest SEM323/324 / ADR-072/073: the user asked to continue without excessive
defensiveness. Reused the existing runtime and geometry checks and evaluated all
remaining105 calibration targets at once. Fixed pose-top3 yields91/105 accessible
targets (rank1=86; folds43/45,39/41,9/19).315 pairs and630 directional matches,
representative replay and independent geometry audit pass. No model changes.

Then separately registered and executed the useful next question: class agreement
and coverage of SEM160 OOF errors.105 target/91 selected-source label observations
use114 unique PINHOLE labels, with source RoMa positions undistorted correctly.
Agreement94.086% and old-error coverage15.836% do not justify whole-label transfer:
it would correct48069 but harm164085 grid points, dominated by background/cable
swaps. Full20-direction decomposition finds foundation4→background0 strongest:
23474 corrections/3707 harms, net-positive in all3 folds, with about1.92pp sampled
foundation recall cost. This post-hoc calibration hypothesis now warrants a
separately registered native foundation-only reviewer, not more generic matching
expansions. Actual native inference/quality is not yet run. SEM161 remains intact;
no predictions, final fitting, materials or upload were produced in these diagnostics.

Latest SEM322 / ADR-071: fixed pose-only source ranking (same-fold train,
axis angle<=60deg, then center distance/angle/name, top3) passes a separately
registered accessibility screen on the same3 calibration targets. Rank1 passes
2/3 targets; top3 finds an accessible source for3/3;6/9 pairs pass. Selected
017→013,001→103,032→029 have reliable512-grid coverage6.893%/6.322%/3.218%
and post-cycle epipolar retention82.950%/81.471%/94.258%. Independent pose and
geometry recomputation, new-process GPU replay and18 frozen hashes pass. No
target tracks parsed, labels read, source unions or semantic changes. Historical
check IDs are explicitly permitted as same-fold source RGB only. This is positive
small-sample source evidence, not semantic accuracy or unseen-camera generalization.
Next requires a separate fixed-policy expansion to more preselected training-side
targets; class/label diagnostics require explicit registration before access.
No K/angle/filter tuning, production promotion, final fitting or material creation
was performed. The historical statements below are superseded, not erased.

Latest execution / ADR-069/070: the SEM314 final reviewer failed and was not
promoted. SEM320 now downloaded and hash-verified the complete official RoMa
release, pinned both source trees and passed fixed512 native-PyTorch-fallback
runtime plus exact fresh-process replay (peak CUDA allocated2497.295MiB). Its
overbroad no111-image prose conflicts with046/029 being same-fold training-source
RGB; this wording deviation is retained explicitly. No labels, real target RGB or
74 images were decoded. SEM321 reused only the six maps and official cameras:
reliable coverage9.795%/0.002670%/3.218%, epipolar retention67.426%/63.636%/94.258%,
so the fixed geometric screen failed. Independent Torch sampling/radial inverse/
essential-matrix recomputation agrees with the NumPy/OpenCV/fundamental version.
No label transfer, threshold rescue or SEM161 change. Known-target shared-track
selection is not available for arbitrary novel poses; any source-accessibility
follow-up needs a separate registration and pose/rendered-input-only retrieval.
The not-yet-installed/not-yet-executed statements below are historical planning
records and are superseded by the two experiment reports, not erased.

Later user update / ADR-066: local login succeeded but DINOv3 access was rejected;
the user selected public DINOv2 first. SEM-311 independently registers frozen
DINOv2 ViT-B/14 versus frozen MiT-B2, both native896/448 and the prospective
ADR-065 gates. Full paired OOF completed in13.405min: DINOv2 beats frozen MiT
mIoU+2.038pp/foundation IoU+9.878pp, but loses2.653pp/12.947pp against SEM-160
and fails every spatial safety gate.438-output/6-head native/four full-trained
official-camera probes, independent108 quality recomputation and18 frozen hashes
passed. No111/74 scoring, final fit or rescue; retain feature-source evidence,
drop this full-lane replacement. Original SEM-310 configuration,
code and engineering results remain intact and blocked, not failed for quality.
This update supersedes the first-step ordering below without erasing its history.

The user prefers further research to immediate manuscript/presentation production,
targets both practical score and interpretable evidence, and keeps the RTX5070
12GB hardware constraint. Existing reproducibility records must still be maintained.

1. First execute SEM-310: frozen DINOv3 ViT-B/16 versus frozen MiT-B2, same
   lightweight semantic decoder, native960 and spatial OOF. See its config/report
   and ADR-065 for prospective quality gates and official access requirements.
2. Next examine RoMa v2 dense correspondence as a separately registered diagnostic.
   Match a rendered target RGB to real training-source images, never to unavailable
   real target RGB or target labels. Quantify valid coverage and correspondence
   reliability before proposing label transfer. The failed sparse SEM-250 recipe
   does not bound dense matching, but it warns against assuming coverage.
3. Only after method selection, register a separate all-released-data final:
   400 RGB/300 labels and rebuilt dependent semantic heads if the RGB distribution
   changes. Do not relabel training scores as held-out validation. Retain the
   original RGB-030-FULL + SEM-161 bundle as an independent fallback.

RoMa v2 source preflight pinned upstream commit
`95c9968145c8906b7b59383258e9f73b02853d89`. Its default feature extractor uses
DINOv3 ViT-L, not SEM-310's B model. Importantly, default `weights_path=None`
constructs DINOv3 without pretrained downloading, then RoMa strictly loads the
author's complete release. The upstream DINOv3 code reference is
`adc254450203739c8149213a7a69d8d905b4fcfa`. Therefore the source indicates the
SEM-310 HF gate need not block RoMa's intended public release route; no runtime
verification of that inference has occurred, and extracting substitute SEM-310
weights is neither intended nor attempted.

Official GitHub release metadata reports `romav2.0.1.pt`,1,095,883,548 bytes,
SHA256 `1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7`.
Its current pyproject declares Linux `fused-local-corr`, although README presents
an optional extra. Do not assume a simple pip install or a published H200 memory
figure transfers to12GB. No RoMa matcher, weight or native kernel has been installed
by this continuation. Isolate dependencies and verify the release before a real
GPU correspondence preflight; stop if source access or memory fails. MIT code
licensing does not supersede the custom DINOv3 component license.

Official references checked during planning: [DINOv3](https://github.com/facebookresearch/dinov3),
[pinned candidate model](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m),
[RoMa v2](https://github.com/Parskatt/RoMaV2),
[descriptor loading](https://github.com/Parskatt/RoMaV2/blob/95c9968145c8906b7b59383258e9f73b02853d89/src/romav2/features.py),
[complete-model loading](https://github.com/Parskatt/RoMaV2/blob/95c9968145c8906b7b59383258e9f73b02853d89/src/romav2/romav2.py),
[release metadata](https://api.github.com/repos/Parskatt/RoMaV2/releases/tags/v2.0.1).

No paper/slides, official upload, Git commit/push, frozen module replacement or
automatic candidate promotion is authorized by this research ordering alone.
