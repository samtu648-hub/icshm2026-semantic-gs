# Development evidence and paper outline

This is a working manuscript outline, not a finished submission or a claim of
hidden-set performance. Team identities, contribution statements, distribution
permissions and the final organiser protocol remain to be supplied by the team.

## Proposed technical narrative

Bridge semantics do not align perfectly with Gaussian surface primitives:
thin structures and broad polygon annotations produce conflicting projected
supervision. On this dataset, a reliable practical system combines a frozen
Scaffold-GS renderer and static semantic baseline with rendered-domain specialist
segmenters. Protected fusion exploits their complementarity without retraining
the RGB scene. Protection is a deployment constraint, not a declaration that
all historical labels are correct.

The successful sequence is SEM-001 → SEM-111 → SEM-141 → SEM-151 → SEM-161.
Use each direct parent as the ablation control. Do not add the component gains
to an unrelated baseline or compare metrics from different partitions.

## Evidence to include

1. RGB-060: equal-protocol Standard/Mip/Scaffold comparison, full training budget
   and the same spatial split. Distinguish a 400-image source pool from the303
   actual training cameras after split/guard exclusion.
2. SEM-102 versus SEM-111: whole-map replacement harms previously correct
   structures, while protected fusion retains them. This is the central fusion
   ablation, not evidence that arbitrary 2D segmentation is always preferable.
3. SEM-141, SEM-151, SEM-161: class-specific direct-parent ablations, all five
   IoU/precision/recall values, RGB exactness, latency and memory. Describe74-view
   results as reused development data. Current mIoU is83.387%, not a hidden score.
4. Negative families: one representative each for semantic conflict modelling,
   hard structural filtering, geometry-derived deck repair and hard contribution
   responsibility. Group failures by hypothesis rather than listing every sweep.
5. SEM-260/261: native-boundary error budget followed by same-size coarse/fine
   point-head comparison. Report it even if it fails. A boundary-distance
   association alone cannot prove fine features will generalise.
6. SUBMIT-030: external COLMAP interface, checkpoint reload, immutable artifacts,
   no target RGB/GT at inference and fallback hierarchy. Fresh processes are
   not clean-machine reproduction; report the environment scope accurately.

## Required limitations

- The inherited SEM-001 static semantic base saw all219 training labels.
  Later three-fold head experiments are incremental OOF, not full-chain OOF.
- Both108 calibration and111 check have been reused; SEM-250 evaluated all219
  targets despite its historical `check_metrics_opened: false` flag. Cite the
  erratum and preserve the original result hash. No new blind-test claim.
- SEM-230 lacks the matched anchor/offset control needed for a pure granularity
  causal claim. SEM-240 tests hard maximum responsibility, not every possible
  soft-visibility method. SEM-250 only bounds direct sparse-point corrections.
- Official class2 is always named `stay_cable`, including wide annotated
  envelopes; do not silently replace the annotation definition with physical
  wire centre lines.
- Protected predictions may themselves be wrong. Development can challenge
  them in separately registered class-specific experiments; deployment still
  requires comparing gains and harms across all classes.
- SegFormer weights have NVIDIA research/evaluation terms, not an Apache-2.0
  grant. Include original license/source and obtain team distribution decisions.

## Material remaining

Write the full English paper in the official template, select native paired
figures with honest failure examples, prepare the demonstration/PPT, assign the
three members' roles, validate a clean installation and obtain the organiser's
final output naming protocol. None of these is automatically completed by this
outline; no files are uploaded or submitted by this task.
