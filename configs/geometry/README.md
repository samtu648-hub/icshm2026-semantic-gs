# GeometryPrior runtime contract

[evaluation_v2.json](evaluation_v2.json) is the machine-readable four-model protocol. It fixes the common data/split/camera/Scaffold controls while allowing model-native input organization.

The frozen submission backbone remains Scaffold-GS. Optional forward models are
offline providers of depth, confidence, point proposals, or pose proposals;
they never receive hidden test RGB/labels and never replace the final external-
camera renderer.

The 5070/12 GB contract is windowed inference only:

- MapAnything: `memory_efficient_inference=true`, batch 1, AMP, 16-image windows,
  overlap 4–8;
- DA3: `da3-large`, 8–16-image windows, official intrinsics/extrinsics supplied;
- LingBot-Map: segmented trajectory input, 518 image size, SDPA first, CPU
  offload, camera iterations 1 for smoke and 4 only after compatibility passes.

No model source, commit, license, or checkpoint is invented in this repository.
MapAnything and DA3-Large have now completed their separately registered real
P0 executions; their local source trees, weights and raw outputs remain Git
ignored. Their recorded P0 initialization recipes are negative results and
must not be rerun under the same experiment IDs. LingBot-Map and VGGT-Ω remain
uninstalled/unexecuted candidates: a real runner must be given an approved
`source_root` and checkpoint path and must record them in its experiment config
before downloading or executing.

The runners in `scripts/run_mapanything.py`, `scripts/run_da3.py`, and
`scripts/run_lingbot.py` support deterministic window-manifest dry-runs and
fail closed when the optional package/API is not installed.
