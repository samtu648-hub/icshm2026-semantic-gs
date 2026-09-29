# Third-party dependencies

## RGB-151 ImprovedGS LAS method reference (2026-09-17)

- Long-Axis Split is referenced from the [ImprovedGS paper](https://arxiv.org/abs/2508.12313)
  and official [XiaoBin2001/Improved-GS](https://github.com/XiaoBin2001/Improved-GS)
  repository at commit `20b5db343a87c2f9fb4b70c6e39ff1f042a8a47a`. The official
  repository declares the `ImprovedGS Non-Commercial Research License`; research-only
  use does not clear redistribution or commercial use.
- RGB151 independently implements only the cited LAS child-center, scale and opacity
  equations. No upstream source or repository is copied. Candidate sampling, importance
  budgets, clone replacement, RAP, MU and GC are excluded, so RGB151 is not a complete
  ImprovedGS reproduction.
- Standard 3DGS remains pinned as documented below and retains its own research-only
  license. The RGB151 process-local hook leaves the explicit `native` worker path unchanged.

## RGB-138 StopThePop render-only diagnostic (2026-09-15)

- Official [r4dl/StopThePop](https://github.com/r4dl/StopThePop) source is pinned at
  `859b11bde9195e19a1c40536e1ab16765b64b365`; its rasterizer submodule is pinned at
  `d8cad09919ff49b11be3d693d1e71fa792f559bb`. The main implementation retains the
  Gaussian-Splatting research/evaluation license; the StopThePop headers are MIT.
- The pristine recursive clone stays outside Git under the existing external
  `$R1_OUTPUT_ROOT/r1/third_party`. RGB138 uses a separate ignored
  `outputs/quality/rgb138/source-r1` copy and records its exact diff and compiled binary hash.
  No upstream repository or compiled extension is added to Git.
- `patches/stopthepop-py312.patch` is the compatibility delta: Python 3.12 dataclass
  fields use factories and the small `dacite` dependency is replaced by an equivalent
  explicit parser for the fixed settings schema. CUDA code, sorting queues, compositing,
  camera transforms and Gaussian parameters are unchanged. The rasterizer is compiled
  locally for sm120 with the existing Torch 2.11/CUDA 12.8 toolchain.
- RGB138 renders the frozen RGB130 PLY with StopThePop global and hierarchical sorting
  in the same process. Culling, EWA, load balancing and all model parameters remain off/
  fixed in both arms, so the arm difference is sorting only. Compilation/import success
  is engineering evidence, not a quality result.

## SEM-340 Mask2Former-Swin-L preflight (2026-09-11)

- [Mask2Former](https://github.com/facebookresearch/Mask2Former) commit
  `9b0651c6c1d5b3af2e6da0589b719c514ec0d69a`, MIT code; inherited Deformable-DETR
  native operator retains its Apache-2.0 notices. [Detectron2](https://github.com/facebookresearch/detectron2)
  commit `a2f4a8771ab77e8411c26b27f24f9489a28a2453`, Apache-2.0.
- Official complete ADE20K **semantic** Swin-L checkpoint `model_final_6b4a3a.pkl`,
  866036678 bytes; SHA256 `f0c081e4fb76d3387141341cd799f1a615aa8178d6873157e1d11e0c8773ef38`.
  The [model zoo](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md)
  lists its weights as **CC-BY-NC-4.0**, not the code's MIT license. Internal technical
  preflight is not clearance for competition redistribution or future commercial reuse.
- Source, licenses, dependencies, compiled binaries and checkpoint stay under ignored
  `outputs/quality/mask2former/SEM-340`. Existing Torch2.11/CUDA12.8 and Pillow12 overlay
  are shared read-only. New lightweight packages are installed with `pip --target` and
  `--no-deps`; exact versions are in the experiment report/runtime manifest. This is not
  an independent clean installation, nor all optional upstream dataset/evaluator dependencies.
- Apply `patches/quality_mask2former_torch211.patch` to the pinned Mask2Former source
  with `git apply --unidiff-zero`; it replaces removed THC header/type dispatch APIs and
  makes output gradients contiguous. It does not change CUDA interpolation or gradients.
  `scripts/build_quality_mask2former.py` takes explicit experiment root/CUDA/output paths,
  discovers existing NVIDIA include/library directories, and builds native sm120 extensions
  locally. First missing-header build failures are retained; no driver/system installation.
- The project adapter explicitly preserves the author's legacy query tensor rename and v2
  state metadata, verifies every loaded non-classifier tensor, and initializes only the
  five-class classifier plus class-dependent criterion buffer. Config/data/weights/code,
  RGB checkpoint/topology and PINHOLE identity are recorded and checked on replay.
  Native768 crops adapt the original ADE640 recipe; no architectural reduction is hidden.
- Preflight passed two effective AMP updates and independent native replay on RTX5070.
  A standard finite AMP backoff was needed; long-term training quality remains untested.
  No old model, rendering, semantic probability, environment or submission package changed.

## QUAL-001 / RGB110–113 isolated quality route (2026-09-08)

- 2026-09-09 RGB114 numeric diagnosis keeps the original PGSR checkout intact. A separate Git-ignored `outputs/quality/source/pgsr-safe-cycle` copy adds only `third_party/patches/quality_pgsr_safe_cycle.patch` to `train.py` and calls project helper `src/quality_pgsr_geometry.py`. Apply the existing compatibility patch first, then this delta; Python helper must also be on PYTHONPATH. Commit and licenses remain those of PGSR below. No CUDA kernel or native rendering/compositing change is made. The delta computes the original visibility/occlusion mask without autograd and differentiates only accepted cycles, preventing rejected invalid projections from poisoning gradients; it is a distinct numerical adaptation, not an untouched-author reproduction. Diagnostic and future full-run provenance must include the helper hash as well as the source delta. No upstream copy is committed or distributed.

- PGSR author source `de24f1a38b350387e8d8fe381b2cd70c1ae946e7`, educational/research/nonprofit license; derivative modifications must remain open-source/noncommercial and retain notices.
- SOF author source `b9eb4170c843014f5f96d54924976161bd675469`, Gaussian-Splatting research-only license; retain LICENSE.md/NOTICE.md and permissive subcomponent notices.
- PGSR imports the pure-Python transforms from official PyTorch3D `0a7d4c1a171e8b768c63f15b17564f9ad495f49b` (BSD-3-Clause); this is not a full compiled PyTorch3D installation.
- SOF uses official fused-ssim `a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38` (MIT), plus isolated dacite1.9.2 (MIT) and trimesh4.7.4 (MIT). No existing environment is modified.
- New code/builds live under Git-ignored `outputs/quality/`. Reproduction uses `scripts/prepare_quality_sources.py` and the two `third_party/patches/quality_*_compat.patch` files. Patches cover missing standard headers, explicit sm120 compilation, Python3.12 dataclass factories, package layout, and checkpoint save/load when SOF appearance is disabled. They do not change losses or enable appearance fitting.
- An include path containing `research` triggers PyTorch's broad `arch` substring check and suppresses automatic architecture flags. Explicit `QUALITY_CUDA_ARCHS=120` avoids a successful-but-wrong-architecture build; fused-ssim separately uses `CUDA_ARCHITECTURES=120`.
- Earlier failed/incorrect-target builds remain logged; compilation/import success is not a quality result. Additional runtime modifications must be fingerprinted before training. Competition redistribution and future commercial reuse require separate license review; no upload is authorized.

## SEM-320 RoMa v2 isolated preflight (2026-09-08)

- [Author source](https://github.com/Parskatt/RoMaV2) pinned to
  `95c9968145c8906b7b59383258e9f73b02853d89`; its RoMa code is MIT except the
  [DINOv3 component](https://github.com/facebookresearch/dinov3) pinned to
  `adc254450203739c8149213a7a69d8d905b4fcfa`, under its custom LICENSE.md.
  Both notices are retained in the Git-ignored source checkouts; public release
  availability does not remove acknowledgment/use/redistribution obligations.
- Official [complete release](https://github.com/Parskatt/RoMaV2/releases/tag/v2.0.1)
  `romav2.0.1.pt`,1,095,883,548 bytes, downloaded and independently hashed:
  `1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7`.
  No gated standalone DINOv3 download, credential use or backbone extraction.
- Runs from the unmodified source path with the author's native PyTorch
  local-correlation fallback. The Linux declared fused dependency is intentionally
  absent; this is not a normal dependency-complete package installation. Only
  Pillow12.0.0 (MIT-CMU as declared by its distribution metadata) is installed in
  the isolated ignored runtime overlay. Existing GPU packages are unchanged.
- Source/weight/dependency/runtime artifacts live under `outputs/sem320/`, not
  Git. Exact source manifests are in its preparation record. The old SEM310
  source-only note below is historical; RoMa now passes actual512 runtime/replay
  engineering, with a documented source/check-identifier wording deviation.
  The separate SEM321 geometric screen failed; neither experiment changes SEM161.

## SEM-311 public DINOv2 (2026-09-08)

- [Official facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base),
  fixed revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415`, ungated, Apache-2.0
  as declared in its pinned model card. Expected model.safetensors SHA256
  `d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841`.
- Existing external Transformers5.5.3 implementation is read-only. The new source
  manifest verifies official Git/LFS fingerprints; source/model files and outputs
  remain Git-ignored. No upstream whole repository is vendored, no DINOv3 access
  is attempted, and no dependency installation is performed.

## SEM-310 official DINOv3 source registration (2026-09-08)

- Candidate: [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m),
  pinned revision `5931719e67bbdb9737e363e781fb0c67687896bc`, DINOv3 License.
  Official gated access is pending; no candidate pretrained weight is claimed downloaded.
  The preparation script verifies official Git/LFS digests and records local SHA256.
- Implementation: the existing external Transformers5.5.3 runtime, read-only;
  no upstream DINOv3 repository is vendored or new dependency installed. Random
  architecture smoke is explicitly not a pretrained-quality experiment.
- Frozen MiT-B2 control reuses the existing pinned NVIDIA source and its license.
  Learned heads/source manifests remain in Git-ignored output directories.
- RoMa v2 is a source-level research reference only at this stage; no installation
  or weight download is asserted. Pinned source commit
  `95c9968145c8906b7b59383258e9f73b02853d89`, MIT except the custom-licensed DINOv3
  component; its code is pinned upstream at `adc254450203739c8149213a7a69d8d905b4fcfa`.
  Default construction strictly loads the complete author release, not a separate
  HF DINO-L download. Official release metadata declares `romav2.0.1.pt`,
  1,095,883,548 bytes, SHA256
  `1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7`.
  This is metadata verification only, not a downloaded-content checksum assertion.

## SEM-381 local DINOv3 source update (2026-09-16)

- A separately prepared local copy of the same official ViT-B/16 model is now
  available for SEM381. It was obtained through ModelScope and cross-checked
  against Hugging Face revision
  `5931719e67bbdb9737e363e781fb0c67687896bc` and the official LFS fingerprint.
- `model.safetensors` is 342,662,192 bytes with SHA256
  `9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b`.
  Config, preprocessor and license hashes are fixed in the SEM381 config. Runtime
  access is local-only through `DINOV3_VITB16_DIR`; no credentials, URL fallback,
  model file or personal absolute path are committed.
- The DINOv3 License applies. Research publications must acknowledge the model;
  redistribution must include and follow that agreement. Local research use and
  successful hash verification do not by themselves clear every delivery use.

## SEM-261/262 point refinement and clean reproduction (2026-09-07)

- Method reference: [PointRend](https://arxiv.org/abs/1912.08193), CVPR2020.
  The point MLP is independently implemented here; no Detectron2 source or
  pretrained weight is imported. It is not a full adaptive-subdivision port.
- SegFormer retains the existing NVIDIA research/evaluation weight license.
- Clean reproduction additionally pins the installed `torch_scatter`
  `2.1.2+pt211cu128` PyG binary, SHA256
  `04f27281c6598a5747cab1e57e7e1ffd38e53ed11aee73499328094f14412c15`;
  source URL is in `reproduction/sem161/requirements.txt`. Existing upstream
  license files remain in the package; no vendored source is added to Git.

## RGB-092 DINO feature uncertainty (2026-09-06)

- Feature model: `facebook/dinov2-small`, revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`; model card license Apache-2.0. Cached outside Git under `$R1_OUTPUT_ROOT/r2/rgb_092/dino`. Weight SHA256 `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`; config SHA256 `1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d`.
- Method reference: [WildGaussians paper](https://arxiv.org/abs/2407.08447) and [author code](https://github.com/jkulhanek/wild-gaussians), reference commit `66fa22ac74a6ffba024842dff29ded114d41f4d0`. Equations 8–11 motivate an independent implementation; no WildGaussians source repository, sky model, appearance module or sparsification code is imported.
- Scaffold remains pinned, with R1 compatibility patches plus isolated training hooks in `experiments/runs/RGB-092-TRANSIENT/scaffold_transient.patch`; no frozen original source or checkpoint is changed.

不要把完整上游仓库直接提交到本目录。骨干冻结后记录：

| 项目 | 上游 | commit | 许可证 | 状态 |
|---|---|---|---|---|
| Standard 3DGS | https://github.com/graphdeco-inria/gaussian-splatting | `54c035f7834b564019656c3e3fcc3646292f727d` | Gaussian-Splatting License；仅研究/非商业使用 | cloned externally; CUDA 12.8 preflight passed |
| Mip-Splatting | https://github.com/autonomousvision/mip-splatting | `dda02ab5ecf45d6edb8c540d9bb65c7e451345a9` | 同上游 3DGS Gaussian-Splatting License | cloned externally; CUDA 12.8 preflight passed |
| Scaffold-GS | https://github.com/city-super/Scaffold-GS | `59c833b56bbbf510f3f64d40f81721995caced66` | Gaussian-Splatting License；仅研究/非商业使用 | cloned externally; CUDA 12.8 preflight passed; checkpoint compatibility patch verified |
| 3DGS-MCMC | https://github.com/ubc-vision/3dgs-mcmc | `7b4fc9f76a1c7b775f69603cb96e70f80c7e6d13` | Gaussian-Splatting License；仅研究/非商业使用 | externally cloned; CUDA 12.8 / sm_120 and interface preflight passed; dropped after matched 1k RGB screen |

冻结后建议使用Git submodule、独立clone脚本或明确安装步骤；任何方式都必须固定commit并记录本地补丁。

这些 commit 于 2026-07-30 固定并已克隆到 Git 忽略的`$R1_OUTPUT_ROOT/r1/third_party`；完整仓库、子模块、编译产物和训练输出绝不复制到本仓库。R1 最小本地补丁包括：显式`test.txt` split 读取、CUDA 12.8 头文件兼容、Scaffold 延迟 LPIPS 初始化/禁用其非冻结 VGG 指标，以及它的 checkpoint capture/restore 修复。每份实际 diff 与 hash 将保存在 Git 忽略的运行日志中；若比赛要求分发补丁或 checkpoint，须先按上述许可证确认再分发义务与竞赛用途兼容性。
