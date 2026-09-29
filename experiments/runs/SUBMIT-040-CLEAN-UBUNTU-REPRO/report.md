# SUBMIT-040-CLEAN-UBUNTU-REPRO

Status: completed clean-container reproduction (`full-20260907-r2`).

The dedicated context contains only the Dockerfile and dependency recipe. No
images, labels, models, repository outputs or user configuration are sent to the
builder. No existing image tag or running user container is replaced. Original
frozen bundle and all SUBMIT-020 artifacts stay read-only.

The old dependency list omitted `torch_scatter`, required by Scaffold's
`gaussian_model.py`. A new recipe pins its recorded PyG binary source/hash.
The bundled rasterization and KNN extensions will be rebuilt from source copies
excluding old `.so`/build products. Their installed modules are preloaded and
path-checked to prevent the bundle's old binaries shadowing the new build.

Verification uses a fresh Ubuntu24.04 container with fresh Python dependencies,
but the same physical RTX5070/host kernel and a read-only existing CUDA12.8
compiler. This is not a second-machine or fresh compiler installation claim.
Inference receives only official camera text metadata; frozen historical RGB
outputs are used solely for the subsequent byte comparison. No labels are read.

The first build command failed before execution because the legacy Docker
builder rejected `--progress`; the compatible invocation then proceeded. This
does not affect any model or research result.

First actual clean verification (`full-20260907-r1`) successfully built both CUDA
extensions and verified their fresh import paths, then stopped at the first
render because `scene.dataset_readers` also imports the undeclared `colorama`.
The original logs and output directory remain untouched. A separate r2 image
adds only host-recorded `colorama==0.4.6`; no model/source behavior is patched.

## Verified completion

The r2 container freshly installed Python packages and rebuilt the two CUDA
extensions, then passed all eight checks: two independent native processes with
RGB and semantic output exactly equal to the frozen reference; both outputs exact
at320×240; nonempty output rejection; and all2,161 frozen bundle files unchanged.
This uses camera ID/name140.png, not140 different cameras. The fresh extension
paths are logged under `/opt/icshm-runtime`, never the old bundled `.so` files.

Verification runtime70.671s, excluding the image build/dependency downloads.
Runtime Torch2.11.0+cu128, Transformers5.5.3, NumPy1.26.4 and
torch_scatter2.1.2+pt211cu128. The full resolved package list and image ID are
stored with hashes in the output provenance. The task retains both r1 and r2
image tags; shared Docker layers occupy roughly8GiB additional host disk.
No driver, existing Python environment, existing user container or frozen model
was changed. Available host disk remained about250GB after the exercise.

Output: `outputs/submit040/full-20260907-r2/`; r1 failure logs remain in
`outputs/submit040/full-20260907-r1/`. This closes the fresh Ubuntu userspace,
fresh Python and fresh-extension exercise, but not second-machine/different-GPU
reproduction, fresh compiler installation, full paper/video or official submission.
