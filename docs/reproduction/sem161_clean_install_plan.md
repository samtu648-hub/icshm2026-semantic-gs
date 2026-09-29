# SEM-161 clean-install reproduction

Status: completed by SUBMIT-040 r2. SUBMIT-030 demonstrated new-process reload
on the existing host, not a fresh dependency installation.

The host has cached Ubuntu24.04 and NVIDIA CUDA12.8.1-base-ubuntu24.04 images,
and an NVIDIA container runtime. This makes a fresh container reproduction
possible without replacing drivers or touching existing containers. The cached
CUDA base image does not by itself contain Python, the pinned Python packages,
the compiler toolchain or rebuilt Scaffold extensions.

Use `requirements/submission_sem161_cuda128.txt` for the actual recorded runtime.
The current installed Transformers is5.5.3; older training reports record4.57.1.
Record both the parent training version and the reproduction version, and verify
output equality instead of assuming those versions are equivalent.

The fresh recipe is in `reproduction/sem161/`: it additionally includes the
previously omitted runtime dependency `torch_scatter` with its recorded PyG wheel
URL and SHA-256. The native extension source copies deliberately exclude old
`.so` and build products; fresh imports must resolve inside the new environment,
not the read-only bundle's old binaries. The first Docker invocation rejected
the unsupported `--progress` flag before building; the legacy-compatible command
then began successfully. No existing user container or image tag was replaced.

The clean-install exercise should:

1. Create a dedicated image/container and isolated output directory, never
   modify a running user container or mount the entire home directory.
2. Install the pinned dependencies and build only the bundled Scaffold CUDA
   extensions. Mount the SEM-161 bundle read-only at a relocated path.
3. Give inference only official camera metadata; no dataset images or labels.
   Use an independently supplied frozen output strictly for the subsequent audit.
4. Render native and320×240 cameras in separate processes. Check five-class
   range, shapes, RGB/semantic equality, nonempty-output rejection and immutable
   bundle/source hashes. Save image digest, package lock and build logs.
5. State whether success covers clean OS, clean dependencies, clean extension
   build and different GPU. These are distinct claims; host-runtime bind mounts
   do not count as clean dependency installation.

SUBMIT-040 r2 executed these checks successfully: two native processes and one
320×240 render of camera140 exactly replayed frozen RGB and semantics, rejected
nonempty output, and preserved all2,161 bundle files. Both extension imports were
verified in the fresh environment. `colorama==0.4.6` was another missing import
found by r1; its failure logs remain and r2 adds the locked dependency.

SUBMIT-002 is complete at the fresh-container/userspace/dependency scope, not a
second machine or independent driver installation. The external read-only CUDA
compiler and same host GPU/kernel are explicit limitations. No external
distribution or official hidden evaluation occurred.
