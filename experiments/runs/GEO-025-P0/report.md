# GEO-025-P0 - VGGT-Omega full official-P0 geometry

Status: `failed_preserved_coordinate_roundtrip_gate`.

This node covers the authorized full 303-training-image geometry generation,
operational-v2 fusion, and Scaffold initialization interface gate. It is not an
RGB result. The 303 fixed 25-view windows use every training image as an anchor
and never read validation, guard, or test RGB.

To keep the forward model as the only major candidate variable, fusion reuses
the exact historical implementation and parameters executed for earlier v2
candidates. Its known differences from some normative prose in the v2 document
are preserved and will be reported as a conclusion boundary, not silently
changed after the VGGT smoke result.

## Full Run Result

`full_v25_bf16_attempt1` completed all 303 windows and wrote 7,575 unique
window/view NPZ files covering all 303 training images. Generation-time records
reported full-grid finite depth/confidence/world points and valid fraction 1.0
for every record; sparse scale correspondence counts were 251/1,334/5,393
min/median/max. Peak allocated/reserved GPU memory was
6,680,261,632/8,279,556,096 bytes. The run reached the final strict NPZ audit
after 1,961.36 seconds.

The audit then stopped at its first coordinate hard-gate failure:
`vggt-omega-smoke-001-v25/365.npz`. Three shallow-depth pixels exceeded the
pre-forward 0.001-pixel official-P0 project/backproject limit; the maximum was
0.002058999 pixels at depth 0.0610960. The maximum depth roundtrip error was
1.52685e-6 and passed its fixed allclose check. The diagnostic is consistent
with the required float32 world-point serialization being most sensitive at
very shallow depth, but this observation does not turn the failed gate into a
pass.

Per policy, the tolerance, input, window, camera, weight, and evaluator were
not changed and no retry was launched. The strict audit did not complete the
remaining records. Fusion, candidate-scene construction, Scaffold reload, and
`GEO-025-10K` were not started. All 33 GiB of generated outputs and the failure
site remain preserved. Evidence hashes are recorded in `config.yaml`; the
read-only diagnosis SHA256 is
`07c8f87a4d2dc1dc0052e9e8b575dd46574042a2c01fac4557fd802b1f2fabe4`.
