# SEM-020-EQ — disabled-gate equivalence

- Status: completed; parent: `SEM-001`; code was run from the pre-freeze worktree and then frozen as `5669518`.
- Output: Git-ignored `$R1_OUTPUT_ROOT/r2/sem_002/eq/`; `result.json` SHA-256 `d638a19804c51169c279c0a875f558d067586d2e9415a6911a13c9f042356587`.

The frozen SEM-001 semantic checkpoint (`8e3dbe30e17f67081a72380018fc281c155ec00eed83cd5a19b3285bf4fb8aa3`) was loaded with RGB-030-FULL.  With visibility disabled, SEM-002 deliberately invokes the unchanged SEM-001 renderer: five-class logits and class-ID PNG are bit-identical and RGB maximum absolute pixel difference is zero.

The isolated contribution adapter was also tested in its all-contributor cumulative mode (`threshold=1.0`).  Its maximum semantic-probability difference from SEM-001 was `1.03116e-5` (registered tolerance `2e-5`) and class IDs were identical.  This is an adapter numerical cross-check, not an S1 quality result.
