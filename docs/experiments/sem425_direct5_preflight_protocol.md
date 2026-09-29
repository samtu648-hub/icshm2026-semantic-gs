# SEM-425 direct five-logit engineering preflight

## Question

Can the sealed `RGB-152-STANDARD-ABSGRAD-SPATIAL30` Gaussian field support the
minimal closed-set semantic representation already established conceptually by
`SEM-001`, while preserving the current spatial-confirmation roles and keeping
every RGB/reconstruction parameter immutable?

This is an engineering question only. It does not compare quality, choose a
model, or authorize a full semantic fit.

## Fixed parent and roles

- Parent RGB: sealed `RGB-152` step 30,000 endpoint.
- Split: `spatial_confirmation_v1`.
- Semantic training role: 256 labeled cameras.
- Confirmation role: 30 labeled cameras; neither RGB nor labels may be opened.
- Buffer role: 17 cameras; neither RGB nor labels may be opened.
- Unlabeled released views are not semantic supervision.

## Single variable

Add exactly five trainable logits to every frozen Gaussian. Convert per-Gaussian
logits to probabilities before native alpha compositing. Residual transmittance
is assigned to official class 0 (`background`). There is no decoder, latent
feature, DINO teacher, pseudo-label, post-processing, fusion or geometry update.

The preflight runs eight deterministic 768-pixel crop updates with
training-role-only median-frequency class weights. All five logits start at
zero; Adam learning rate is `0.05` and weight decay is zero. These values match
the intent of `SEM-001` and are not a quality search.

## Required gates

1. Registered 256/30/17 roles are disjoint and counts match.
2. Only training labels are hashed/read; confirmation and buffer accesses are zero.
3. The first direct-logit gradient is finite and nonzero.
4. Rendered five-class probabilities are finite and sum to one per pixel.
5. Saved logits reload exactly and reproduce the semantic render.
6. All frozen RGB parameter hashes are identical before/after.
7. A fixed fresh RGB render is pixel-identical before/after.
8. Peak allocated/reserved CUDA memory is recorded and below 12 GiB.

Passing these gates authorizes only a separately registered spatial training
comparison. Failure is preserved and does not alter `RGB140+SEM378`,
`RGB130+SEM371`, `RGB152+SEM396` or any earlier artifact.
