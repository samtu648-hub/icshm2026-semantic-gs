# SEM-006 — Cable3D semantic micro-splat single-fold experiment not started

Status: `not_started_prerequisite_geometry_failed`; quality hypothesis remains unproven.

The mandatory `SEM-060-CABLE-GEO` preflight achieved 2 px recall 0.410899 and 40/40 check-view coverage, but 2 px precision was only 0.104230, median/P90 projected-curve error was 16.553/65.437 px, 69.11% of projected pixels were farther than 8 px from a check centerline, and the coordinate-scale audit failed. It was therefore dropped before any micro-splat representation, equivalence run, CUDA smoke or 10k training was implemented.

No fixed 74-view SEM-006 evaluation, iteration selection, three-fold/three-seed confirmation, pseudo-label use or unlabeled-image use occurred. `SEM-001` remains the frozen semantic baseline. This failure concerns the preregistered training-side cable geometry recipe; it does not show that 2D images lack cable information, and it does not claim any RGB improvement or degradation.
