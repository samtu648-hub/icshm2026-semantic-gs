#!/usr/bin/env python3
"""GPU zero-background equivalence, geometry gradients and camera directions."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"scripts"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import torch
    import run_sem001 as s
    from src.scaffold_semantic_sidecar import load_frozen_scaffold, ScaffoldSettings, pipeline, render_frozen_rgb, frozen_generated_geometry, _render_color_channels
    from src.scaffold_environment import Environment, install
    root = args.artifact_root
    model = load_frozen_scaffold(root/"r1/third_party/scaffold-gs", root/"r1/runs/RGB-030-FULL/point_cloud/iteration_30000/point_cloud.ply", ScaffoldSettings(train_camera_count=303,n_offsets=10))
    pipe = pipeline()
    _, _, view = s.official_view(args.data_root,"001.png",0)
    old = render_frozen_rgb(view, model, pipe)
    environment = Environment()
    install(environment)
    new = render_frozen_rgb(view, model, pipe)
    exact = torch.equal(old,new)
    geometry, _ = frozen_generated_geometry(view,model,pipe)
    # Two copies of a differentiable fixed geometry: identical semantic colors,
    # with/without the additional zero-environment alpha derivative path.
    gradients = []
    for auxiliary in (False, False, True):
        geom = tuple(t.clone().requires_grad_(True) for t in geometry[:4]) + (geometry[4],)
        colors = torch.full_like(geom[0],.3)
        image = _render_color_channels(geom,colors,view,torch.zeros(3,device="cuda"))
        if auxiliary:
            alpha = _render_color_channels(geom,torch.ones_like(colors),view,torch.zeros(3,device="cuda"))
            image = image+(1-alpha)*environment(view)
        gradients.append(torch.autograd.grad(image.mean(),geom[:4]))
    maxdiff = max(float((a-b).abs().max()) for a,b in zip(gradients[0], gradients[2]))
    repeatdiff = max(float((a-b).abs().max()) for a,b in zip(gradients[0], gradients[1]))
    # Atomic additions in the native backward are not bit-reproducible even
    # without this module. Test the added derivative in isolation, exactly.
    geom = tuple(t.clone().requires_grad_(True) for t in geometry[:4]) + (geometry[4],)
    alpha = _render_color_channels(geom,torch.ones_like(geom[0]),view,torch.zeros(3,device="cuda"))
    added_gradient = torch.autograd.grad(((1-alpha)*environment(view)).mean(),geom[:4])
    zero_added = all(torch.count_nonzero(g).item() == 0 for g in added_gradient)
    report = {"rgb_tensor_exact": exact, "added_geometry_gradient_exact_zero": zero_added, "native_backward_repeat_max_difference": repeatdiff, "combined_backward_max_difference": maxdiff, "input_camera": "official_COLMAP_001", "learned_environment_parameters": environment.rgb.numel(), "test_correction": "First attempt compared independent native atomic backward kernels for bit equality. Revised pretraining test checks the module added derivative is exactly zero, and reports native repeat noise without asserting bit-reproducible native gradients."}
    (args.output_dir/"result.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))
    assert exact and zero_added


if __name__ == "__main__":
    main()
