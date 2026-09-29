#!/usr/bin/env python3
"""Fresh-process optimizer+environment reload and external COLMAP rendering."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"scripts"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("source-root","run-dir","data-root","output-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--iteration",type=int,required=True)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    from src.scaffold_semantic_sidecar import configure_scaffold_import
    configure_scaffold_import(args.source_root)
    import torch
    import numpy as np
    from PIL import Image
    from scene.gaussian_model import GaussianModel
    from src.scaffold_environment import Environment, install
    from src.scaffold_semantic_sidecar import pipeline, render_frozen_rgb
    from src.camera_adapter import redistort_rgb
    from validate_candidate_checkpoint import optimizer_params
    from run_sem001 import official_view, tensor_to_rgb
    from package_submission_baseline import sha256
    payload=torch.load(args.run_dir/"environment.pth",map_location="cuda",weights_only=True)
    environment=Environment(enabled=payload["experiment_id"]=="RGB-101-ENVIRONMENT")
    environment.load_state_dict(payload["state_dict"])
    install(environment)
    stored,iteration=torch.load(args.run_dir/f"chkpnt{args.iteration}.pth",map_location="cuda",weights_only=False)
    assert iteration==args.iteration
    model=GaussianModel(n_offsets=10)
    model.set_appearance(303)
    model.restore(stored,optimizer_params())
    model.eval()
    _,camera,view=official_view(args.data_root,"140.png",0)
    rgb=tensor_to_rgb(render_frozen_rgb(view,model,pipeline()))
    official=np.rint(redistort_rgb(rgb.astype(np.float32)/255.,camera).clip(0,1)*255).astype(np.uint8)
    assert official.shape==(camera.height,camera.width,3)
    Image.fromarray(official).save(args.output_dir/"rgb.png")
    report={"experiment_id":payload["experiment_id"],"iteration":iteration,"optimizer_reload":True,"environment_reload":True,"anchor_count":len(model.get_anchor),"official_camera_grid":[camera.height,camera.width],"rgb_sha256":sha256(args.output_dir/"rgb.png"),"checkpoint_sha256":sha256(args.run_dir/f"chkpnt{iteration}.pth"),"environment_sha256":sha256(args.run_dir/"environment.pth"),"uses_reference_rgb_or_labels":False}
    (args.output_dir/"result.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__=="__main__":
    main()
