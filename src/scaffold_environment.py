"""Explicit world-direction background, isolated from the frozen renderer."""
import math
import torch
import torch.nn.functional as F


class Environment(torch.nn.Module):
    def __init__(self, height=16, width=32, *, enabled=True, device="cuda"):
        super().__init__()
        self.rgb = torch.nn.Parameter(torch.zeros((1, 3, height, width), device=device), requires_grad=enabled)

    def sample_directions(self, directions):
        direction = F.normalize(directions, dim=-1)
        longitude = torch.atan2(direction[..., 2], direction[..., 0]) / (2*math.pi) + .5
        latitude = torch.asin(direction[..., 1].clamp(-1,1)) / math.pi + .5
        height, width = self.rgb.shape[-2:]
        # Pad the periodic longitude by one pixel on each side.
        texture = torch.cat((self.rgb[..., -1:], self.rgb, self.rgb[..., :1]), dim=-1)
        gx = (longitude*width + 1) / (width+2)*2-1
        gy = latitude*2-1
        grid = torch.stack((gx, gy), -1).unsqueeze(0)
        return F.grid_sample(texture, grid, mode="bilinear", padding_mode="border", align_corners=False)[0]

    def forward(self, view):
        h, w = int(view.image_height), int(view.image_width)
        with torch.no_grad():
            ys, xs = torch.meshgrid(torch.arange(h, device=self.rgb.device), torch.arange(w, device=self.rgb.device), indexing="ij")
            x = ((2*xs+1)/w-1)*math.tan(view.FoVx*.5)
            y = ((2*ys+1)/h-1)*math.tan(view.FoVy*.5)
            camera_rays = torch.stack((x, y, torch.ones_like(x)), -1)
            # The Scaffold transform is transposed: row-camera to row-world.
            rotation = view.world_view_transform[:3, :3]
            directions = camera_rays @ rotation.T
        return self.sample_directions(directions)


def install(environment, *, checkpoint_path=None):
    """Patch imports in this process only; never edit the upstream checkout."""
    import gaussian_renderer as renderer
    from scene.gaussian_model import GaussianModel
    from src.scaffold_semantic_sidecar import _raster_settings
    from diff_gaussian_rasterization import GaussianRasterizer

    original_render = renderer.render
    original_setup = GaussianModel.training_setup
    original_capture = GaussianModel.capture

    def render(view, pc, pipe, bg, scaling_modifier=1., visible_mask=None, retain_grad=False):
        output = original_render(view, pc, pipe, bg, scaling_modifier, visible_mask, retain_grad)
        generated = renderer.generate_neural_gaussians(view, pc, visible_mask, is_training=pc.get_color_mlp.training)
        xyz, color, opacity, scale, rotation = generated[:5]
        settings = _raster_settings(view, torch.zeros_like(bg))._replace(scale_modifier=scaling_modifier)
        alpha, _ = GaussianRasterizer(raster_settings=settings)(means3D=xyz, means2D=output["viewspace_points"], shs=None, colors_precomp=torch.ones_like(color), opacities=opacity, scales=scale, rotations=rotation, cov3D_precomp=None)
        output["render"] = output["render"] + (1-alpha[:1])*environment(view)
        return output

    def setup(model, args):
        original_setup(model, args)
        if environment.rgb.requires_grad:
            model.optimizer.add_param_group({"params": [environment.rgb], "lr": .01, "name": "mlp_environment"})
            def project(_optimizer, _args, _kwargs):
                with torch.no_grad():
                    environment.rgb.clamp_(0, 1)
            model.optimizer.register_step_post_hook(project)

    def capture(model):
        if checkpoint_path:
            torch.save({"experiment_id": "RGB-101-ENVIRONMENT" if environment.rgb.requires_grad else "RGB-100-ENV-CTRL", "state_dict": environment.state_dict()}, checkpoint_path)
        return original_capture(model)

    renderer.render = render
    GaussianModel.training_setup = setup
    GaussianModel.capture = capture
    return original_render
