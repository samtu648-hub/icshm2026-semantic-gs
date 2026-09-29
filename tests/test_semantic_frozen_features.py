import copy
from types import SimpleNamespace
import unittest

try:
    import torch
    from torch import nn
    from src.semantic_frozen_features import ARMS, FrozenFeatureSegmenter, quality_gates, state_hash
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional model runtime unavailable")
class FrozenFeaturesTest(unittest.TestCase):
    def backbone(self, arm):
        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Conv2d(3, 16, 16, stride=16)
                self.drop = nn.Dropout(0.9)
                self.config = SimpleNamespace(hidden_size=16, hidden_sizes=[16])

            def forward(self, *, pixel_values):
                feature = self.drop(self.layer(pixel_values))
                if arm == ARMS[1]:
                    feature = feature.flatten(2).transpose(1, 2)
                    registers = torch.full((len(feature), 5, 16), 999., device=feature.device)
                    feature = torch.cat((registers, feature), dim=1)
                return SimpleNamespace(last_hidden_state=feature)
        return Backbone()

    def test_frozen_eval_backbone_and_trainable_head_reload(self):
        for arm in ARMS:
            model = FrozenFeatureSegmenter(self.backbone(arm), arm, width=16).train()
            before = state_hash(model.backbone)
            self.assertFalse(model.backbone.training)
            image = torch.randn(1, 3, 32, 48)
            original = image.clone()
            opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.01)
            logits = model(pixel_values=image).logits
            self.assertEqual(tuple(logits.shape), (1, 5, 2, 3))
            logits.square().mean().backward()
            self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
            opt.step()
            self.assertEqual(before, state_hash(model.backbone))
            self.assertTrue(torch.equal(image, original))
            fresh = FrozenFeatureSegmenter(copy.deepcopy(model.backbone), arm, width=16)
            fresh.load_head(model.head_state())
            self.assertTrue(torch.equal(model.eval()(pixel_values=image).logits,
                                        fresh.eval()(pixel_values=image).logits))

    def test_common_decoder_initialization_and_bad_grid(self):
        models = [FrozenFeatureSegmenter(self.backbone(a), a, width=16) for a in ARMS]
        self.assertEqual(state_hash(models[0].decoder), state_hash(models[1].decoder))
        with self.assertRaises(ValueError):
            models[1](pixel_values=torch.zeros(1,3,31,48))

    def test_composite_gate_accepts_small_recall_trade_but_rejects_spatial_collapse(self):
        from pathlib import Path
        import json
        cfg = json.loads((Path(__file__).parents[1]/"experiments/runs/SEM-310-DINOV3-FROZEN-FEATURE-OOF/config.yaml").read_text())["gates"]
        def metric(miou, iou, recall):
            return {"mIoU_with_background": miou, "per_class": {"foundation": {"iou":iou,"recall":recall}}}
        row = {"baseline":metric(.85,.65,.96), ARMS[0]:metric(.852,.66,.96), ARMS[1]:metric(.86,.70,.95)}
        folds = {str(i):copy.deepcopy(row) for i in range(3)}
        self.assertTrue(all(quality_gates(row,folds,cfg).values()))
        folds["2"][ARMS[1]]["per_class"]["foundation"]["iou"] = .60
        self.assertFalse(all(quality_gates(row,folds,cfg).values()))


if __name__ == "__main__":
    unittest.main()
