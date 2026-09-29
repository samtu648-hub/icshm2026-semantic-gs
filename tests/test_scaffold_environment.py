import unittest
try:
    import torch
except ModuleNotFoundError:
    torch = None
if torch is not None:
    from src.scaffold_environment import Environment


@unittest.skipIf(torch is None, "requires optional Torch runtime")
class EnvironmentTests(unittest.TestCase):
    def test_zero_constant_and_periodic(self):
        environment = Environment(device="cpu")
        directions = torch.tensor([[[1., 0., 0.], [-1., 0., 1e-7], [-1., 0., -1e-7]]])
        self.assertTrue(torch.equal(environment.sample_directions(directions), torch.zeros(3,1,3)))
        with torch.no_grad():
            environment.rgb.fill_(.25)
        torch.testing.assert_close(environment.sample_directions(directions), torch.full((3,1,3), .25))

    def test_zero_compositing_gradient(self):
        rgb = torch.rand((3, 5, 6), requires_grad=True)
        alpha = torch.rand((1,5,6), requires_grad=True)
        environment = torch.zeros((3,5,6), requires_grad=True)
        result = rgb + (1-alpha)*environment
        self.assertTrue(torch.equal(result, rgb))
        grad = torch.autograd.grad(result.sum(), (rgb, alpha, environment))
        self.assertTrue(torch.equal(grad[0], torch.ones_like(rgb)))
        self.assertTrue(torch.equal(grad[1], torch.zeros_like(alpha)))
        self.assertGreater(float(grad[2].sum()), 0)


if __name__ == "__main__":
    unittest.main()
