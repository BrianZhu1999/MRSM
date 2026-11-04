import unittest
from types import SimpleNamespace

import torch
from sampler.sde import VESDE
from trainer.loss import compute_fpe_terms, fpe_regularizer_loss


class GaussianScore(torch.nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale, dtype=torch.float64))

    def forward(self, x, timesteps, **kwargs):
        variance = (0.7 + timesteps.square()).reshape(-1, *([1] * (x.ndim - 1)))
        return -self.scale * x / variance, None


class FPETests(unittest.TestCase):
    def setUp(self):
        self.sde = VESDE(SimpleNamespace(device="cpu"), sigma_min=.1, sigma_max=2, N=100)
        self.x = torch.linspace(-1, 1, 24, dtype=torch.float64).reshape(3, 2, 1, 4).requires_grad_()
        self.t = torch.tensor([.2, .4, .7], dtype=torch.float64)

    def residual(self, step):
        net = GaussianScore()
        sigma = self.sde.marginal_prob(torch.zeros_like(self.t), self.t)[1]
        score = net(self.x, timesteps=sigma)[0]
        return compute_fpe_terms(net, self.sde, self.x, torch.zeros_like(self.x), self.t,
                                 score, step, step / 2)[0].detach().norm()

    def test_gaussian_score_fpe_residual_and_stencil_refinement(self):
        coarse = self.residual(.002)
        fine = self.residual(.001)
        self.assertLess(float(fine), float(coarse) * .35)
        self.assertLess(float(fine), 2e-5)

    def test_regularizer_backpropagates_for_perturbed_score(self):
        torch.manual_seed(123)
        net = GaussianScore(.9)
        loss = fpe_regularizer_loss(net, self.sde, self.x.detach(), alpha=1e-4, beta=0, m=1)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0)
        self.assertTrue(torch.isfinite(net.scale.grad))
        self.assertGreater(abs(float(net.scale.grad)), 0)


if __name__ == "__main__":
    unittest.main()
