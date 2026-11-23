from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from sampler import utils
from sampler.sde import VESDE
from sampler.tube import pressure_only_observation, tube_scaler


class TubePhysicsTests(unittest.TestCase):
    def sample(self, channels, coefficient=0.1, boundary_coefficient=0.0):
        cfg = SimpleNamespace(num_frames=4, num_components=channels,
                              num_conditions=0, image_size=7, device="cpu")
        utils.setup_seed(123)
        return utils.generate_parallel_1d_ISS_1dtube_res(
            cfg, lambda x, t: torch.zeros_like(x),
            VESDE(cfg, sigma_min=0.1, sigma_max=0.2, N=3),
            np.zeros((1, 6, channels, 7), dtype=np.float32), lambda x: x,
            corrector=utils.NoneCorrector, num_steps=6, overlap=2,
            denoising_steps=3, physics_start_step=3,
            alpha=0, beta1=0, beta2=0, reg_coef_pa=0,
            data_scalar=lambda x: x, reg_coef_wave=coefficient,
            reg_coef_cont=coefficient, reg_coef_euler=coefficient,
            reg_coef_bc=boundary_coefficient,
            physics_c=2.0, physics_rho0=1.0, physics_dt=0.1, physics_dx=0.2,
        )

    def test_two_channel_residual_reaches_sampling_gradient(self):
        residual = utils.calculate_physics_residual_loss_1d_3
        gradients = []

        def record_gradient(data, **kwargs):
            losses = residual(data, **kwargs)
            gradients.append(torch.autograd.grad(sum(losses), data, retain_graph=True)[0])
            return losses

        with patch.object(utils, "calculate_physics_residual_loss_1d_3", side_effect=record_gradient) as tube:
            with patch.object(utils, "calculate_physics_residual_loss_1d_full") as full:
                active, _, losses = self.sample(2)
        inactive, _, _ = self.sample(2, coefficient=0)
        self.assertEqual(tube.call_count, 3)
        full.assert_not_called()
        self.assertTrue(all(value > 0 for value in losses["loss_phy"]))
        self.assertTrue(all(torch.isfinite(gradient).all() and gradient.abs().sum() > 0
                            for gradient in gradients))
        self.assertFalse(np.array_equal(active, inactive))

    def test_four_channel_residual_and_boundary_behavior_are_preserved(self):
        residual = utils.calculate_physics_residual_loss_1d_full
        with patch.object(utils, "calculate_physics_residual_loss_1d_full", wraps=residual) as full:
            with patch.object(utils, "calculate_physics_residual_loss_1d_3") as tube:
                reference, _, reference_losses = self.sample(4)
        unchanged, _, unchanged_losses = self.sample(4, boundary_coefficient=10)
        self.assertEqual(full.call_count, 3)
        tube.assert_not_called()
        np.testing.assert_array_equal(reference, unchanged)
        self.assertEqual(reference_losses["loss_phy"], unchanged_losses["loss_phy"])
        self.assertTrue(all(value > 0 for value in reference_losses["loss_phy"]))

    def test_independent_scaling_ignores_reference_velocity(self):
        data = np.random.default_rng(123).normal(size=(1, 12, 2, 5))
        perturbed = data.copy()
        perturbed[:, :, 1, :] = perturbed[:, :, 1, :] * 100 + 20
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scalar.npy"
            np.save(path, np.array([[0.2, 0.0], [0.8, 0.002]]))
            first = tube_scaler(data, "cpu", "training", path)
            second = tube_scaler(perturbed, "cpu", "training", path)
            np.testing.assert_array_equal(first.mean_np, second.mean_np)
            np.testing.assert_array_equal(first.std_np, second.std_np)
            np.testing.assert_array_equal(pressure_only_observation(data, first),
                                          pressure_only_observation(perturbed, second))
            record = tube_scaler(data, "cpu")
            changed_record = tube_scaler(perturbed, "cpu")
            self.assertFalse(np.array_equal(record.std_np, changed_record.std_np))

    def test_invalid_training_scale_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scalar.npy"
            np.save(path, [[0.0, 0.0], [1.0, 0.0]])
            with self.assertRaisesRegex(ValueError, "positive"):
                tube_scaler(None, "cpu", "training", path)


if __name__ == "__main__":
    unittest.main()
