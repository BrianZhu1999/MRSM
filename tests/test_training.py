import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scripts import train_score


class TinyScore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(()))

    def forward(self, x, timesteps, **kwargs):
        sigma = timesteps.reshape(-1, 1, 1, 1)
        return self.weight * x / (1 + sigma.square()), None


class TrainingTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_updates_and_protects_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = np.random.default_rng(3).normal(size=(6, 30, 8, 4)).astype("float32")
            np.save(root / "data.npy", data)
            config = dict(data="line_array", data_location=str(root / "data.npy"), train_split=.8,
                num_samples_train=3, num_samples_val=2, num_frames=2, num_interval=1,
                num_conditions=0, num_components=4, is_scalar=True, lr=1e-3,
                seed=7, batch_size=1, ema_rate=.99, beta_min=.1, beta_max=2, num_scales=10)
            (root / "config.json").write_text(json.dumps(config))
            def run(output, steps, resume=False):
                argv = ["train.py", "--config", str(root / "config.json"), "--output-dir", str(root / output),
                        "--total-steps", str(steps), "--device", "cpu", "--checkpoint-every", "2"]
                if resume:
                    argv.append("--resume")
                with patch("sys.argv", argv), patch.object(train_score, "get_model", lambda cfg: TinyScore()):
                    train_score.main("dsm")
            run("full", 4)
            run("split", 2)
            with self.assertRaises(FileExistsError):
                run("split", 2)
            run("split", 4, resume=True)
            full = torch.load(root / "full/checkpoint.pth", weights_only=False)
            split = torch.load(root / "split/checkpoint.pth", weights_only=False)
            torch.testing.assert_close(full["model"]["weight"], split["model"]["weight"], rtol=0, atol=0)
            torch.testing.assert_close(full["ema"]["shadow_params"][0], split["ema"]["shadow_params"][0], rtol=0, atol=0)
            self.assertEqual(full["loss_train"], split["loss_train"])


if __name__ == "__main__":
    unittest.main()
