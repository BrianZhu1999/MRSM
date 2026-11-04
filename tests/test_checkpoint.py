import tempfile
import unittest
from pathlib import Path

import torch
from models.ema import ExponentialMovingAverage
from trainer.utils import restore_checkpoint, save_checkpoint


class CheckpointTests(unittest.TestCase):
    def state(self, wrapped=False):
        model = torch.nn.Linear(3, 2)
        if wrapped:
            model = torch.nn.DataParallel(model)
        return dict(model=model, ema=ExponentialMovingAverage(model.parameters(), .99),
                    optimizer=torch.optim.Adam(model.parameters()), epoch=4, iteration=12,
                    loss_train=[1.0], loss_val=[])

    def test_roundtrip_both_dataparallel_directions(self):
        for source_wrapped, target_wrapped in [(True, False), (False, True)]:
            with self.subTest(source=source_wrapped), tempfile.TemporaryDirectory() as directory:
                source = self.state(source_wrapped)
                target = self.state(target_wrapped)
                path = Path(directory) / "checkpoint.pth"
                save_checkpoint(path, source)
                restore_checkpoint(path, target, "cpu")
                for left, right in zip(source["model"].parameters(), target["model"].parameters()):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                for shadow, parameter in zip(target["ema"].shadow_params, target["model"].parameters()):
                    self.assertEqual(shadow.device, parameter.device)
                    self.assertEqual(shadow.shape, parameter.shape)
                self.assertEqual(target["iteration"], 12)
                self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_missing_file_does_not_create_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing" / "checkpoint.pth"
            with self.assertRaises(FileNotFoundError):
                restore_checkpoint(path, self.state(), "cpu")
            self.assertFalse(path.parent.exists())

    def test_missing_parameter_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.pth"
            state = self.state()
            torch.save(dict(model={"weight": state["model"].weight.detach()}), path)
            with self.assertRaisesRegex(ValueError, "checkpoint keys"):
                restore_checkpoint(path, state, "cpu")

    def test_wrong_ema_shape_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self.state()
            ema = state["ema"].state_dict()
            ema["shadow_params"][0] = torch.zeros(1)
            path = Path(directory) / "broken.pth"
            torch.save(dict(model=state["model"].state_dict(), ema=ema), path)
            with self.assertRaisesRegex(ValueError, "EMA parameter shape"):
                restore_checkpoint(path, dict(model=state["model"], ema=state["ema"]), "cpu")

    def test_wrong_model_shape_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self.state()
            saved = state["model"].state_dict()
            saved["weight"] = torch.zeros(1)
            path = Path(directory) / "broken.pth"
            torch.save(dict(model=saved), path)
            with self.assertRaisesRegex(ValueError, "parameter shape"):
                restore_checkpoint(path, state, "cpu")


if __name__ == "__main__":
    unittest.main()
