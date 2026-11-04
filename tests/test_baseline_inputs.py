import importlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch


class BaselineInputTests(unittest.TestCase):
    def test_original_layout_and_channel_normalizer(self):
        for model in ("FNO", "LNO"):
            for suffix in ("23456", "channel_nor_23456"):
                module = importlib.import_module(f"scripts.baselines.train_{model}_{suffix}")
                for channels in (2, 3):
                    with self.subTest(model=model, suffix=suffix, channels=channels), tempfile.TemporaryDirectory() as directory:
                        values = np.random.default_rng(123).normal(size=(1,20,64,64,channels)).astype("float32")
                        path = Path(directory) / "input.npy"
                        np.save(path, values)
                        config = SimpleNamespace(SEQUENCE_LENGTH=10, NUM_CHANNELS=channels, OUTPUT_RES=64)
                        actual = module.load_data_file(path, config)
                        self.assertEqual(actual.shape, (2,64,64,10*channels))
                        torch.testing.assert_close(actual[0,3,7], torch.from_numpy(values[0,:10,3,7,:].reshape(-1)))
                        if hasattr(module, "UnitGaussianNormalizer"):
                            normalization = module.UnitGaussianNormalizer(actual, channels)
                            torch.testing.assert_close(normalization.decode(normalization.encode(actual)), actual, rtol=1e-5, atol=1e-6)

    def test_channel_count_mismatch_is_rejected(self):
        module = importlib.import_module("scripts.baselines.train_FNO_23456")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.npy"
            np.save(path, np.zeros((1,10,64,64,3), dtype="float32"))
            config = SimpleNamespace(SEQUENCE_LENGTH=10, NUM_CHANNELS=2, OUTPUT_RES=64)
            with self.assertRaisesRegex(ValueError, "expected"):
                module.load_data_file(path, config)


if __name__ == "__main__":
    unittest.main()
