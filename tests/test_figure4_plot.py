import unittest

import numpy as np

from scripts.plot_figure4 import fit_velocity_waveforms


class Figure4PlotTests(unittest.TestCase):
    def test_circular_shift_gain_and_bias_preserve_raw_waveforms(self):
        raw = np.array([0, 1, -1, 3, -2, 2, 0, -1, 0, 0], dtype=float)[:, None]
        reference = 2.5 * np.roll(raw, 2, axis=0) - 0.3
        original = raw.copy()
        result = fit_velocity_waveforms(reference, raw)
        np.testing.assert_array_equal(result["lag_samples"], [2])
        np.testing.assert_allclose(result["gain"], [2.5], rtol=0, atol=1e-14)
        np.testing.assert_allclose(result["bias"], [-0.3], rtol=0, atol=1e-14)
        np.testing.assert_allclose(result["processed_prediction_velocity"], reference, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(result["raw_prediction_velocity"], original)
        np.testing.assert_array_equal(raw, original)


if __name__ == "__main__":
    unittest.main()
