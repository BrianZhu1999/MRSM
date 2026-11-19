import json
import math
from pathlib import Path
import unittest

import numpy as np

from scripts.line_array_metrics import matched_source_errors


class Figure5PublicSettingsTests(unittest.TestCase):
    def test_condition_manifest_counts(self):
        path = Path(__file__).parents[1] / "configs" / "figure5_line_array.json"
        settings = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(settings["dual_frequency_hz"]), 3)
        self.assertEqual(len(settings["source_configurations"]), 4)
        self.assertEqual(len(settings["localization_methods"]), 4)
        self.assertEqual(len(settings["localization_arrays"]), 5)
        self.assertEqual(len(settings["held_out_field_methods"]), 5)

    def test_two_source_rmse_uses_squared_distance_assignment(self):
        x_grid = np.array([[0.0, 1.0]])
        z_grid = np.zeros_like(x_grid)
        spectrum_db = np.array([[0.0, -1.0]])
        truth = np.array([[0.1, 0.0], [1.2, 0.0]])
        order = np.argsort(spectrum_db.reshape(-1))[::-1]
        estimates = np.column_stack((x_grid.reshape(-1)[order[:2]], z_grid.reshape(-1)[order[:2]]))
        errors = matched_source_errors(truth, estimates)
        self.assertAlmostEqual(float(errors.mean()), 0.15)
        self.assertAlmostEqual(float(np.sqrt(np.mean(errors ** 2))), math.sqrt(0.025))


if __name__ == "__main__":
    unittest.main()
