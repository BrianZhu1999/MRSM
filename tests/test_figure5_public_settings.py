import ast
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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

    def test_condition_files_cover_public_cases(self):
        root = Path(__file__).parents[1]
        settings_text = (root / "configs/figure5_line_array.json").read_text(encoding="utf-8")
        settings = json.loads(settings_text)
        notebook = json.loads((root / "notebooks/line_array/reconstruction_music.ipynb").read_text(encoding="utf-8"))
        cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        setup = ast.parse(next(cell for cell in cells if "FIGURE5_CONFIG_PATH" in cell))
        setup_nodes = []
        for node in setup.body:
            targets = {target.id for target in node.targets if isinstance(target, ast.Name)} if isinstance(node, ast.Assign) else set()
            if not targets & {"PAIR_INDEX", "CONFIGURATION_INDEX"}:
                setup_nodes.append(node)
            if "COORD_GROUND_TRUTH" in targets:
                break
        setup_code = compile(ast.Module(body=setup_nodes, type_ignores=[]), "notebook_config", "exec")
        coordinate_read = next(
            node.iter for cell in cells for node in ast.walk(ast.parse(cell))
            if isinstance(node, ast.For)
            and any(isinstance(part, ast.Name) and part.id == "COORD_GROUND_TRUTH" for part in ast.walk(node.iter))
        )
        coordinate_code = compile(ast.Expression(body=coordinate_read), "notebook_coordinates", "eval")
        expected = {
            f"Aligned_{f1}_{f2}_{position}.npy"
            for f1, f2 in ((2800, 2900), (2800, 3000), (2900, 3000))
            for position in ("pos3", "pos4", "pos7", "pos8")
        }
        checker = ast.parse((root / "tests/check_release_assets.py").read_text(encoding="utf-8"))
        array_nodes = [node for node in checker.body if isinstance(node, ast.For)
                       or isinstance(node, ast.Assign) and any(
                           isinstance(target, ast.Name) and target.id == "ARRAYS" for target in node.targets)]
        assets = {}
        exec(compile(ast.Module(body=array_nodes, type_ignores=[]), "release_arrays", "exec"), assets)
        self.assertEqual({Path(name).name for name in assets["ARRAYS"] if Path(name).name.startswith("Aligned_")}, expected)
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            (temporary_root / "configs").mkdir()
            (temporary_root / "configs/figure5_line_array.json").write_text(settings_text, encoding="utf-8")
            (temporary_root / "data").mkdir()
            for name in expected:
                (temporary_root / "data" / name).touch()
            with patch.object(np, "load", return_value=np.zeros((1, 1, 1))) as load_array:
                for pair_index in range(3):
                    for configuration_index in range(4):
                        with self.subTest(pair=pair_index, configuration=configuration_index):
                            namespace = {"ROOT": temporary_root, "json": json, "np": np,
                                         "PAIR_INDEX": pair_index, "CONFIGURATION_INDEX": configuration_index}
                            exec(setup_code, namespace)
                            self.assertEqual(namespace["f_target"], sum(settings["dual_frequency_hz"][pair_index]) / 2)
                            coordinates = settings["source_configurations"][configuration_index]["sources_xz_m"]
                            self.assertEqual(dict(eval(coordinate_code, namespace)),
                                             {f"s{i + 1}": tuple(source) for i, source in enumerate(coordinates)})
                accessed = [Path(call.args[0]).name for call in load_array.call_args_list]
            self.assertEqual(len(accessed), 12)
            self.assertEqual(set(accessed), expected)

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
