import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from sampler.line_guidance import LineGuidance, join_windows
from scripts.evaluate_figure5 import load_settings, conditions, network_observation, prediction
from scripts.line_array_baselines import interpolate, tone_coefficients, synthesize, source_dictionary
from scripts.line_array_localization import response, peak_starts, refine_plane
from scripts.line_array_metrics import (matched_source_errors, lobe_measurements,
    waveform_metrics, double_tone_projection, average_seeds, grouped_median_interval,
    grouped_paired_intervals)


class Figure5PublicSettingsTests(unittest.TestCase):
    def test_twelve_conditions_and_real_sampling_settings(self):
        settings = load_settings()
        self.assertEqual(len(list(conditions(settings))), 12)
        self.assertEqual({c['configuration'] for c in conditions(settings)}, {'A', 'B', 'C', 'D'})
        self.assertEqual({c['case'] for c in conditions(settings)},
                         {f'{pair}_{label}' for pair in ('2800_2900', '2800_3000', '2900_3000')
                          for label in ('A', 'B', 'C', 'D')})
        self.assertEqual(settings['localization_seeds'], [3, 4, 5])
        self.assertEqual(settings['heldout_seeds'], [1, 2, 3])
        self.assertEqual(settings['heldout_indices'], [1, 2, 3, 4, 5])
        self.assertEqual(settings['sampling']['reverse_steps'], 500)
        self.assertEqual(settings['sampling']['wave_weight'], 0)
        self.assertEqual(settings['sampling']['continuity_weight'], 0)
        self.assertEqual(settings['sampling']['euler_weight'], 0)
        self.assertEqual(settings['sampling']['variation_weight'], 1)
        self.assertEqual(settings['sampling']['tone_weight'], .01)
        self.assertEqual(settings['sampling']['tone_start_step'], 250)
        self.assertEqual(settings['model']['fine_tune_step'], 5000)
        observed = np.array(settings['sensor_coordinates_m'])
        generated = np.array(settings['generated_coordinates_m'])
        np.testing.assert_allclose(generated[settings['observed_generated_indices']], observed, atol=1e-14)
        self.assertAlmostEqual(generated[-1, 0], -.512)
        self.assertEqual(len(settings['localization_methods']), 4)
        self.assertEqual(len(settings['localization_arrays']), 5)
        self.assertEqual(len(settings['held_out_field_methods']), 5)

    def test_notebook_uses_configuration_a_and_shared_entry(self):
        root = Path(__file__).parents[1]
        notebook = json.loads((root/'notebooks/line_array/reconstruction_music.ipynb').read_text(encoding='utf-8'))
        self.assertTrue(all(not c.get('outputs') for c in notebook['cells']))
        settings = load_settings()
        self.assertEqual(settings['source_configurations'][0]['name'], 'A')
        self.assertEqual(settings['dual_frequency_hz'][1], [2800, 3000])

    def test_source_matching_minimizes_squared_distance(self):
        truth = np.array([[.1, 0], [1.2, 0]])
        errors = matched_source_errors(truth, np.array([[1., 0], [0., 0]]))
        np.testing.assert_allclose(errors, [.1, .2])
        self.assertAlmostEqual(float(np.sqrt(np.mean(errors**2))), np.sqrt(.025))
        for invalid in (np.array([[0., 0.]]), np.array([[np.nan, 0], [1, 0]])):
            with self.assertRaises(ValueError):
                matched_source_errors(truth, invalid)

    def test_zero_prediction_has_unit_nrmse(self):
        time = np.arange(1002)/100000
        wave = np.sin(2*np.pi*2800*time)+.4*np.cos(2*np.pi*3000*time)
        target = np.tile(wave, (1, 4, 1))
        result = waveform_metrics(np.zeros_like(target), target, [2800, 3000])
        self.assertEqual(result['full_nrmse'], 1)
        self.assertEqual(result['double_tone_projection_nrmse'], 1)
        for channel in ('p', 'vx', 'vy', 'vz'):
            self.assertEqual(result[f'double_tone_projection_{channel}_nrmse'], 1)

    def test_tone_fit_removes_offset_and_trend(self):
        rng = np.random.default_rng(71)
        coefficients = rng.normal(size=(2, 28))+1j*rng.normal(size=(2, 28))
        field = synthesize(coefficients, 1002, [2800, 3000])
        trend = np.arange(1002)/100000
        contaminated = field+2+3*trend
        np.testing.assert_allclose(tone_coefficients(contaminated, [2800, 3000]), coefficients, atol=2e-10)
        np.testing.assert_allclose(double_tone_projection(field, [2800, 3000]), field, atol=1e-12)

    def test_interpolation_has_correct_descending_coordinates(self):
        x = np.array([.48, .32, 0., -.16, -.32, -.48])
        coords = np.column_stack((x, np.zeros((len(x), 2))))
        target = np.array([[.16, 0., 0.]])
        observed = np.tile((2*x+3)[:, None, None], (1, 4, 20))
        for cubic in (False, True):
            np.testing.assert_allclose(interpolate(observed, coords, target, cubic=cubic), 3.32, atol=1e-12)

    def test_heldout_values_do_not_enter_observation_or_scale(self):
        field = np.random.default_rng(32).normal(size=(7, 4, 1002))
        keep = np.array([0, 1, 2, 4, 5, 6])
        observation, scale, rms = network_observation(field, keep, .3)
        field[3] *= 1e6
        changed, changed_scale, changed_rms = network_observation(field, keep, .3)
        np.testing.assert_array_equal(observation, changed)
        self.assertEqual(scale, changed_scale)
        self.assertEqual(rms, changed_rms)

    def test_heldout_prediction_contains_the_requested_single_position(self):
        settings = load_settings()
        condition = next(conditions(settings))
        field = np.zeros((7, 4, 1002))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/settings['files']['heldout_prediction'].format(
                case=condition['case'], seed=1, heldout=3)
            path.parent.mkdir(parents=True)
            target = np.full((1, 4, 1002), .25)
            np.savez_compressed(path, physical=target, sensor_index=3, generated_index=15)
            result = prediction(root, root/'output', settings, condition, field, 1, 3, False, 'cpu')
            np.testing.assert_array_equal(result, target)
            np.savez_compressed(path, physical=target, sensor_index=2, generated_index=10)
            with self.assertRaises(ValueError):
                prediction(root, root/'output', settings, condition, field, 1, 3, False, 'cpu')

    def test_fresh_heldout_prediction_saves_only_its_target_position(self):
        settings = load_settings()
        condition = next(conditions(settings))
        field = np.zeros((7, 4, 1002))
        full = np.broadcast_to(np.arange(32)[:, None, None], (32, 4, 1002)).copy()
        def sampled(*args):
            args[-1].mkdir(parents=True)
            return full, {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('scripts.evaluate_figure5.sample_prediction', side_effect=sampled):
                result = prediction(root, root/'output', settings, condition, field, 1, 3, True, 'cpu')
            np.testing.assert_array_equal(result, full[15:16])
            path = root/'output/predictions/heldout'/condition['case']/'h3_seed1/prediction.npz'
            with np.load(path) as saved:
                self.assertEqual(saved['physical'].shape, (1, 4, 1002))
                self.assertEqual(int(saved['sensor_index']), 3)
                self.assertEqual(int(saved['generated_index']), 15)

    def test_response_derivatives_and_equivalent_source_dictionary(self):
        coords = np.array(load_settings()['sensor_coordinates_m'])
        point = np.array([.71, .93])
        for nearfield in (False, True):
            value, derivative = response(point, coords, 2900, nearfield)
            numerical = np.column_stack([(response(point+direction, coords, 2900, nearfield)[0]
                -response(point-direction, coords, 2900, nearfield)[0])/2e-6
                for direction in np.eye(2)*1e-6])
            np.testing.assert_allclose(derivative, numerical, atol=2e-7)
            dictionary = source_dictionary(point[None], coords, 2900, nearfield)[:, 0]
            np.testing.assert_allclose(value, dictionary/np.linalg.norm(dictionary), atol=1e-12)

    def test_complete_plane_sidelobe_uses_weaker_target(self):
        xx, zz = np.meshgrid(np.linspace(-1, 1, 201), np.linspace(-1, 1, 201))
        peaks = np.array([[-.6, -.4], [.5, .3], [0, .9]])
        peak_db = np.array([0., -3., -9.])
        db = np.max(np.stack([v-100*((xx-p[0])**2+(zz-p[1])**2)
                              for p, v in zip(peaks, peak_db)]), axis=0)
        result, _, _ = lobe_measurements(db, peaks, peak_db, peaks[:2], xx, zz)
        self.assertEqual(result['extra_peak_to_weak_target_db'], -6)
        self.assertEqual(result['strongest_extra_peak_index'], 2)

    def test_continuous_refinement_finds_tilted_offgrid_peak(self):
        xx, zz = np.meshgrid(np.linspace(-1.5, 1.5, 101), np.linspace(.01, 1.8, 101))
        centre = np.array([.2537, .9143])
        rotation = np.array([[.8, -.6], [.6, .8]])
        hessian = rotation@np.diag([1., 1000.])@rotation.T
        def objective(point):
            delta = point-centre
            return float(delta@hessian@delta), 2*hessian@delta
        grid = np.column_stack((xx.ravel(), zz.ravel()))
        q = np.array([objective(point)[0] for point in grid])
        peaks, _ = refine_plane(objective, q, xx, zz)
        np.testing.assert_allclose(peaks[0], centre, atol=2e-6)

    def test_guidance_is_zero_on_known_tones_and_has_gradient_off_tones(self):
        samples = 1002
        time = torch.arange(samples, dtype=torch.float64)/100000
        wave = torch.cos(2*torch.pi*2800*time)+.4*torch.sin(2*torch.pi*3000*time)
        field = wave[None, :, None, None].expand(1, samples, 4, 32).clone().requires_grad_(True)
        guidance = LineGuidance(samples, [2800, 3000], dtype=torch.float64, tone_weight=.01)
        self.assertLess(guidance(field).item(), 1e-20)
        perturbed = field+torch.cos(2*torch.pi*1000*time)[None, :, None, None]
        loss = guidance(perturbed)
        self.assertGreater(torch.linalg.vector_norm(torch.autograd.grad(loss, field)[0]).item(), 0)

    def test_window_join_values_and_gradients(self):
        x = torch.arange(2*4*10*3*5, dtype=torch.float64).reshape(2, 4, 10, 3, 5).requires_grad_(True)
        reference = torch.zeros((2, 34, 3, 5), dtype=x.dtype)
        for j in reversed(range(4)):
            reference[:, j*8:j*8+10] = x[:, j]
        joined = join_windows(x, 2)
        self.assertTrue(torch.equal(reference, joined))
        weights = torch.arange(reference.numel(), dtype=x.dtype).reshape(reference.shape)
        a = torch.autograd.grad((reference*weights).sum(), x, retain_graph=True)[0]
        b = torch.autograd.grad((joined*weights).sum(), x)[0]
        self.assertTrue(torch.equal(a, b))

    def test_seed_metrics_average_before_configuration_resampling(self):
        rows = [dict(pair='2800_3000', configuration='A', seed=seed, value=value)
                for seed, value in [(3, 1.), (4, 2.), (5, 6.)]]
        units = average_seeds(rows, ['pair', 'configuration'], ['value'])
        self.assertEqual(units[0]['value'], 3.)
        configurations = np.repeat(['A', 'B', 'C', 'D'], 3)
        values = np.arange(12, dtype=float)
        summary = grouped_median_interval(values, configurations)
        self.assertEqual(summary['resamples'], 256)
        paired = grouped_paired_intervals(values, values, configurations)
        self.assertEqual(paired['difference_of_medians_ci95'], [0., 0.])
        self.assertEqual(paired['median_of_paired_differences_ci95'], [0., 0.])


if __name__ == '__main__':
    unittest.main()
