#!/usr/bin/env python
"""Evaluate the twelve Figure 5 conditions from saved or newly sampled fields."""

import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.line_array_baselines import (Z0, interpolate, ridge_predict,
                                         heldout_predictions)
from scripts.line_array_localization import localize
from scripts.line_array_metrics import (waveform_metrics, double_tone_projection,
                                       average_seeds, grouped_median_interval,
                                       grouped_paired_intervals)


def load_settings(path=None):
    settings = json.loads(Path(path or ROOT/'configs/figure5_line_array.json').read_text(encoding='utf-8'))
    pairs = settings['dual_frequency_hz']
    positions = [item['position'] for item in settings['source_configurations']]
    if pairs != [[2800, 2900], [2800, 3000], [2900, 3000]] or positions != [3, 4, 7, 8]:
        raise ValueError("Figure 5 requires the three specified frequency pairs and positions 3,4,7,8")
    return settings


def conditions(settings):
    for frequencies in settings['dual_frequency_hz']:
        pair = '_'.join(map(str, frequencies))
        for configuration in settings['source_configurations']:
            yield dict(pair=pair, position=configuration['position'],
                       case=f"{pair}_pos{configuration['position']}",
                       frequencies=frequencies,
                       truth=np.asarray(configuration['sources_xz_m'], float))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def load_input(root, settings, condition, task):
    path = root/settings['files'][task+'_input'].format(**condition)
    with np.load(path, allow_pickle=False) as saved:
        field = saved['physical7'].copy()
        frequencies = saved['frequencies'].astype(int).tolist()
    if field.shape != (7, 4, 1002) or not np.isfinite(field).all() or frequencies != condition['frequencies']:
        raise ValueError(f"invalid Figure 5 input: {path}")
    return field


@lru_cache(maxsize=1)
def inference_model(config_path, checkpoint_path, device):
    import torch
    from scripts.evaluate_fpe_sampling import load_config, load_model, make_net_fn

    cfg = load_config(config_path, device)
    model, state = load_model(cfg, checkpoint_path, 'ema', torch.device(device))
    if state.get('fine_tune_iteration') != 5000:
        raise ValueError("Figure 5 requires the released 5000-step FPE checkpoint")
    return cfg, make_net_fn(model, cfg, torch.device(device))


def network_observation(field, keep, target_rms):
    """Scale the selected input sensors to pressure-equivalent RMS units."""
    observed = field[keep].copy()
    observed[:, 1:] *= Z0
    rms = float(np.sqrt(np.mean(observed**2)))
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError("observed field must have positive finite RMS")
    scale = target_rms/rms
    observation = (observed*scale).transpose(2, 1, 0)[None].astype(np.float32)
    return observation, scale, rms


def sample_prediction(root, settings, field, frequencies, keep, seed, device, output_dir):
    import torch
    from scripts.evaluate_fpe_sampling import PowerWatchdog
    from sampler import VESDE
    from sampler.line_guidance import sample_line_array

    if not device.startswith('cuda:'):
        raise ValueError("sampling requires an explicit CUDA device")
    torch.cuda.set_device(torch.device(device))
    cfg, net = inference_model(root/settings['model']['configuration'],
                               root/settings['model']['checkpoint'], device)
    observation, scale, rms = network_observation(field, keep, settings['sampling']['target_rms'])
    sde = VESDE(cfg, sigma_min=cfg.beta_min, sigma_max=cfg.beta_max,
                N=settings['sampling']['reverse_steps'])
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir/'power_event.json'
    if marker.exists():
        raise RuntimeError(f"existing GPU monitoring event: {marker}")
    watchdog = PowerWatchdog(torch.device(device).index, 400., .25, marker)
    watchdog.start()
    def checked_net(x, t):
        if marker.exists():
            raise RuntimeError("GPU power monitoring event")
        return net(x, t)
    try:
        prediction, losses = sample_line_array(
            cfg, checked_net, sde, observation,
            np.asarray(settings['observed_generated_indices'])[keep], frequencies,
            settings['sampling'], device, seed)
    finally:
        watchdog.stop()
    inverse_units = np.array([1., 1/Z0, 1/Z0, 1/Z0]).reshape(1, 1, 4, 1)
    physical = (prediction*inverse_units)[0].transpose(2, 1, 0)/scale
    return physical, dict(seed=seed, amplitude_scale=scale, input_rms=rms,
                          input_sensor_indices=keep.tolist(), peak_power_w=watchdog.peak_w,
                          losses=losses)


def prediction(root, output_dir, settings, condition, field, seed, heldout, resample, device):
    task = 'localization' if heldout is None else 'heldout'
    if heldout is not None and heldout not in settings['heldout_indices']:
        raise ValueError("held-out sensor must be one of the five internal positions")
    generated_index = None if heldout is None else settings['observed_generated_indices'][heldout]
    tokens = dict(condition, seed=seed, heldout=heldout)
    if resample:
        keep = np.asarray([i for i in range(7) if i != heldout])
        suffix = f'seed{seed}' if heldout is None else f'h{heldout}_seed{seed}'
        path = output_dir/'predictions'/task/condition['case']/suffix/'prediction.npz'
        if path.exists():
            raise FileExistsError(path)
        values, report = sample_prediction(root, settings, field, condition['frequencies'],
                                           keep, seed, device, path.parent)
        if heldout is None:
            np.savez_compressed(path, physical=values)
        else:
            values = values[generated_index:generated_index+1]
            np.savez_compressed(path, physical=values, sensor_index=heldout,
                                generated_index=generated_index)
        save_json(path.with_name('sampling.json'), report)
    else:
        path = root/settings['files'][task+'_prediction'].format(**tokens)
        with np.load(path, allow_pickle=False) as saved:
            values = saved['physical'].copy()
            if heldout is not None and (int(saved['sensor_index']) != heldout
                                       or int(saved['generated_index']) != generated_index):
                raise ValueError(f"held-out prediction coordinates do not match: {path}")
    expected_shape = (32 if heldout is None else 1, 4, 1002)
    if values.shape != expected_shape or not np.isfinite(values).all():
        raise ValueError(f"invalid prediction: {path}")
    return values


def evaluate_localization(root, output_dir, settings, selected, resample=False, device='cuda:2'):
    coords = np.asarray(settings['sensor_coordinates_m'])
    targets = np.asarray(settings['generated_coordinates_m'])
    rows = []
    fields = ('Direct seven AVSs', 'Linear interpolation', 'Cubic spline interpolation',
              'Helmholtz ridge reconstruction', 'FPE reconstruction')
    for condition in selected:
        observed = load_input(root, settings, condition, 'localization')
        ridge, ridge_details = ridge_predict(observed, coords, targets, condition['frequencies'])
        outputs = [(fields[0], None, observed),
                   (fields[1], None, interpolate(observed, coords, targets)),
                   (fields[2], None, interpolate(observed, coords, targets, cubic=True)),
                   (fields[3], None, ridge)]
        for seed in settings['localization_seeds']:
            outputs.append((fields[4], seed, prediction(root, output_dir, settings, condition,
                            observed, seed, None, resample, device)))
        folder = output_dir/'localization'/condition['case']
        save_json(folder/'ridge_selection.json', ridge_details)
        for name, seed, field in outputs:
            for method in settings['localization_methods']:
                metrics, arrays = localize(field, coords if len(field) == 7 else targets,
                    condition['frequencies'], condition['truth'], method,
                    grid_size=settings['localization']['grid_sizes'][0])
                token = f"{fields.index(name)}_seed{seed or 0}_{method}"
                np.savez_compressed(folder/f'{token}.npz', **arrays)
                rows.append(dict(case=condition['case'], pair=condition['pair'],
                    position=condition['position'], method=method, field=name, seed=seed,
                    geometry='plane', input_sensor_budget=7, output_positions=len(field),
                    **metrics))
        save_csv(output_dir/'localization_runs.csv', rows)
    units = average_seeds(rows, ['pair', 'position', 'method', 'field'],
                          ['localization_rmse_cm', 'extra_to_weak_db'])
    save_csv(output_dir/'boxplot_localization_by_condition.csv', units)
    if len(selected) == 12:
        summaries, paired = [], []
        for method in settings['localization_methods']:
            for metric in ('localization_rmse_cm', 'extra_to_weak_db'):
                reference = [r for r in units if r['method'] == method and r['field'] == fields[-1]]
                reference.sort(key=lambda row: (row['pair'], row['position']))
                reference_values = [r[metric] for r in reference]
                groups = [r['position'] for r in reference]
                for field_name in fields:
                    values = [r for r in units if r['method'] == method and r['field'] == field_name]
                    values.sort(key=lambda row: (row['pair'], row['position']))
                    if any(r[metric] is None for r in values):
                        continue
                    summaries.append(dict(method=method, field=field_name, metric=metric,
                        **grouped_median_interval([r[metric] for r in values], groups)))
                    if field_name != fields[-1] and all(value is not None for value in reference_values):
                        paired.append(dict(method=method, comparison=f'PI-ISS minus {field_name}', metric=metric,
                            **grouped_paired_intervals(reference_values, [r[metric] for r in values], groups)))
        save_json(output_dir/'localization_configuration_intervals.json', summaries)
        save_json(output_dir/'localization_paired_intervals.json', paired)
    return rows, units


def evaluate_heldout(root, output_dir, settings, selected, resample=False, device='cuda:2'):
    coords = np.asarray(settings['sensor_coordinates_m'])
    rows = []
    for condition in selected:
        field = load_input(root, settings, condition, 'heldout')
        for heldout in settings['heldout_indices']:
            keep = np.asarray([i for i in range(7) if i != heldout])
            observed, reference = field[keep], field[heldout:heldout+1]
            baseline, details = heldout_predictions(observed, coords[keep], coords[heldout:heldout+1],
                                                    condition['frequencies'])
            outputs = [(name, 0, values) for name, values in baseline.items()]
            for seed in settings['heldout_seeds']:
                values = prediction(root, output_dir, settings, condition, field,
                                    seed, heldout, resample, device)
                outputs.append(('PI-ISS', seed, values))
            folder = output_dir/'heldout'/condition['case']/f'h{heldout}'
            folder.mkdir(parents=True, exist_ok=True)
            save_json(folder/'ridge_selection.json', details)
            for name, seed, values in outputs:
                token = name.replace(' ', '_')+f'_seed{seed}'
                np.savez_compressed(folder/f'{token}.npz', prediction_raw=values,
                    reference_raw=reference,
                    prediction_projected=double_tone_projection(values, condition['frequencies']),
                    reference_projected=double_tone_projection(reference, condition['frequencies']),
                    observed=observed, sensor_indices=keep, frequencies=condition['frequencies'])
                rows.append(dict(pair=condition['pair'], position=condition['position'],
                    heldout_index=heldout, sensor_id=settings['sensor_ids'][heldout],
                    method=name, seed=seed, calibration='per_case_alignment',
                    **waveform_metrics(values, reference, condition['frequencies'])))
            save_csv(output_dir/'heldout_per_run_metrics.csv', rows)
    metric_names = [f'double_tone_projection_{c}_nrmse' for c in settings['channels']]
    keys = ['pair', 'position', 'heldout_index', 'sensor_id', 'method']
    units = average_seeds(rows, keys, metric_names)
    save_csv(output_dir/'boxplot_heldout_by_sensor.csv', units)
    channels = [dict(**{key: row[key] for key in keys}, channel=channel,
                    channel_nrmse=row[f'double_tone_projection_{channel}_nrmse'])
                for row in units for channel in settings['channels']]
    save_csv(output_dir/'boxplot_heldout_per_channel.csv', channels)
    if len(selected) == 12:
        summaries, paired = [], []
        for channel in settings['channels']:
            reference = [r for r in channels if r['method'] == 'PI-ISS' and r['channel'] == channel]
            reference.sort(key=lambda row: (row['pair'], row['position'], row['heldout_index']))
            groups = [r['position'] for r in reference]
            for method in ('Linear interpolation', 'Cubic spline', 'Plane-wave ridge',
                           'Equivalent sources', 'PI-ISS'):
                values = [r for r in channels if r['method'] == method and r['channel'] == channel]
                values.sort(key=lambda row: (row['pair'], row['position'], row['heldout_index']))
                summaries.append(dict(channel=channel, method=method,
                    **grouped_median_interval([r['channel_nrmse'] for r in values], groups)))
                if method != 'PI-ISS':
                    paired.append(dict(channel=channel, comparison=f'PI-ISS minus {method}',
                        **grouped_paired_intervals([r['channel_nrmse'] for r in reference],
                                                 [r['channel_nrmse'] for r in values], groups)))
        save_json(output_dir/'heldout_configuration_intervals.json', summaries)
        save_json(output_dir/'heldout_paired_intervals.json', paired)
    return rows, units


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--task', choices=['localization', 'heldout', 'all'], default='all')
    parser.add_argument('--case', choices=[c['case'] for c in conditions(load_settings())])
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/figure5_evaluation')
    parser.add_argument('--resample', action='store_true', help='sample fresh fields from the released FPE checkpoint')
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--check-settings', action='store_true')
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    selected = [c for c in conditions(settings) if not args.case or c['case'] == args.case]
    if args.check_settings:
        print(json.dumps(dict(conditions=len(selected), sampling=settings['sampling'], model=settings['model'])))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir/'settings.json', settings)
    if args.task in ('localization', 'all'):
        rows, units = evaluate_localization(args.root, args.output_dir, settings, selected, args.resample, args.device)
        if not args.case:
            assert len(rows) == 336 and len(units) == 240
    if args.task in ('heldout', 'all'):
        rows, units = evaluate_heldout(args.root, args.output_dir, settings, selected, args.resample, args.device)
        if not args.case:
            assert len(rows) == 420 and len(units) == 300


if __name__ == '__main__':
    main()
