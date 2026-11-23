"""Metrics shared by the measured line-array examples."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import label
from skimage.segmentation import watershed
from itertools import product

from scripts.line_array_baselines import Z0


def match_source_positions(true_positions: np.ndarray,
                           estimated_positions: np.ndarray):
    """Match source positions by the minimum sum of squared distances."""
    true_positions = np.asarray(true_positions, dtype=float)
    estimated_positions = np.asarray(estimated_positions, dtype=float)
    if (true_positions.ndim != 2 or true_positions.shape != estimated_positions.shape
            or len(true_positions) == 0 or not np.isfinite(true_positions).all()
            or not np.isfinite(estimated_positions).all()):
        raise ValueError("source coordinates must be finite arrays of the same nonempty shape")
    distances = np.linalg.norm(
        true_positions[:, np.newaxis, :] - estimated_positions[np.newaxis, :, :],
        axis=2,
    )
    true_indices, estimate_indices = linear_sum_assignment(distances ** 2)
    return true_indices, estimate_indices, distances[true_indices, estimate_indices]


def matched_source_errors(true_positions: np.ndarray,
                          estimated_positions: np.ndarray) -> np.ndarray:
    """Return matched source distances."""
    return match_source_positions(true_positions, estimated_positions)[2]


def double_tone_projection(field, frequencies, fs=100000.):
    """Fit the two sine/cosine pairs, without the offset and trend used in inference."""
    time = np.arange(field.shape[-1])/fs
    basis = np.column_stack([f(2*np.pi*frequency*time) for frequency in frequencies
                             for f in (np.cos, np.sin)])
    samples = field.reshape(-1, len(time)).T
    fitted = basis@np.linalg.lstsq(basis, samples, rcond=None)[0]
    return fitted.T.reshape(field.shape)


def waveform_metrics(prediction, target, frequencies):
    """Report full-waveform and two-tone errors, before and after projection."""
    prediction, target = np.asarray(prediction), np.asarray(target)
    if (prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1] != 4
            or not np.isfinite(prediction).all() or not np.isfinite(target).all()):
        raise ValueError("expected matching sensor,4,time arrays")
    scale = np.array([1., Z0, Z0, Z0]).reshape(1, 4, 1)
    p, t = prediction*scale, target*scale
    pp, tp = double_tone_projection(p, frequencies), double_tone_projection(t, frequencies)
    result = {}
    for prefix, a, b in [('full', p, t), ('double_tone_projection', pp, tp)]:
        error, energy = float(np.sum((a-b)**2)), float(np.sum(b**2))
        if energy <= 0:
            raise ValueError("reference has zero energy")
        result.update({prefix+'_squared_error': error, prefix+'_squared_target': energy,
                       prefix+'_nrmse': float(np.sqrt(error/energy))})
        for j, channel in enumerate(('p', 'vx', 'vy', 'vz')):
            energy = float(np.sum(b[:, j]**2))
            if energy <= 0:
                raise ValueError(f"reference channel {channel} has zero energy")
            error = float(np.sum((a[:, j]-b[:, j])**2))
            result.update({prefix+'_'+channel+'_nrmse': float(np.sqrt(error/energy)),
                           prefix+'_'+channel+'_squared_error': error,
                           prefix+'_'+channel+'_squared_target': energy})
    return result


def average_seeds(rows, keys, metrics):
    """Average already-computed per-run metrics within each condition."""
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    return [dict(zip(keys, group), **{
        metric: float(np.mean([row[metric] for row in values]))
        if all(row[metric] is not None for row in values) else None
        for metric in metrics}) for group, values in groups.items()]


def grouped_median_interval(values, configurations):
    """Enumerate the 4^4 configuration resamples, retaining frequencies together."""
    values, configurations = np.asarray(values, float), np.asarray(configurations)
    groups = np.unique(configurations)
    if len(groups) != 4 or not np.isfinite(values).all():
        raise ValueError("four configurations with finite values are required")
    samples = [np.median(np.concatenate([values[configurations == g] for g in draw]))
               for draw in product(groups, repeat=4)]
    return dict(median=float(np.median(values)),
                ci95=np.percentile(samples, [2.5, 97.5]).tolist(),
                resamples=len(samples),
                leave_one_configuration_out=[float(np.median(values[configurations != g]))
                                             for g in groups])


def grouped_paired_intervals(first, second, configurations):
    """Use identical configuration draws for both paired median definitions."""
    first, second = np.asarray(first, float), np.asarray(second, float)
    configurations = np.asarray(configurations)
    if first.shape != second.shape or first.shape != configurations.shape:
        raise ValueError("paired values must have matching shapes")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("paired values must be finite")
    groups = np.unique(configurations)
    if len(groups) != 4:
        raise ValueError("four configurations are required")
    draws = [np.concatenate([np.flatnonzero(configurations == g) for g in draw])
             for draw in product(groups, repeat=4)]
    differences = [np.median(first[ids])-np.median(second[ids]) for ids in draws]
    paired = [np.median(first[ids]-second[ids]) for ids in draws]
    return dict(difference_of_medians=float(np.median(first)-np.median(second)),
                difference_of_medians_ci95=np.percentile(differences, [2.5, 97.5]).tolist(),
                median_of_paired_differences=float(np.median(first-second)),
                median_of_paired_differences_ci95=np.percentile(paired, [2.5, 97.5]).tolist(),
                resamples=len(draws))


def grid_index(point, xx, zz):
    return int(np.argmin(abs(zz[:, 0] - point[1]))), int(np.argmin(abs(xx[0] - point[0])))

def lobe_measurements(db, peak_xy, peak_db, truth, xx, zz):
    """Map each source to its watershed lobe; measure all remaining local peaks."""
    markers = np.zeros(db.shape, np.int32)
    for k, point in enumerate(peak_xy):
        ij = grid_index(point, xx, zz)
        if markers[ij]:
            raise ValueError('Two refined peaks share a grid cell; increase scan resolution')
        markers[ij] = k + 1
    basins = watershed(-db, markers=markers, connectivity=2)
    targets = [int(basins[grid_index(point, xx, zz)]) - 1 for point in truth]
    assert min(targets) >= 0
    resolved = len(set(targets)) == len(truth)
    extra = [i for i in range(len(peak_xy)) if i not in targets]
    strongest = max(extra, key=lambda i: peak_db[i]) if extra else None
    target_values = np.asarray([peak_db[i] for i in targets])
    lobe_rows = []
    half_labels = np.zeros(db.shape, np.int8)
    raw_halves = []
    dx, dz = xx[0, 1] - xx[0, 0], zz[1, 0] - zz[0, 0]
    for source, k in enumerate(targets):
        # Continuous peak height defines the half-power level; its connected lobe is measured.
        threshold = peak_db[k] - 10 * np.log10(2)
        raw_components, _ = label(db >= threshold, structure=np.ones((3, 3)))
        ij = grid_index(peak_xy[k], xx, zz)
        raw_id = raw_components[ij]
        if raw_id == 0:
            raise ValueError('Peak is narrower than the grid resolution')
        raw_half = raw_components == raw_id
        components, _ = label(raw_half & (basins == k + 1), structure=np.ones((3, 3)))
        half = components == components[ij]
        assert components[ij] != 0
        half_labels[half] = source + 1
        raw_halves.append(raw_half)
        # Trapezoidal weights keep boundary pixels within the finite scan domain.
        weights = np.ones(db.shape)
        weights[[0, -1], :] *= .5
        weights[:, [0, -1]] *= .5
        lobe_rows.append({'source': source + 1, 'peak_index': k, 'peak_xz_m': peak_xy[k].tolist(),
            'target_association_error_cm': float(np.linalg.norm(peak_xy[k] - truth[source]) * 100),
            'half_power_area_m2': float(np.sum(weights[half]) * dx * dz),
            'half_power_x_span_m': float(np.ptp(xx[half])),
            'half_power_z_span_m': float(np.ptp(zz[half])),
            'half_power_touches_scan_boundary': bool(half[0].any() or half[-1].any() or half[:, 0].any() or half[:, -1].any())})
    cost = np.linalg.norm(truth[:, None] - peak_xy[None], axis=-1)
    ri, ci = linear_sum_assignment(cost ** 2)
    nearest_set = set(int(k) for k in ci)
    nearest_extra = [k for k in range(len(peak_xy)) if k not in nearest_set]
    nearest_strongest = max(nearest_extra, key=lambda i: peak_db[i]) if nearest_extra else None
    widths = [r['half_power_area_m2'] for r in lobe_rows]
    result = {'two_target_lobes_resolved': resolved, 'target_peak_indices': targets,
        'target_lobes': lobe_rows, 'half_power_regions_overlap': bool((raw_halves[0] & raw_halves[1]).any()),
        'mean_half_power_area_m2': float(np.mean(widths)),
        'extra_peak_count': len(extra), 'strongest_extra_peak_index': strongest,
        'strongest_extra_peak_xz_m': peak_xy[strongest].tolist() if strongest is not None else None,
        'extra_peak_to_strong_target_db': float(peak_db[strongest] - max(target_values)) if strongest is not None else None,
        'extra_peak_to_weak_target_db': float(peak_db[strongest] - min(target_values)) if strongest is not None else None,
        'nearest_peak_target_indices': sorted(nearest_set),
        'association_agrees_with_nearest_peaks': resolved and set(targets) == nearest_set,
        'nearest_association_extra_to_strong_db': float(peak_db[nearest_strongest] - max(peak_db[list(nearest_set)])) if nearest_strongest is not None else None,
        'nearest_association_extra_to_weak_db': float(peak_db[nearest_strongest] - min(peak_db[list(nearest_set)])) if nearest_strongest is not None else None}
    return result, basins, half_labels
