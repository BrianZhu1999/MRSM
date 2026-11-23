"""Spatial interpolation and input-selected ridge reconstruction of AVS signals.

Fields use sensor, channel, time order and physical [Pa, m/s, m/s, m/s] units.
"""

from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicSpline, interp1d


Z0 = 1.21 * 343
FS = 100000.
LAMBDAS = np.logspace(-7, 0, 8)


def tone_coefficients(field, frequencies, fs=FS):
    """Fit two tones, an offset and a linear trend in pressure-equivalent units."""
    field = np.asarray(field, dtype=float)
    if field.ndim != 3 or field.shape[1] != 4 or not np.isfinite(field).all():
        raise ValueError("field must be a finite sensor,4,time array")
    x = field.copy()
    x[:, 1:] *= Z0
    t = np.arange(x.shape[-1]) / fs
    design = np.column_stack([v for f in frequencies for v in
        (np.cos(2*np.pi*f*t), np.sin(2*np.pi*f*t))] + [np.ones(len(t)), t-t.mean()])
    coef = np.linalg.lstsq(design, x.reshape(-1, len(t)).T, rcond=None)[0]
    return np.stack([coef[2*j]-1j*coef[2*j+1] for j in range(len(frequencies))])


def synthesize(coefficients, samples, frequencies, fs=FS):
    """Convert complex pressure-equivalent tone amplitudes to physical waveforms."""
    t = np.arange(samples) / fs
    x = np.real(np.einsum('fc,ft->ct', coefficients,
        np.exp(2j*np.pi*np.asarray(frequencies)[:, None]*t)))
    physical = x.reshape(-1, 4, samples)
    physical[:, 1:] /= Z0
    return physical


def interpolate(observation, coords, target_coords, cubic=False):
    order = np.argsort(coords[:, 0])
    x = coords[order, 0]
    values = observation[order]
    if cubic:
        return CubicSpline(x, values, axis=0, bc_type='not-a-knot',
                           extrapolate=True)(target_coords[:, 0])
    return interp1d(x, values, axis=0, fill_value='extrapolate')(target_coords[:, 0])


@lru_cache(maxsize=1)
def plane_directions():
    i = np.arange(1024)
    z = 1-2*(i+.5)/len(i)
    phase = i*np.pi*(3-np.sqrt(5))
    return np.column_stack((np.sqrt(1-z*z)*np.cos(phase),
                            np.sqrt(1-z*z)*np.sin(phase), z))


def plane_dictionary(coords, frequency, normalization=1.):
    d = plane_directions()
    propagation = np.exp(2j*np.pi*frequency/343*(coords@d.T))
    return (propagation[:, None, :]*np.vstack((np.ones(len(d)), -d.T))[None]).reshape(4*len(coords), -1)/normalization


def source_dictionary(points, coords, frequency, nearfield=True):
    """Equivalent monopoles on y=1 m, with the original velocity response."""
    points = np.atleast_2d(points)
    xyz = np.column_stack((points[:, 0], np.ones(len(points)), points[:, 1]))
    delta = coords[None]-xyz[:, None]
    radius = np.linalg.norm(delta, axis=-1)
    k = 2*np.pi*frequency/343
    block = np.concatenate((np.ones((*radius.shape, 1)), delta/radius[..., None]), axis=-1).astype(complex)
    if nearfield:
        block[:, :, 1:] *= (1-1j/(k*radius))[..., None]
    return ((np.exp(-1j*k*radius)/radius)[..., None]*block).reshape(len(points), -1).T


def ridge_predict(observation, coords, target_coords, frequencies,
                  equivalent_sources=False, normalize_columns=False):
    """Select regularization by leaving out each input sensor's four channels."""
    coefficients = tone_coefficients(observation, frequencies)
    predictions, details = [], []
    for frequency, original_b in zip(frequencies, coefficients):
        if equivalent_sources:
            sx, sz = np.meshgrid(np.arange(-1.5, 1.5001, .05), np.arange(.01, 1.8001, .05))
            sources = np.column_stack((sx.ravel(), sz.ravel()))
            a = source_dictionary(sources, coords, frequency)
            target = source_dictionary(sources, target_coords, frequency)
        else:
            normalization = 1. if normalize_columns else np.sqrt(14.)
            a = plane_dictionary(coords, frequency, normalization)
            target = plane_dictionary(target_coords, frequency, normalization)
        if normalize_columns:
            scale = np.linalg.norm(a, axis=0)
            a, target = a/scale, target/scale
        scale = np.linalg.norm(original_b)
        if scale == 0:
            raise ValueError("input tone has zero energy")
        b = original_b/scale
        scores = np.zeros(len(LAMBDAS))
        for held in range(len(coords)):
            test = np.arange(4*held, 4*held+4)
            train = np.setdiff1d(np.arange(4*len(coords)), test)
            training, testing = a[train], a[test]
            gram = training@training.conj().T
            eigen, vectors = np.linalg.eigh(gram)
            coordinates = vectors.conj().T@b[train]
            cross = testing@training.conj().T
            for j, fraction in enumerate(LAMBDAS):
                solution = vectors@(coordinates/(np.maximum(eigen, 0)+fraction*eigen[-1]))
                scores[j] += np.linalg.norm(cross@solution-b[test])**2
        best = int(np.argmin(scores))
        gram = a@a.conj().T
        largest = float(np.linalg.eigvalsh(gram)[-1])
        weights = a.conj().T@np.linalg.solve(gram+LAMBDAS[best]*largest*np.eye(len(b)), b)
        predictions.append(scale*(target@weights))
        details.append(dict(frequency_hz=float(frequency), lambda_fraction=float(LAMBDAS[best]),
                            input_only_cv_squared_relative_error=scores.tolist()))
    return synthesize(np.asarray(predictions), observation.shape[-1], frequencies), details


def heldout_predictions(observed, coords, target_coords, frequencies):
    outputs = {'Linear interpolation': interpolate(observed, coords, target_coords),
               'Cubic spline': interpolate(observed, coords, target_coords, cubic=True)}
    details = {}
    for name, equivalent in [('Plane-wave ridge', False), ('Equivalent sources', True)]:
        outputs[name], details[name] = ridge_predict(
            observed, coords, target_coords, frequencies,
            equivalent_sources=equivalent, normalize_columns=True)
    return outputs, details
