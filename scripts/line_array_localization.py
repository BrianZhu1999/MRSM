"""The four Figure 5 spatial spectra and continuous local-peak refinement."""

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize

from scripts.line_array_baselines import tone_coefficients, source_dictionary, FS
from scripts.line_array_metrics import matched_source_errors, lobe_measurements

BOUNDS = [(-1.5, 1.5), (.01, 1.8)]

def response(point, coords, frequency=2950., nearfield=False, pressure_only=False):
    """Unit-norm steering vector and its analytic x/z derivatives."""
    d = coords - np.array([point[0], 1., point[1]])
    r = np.linalg.norm(d, axis=1)
    u = d / r[:, None]
    k = 2 * np.pi * frequency / 343.
    g = np.exp(-1j * k * r) / r
    h = 1 - 1j / (k * r) if nearfield else np.ones(len(r))
    block = np.column_stack([np.ones(len(r)), u * h[:, None]])
    db = []
    for axis in (0, 2):
        du = (u * u[:, axis, None] - np.eye(3)[axis]) / r[:, None]
        dh = -1j * u[:, axis] / (k * r**2) if nearfield else np.zeros(len(r))
        db.append(np.column_stack([np.zeros(len(r)), du * h[:, None] + u * dh[:, None]]))
    if pressure_only:
        block, db = block[:, :1], [b[:, :1] for b in db]
    v = (g[:, None] * block).ravel()
    norm = np.linalg.norm(v)
    a = v / norm
    jac = []
    for axis, b in zip((0, 2), db):
        dg = (1j * k + 1 / r) * u[:, axis]
        dv = (g[:, None] * (dg[:, None] * block + b)).ravel()
        jac.append((dv - a * np.vdot(a, dv).real) / norm)
    return a, np.column_stack(jac)

def peak_starts(q, xx, zz):
    """Include minima on tilted narrow ridges and on every scan edge."""
    plane = q.reshape(xx.shape)
    ids = np.flatnonzero(-plane == maximum_filter(-plane, 3, mode='constant', cval=-np.inf))
    dx, dz = xx[0, 1]-xx[0, 0], zz[1, 0]-zz[0, 0]
    gz, gx = np.gradient(plane, dz, dx)
    angle = np.arctan2(gz, gx)
    corners = [angle[:-1, :-1], angle[:-1, 1:], angle[1:, 1:], angle[1:, :-1]]
    winding = sum((corners[(j+1)%4]-corners[j]+np.pi)%(2*np.pi)-np.pi for j in range(4))
    lap = np.gradient(gx, dx, axis=1) + np.gradient(gz, dz, axis=0)
    positive_curvature = (lap[:-1,:-1]+lap[:-1,1:]+lap[1:,1:]+lap[1:,:-1]) > 0
    iz, ix = np.where((abs(winding)>np.pi) & positive_curvature)
    starts = [np.column_stack((xx.ravel(), zz.ravel()))[ids], np.column_stack((xx[iz, ix]+dx/2, zz[iz, ix]+dz/2))]
    for edge_points, edge_q in [(np.column_stack((xx[0],zz[0])),plane[0]),
                                (np.column_stack((xx[-1],zz[-1])),plane[-1]),
                                (np.column_stack((xx[:,0],zz[:,0])),plane[:,0]),
                                (np.column_stack((xx[:,-1],zz[:,-1])),plane[:,-1])]:
        edge_ids = np.flatnonzero(-edge_q == maximum_filter(-edge_q,3,mode='constant',cval=-np.inf))
        starts.append(edge_points[edge_ids])
    return np.unique(np.vstack(starts), axis=0)

def refine_plane(fun, q, xx, zz):
    candidates, failures = [], []
    for start in peak_starts(q, xx, zz):
        opt = minimize(fun, start, jac=True, method='L-BFGS-B', bounds=BOUNDS,
                       options={'ftol':1e-15,'gtol':1e-9,'maxiter':250,'maxls':40})
        value, grad = fun(opt.x)
        projected = opt.x-np.clip(opt.x-grad, [-1.5,.01], [1.5,1.8])
        if np.linalg.norm(projected)>2e-6:
            opt = minimize(fun,opt.x,jac=True,method='SLSQP',bounds=BOUNDS,
                           options={'ftol':1e-14,'maxiter':250})
            value,grad=fun(opt.x)
            projected=opt.x-np.clip(opt.x-grad,[-1.5,.01],[1.5,1.8])
        if np.linalg.norm(projected)>2e-6:
            failures.append(start.tolist()); continue
        offsets=np.array([[i,j] for i in (-1,0,1) for j in (-1,0,1) if i or j])*1e-4
        if any(fun(p)[0]<value-1e-11 for p in np.clip(opt.x+offsets,[-1.5,.01],[1.5,1.8])):
            continue
        if all(np.linalg.norm(opt.x-p[1])>.002 for p in candidates):
            candidates.append((float(value),opt.x.copy()))
    candidates.sort(key=lambda p:p[0])
    assert candidates and not failures, failures
    return np.asarray([p for _,p in candidates]),np.asarray([v for v,_ in candidates])


def covariance(field, frequencies):
    coefficients = tone_coefficients(field, frequencies)
    time = np.arange(field.shape[-1])/FS
    x = coefficients.T@np.exp(2j*np.pi*np.asarray(frequencies)[:, None]*time)
    r = x@x.conj().T/len(time)
    return (r+r.conj().T)/2, coefficients


def subspace_cost(point, coords, frequency, signal):
    a, jac = response(point, coords, frequency, nearfield=False)
    value = signal@(signal.conj().T@a)
    return float(1-np.vdot(a, value).real), -2*np.real(jac.conj().T@value)


def srp_cost(points, coords, frequencies, phase, gradient=False):
    points = np.atleast_2d(points)
    delta = coords[None]-np.column_stack((points[:, 0], np.ones(len(points)), points[:, 1]))[:, None]
    radius = np.linalg.norm(delta, axis=-1)
    direction = delta/radius[..., None]
    value, derivative = np.ones(len(points)), np.zeros((len(points), 2))
    for frequency, ph in zip(frequencies, phase):
        k = 2*np.pi*frequency/343
        a = np.exp(-1j*k*radius)/len(coords)
        cross = a.conj()@ph
        value -= abs(cross)**2/2
        if gradient:
            da = 1j*k*direction[..., [0, 2]]*a[..., None]
            dc = np.einsum('mni,n->mi', da.conj(), ph)
            derivative -= np.real(cross.conj()[:, None]*dc)
    return (float(value[0]), derivative[0]) if gradient else value


def localize(field, coords, frequencies, truth, method, grid_size=301):
    """Search the complete y=1 m plane, then associate peaks with known sources."""
    r, coefficients = covariance(field, frequencies)
    frequency = float(np.mean(frequencies))
    if method == 'SRP-PHAT':
        pressure = coefficients.reshape(2, len(coords), 4)[:, :, 0]
        phase = pressure/np.maximum(abs(pressure), 1e-30)
        fun = lambda point: srp_cost(point, coords, frequencies, phase, gradient=True)
        kind = 'power'
    else:
        eigen, vectors = np.linalg.eigh(r)
        eigen, vectors = np.maximum(eigen[-2:], 0), vectors[:, -2:]
        if eigen[0] <= eigen[-1]*1e-12:
            raise ValueError("two-source covariance has fewer than two nonzero eigenvalues")
        if method == 'CBF':
            weights, kind = eigen/eigen.max(), 'power'
        elif method == 'MUSIC':
            weights, kind = np.ones(2), 'inverse'
        elif method == 'MVDR':
            loading = .01*r.trace().real/len(r)
            weights, kind = eigen/(eigen+loading), 'inverse'
        else:
            raise ValueError(f"unknown localization method: {method}")
        signal = vectors*np.sqrt(weights)
        fun = lambda point: subspace_cost(point, coords, frequency, signal)
    while True:
        xx, zz = np.meshgrid(np.linspace(-1.5, 1.5, grid_size),
                             np.linspace(.01, 1.8, grid_size))
        grid = np.column_stack((xx.ravel(), zz.ravel()))
        blocks = []
        for first in range(0, len(grid), 8192):
            points = grid[first:first+8192]
            if method == 'SRP-PHAT':
                blocks.append(srp_cost(points, coords, frequencies, phase))
            else:
                a = source_dictionary(points, coords, frequency, nearfield=False)
                a /= np.linalg.norm(a, axis=0, keepdims=True)
                blocks.append(1-np.sum(abs(signal.conj().T@a)**2, axis=0))
        q = np.concatenate(blocks)
        if method != 'SRP-PHAT':
            q = np.maximum(q, 1e-15)
        peaks, values = refine_plane(fun, q, xx, zz)
        power = np.maximum(1-q, 1e-15) if kind == 'power' else 1/np.maximum(q, 1e-15)
        peak_power = np.maximum(1-values, 1e-15) if kind == 'power' else 1/np.maximum(values, 1e-15)
        reference = max(power.max(), peak_power.max())
        db = (10*np.log10(power/reference)).reshape(xx.shape)
        peak_db = 10*np.log10(peak_power/reference)
        try:
            info, basins, _ = lobe_measurements(db, peaks, peak_db, truth, xx, zz)
            break
        except ValueError:
            if grid_size >= 1201:
                raise
            grid_size = (grid_size-1)*2+1
    if len(peaks) < len(truth):
        raise ValueError("fewer distinct spectrum peaks than source positions")
    errors = matched_source_errors(truth, peaks[:2])
    metrics = dict(localization_rmse_cm=float(np.sqrt(np.mean(errors**2))*100),
                   source_errors_cm=(errors*100).tolist(), estimates=peaks[:2].tolist(),
                   all_peaks=peaks.tolist(), all_peak_db=peak_db.tolist(),
                   extra_to_weak_db=info['extra_peak_to_weak_target_db'],
                   extra_to_strong_db=info['extra_peak_to_strong_target_db'],
                   target_peak_indices=info['target_peak_indices'],
                   strongest_extra_peak_xz_m=info['strongest_extra_peak_xz_m'],
                   grid_size=grid_size,
                   frequency_hz=frequency if method != 'SRP-PHAT' else list(frequencies))
    return metrics, dict(db=db.astype(np.float32), basins=basins.astype(np.int16),
                         peaks=peaks, peak_db=peak_db, x=xx[0], z=zz[:, 0])
