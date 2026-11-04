"""Optional SOFIMA flow + elastic mesh refinement of robust translation fields.

This bounded ROI adapter follows the upstream section-alignment example, using
already warped neighboring images as the reference for each relaxation. It is
not the upstream distributed/block solver. Rejected refinements retain the
translation initializer and are explicitly reported.
"""
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from probe_em.alignment_field import ZElasticField, deformation_limits, load_field, warp_field_slice

ALGORITHM = 'sofima-elastic-roi-v1'
DEFAULTS = dict(resolution_nm=16., patch_nm=768., stride_nm=256.,
                max_residual_nm=160., max_gradient=.45, min_jacobian=.3, max_jacobian=2.,
                min_peak_ratio=1.6, min_peak_sharpness=1.6, min_valid_fraction=.1,
                min_ncc_gain=.001, mesh_k=.1, mesh_k0=.01, mesh_max_iters=10000,
                batch_size=64, max_pair_ncc_regression=.01)


def parameters(options=None):
    options = options or {}
    if not isinstance(options, dict):
        raise ValueError('SOFIMA configuration must be a JSON object')
    unknown = set(options)-set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown SOFIMA options: {sorted(unknown)}')
    result = dict(DEFAULTS, **options)
    if (not np.isfinite(list(result.values())).all() or any(value <= 0 for value in result.values())
            or result['max_gradient'] >= .8 or result['min_valid_fraction'] > 1
            or result['min_jacobian'] >= 1 or result['max_jacobian'] <= 1
            or result['stride_nm'] > result['patch_nm']
            or int(result['mesh_max_iters']) != result['mesh_max_iters']
            or result['mesh_max_iters'] < 250 or result['mesh_max_iters'] % 250
            or int(result['batch_size']) != result['batch_size']):
        raise ValueError('Invalid SOFIMA physical, quality or mesh parameters')
    return result


def runtime_info():
    try:
        import jax
        from sofima import flow_field, flow_utils, mesh
    except ImportError as error:
        raise RuntimeError('Install probe-em/requirements-sofima.txt in a separate environment and '
                           'select it with --sofima-python; saved fields need no SOFIMA installation') from error
    source = {}
    for module in (flow_field, flow_utils, mesh):
        source[module.__name__] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    adapter = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
               for name in ('sofima_registration.py', 'alignment_field.py', 'registration.py', 'z_align.py')}
    return {'versions': {name: importlib.metadata.version(name) for name in
                         ('sofima', 'connectomics', 'jax', 'jaxlib', 'numpy', 'scipy', 'opencv-python')},
            'source_sha256': source, 'adapter_sha256': adapter,
            'devices': [str(device) for device in jax.devices()]}


@lru_cache(maxsize=4)
def describe_runtime(python=None):
    if python is None:
        return runtime_info()
    result = subprocess.run([str(python), str(worker_script()), '--describe'],
                            text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError('SOFIMA worker could not load: ' + result.stderr[-3000:])
    return json.loads(result.stdout)


def worker_script():
    return Path(__file__).resolve().parents[1]/'scripts/sofima_worker.py'


def estimate_with_worker(raw, resolution, offset, *, python=None, options=None, **base_parameters):
    options = parameters(options)
    if python is None:
        return estimate(raw, resolution, offset, options=options, **base_parameters)
    # JAX and Torch can require different NumPy/OpenCV builds. Only numeric NPZ
    # and JSON cross the process boundary; commands never go through a shell.
    with tempfile.TemporaryDirectory(prefix='probe-em-sofima-') as folder:
        folder = Path(folder)
        np.savez(folder/'input.npz', raw=raw, resolution=resolution, offset=offset)
        (folder/'parameters.json').write_text(json.dumps(dict(options=options, **base_parameters)), encoding='utf-8')
        result = subprocess.run([str(python), str(worker_script()), '--input', str(folder/'input.npz'),
                                 '--parameters', str(folder/'parameters.json'), '--output', str(folder/'field.npz')],
                                timeout=7200)
        if result.returncode:
            raise RuntimeError(f'SOFIMA estimation failed (worker exit {result.returncode}); no field cached')
        return load_field(folder/'field.npz'), json.loads((folder/'field.json').read_text())


def estimate(raw, resolution, offset, *, options=None, ref_z=None, **base_parameters):
    from sofima import flow_field, flow_utils, mesh
    from probe_em.registration import estimate_field, ncc
    options = parameters(options)
    runtime = runtime_info()
    resolution, offset = np.asarray(resolution, float), np.asarray(offset)
    base, quality = estimate_field(raw, resolution, offset, ref_z=ref_z, **base_parameters)
    offset = offset.astype(int)
    depth, reference = raw.shape[2], base.ref_z-base.z0
    target = max(options['resolution_nm'], *resolution[:2])
    size = np.maximum(1, np.rint(np.asarray(raw.shape[:2])*resolution[:2]/target).astype(int))
    scale = np.asarray(raw.shape[:2])/size
    effective_resolution = resolution[:2]*scale
    patch_xy = np.maximum(8, np.rint(options['patch_nm']/effective_resolution).astype(int))
    step_xy = np.maximum(1, np.rint(options['stride_nm']/effective_resolution).astype(int))
    grid_xy = (size-patch_xy)//step_xy+1
    if np.any(grid_xy < 3):
        raise ValueError('SOFIMA needs at least 3x3 patches; enlarge the ROI or reduce patch_nm/stride_nm')
    origin = offset[:2]+(patch_xy//2+.5)*scale-.5
    spacing = step_xy*scale
    shifts = np.array([base.transform(base.z0+i)[:2, 2] for i in range(depth)])
    residuals = np.zeros((depth, 2, int(grid_xy[1]), int(grid_xy[0])), np.float32)
    field = ZElasticField(base.z0, shifts, residuals, origin, spacing, base.ref_z, base.window_bbox, resolution)
    native_ones = np.ones(raw.shape[:2], np.uint8)
    images, supports = {}, {}

    def raster(index, current_field):
        z = base.z0+index
        image = warp_field_slice(current_field, raw[:, :, index], z, offset[:2], offset[:2], raw.shape[:2])
        valid = warp_field_slice(current_field, native_ones, z, offset[:2], offset[:2], raw.shape[:2], seg=True)
        # Use AREA for both: partially padded estimation pixels are invalid.
        return (cv2.resize(image.T, tuple(size), interpolation=cv2.INTER_AREA),
                cv2.resize(valid.T.astype(np.float32), tuple(size), interpolation=cv2.INTER_AREA) > .999)

    images[reference], supports[reference] = raster(reference, base)
    calculator = flow_field.JAXMaskedXCorrWithStatsCalculator()
    config = mesh.IntegrationConfig(dt=.001, gamma=0., k0=options['mesh_k0'], k=options['mesh_k'],
                                    stride=tuple(step_xy*effective_resolution), num_iters=250,
                                    max_iters=int(options['mesh_max_iters']), stop_v_max=.05,
                                    dt_max=1000., start_cap=10., final_cap=10., prefer_orig_order=True,
                                    remove_drift=True)
    reports = []
    unobserved = set(quality['interpolated_or_extrapolated_slices'])
    min_ncc, min_overlap = base_parameters.get('min_ncc', .25), base_parameters.get('min_overlap', .70)
    for direction in (1, -1):
        for index in range(reference+direction, depth if direction == 1 else -1, direction):
            z = base.z0+index
            current, current_valid = raster(index, base)
            images[index], supports[index] = current, current_valid
            candidates = [index-direction*gap for gap in (1, 2)
                          if index-direction*gap in images and base.z0+index-direction*gap not in unobserved]
            report = {'z': z, 'accepted': False}
            if z in unobserved or not candidates:
                reports.append(dict(report, reason='no_reliable_section_observation'))
                continue
            neighbor = candidates[0]
            fixed, fixed_valid = images[neighbor], supports[neighbor]
            report['fixed_z'] = base.z0+neighbor
            flow = calculator.flow_field(fixed, current, tuple(patch_xy[::-1]), tuple(step_xy[::-1]),
                                         pre_mask=~fixed_valid, post_mask=~current_valid,
                                         max_masked=.25, batch_size=int(options['batch_size']))
            flow[:2] *= effective_resolution[:, None, None]
            clean = flow_utils.clean_flow(flow[:, None], options['min_peak_ratio'], options['min_peak_sharpness'],
                                           options['max_residual_nm'], options['max_residual_nm']/2)
            valid_nodes = np.isfinite(clean[:2, 0]).all(axis=0)
            report['valid_patch_fraction'] = float(valid_nodes.mean())
            if valid_nodes.sum() < 4 or valid_nodes.mean() < options['min_valid_fraction']:
                reports.append(dict(report, reason='insufficient_reliable_flow'))
                continue
            solved, energy, iterations = mesh.relax_mesh(np.zeros_like(clean), clean, config)
            candidate = np.asarray(solved)[:, 0]/resolution[:2, None, None]
            limits = deformation_limits(candidate, spacing, resolution)
            report.update(limits, mesh_steps=iterations, final_kinetic_energy=energy[-1])
            if (not np.isfinite(candidate).all() or iterations >= config.max_iters
                    or limits['max_gradient_norm'] > options['max_gradient']
                    or limits['min_jacobian'] < options['min_jacobian']
                    or limits['max_jacobian'] > options['max_jacobian']
                    or limits['max_residual_nm'] > options['max_residual_nm']):
                reports.append(dict(report, reason='deformation_or_convergence_limit'))
                continue
            residuals[index] = candidate
            field = ZElasticField(base.z0, shifts, residuals.copy(), origin, spacing,
                                  base.ref_z, base.window_bbox, resolution)
            warped, warped_valid = raster(index, field)
            common = fixed_valid & current_valid & warped_valid
            overlap = float(common.mean())
            before, after = ncc(fixed[common], current[common]), ncc(fixed[common], warped[common])
            report.update(ncc_before=before, ncc_after=after, valid_fraction=overlap)
            if overlap < min_overlap or after < min_ncc or after-before < options['min_ncc_gain']:
                residuals[index] = 0
                reports.append(dict(report, reason='insufficient_shared_support_or_ncc_gain'))
            else:
                images[index], supports[index] = warped, warped_valid
                reports.append(dict(report, accepted=True, reason='measured_flow_and_relaxed_mesh'))
            print(f'[sofima] section {z}: {reports[-1]["reason"]}', flush=True)
    # A failed section must not introduce an abrupt zero-residual seam between
    # neighboring elastic maps. Interpolate bounded residuals and flag inference.
    observed = sorted([reference]+[item['z']-base.z0 for item in reports if item['accepted']])
    inferred = sorted(set(range(depth))-set(observed))
    if inferred:
        flat = residuals.reshape(depth, -1)
        for column in range(flat.shape[1]):
            flat[inferred, column] = np.interp(inferred, observed, flat[observed, column])
    field, validation = validate_refinement(raw, offset, resolution, base, shifts, residuals,
                                            origin, spacing, options, min_overlap)
    quality = dict(quality, initializer_algorithm=quality['algorithm'], algorithm=ALGORITHM,
                   solver='SOFIMA masked cross-correlation and sequential elastic mesh', runtime=runtime,
                   sofima_parameters=options, refinement_sections=reports,
                   refined_slices=[base.z0+i for i in range(depth) if np.any(field.displacement[i])],
                   measured_refinement_slices=[item['z'] for item in reports if item['accepted']],
                   inferred_residual_slices=[base.z0+i for i in inferred],
                   fallback_to_translation_slices=[base.z0+i for i in range(depth)
                                                   if i != reference and not np.any(field.displacement[i])],
                   global_validation=validation, residual_mean_policy='zero mean per section',
                   estimation_resolution_xy_nm=effective_resolution.tolist(), patch_size_xy=patch_xy.tolist(),
                   elastic_geometry=field.stats())
    return field, quality


def validate_refinement(raw, offset, resolution, base, shifts, residuals, origin, spacing, options, min_overlap):
    """Choose per-section strengths using native detail and common support.

    This is an image-consistency safeguard, not a ground-truth accuracy metric.
    The same native pixels are sampled without area averaging. Every adjacent
    pair must stay within the allowed NCC regression, including inferred seams.
    A small dynamic program permits translation-only sections without introducing
    unchecked transitions to neighboring elastic sections. The all-zero path is
    always available, including when the raw stack has missing observations.
    """
    from probe_em.registration import ncc
    x, y = np.meshgrid(np.arange(raw.shape[0])[::2], np.arange(raw.shape[1])[::2], indexing='ij')
    xy = np.column_stack((x.ravel()+offset[0], y.ravel()+offset[1]))

    def sample(index, field):
        if field is None:
            coords = xy-shifts[index]-offset[:2]
        else:
            points = np.column_stack((xy, np.full(len(xy), base.z0+index)))
            coords = field.inverse_pts(points)[:, :2]-offset[:2]
        valid = ((coords >= 0) & (coords <= np.array(raw.shape[:2])-1)).all(axis=1)
        values = cv2.remap(np.ascontiguousarray(raw[:, :, index].T),
                           coords[:, 0].astype(np.float32).reshape(x.shape).T,
                           coords[:, 1].astype(np.float32).reshape(x.shape).T,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).T.ravel()
        return values, valid

    baseline = [sample(i, None) for i in range(raw.shape[2])]
    strengths = np.array([0., .125, .25, .5, 1.])
    samples = [baseline]
    for strength in strengths[1:]:
        field = ZElasticField(base.z0, shifts, residuals*strength, origin, spacing,
                              base.ref_z, base.window_bbox, resolution)
        samples.append([sample(i, field) for i in range(raw.shape[2])])
    depth, states = raw.shape[2], len(strengths)
    reference = base.ref_z-base.z0
    score = np.full((depth, states), -np.inf)
    parent = np.zeros((depth, states), dtype=int)
    score[0] = 0.
    if reference == 0:
        score[0, 1:] = -np.inf
    smoothness = .002
    pair_tables = []
    for index in range(1, depth):
        a, b = baseline[index-1], baseline[index]
        baseline_valid = a[1] & b[1]
        table = []
        for previous_state in range(states):
            previous = samples[previous_state][index-1]
            rows = []
            for state in range(states):
                current = samples[state][index]
                valid = baseline_valid & previous[1] & current[1]
                count = int(valid.sum())
                support = float(count/max(1, baseline_valid.sum()))
                before = ncc(a[0][valid], b[0][valid]) if count >= 32 else -1.
                after = ncc(previous[0][valid], current[0][valid]) if count >= 32 else -1.
                gain = after-before
                unchanged = previous_state == state == 0
                admissible = unchanged or (count >= 32 and support >= min_overlap
                                          and gain >= -options['max_pair_ncc_regression'])
                rows.append({'z': base.z0+index, 'baseline_ncc': before, 'elastic_ncc': after,
                             'gain': gain, 'relative_support': support, 'sample_count': count})
                if not admissible or (index == reference and state != 0):
                    continue
                candidate = (score[index-1, previous_state]+gain
                             - smoothness*(strengths[state]-strengths[previous_state])**2)
                if candidate > score[index, state]:
                    score[index, state], parent[index, state] = candidate, previous_state
            table.append(rows)
        pair_tables.append(table)
    selected = np.zeros(depth, dtype=int)
    selected[-1] = np.argmax(score[-1])
    for index in range(depth-1, 0, -1):
        selected[index-1] = parent[index, selected[index]]

    def selected_pairs():
        return [table[selected[i]][selected[i+1]] for i, table in enumerate(pair_tables)]

    pairs = selected_pairs()
    gain = float(np.mean([row['gain'] for row in pairs])) if pairs else 0.
    accepted = gain >= options['min_ncc_gain'] and bool(np.any(selected))
    candidate_gain = gain
    if not accepted:
        selected[:] = 0
        pairs = selected_pairs()
        gain = 0.
    chosen = strengths[selected]
    worst = min((row['gain'] for row in pairs), default=0.)
    print(f'[sofima validation] refined sections={np.count_nonzero(chosen)}/{depth}, '
          f'mean NCC gain={gain:.4f}, worst pair={worst:.4f}, accepted={accepted}', flush=True)
    return ZElasticField(base.z0, shifts, residuals*chosen[:, None, None, None], origin, spacing,
                          base.ref_z, base.window_bbox, resolution), {
        'metric': 'native-detail NCC sampled every 2 XY pixels on shared support',
        'selection': 'per-section dynamic programming with adjacent-pair regression guards',
        'candidate_strengths': strengths.tolist(), 'smoothness_penalty': smoothness,
        'section_strengths': chosen.tolist(), 'candidate_mean_gain': candidate_gain,
        'mean_gain': gain, 'worst_pair_gain': worst, 'pairs': pairs,
        'status': 'accepted' if accepted else 'retained_translation_initializer'}
