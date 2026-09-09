"""Volume contracts shared by inference, graph cuts and demo exporters.

Arrays use ZYX (volumes) or CZYX (features/affinities). Affinity channels
connect each voxel to its negative z, y and x neighbour, respectively.
Background label 0 is reserved; probabilities are finite numbers in [0, 1].
"""
from pathlib import Path

import numpy as np


def load_volume(path, key="main"):
    path = Path(path)
    if path.suffix.lower() in (".h5", ".hdf", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as handle:
            return handle[key][:]
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(path)
    raise ValueError(f"Unsupported volume format: {path}")


def save_volume(path, data, **metadata):
    import h5py
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        ds = handle.create_dataset("main", data=data, compression="gzip")
        ds.attrs["axes"] = "ZYX" if data.ndim == 3 else "CZYX"
        for key, value in metadata.items():
            ds.attrs[key] = value


def validate_labels(labels):
    labels = np.asarray(labels)
    if labels.ndim != 3 or not all(labels.shape):
        raise ValueError(f"Labels must be a nonempty ZYX volume, got {labels.shape}")
    if labels.dtype.kind not in "ui" or np.any(labels < 0):
        raise ValueError("Labels must be nonnegative integers (0 = background)")
    return labels


def validate_probabilities(values):
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Expected finite probabilities in [0, 1]; apply sigmoid to logits first")
    return values


def neighbour_slices(axis):
    current, previous = [slice(None)] * 3, [slice(None)] * 3
    current[axis], previous[axis] = slice(1, None), slice(None, -1)
    return tuple(current), tuple(previous)


def boundary_to_affinities(boundary):
    boundary = validate_probabilities(boundary)
    if boundary.ndim != 3:
        raise ValueError("Boundary must have shape ZYX")
    affinity = np.zeros((3,) + boundary.shape, dtype=np.float32)
    for axis in range(3):
        current, previous = neighbour_slices(axis)
        affinity[axis][current] = 1.0 - np.maximum(boundary[current], boundary[previous])
    return affinity


def prediction_maps(prediction, input_kind="auto"):
    """Return (boundary ZYX, affinity 3ZYX), without silently guessing logits."""
    prediction = validate_probabilities(prediction)
    if prediction.ndim == 3:
        prediction = prediction[None]
    if prediction.ndim != 4 or prediction.shape[0] not in (1, 3) or not all(prediction.shape):
        raise ValueError(f"Expected ZYX, 1ZYX boundary or 3ZYX affinity, got {prediction.shape}")
    if input_kind == "auto":
        input_kind = "boundary" if prediction.shape[0] == 1 else "affinity"
    if input_kind == "boundary" and prediction.shape[0] == 1:
        boundary = prediction[0]
        return boundary, boundary_to_affinities(boundary)
    if input_kind != "affinity" or prediction.shape[0] != 3:
        raise ValueError("Boundary requires one channel; affinity requires three channels (z,y,x)")
    # In-plane boundaries drive the anisotropic slice watershed. Absent y/x
    # neighbours on the outer faces are excluded from this visualization map.
    count = np.zeros(prediction.shape[1:], dtype=np.float32)
    total = np.zeros_like(count)
    for axis in (1, 2):
        current, _ = neighbour_slices(axis)
        total[current] += 1.0 - prediction[axis][current]
        count[current] += 1
    boundary = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    return boundary, np.ascontiguousarray(prediction)


def compact_labels(labels):
    """Relabel positive IDs, retaining background and uint64 IDs without float casts."""
    labels = np.asarray(labels)
    ids, inverse = np.unique(labels, return_inverse=True)
    values = np.arange(len(ids), dtype=np.uint64) + (0 if ids[0] == 0 else 1)
    return values[inverse].reshape(labels.shape)


def unique_slice_labels(labels):
    """Assign disjoint positive label ranges to independently segmented slices."""
    validate_labels(labels)
    result = np.zeros(labels.shape, dtype=np.uint64)
    offset = 0
    for z, section in enumerate(labels):
        section = compact_labels(section)
        foreground = section > 0
        result[z][foreground] = section[foreground] + offset
        offset += int(section.max())
    return result


def rag_pairs(labels):
    validate_labels(labels)
    pairs = []
    for axis in range(3):
        current, previous = neighbour_slices(axis)
        a, b = labels[current], labels[previous]
        mask = (a != b) & (a != 0) & (b != 0)
        if mask.any():
            pairs.append(np.unique(np.sort(np.stack((a[mask], b[mask]), axis=1), axis=1), axis=0))
    if not pairs:
        return np.empty((0, 2), dtype=np.uint64)
    return np.unique(np.concatenate(pairs), axis=0)


def edge_affinity_means(labels, affinities, uv):
    """Accumulate nearest-neighbour evidence in O(voxels), without vigra histograms."""
    validate_labels(labels)
    if affinities.shape != (3,) + labels.shape:
        raise ValueError("Affinity and fragment shapes do not match")
    sums = np.zeros(len(uv), dtype=np.float64)
    counts = np.zeros(len(uv), dtype=np.float64)
    if not len(uv):
        return sums, counts
    # Structured pair keys avoid u * max_label overflow for uint64 IDs.
    dtype = np.dtype([("u", "<u8"), ("v", "<u8")])
    def keys(pairs):
        return np.ascontiguousarray(np.sort(pairs, axis=1), dtype=np.uint64).view(dtype).ravel()
    edge_keys = keys(uv)
    order = np.argsort(edge_keys)
    for axis in range(3):
        current, previous = neighbour_slices(axis)
        a, b = labels[current], labels[previous]
        mask = a != b
        if not mask.any():
            continue
        positions = np.searchsorted(edge_keys[order], keys(np.stack((a[mask], b[mask]), axis=1)))
        if np.any(positions >= len(order)):
            raise ValueError("RAG does not contain all volume interfaces")
        indices = order[positions]
        if np.any(edge_keys[indices] != keys(np.stack((a[mask], b[mask]), axis=1))):
            raise ValueError('RAG does not contain all volume interfaces')
        sums += np.bincount(indices, weights=affinities[axis][current][mask], minlength=len(uv))
        counts += np.bincount(indices, minlength=len(uv))
    return np.divide(sums, counts, out=np.full_like(sums, 0.5), where=counts > 0), counts
