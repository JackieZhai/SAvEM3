"""Optional z-axis affine alignment module for Probe-EM.

EM volumes often suffer from slice-to-slice drift/rotation along z. When
enabled, Probe-EM registers every z-slice of the trace region to a reference
slice with a 2D affine transform (OpenCV ECC), reads all cutouts through the
aligned space while tracing/segmenting, and maps coordinate outputs (e.g.
contact points) back to the original space before saving.

Pairwise transforms are composed toward one reference section, giving a common
aligned frame inside the trace region. Long chains can accumulate estimation error.

Coordinate conventions (all in mip0 voxel space unless noted):

- ``field.transform(z)`` is the 3x3 affine mapping *original* slice ``z``
  coordinates to *aligned* coordinates.
- ``field.forward_pts`` maps points original -> aligned (z unchanged).
- ``field.inverse_pts`` maps points aligned -> original.
"""

import os

import cv2
import numpy as np

MIP0_RES = np.array([8.0, 8.0, 30.0])  # nm, mip0 of the zebrafish volumes


# --------------------------------------------------------------------------
# affine helpers
# --------------------------------------------------------------------------

def apply_affine_to_points(m, pts):
    """Apply 3x3 (or 2x3) affine to (N,2) points -> (N,2)."""
    pts = np.asarray(pts, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    if m.shape == (2, 3):
        m = np.vstack([m, [0, 0, 1]])
    return (m[:2, :2] @ pts.T + m[:2, 2:]).T


def invert_affine(m):
    m = np.asarray(m, dtype=np.float64)
    if m.shape == (2, 3):
        m = np.vstack([m, [0, 0, 1]])
    return np.linalg.inv(m)


def warp_slice(slice2d, m, seg=False, output_shape=None, border_mode=None):
    """Warp one 2D slice by affine m (3x3 or 2x3).

    ``slice2d`` is in cloud-volume (x, y) axis order; cv2.warpAffine expects
    (y, x) images with the matrix in (x, y) space, so the slice is transposed
    around the warp. Labels use compact float32 indices and INTER_NEAREST,
    then map back to their exact original IDs (including full-range uint64).
    output_shape is optional (X,Y). Raw defaults to replicated borders; callers
    showing an expanded registered ROI should use border_mode='constant'.
    """
    if m is None and output_shape is None:
        return slice2d
    if m is None:
        m = np.eye(3)
    m = np.asarray(m, dtype=np.float64)
    if m.shape == (3, 3):
        m = m[:2]
    x_extent, y_extent = output_shape or slice2d.shape[:2]
    img = np.ascontiguousarray(slice2d.T)   # (y, x) for cv2
    if not seg and (slice2d.dtype == np.uint8 or slice2d.dtype.kind == 'f'):
        out = cv2.warpAffine(img, m, (x_extent, y_extent), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT if border_mode == 'constant'
                             else cv2.BORDER_REPLICATE, borderValue=0)
    elif seg or slice2d.dtype in (np.uint32, np.uint64, np.int32, np.int64):
        # Warp compact indices to preserve IDs beyond int32 and float64 precision.
        ids, inverse = np.unique(np.concatenate((np.zeros(1, dtype=img.dtype), img.ravel())),
                                 return_inverse=True)
        work = inverse[1:].reshape(img.shape).astype(np.float32)
        if len(ids) >= 2**24:
            raise ValueError('Too many distinct labels in one slice for exact OpenCV index warping')
        out = cv2.warpAffine(work, m, (x_extent, y_extent), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out = ids[out.astype(np.int64)]
    else:
        out = cv2.warpAffine(img, m, (x_extent, y_extent), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.ascontiguousarray(out.T)  # back to (x, y) order


# --------------------------------------------------------------------------
# alignment field
# --------------------------------------------------------------------------

class ZAffineField:
    """Per-z-slice 2D affine transforms (mip0 voxel space)."""

    def __init__(self, z0, z1, mats, ref_z, window_bbox):
        self.z0 = int(z0)
        self.z1 = int(z1)
        self.ref_z = int(ref_z)
        self.mats = {int(z): np.asarray(m, dtype=np.float64) for z, m in mats.items()}
        self.window_bbox = np.asarray(window_bbox, dtype=np.int64)

    def transform(self, z):
        return self.mats.get(int(z))

    def forward_pts(self, pts):
        """original-space (N,3) mip0 vox -> aligned-space (N,3)."""
        pts = np.asarray(pts, dtype=np.float64)
        out = pts.copy()
        zs = np.unique(np.round(pts[:, 2]).astype(int))
        for z in zs:
            m = self.transform(z)
            if m is None:
                continue
            sel = np.round(pts[:, 2]).astype(int) == z
            out[sel, :2] = apply_affine_to_points(m, pts[sel, :2])
        return out

    def inverse_pts(self, pts):
        """aligned-space (N,3) mip0 vox -> original-space (N,3)."""
        pts = np.asarray(pts, dtype=np.float64)
        out = pts.copy()
        zs = np.unique(np.round(pts[:, 2]).astype(int))
        for z in zs:
            m = self.transform(z)
            if m is None:
                continue
            sel = np.round(pts[:, 2]).astype(int) == z
            out[sel, :2] = apply_affine_to_points(invert_affine(m), pts[sel, :2])
        return out

    def align_cutout(self, data, z0, seg=False, origin_xy=(0, 0)):
        """Align a cutout fetched in original space (z0 = its first z index).

        data shape: (nx, ny) single slice or (nx, ny, nz). Slices outside the
        field z-range are returned unchanged. origin_xy locates the cutout in
        the global voxel frame used by the field (essential for rotations).
        """
        data = np.asarray(data)
        shift = np.eye(3)
        shift[:2, 2] = origin_xy
        def local_transform(z):
            m = self.transform(z)
            return invert_affine(shift) @ m @ shift if m is not None else None
        if data.ndim == 2:
            m = local_transform(int(round(z0)))
            return warp_slice(data, m, seg=seg) if m is not None else data
        out = np.empty_like(data)
        for i in range(data.shape[2]):
            m = local_transform(z0 + i)
            out[..., i] = warp_slice(data[..., i], m, seg=seg) if m is not None else data[..., i]
        return out

    def inverse_connections(self, connections):
        """Map a list of conn dicts (x/y/z) back to original space."""
        out = []
        for c in connections:
            c = dict(c)
            p = self.inverse_pts(np.array([[c["x"], c["y"], c["z"]]], dtype=np.float64))[0]
            c["x"], c["y"], c["z"] = float(p[0]), float(p[1]), float(p[2])
            out.append(c)
        return out

    def stats(self):
        t = np.array([self.mats[z][:2, 2] for z in sorted(self.mats)])  # translations
        angs = np.array(
            [np.degrees(np.arctan2(m[1, 0], m[0, 0])) for m in
             (self.mats[z] for z in sorted(self.mats))]
        )
        # actual displacement the transform causes at the window corners
        w = self.window_bbox
        corners = np.array([[w[0], w[1]], [w[2], w[1]], [w[0], w[3]], [w[2], w[3]]],
                           dtype=np.float64)
        disp = np.array([
            np.abs(apply_affine_to_points(self.mats[z], corners) - corners).max()
            for z in sorted(self.mats)
        ])
        return {
            "z_range": [self.z0, self.z1],
            "ref_z": self.ref_z,
            "n_slices": len(self.mats),
            "mean_abs_xy_shift_vox": float(np.abs(t).mean(axis=0).max()),
            "max_abs_xy_shift_vox": float(np.abs(t).max()),
            "rotation_deg_minmax": [float(angs.min()), float(angs.max())],
            "max_corner_displacement_vox": float(disp.max()),
        }

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        zs = sorted(self.mats)
        np.savez(
            path,
            z0=self.z0, z1=self.z1, ref_z=self.ref_z,
            window_bbox=self.window_bbox,
            zs=np.array(zs),
            mats=np.stack([self.mats[z] for z in zs]),
        )

    @classmethod
    def load(cls, path):
        d = np.load(path)
        mats = {int(z): m for z, m in zip(d["zs"].tolist(), d["mats"])}
        return cls(int(d["z0"]), int(d["z1"]), mats, int(d["ref_z"]), d["window_bbox"])


class AlignedVolume:
    """Transparent wrapper: ``vol[x0:x1, y0:y1, z0:z1]`` returns the ALIGNED
    cutout; attributes resolution/bounds/mip pass through to the wrapped
    CloudVolume.

    The underlying fetch is expanded to the per-slice preimage of the
    requested output box (``T_z^-1`` of the box corners, union over z), so
    large cumulative translations/rotations of the field never push content
    outside the fetched window.
    """

    def __init__(self, vol, field):
        self.vol = vol
        self.field = field
        self.resolution = vol.resolution
        self.bounds = vol.bounds
        self.mip = vol.mip

    def __getattr__(self, name):
        return getattr(self.vol, name)

    def _preimage_box(self, x0, x1, y0, y1, z0, z1):
        """Fetch box covering the requested aligned box AND its per-slice
        preimages: the warp output grid covers the fetch coordinate range, so
        the fetch must include both [x0,x1]x[y0,y1] and T_z^-1 of it."""
        corners = np.array(
            [[x0, y0], [x1 - 1, y0], [x0, y1 - 1], [x1 - 1, y1 - 1]],
            dtype=np.float64,
        )
        lo = np.array([x0, y0], dtype=np.float64)
        hi = np.array([x1 - 1, y1 - 1], dtype=np.float64)
        for z in range(z0, z1):
            m = self.field.transform(z)
            if m is None:
                c = corners
            else:
                c = apply_affine_to_points(invert_affine(m), corners)
            lo = np.minimum(lo, c.min(axis=0))
            hi = np.maximum(hi, c.max(axis=0))
        return np.floor(lo).astype(int) - 2, np.ceil(hi).astype(int) + 2

    def __getitem__(self, slices):
        x0, x1 = slices[0].start, slices[0].stop
        y0, y1 = slices[1].start, slices[1].stop
        z0, z1 = slices[2].start, slices[2].stop
        lo, hi = self._preimage_box(x0, x1, y0, y1, z0, z1)
        bounds = np.array(self.vol.bounds.maxpt)
        lo = np.maximum(lo, np.asarray(self.vol.bounds.minpt)[:2])
        hi = np.minimum(hi, bounds[:2])
        if lo[0] >= hi[0] or lo[1] >= hi[1]:
            return np.zeros((x1 - x0, y1 - y0, z1 - z0, 1), dtype=self.vol.dtype)
        data = np.asarray(self.vol[lo[0]:hi[0], lo[1]:hi[1], z0:z1])
        if data.ndim != 4 or data.shape[3] != 1:
            raise ValueError('AlignedVolume expects a single-channel XYZC CloudVolume')
        data = self.field.align_cutout(data[..., 0], z0,
                                      seg=self.vol.layer_type == 'segmentation', origin_xy=lo)
        # crop to the requested output box
        ox0, oy0 = x0 - lo[0], y0 - lo[1]
        return data[ox0:ox0 + (x1 - x0), oy0:oy0 + (y1 - y0), :, None]


# --------------------------------------------------------------------------
# field estimation (OpenCV ECC, chained slice-to-slice with motion cascade)
# --------------------------------------------------------------------------

def _estimate_pair_affine(template, image, init=None, max_iter=100, eps=1e-6,
                          gauss_filt=5, max_rot_deg=10.0, max_scale_dev=0.15,
                          max_trans=10.0, motion="translation"):
    """Robustly estimate the 3x3 affine mapping image-coords -> template-coords.

    ``template``/``image`` are (x, y)-ordered arrays (cloud-volume convention).
    They are transposed into OpenCV's (row=y, col=x) layout. The estimate is initialized
    with the phase-correlation translation and refined by an ECC cascade whose
    extent depends on ``motion``:

    - "translation" (default): pure 2D translation — the most robust; chained
      composition accumulates no rotation/scale distortion.
    - "euclidean": + rigid rotation.
    - "affine": + scale/shear (note: scale/shear noise compounds multiplicatively
      over long chains).

    Returns None if the result violates the sanity clamps.
    """
    cascade = [cv2.MOTION_TRANSLATION]
    if motion in ("euclidean", "affine"):
        cascade.append(cv2.MOTION_EUCLIDEAN)
    if motion == "affine":
        cascade.append(cv2.MOTION_AFFINE)

    template = np.ascontiguousarray(template.T, dtype=np.float32)
    image = np.ascontiguousarray(image.T, dtype=np.float32)
    if template.shape != image.shape or min(template.shape) < 8:
        return None
    if not np.isfinite(template).all() or not np.isfinite(image).all():
        return None
    if min(float(template.std()), float(image.std())) < 1e-6:
        return None

    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
    # phase-correlation translation init (in (x, y) units)
    try:
        (dx, dy), _ = cv2.phaseCorrelate(template, image)
        w = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        if not np.all(np.isfinite(w)):
            w = np.eye(2, 3, dtype=np.float32)
    except cv2.error:
        w = np.eye(2, 3, dtype=np.float32)
    if init is not None:
        # The public transform is image -> template; ECC expects its inverse.
        w = invert_affine(init)[:2].astype(np.float32)
    converged = False
    for model in cascade:
        try:
            _, w = cv2.findTransformECC(template, image, w, model, crit, None, gauss_filt)
            converged = True
        except cv2.error:
            break
    if not converged or not np.isfinite(w).all():
        return None
    m_yx = np.eye(3, dtype=np.float64)
    m_yx[:2] = w.astype(np.float64)
    rot = abs(float(np.degrees(np.arctan2(m_yx[1, 0], m_yx[0, 0]))))
    scale = float(np.linalg.norm(m_yx[:2, 0]))
    trans = float(np.linalg.norm(m_yx[:2, 2]))
    if rot > max_rot_deg or abs(scale - 1.0) > max_scale_dev or trans > max_trans:
        return None
    return invert_affine(m_yx)   # image -> template in (x, y) space


def estimate_z_affine_field(raw_vol, z0, z1, window_mip0, ref_z=None,
                            est_mip=1, max_iter=100, eps=1e-6, gauss_filt=5,
                            motion="translation"):
    """Estimate per-z-slice affine transforms over [z0, z1].

    Alignment is chained slice-to-slice (each slice is registered to its
    already-aligned neighbor and composed towards the reference slice), which
    is robust to small per-slice drift; the returned field still maps every
    slice into the single reference frame.

    Parameters
    ----------
    raw_vol : CloudVolume
        raw volume opened at ``est_mip`` (used for estimation; the returned
        field is always in mip0 voxel space).
    window_mip0 : (x0, y0, x1, y1)
        xy estimation window in mip0 voxels.
    est_mip : int
        mip used for estimation (default 1 = faster). Transforms are rescaled
        to mip0 before returning.
    motion : str
        "translation" (default), "euclidean" or "affine" — see
        ``_estimate_pair_affine``.
    """
    res_est = np.array(raw_vol.resolution, dtype=np.float64)
    scale = res_est[:2] / MIP0_RES[:2]          # est-mip voxels per mip0 voxel
    x0, y0, x1, y1 = [int(v) for v in window_mip0]
    wx0, wy0 = int(x0 / scale[0]), int(y0 / scale[1])
    wx1, wy1 = int(x1 / scale[0]) + 1, int(y1 / scale[1]) + 1

    if ref_z is None:
        ref_z = (int(z0) + int(z1)) // 2
    z0, z1 = int(z0), int(z1)

    cache = {}

    def fetch(z):
        if z not in cache:
            img = np.asarray(raw_vol[wx0:wx1, wy0:wy1, z:z + 1]).squeeze()
            cache[z] = img.astype(np.float32)
        return cache[z]

    ref_img = fetch(ref_z)
    if ref_img.size == 0 or ref_img.std() < 1e-6:
        raise ValueError("reference slice is empty/constant; cannot estimate alignment")

    mats = {ref_z: np.eye(3, dtype=np.float64)}

    def estimate(z, neighbor):
        """T_z = T_neighbor @ M_z, with M_z aligning slice z onto slice neighbor.

        The pair matrix is estimated from a per-pair init only (phase
        correlation inside _estimate_pair_affine); seeding ECC with the
        neighbor's cumulative transform would lock it onto the cumulative
        translation instead of the small per-step shift.
        """
        m = _estimate_pair_affine(fetch(neighbor), fetch(z), motion=motion)
        if m is None or np.linalg.norm(m[:2, 2]) > 8.0:
            print(f"[align-z][warn] ECC rejected for slice {z} (clamped); "
                  f"reusing neighbor transform")
            return np.asarray(mats[neighbor], dtype=np.float64)
        return np.asarray(mats[neighbor], dtype=np.float64) @ m

    for z in range(ref_z + 1, z1 + 1):
        mats[z] = estimate(z, z - 1)
        if z % 20 == 0 or z == z1:
            print(f"[align-z] estimated slice {z}/{z1}")

    for z in range(ref_z - 1, z0 - 1, -1):
        mats[z] = estimate(z, z + 1)
        if z % 20 == 0 or z == z0:
            print(f"[align-z] estimated slice {z}/{z0}")

    # rescale to mip0: T_mip0 = S @ T_est @ S^-1, S = diag(scale_xy, scale_xy, 1)
    s = np.diag([scale[0], scale[1], 1.0])
    s_inv = np.diag([1.0 / scale[0], 1.0 / scale[1], 1.0])
    # Pair estimates are local to the estimation window; rotations must pivot
    # around that window before converting to global mip0 voxel coordinates.
    origin = np.eye(3)
    origin[:2, 2] = [wx0, wy0]
    mats = {z: s @ origin @ m @ invert_affine(origin) @ s_inv for z, m in mats.items()}

    field = ZAffineField(z0, z1, mats, ref_z,
                         [int(x0), int(y0), int(x1), int(y1)])
    print(f"[align-z] field stats: {field.stats()}")
    return field


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import cv2 as _cv2
    rng = np.random.default_rng(1)
    print("== synthetic pair-alignment check ==")
    # draw in cv2 (y, x) space, then transpose -> (x, y) module convention
    _img = np.zeros((200, 240), np.float32)
    for _ in range(40):
        cx, cy = rng.uniform(0, 240, 2)
        r = rng.uniform(5, 25)
        v = rng.uniform(100, 255)
        _cv2.circle(_img, (int(cx), int(cy)), int(r), v, -1)
    for i in range(0, 240, 12):
        _cv2.line(_img, (i, 0), (i + 30, 200), 60 + 30 * np.sin(i), 2)
    img = _img.T.copy()
    img = _cv2.GaussianBlur(img, (5, 5), 0) + rng.normal(0, 8, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.float32)
    t_true = np.array([[1.0, -0.008, 4.0], [0.006, 1.0, -2.5], [0, 0, 1]], dtype=np.float64)
    shifted = warp_slice(img, t_true)
    before = float(np.abs(shifted.astype(np.float32) - img).mean())
    m_est = _estimate_pair_affine(img, shifted, motion="affine")
    assert m_est is not None
    aligned = warp_slice(shifted, m_est)
    err = float(np.abs(aligned.astype(np.float32) - img).mean())
    print(f"mean abs diff: before={before:.2f} after={err:.2f}")
    assert err < 0.35 * before, "ECC alignment failed to improve!"
    # m_est should be image->template, i.e. close to the inverse of t_true
    # (sub-voxel ECC residual is expected)
    assert np.abs(m_est - invert_affine(t_true)).max() < 0.5

    print("== synthetic chained-field check ==")
    # slices shifted by z-dependent translations; recover the field chained
    shifts = {z: (0.5 * z, -0.3 * z) for z in range(0, 11)}
    stack = {}
    for z in range(0, 11):
        t = np.eye(3)
        t[0, 2], t[1, 2] = shifts[z]
        stack[z] = warp_slice(img, t)
    # estimate field with the chained routine on a fake raw_vol
    class FakeVol:
        resolution = (8, 8, 30)
        def __getitem__(self, sl):
            z = sl[2].start
            return stack[z][None, :, :, None]  # (1, h, w, 1) like cloud-volume
    field = estimate_z_affine_field(FakeVol(), 0, 10, [0, 0, 240, 200],
                                    ref_z=5, est_mip=0)
    # T_z should map slice z coords -> ref coords: translation (x,y) -> ref
    for z in range(0, 11):
        expected_t = np.array([shifts[5][0] - shifts[z][0],
                               shifts[5][1] - shifts[z][1]])
        got_t = field.mats[z][:2, 2]
        assert np.abs(got_t - expected_t).max() < 1.5, (z, got_t, expected_t)
    print("chained field translations match expected (sub-pixel ECC tolerance)")

    print("== roundtrip check ==")
    field2 = ZAffineField(0, 4, {0: np.eye(3), 1: t_true,
                                 2: np.linalg.inv(t_true) @ t_true, 3: t_true @ t_true,
                                 4: np.eye(3)}, ref_z=0, window_bbox=[0, 0, 10, 10])
    pts = np.array([[10.0, 20.0, 1.0], [30.0, 40.0, 3.0]])
    back = field2.inverse_pts(field2.forward_pts(pts))
    print("roundtrip max err:", float(np.abs(back - pts).max()))
    assert np.abs(back - pts).max() < 1e-9

    print("== save/load check ==")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "field.npz")
        field2.save(p)
        f2 = ZAffineField.load(p)
        assert f2.z0 == field2.z0 and f2.mats[1].shape == (3, 3)
    print("all self-tests passed")
