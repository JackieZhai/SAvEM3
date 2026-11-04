"""Validate candidate topology before accepting it into a budgeted trace."""


def candidate_topology(segment_id, mip, seg_path, reader=None):
    from probe_em.get_endpoints_vectors import get_endpoints_vectors_precomputed, is_messy_segment
    reader = reader or get_endpoints_vectors_precomputed
    endpoints, vectors, resolution, _ = reader(segment_id, mip, seg_path)
    if endpoints is None or vectors is None or resolution is None:
        raise RuntimeError(f'Unreadable candidate skeleton: {segment_id}')
    messy = is_messy_segment(endpoints, resolution, max_endpoints=40)
    rejected = len(endpoints) > 15 and messy
    return {'segment': str(segment_id), 'endpoint_count': len(endpoints),
            'density_warning': bool(messy), 'rejected': bool(rejected),
            'reason': 'complex_topology' if rejected else 'topology_pass'}
