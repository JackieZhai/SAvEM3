"""Auditable safeguards for generic SAM 2; paper thresholds remain selectable."""
import numpy as np


def occupancy_evidence(predictions, segmentations, target_id, allowed_ids,
                       policy='guarded', coverage_threshold=.6, min_pixels=50):
    """Return per-frame coverage AND prediction precision, including failed gates.

    One-sided coverage alone rewards a large leaked prediction swallowing tiny
    labels. Guarded acceptance needs two consecutive unprompted observations and
    substantial prediction ownership; short fragments remain review proposals.
    """
    if policy not in ('paper', 'guarded'):
        raise ValueError('Unknown verification policy')
    allowed_ids = {int(value) for value in allowed_ids}
    evidence = {}
    for index in sorted(predictions):
        mask = np.asarray(predictions[index], dtype=bool)
        labels = segmentations[index]
        if mask.shape != labels.shape:
            rows = np.arange(mask.shape[0])*labels.shape[0]//mask.shape[0]
            cols = np.arange(mask.shape[1])*labels.shape[1]//mask.shape[1]
            labels = labels[rows[:, None], cols[None, :]]
        ids, counts = np.unique(labels[mask], return_counts=True)
        for uid, count in zip(ids, counts):
            uid, count = int(uid), int(count)
            if uid in (0, int(target_id)) or count < min_pixels:
                continue
            area = int(np.count_nonzero(labels == uid))
            item = evidence.setdefault(uid, {'neighbor_id': uid, 'frames': []})
            item['frames'].append({'frame': int(index), 'intersection': count,
                                   'label_area': area, 'prediction_area': int(mask.sum()),
                                   'coverage': count/area, 'precision': count/max(1, int(mask.sum()))})
    for uid, item in evidence.items():
        frames = item['frames']
        item['score'] = max(row['coverage'] for row in frames)
        eligible = [row['frame'] for row in frames if row['frame'] > 0
                    and row['coverage'] > coverage_threshold and row['precision'] >= .25]
        consecutive = any(b == a+1 for a, b in zip(eligible, eligible[1:]))
        reasons = []
        if uid not in allowed_ids:
            reasons.append('outside_hss_candidates')
        if item['score'] <= coverage_threshold:
            reasons.append('low_coverage')
        if not consecutive:
            reasons.append('no_two_frame_precision_support')
        item.update(supported_frames=eligible, reasons=reasons,
                    paper_accepted=item['score'] > coverage_threshold,
                    is_connected=int(item['score'] > coverage_threshold if policy == 'paper' else not reasons))
    return list(evidence.values())


def cycle_overlap(returned_mask, prompt_mask):
    returned, prompt = np.asarray(returned_mask, bool), np.asarray(prompt_mask, bool)
    intersection = int(np.count_nonzero(returned & prompt))
    return {'coverage': intersection/max(1, int(prompt.sum())),
            'iou': intersection/max(1, int(np.count_nonzero(returned | prompt)))}
