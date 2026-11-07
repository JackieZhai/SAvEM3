"""Select unmodified SAM2 outputs using point agreement and iterative masks."""
import numpy as np


def point_agreement(mask, points, labels):
    """Count foreground/background disagreements at native pixel centers."""
    mask = np.asarray(mask, dtype=bool)
    points, labels = np.asarray(points), np.asarray(labels)
    xy = np.floor(points).astype(int)
    inside = mask[xy[:, 1], xy[:, 0]]
    return {'positive_points': int(np.count_nonzero(labels == 1)),
            'negative_points': int(np.count_nonzero(labels == 0)),
            'positive_missed': int(np.count_nonzero(~inside & (labels == 1))),
            'negative_included': int(np.count_nonzero(inside & (labels == 0)))}


def predict_refined(predictor, image_shape, points, labels, mask_input=None):
    """Refine an existing mask, checking alternative SAM2 masks if necessary.

    A previous low-resolution mask belongs to one controller, object and section;
    the caller must validate that context before passing it here. If every model
    candidate contradicts a point, return the least contradictory candidate and
    its counts. No pixels are forced to match clicks.
    """
    points, labels = np.asarray(points, np.float32), np.asarray(labels, np.int32)
    height, width = image_shape[:2]
    if (points.ndim != 2 or points.shape[1] != 2 or labels.shape != (len(points),)
            or not len(points) or not np.isfinite(points).all() or not np.isin(labels, [0, 1]).all()
            or not np.any(labels == 1) or np.any(points < 0)
            or np.any(points >= [width, height])):
        raise ValueError('Point prediction requires valid native XY points and binary labels with a positive point')
    if mask_input is not None:
        mask_input = np.asarray(mask_input, np.float32)
        if mask_input.shape != (1, 256, 256) or not np.isfinite(mask_input).all():
            raise ValueError('SAM2 refinement requires finite 1x256x256 mask logits')
    candidates = []
    calls = 0

    def infer(coords, point_labels, multiple, prior, source, collect=True):
        nonlocal calls
        masks, scores, logits = predictor.predict(point_coords=coords, point_labels=point_labels,
                                                  multimask_output=multiple, mask_input=prior)
        calls += 1
        masks, scores, logits = np.asarray(masks), np.asarray(scores), np.asarray(logits)
        count = 3 if multiple else 1
        if (masks.shape != (count, height, width) or scores.shape != (count,)
                or logits.shape != (count, 256, 256) or not np.isfinite(scores).all()
                or not np.isfinite(logits).all() or not np.isin(masks, [0, 1]).all()):
            raise ValueError('SAM2 returned invalid image masks, scores or refinement logits')
        values = []
        for i in range(count):
            agreement = point_agreement(masks[i], points, labels)
            values.append(dict(mask=masks[i].astype(bool), score=float(scores[i]),
                               low_res_logits=logits[i:i+1].copy(), agreement=agreement, source=source))
        if collect:
            candidates.extend(values)
        return values

    def rank(item):
        counts = item['agreement']
        return counts['positive_missed']+counts['negative_included'], -item['score']

    def consistent():
        return any(rank(item)[0] == 0 for item in candidates)

    # A consistent fresh result avoids accumulating an earlier mask's missed
    # branches. Use history to resolve contradictory clicks rather than forcing
    # every edit to inherit the previous object's extent.
    infer(points, labels, False, None, 'points')
    previous_used = False
    if not consistent() and mask_input is not None:
        infer(points, labels, False, mask_input, 'previous_mask')
        previous_used = True
        if not consistent():
            infer(points, labels, True, mask_input, 'previous_mask_candidates')
    if not consistent():
        infer(points, labels, True, None, 'point_candidates')
    if not consistent() and np.any(labels == 0):
        # Initialize plausible foreground objects before applying the full set
        # of corrections, as when the user first predicts and then adds N.
        positive = labels == 1
        seeds = infer(points[positive], labels[positive], True, None, 'positive_seed', collect=False)
        for seed in sorted(seeds, key=lambda item: -item['score']):
            if seed['agreement']['positive_missed']:
                continue
            infer(points, labels, False, seed['low_res_logits'], 'positive_mask_refinement')
    result = min(candidates, key=rank)
    result['diagnostics'] = dict(result.pop('agreement'), candidate_source=result.pop('source'),
                                 previous_mask_used=previous_used, decoder_calls=calls,
                                 candidates_checked=len(candidates), strategy='iterative-point-agreement-v1')
    return result
