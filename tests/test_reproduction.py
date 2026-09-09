"""Regression tests for scientific invariants, without model downloads."""
from types import SimpleNamespace

import numpy as np
import pytest

from repro.volume import (prediction_maps, unique_slice_labels, edge_affinity_means,
                          rag_pairs, boundary_to_affinities)
from repro.sam import mask_prompt, choose_mask
from repro.graph_cut.prompt_graph_cut import node_features, prompt_iou_edges
from repro.graph_cut.waterz_iou import boost_interfaces


def test_boundary_polarity_and_negative_offset():
    boundary = np.zeros((2, 4, 6), dtype=np.float32)
    boundary[:, :, 3] = 1
    _, aff = prediction_maps(boundary)
    assert aff.shape == (3, 2, 4, 6)
    assert np.all(aff[2, :, :, 3:5] == 0)
    assert np.all(aff[2, :, :, 1:3] == 1)
    assert np.all(aff[0, 0] == 0)
    with pytest.raises(ValueError, match='probabilities'):
        prediction_maps(boundary + 2)


def test_sparse_stitch_uses_local_origin_and_marks_uncovered_voxels():
    from repro.evaluate.sparse_eval import stitch
    chunks = [np.full((2, 2, 2), 0.2), np.full((2, 2, 2), 0.8)]
    output, coverage, origin = stitch(chunks, [[10000, 20, 30], [10001, 21, 31]], return_metadata=True)
    assert output.shape == (3, 3, 3)
    np.testing.assert_array_equal(origin, [10000, 20, 30])
    assert output[1, 1, 1] == pytest.approx(0.5)
    assert coverage[1, 1, 1] == 2
    assert np.all(output[coverage == 0] == 1)


def test_slice_ids_never_merge_by_numeric_coincidence():
    slices = np.array([[[0, 7], [2, 7]], [[0, 7], [2, 7]]], dtype=np.uint64)
    labels = unique_slice_labels(slices)
    assert set(np.unique(labels[0])) & set(np.unique(labels[1])) == {0}
    assert np.array_equal(labels == 0, slices == 0)


def test_affinity_statistics_support_uint64_ids():
    labels = np.array([[[2**63 + 1, 2**63 + 2, 2**63 + 2]]], dtype=np.uint64)
    uv = rag_pairs(labels)
    affinities = np.zeros((3,) + labels.shape, np.float32)
    affinities[2, 0, 0, 1] = 0.75
    means, counts = edge_affinity_means(labels, affinities, uv)
    np.testing.assert_allclose(means, [0.75])
    np.testing.assert_array_equal(counts, [1])


def test_waterz_evidence_only_boosts_matching_axis_and_pair():
    labels = np.array([[[1, 2, 3], [1, 2, 3]]], dtype=np.uint64)
    affinity = np.full((3,) + labels.shape, 0.2, dtype=np.float32)
    result = boost_interfaces(affinity, labels, {'1,2': {'iou': 0.9}})
    expected = affinity.copy()
    expected[2, 0, :, 1] = 0.9
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(affinity, 0.2)  # input is not modified


def test_feature_means_handle_native_hq_resolution_and_sparse_ids():
    pytest.importorskip('cv2')
    labels = np.array([[[0, 0, 91, 91], [0, 0, 91, 91]]], dtype=np.uint64)
    features = np.ones((32, 1, 1, 2), dtype=np.float32) * 7
    means = node_features(labels, features)
    assert set(means) == {91}
    np.testing.assert_allclose(means[91], 7)


class RecordingPredictor:
    def __init__(self):
        self.model = SimpleNamespace(image_encoder=SimpleNamespace(img_size=1024))
        self.encoded = []

    def set_image(self, image):
        self.original_size = image.shape[:2]
        self.input_size = (512, 1024)
        self.encoded.append(int(image[0, 0, 0]))

    def predict(self, mask_input=None, **kwargs):
        assert mask_input.shape == (1, 256, 256)
        assert mask_input.dtype == np.float32
        assert mask_input.min() < 0 < mask_input.max()
        # Both prompts cover the same object on a common physical section.
        return np.ones((1,) + self.original_size, dtype=bool), np.array([0.9]), mask_input


def test_sam_mask_padding_tuple_and_common_plane_encoding():
    pytest.importorskip('cv2')
    predictor = RecordingPredictor()
    labels = np.zeros((3, 8, 16), dtype=np.uint64)
    labels[0, 2:6, 4:12] = 1
    labels[1:, 2:6, 4:12] = 2
    images = np.stack([np.full((8, 16), z, np.uint8) for z in range(3)])
    evidence = prompt_iou_edges(labels, images, predictor, erode=0)
    assert predictor.encoded == [0, 1]
    assert evidence[(1, 2)]['iou'] == pytest.approx(1)
    assert evidence[(1, 2)]['n_sections'] == 2
    prompt = mask_prompt(np.ones((8, 16), bool), predictor)
    assert np.all(prompt[:, 128:] == -6)  # SAM bottom padding is background
    mask, score = choose_mask((np.ones((1, 8, 16)), [0.7], None))
    assert mask.shape == (8, 16) and score == 0.7


@pytest.mark.parametrize('first_id', [1, 2**63+1])
def test_native_multicut_similarity_encourages_merge(first_id):
    pytest.importorskip('elf.segmentation.multicut')
    from repro.savem3.distill_postprocess import lmc_agglomerate
    fragments = np.full((2, 4, 6), first_id, np.uint64)
    fragments[:, :, 3:] = first_id+1
    affinity = np.full((3,) + fragments.shape, 0.1, np.float32)
    split = lmc_agglomerate(affinity, fragments)
    merge = lmc_agglomerate(affinity, fragments, {f'{first_id},{first_id+1}': {'iou': 1}}, {'iou': 10})
    assert len(np.unique(split)) == 2
    assert len(np.unique(merge)) == 1
    assert merge.min() > 0


def test_native_premerge_preserves_slice_identity_and_background():
    pytest.importorskip('elf.segmentation.multicut')
    from repro.savem3.distill_postprocess import premerge_2d_multicut
    labels = np.array([[[0, 1, 1, 2], [0, 1, 1, 2]]] * 2, np.uint64)
    output = premerge_2d_multicut(labels, np.zeros(labels.shape, np.float32))
    assert set(np.unique(output[0])) & set(np.unique(output[1])) == {0}
    np.testing.assert_array_equal(output == 0, labels == 0)


def test_native_watershed_multicut_waterz_pipeline():
    pytest.importorskip('elf.segmentation.watershed')
    pytest.importorskip('waterz')
    from repro.savem3.distill_postprocess import watershed_oversegment, premerge_2d_multicut, waterz_agglomerate
    boundary = np.zeros((3, 24, 24), np.float32)
    boundary[:, :, 11:13] = 0.99
    fragments = watershed_oversegment(boundary)
    fragments = premerge_2d_multicut(fragments, boundary)
    results = waterz_agglomerate(boundary_to_affinities(boundary), fragments, [0.3, 0.5])
    for result in results.values():
        assert result.shape == boundary.shape
        assert result[1, 5, 4] != result[1, 5, 20]
        assert result[0, 5, 4] == result[2, 5, 4] != 0
