"""Feature-cache isolation, provider selection and shared predictor concurrency."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'probe-em'))
from probe_em.sam2_acceleration import EncoderRuntime, FeatureStore, provider_settings


def test_lru_budget_and_caller_mutation_cannot_change_saved_features():
    store = FeatureStore(32/1024**2)
    features = [torch.arange(4, dtype=torch.float32)]
    store.put('a', features)
    store.put('b', features)
    copy = store.get('a', 'cpu')
    copy[0][:] = -1
    torch.testing.assert_close(store.get('a', 'cpu')[0], features[0])
    store.put('c', features)
    assert store.get('b', 'cpu') is None
    assert store.bytes == 32 and store.evictions == 1
    store.put('large', [torch.zeros(100)])
    assert store.bytes == 32 and 'large' not in store.entries


def test_cache_uses_pixels_and_never_reuses_temporal_outputs():
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward_image(self, image):
            self.calls += 1
            features = [image+1, image+2, image+3]
            return {'backbone_fpn': features, 'vision_pos_enc': [torch.zeros_like(image)]*3,
                    'vision_features': features[-1]}

    runtime = EncoderRuntime('unused', 'unused', feature_cache_mb=1)
    first, second = Encoder().eval(), Encoder().eval()
    runtime.install(first); runtime.install(second)
    image = torch.zeros(1, 3, 2, 2)
    result = first.forward_image(image)
    result['backbone_fpn'][0][:] = 99
    cached = second.forward_image(image.clone())
    assert first.calls == 1 and second.calls == 0
    assert torch.all(cached['backbone_fpn'][0] == 1)
    image[:, :, 0, 0] = 7
    changed = second.forward_image(image)
    assert second.calls == 1 and torch.all(changed['backbone_fpn'][0][:, :, 0, 0] == 8)
    assert runtime.info()['feature_cache']['hits'] == 1


def test_missing_provider_fails_instead_of_silently_using_cpu():
    assert provider_settings('cpu', ['CPUExecutionProvider']) == [('CPUExecutionProvider', {})]
    assert provider_settings('directml', ['DmlExecutionProvider', 'CPUExecutionProvider'])[0][0] == 'DmlExecutionProvider'
    with pytest.raises(RuntimeError, match='unavailable'):
        provider_settings('cuda', ['CPUExecutionProvider'])
    with pytest.raises(ValueError):
        FeatureStore(float('nan'))


def test_encoder_rejects_mismatched_checkpoint_and_corrupt_artifacts(tmp_path, monkeypatch):
    from probe_em import device
    from probe_em.prompt_models import file_digest
    from probe_em.sam2_acceleration import SCHEMA
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace())
    monkeypatch.setattr(device, 'sam2_config_name', lambda name: name)
    checkpoint = tmp_path/'weights.pt'
    checkpoint.write_bytes(b'decoder weights')
    encoder = tmp_path/'encoder.onnx'
    encoder.write_bytes(b'encoder fixture')
    positions = tmp_path/'positions.npz'
    positions.write_bytes(b'position fixture')
    manifest = dict(schema=SCHEMA, status='validated', model_config='tiny', checkpoint_sha256='incorrect',
                    files={path.name: file_digest(path) for path in [encoder, positions]})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    runtime = EncoderRuntime(checkpoint, 'tiny', runtime='onnx', onnx_dir=tmp_path)
    with pytest.raises(ValueError, match='checkpoint differs'):
        runtime._load_onnx()
    manifest['checkpoint_sha256'] = file_digest(checkpoint)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    encoder.write_bytes(b'changed after validation')
    with pytest.raises(ValueError, match='checksum mismatch'):
        runtime._load_onnx()
    assert runtime.session is None


@pytest.mark.parametrize('refined', [False, True])
def test_shared_model_keeps_each_window_image_and_points_together(tmp_path, monkeypatch, refined):
    from probe_em.prompt_models import PointModel
    path = tmp_path/'checkpoint.pt'
    path.write_bytes(b'local predictor fixture')
    model = PointModel('sam2', path, model_config='sam2_hiera_t.yaml', device='cpu')

    class Predictor:
        def set_image(self, image):
            self.value = bool(image[0, 0, 0])

        def predict(self, **_):
            time.sleep(.002)
            mask = np.full((1,3,4), self.value, bool)
            mask[0,1,1] = True
            return mask, np.array([.8]), np.full((1,256,256),self.value,np.float32)

    monkeypatch.setattr(model, '_load', lambda: setattr(model, 'predictor', Predictor()))

    def run(index):
        value = bool(index % 2)
        arguments = (np.full((3,4,3),value,np.uint8),np.array([[1,1]]),np.array([1]))
        mask = model.predict_refined(*arguments)['mask'] if refined else model.predict(*arguments)[0]
        return bool(mask[0,0] == value and mask[1,1])

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert all(executor.map(run, range(24)))
