import argparse
from pathlib import Path

import h5py
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from repro.savem3.distill import build_student, distillation_loss, predict_volume, sample_batch, train


def teacher_bundle(path):
    rng = np.random.default_rng(42)
    with h5py.File(path, 'w') as handle:
        handle.attrs['complete'] = True
        handle.create_dataset('raw', data=rng.integers(0, 255, (2, 16, 16), dtype=np.uint8))
        handle.create_dataset('boundary', data=rng.random((2, 16, 16), dtype=np.float32))
        handle.create_dataset('features', data=rng.random((32, 2, 8, 8), dtype=np.float32))
        handle.create_dataset('embeddings', data=rng.random((256, 2, 2, 2), dtype=np.float32))


def test_original_student_forward_backward_resume_inference(tmp_path):
    torch.set_num_threads(2)
    bundle = tmp_path / 'teacher.h5'
    teacher_bundle(bundle)
    args = argparse.Namespace(teacher=str(bundle), output=str(tmp_path / 'run'), seed=42,
             device='cpu', steps=1, batch=1, patch=[2, 16, 16], lr=1e-4, ablation='full',
             resume=None, init_encoder=None, secondary_weight=0.0, epsilon=0.1,
             log_every=1, save_every=1)
    train(args)
    checkpoint = torch.load(Path(args.output) / 'last.pt', weights_only=False)
    assert checkpoint['step'] == 1
    args.resume = str(Path(args.output) / 'last.pt')
    args.steps = 2
    train(args)
    checkpoint = torch.load(args.resume, weights_only=False)
    assert checkpoint['step'] == 2
    model = build_student()
    model.load_state_dict(checkpoint['model'])
    # Odd shape and a volume smaller than a tile exercise padding and final tiles.
    raw = np.full((1, 19, 17), 127, np.uint8)
    prediction = predict_volume(model, raw, [2, 16, 16], [1, 8, 8], 'cpu')
    assert prediction.shape == raw.shape
    assert np.isfinite(prediction).all() and 0 <= prediction.min() <= prediction.max() <= 1


def test_secondary_loss_flow_direction_and_boundary_mask():
    features = torch.zeros((1, 64, 2, 4, 4), requires_grad=True)
    boundary = torch.full((1, 1, 2, 8, 8), 0.5, requires_grad=True)
    embeddings = torch.zeros((1, 256, 2, 1, 1), requires_grad=True)
    targets = {'boundary': torch.ones_like(boundary), 'features': features[:, :32].detach(),
               'embeddings': embeddings.detach()}
    zero_flow = torch.zeros(1, 2, 1, 4, 4)
    loss, components = distillation_loss((boundary, features, embeddings), targets,
                                         secondary_weight=0.05, flow=zero_flow)
    assert components['motion'] == 0 and components['regularization'] == 0
    loss.backward()
    assert torch.isfinite(features.grad).all()
    with pytest.raises(ValueError, match='flow'):
        distillation_loss((boundary, features, embeddings), targets, secondary_weight=0.05)
    # current(x)=previous(x+1); f_x=+1 aligns x-1 and excludes the left border.
    shifted = features.detach().clone()
    ramp = torch.arange(4).float().repeat(4, 1)
    shifted[:, :, 0] = ramp
    shifted[:, :, 1] = ramp+1
    flow = zero_flow.clone()
    flow[:, 0] = 1
    _, components = distillation_loss((boundary, shifted, embeddings), targets,
                                      secondary_weight=0.05, flow=flow)
    assert components['motion'] == pytest.approx(0, abs=1e-6)


def test_membrane_decoder_export_interface():
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'saem2'))
    from saem2.modeling import MaskDecoderHQ
    decoder = MaskDecoderHQ('vit_b', initialize_from_sam=False).eval()
    # Smaller feature maps exercise the same prompt batching interface cheaply.
    with torch.inference_mode():
        masks, membrane = decoder(torch.zeros(2, 256, 4, 4), torch.zeros(2, 256, 4, 4),
                    torch.zeros(2, 1, 2, 256), torch.zeros(2, 1, 256, 4, 4),
                    False, True, [torch.zeros(2, 4, 4, 768)])
    assert membrane.shape == masks.shape == (2, 1, 16, 16)
