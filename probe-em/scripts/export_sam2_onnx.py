"""Export and validate SAM2 image features for the optional hybrid ONNX runtime."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe_em.device import sam2_config_name
from probe_em.prompt_models import file_digest
from probe_em.sam2_acceleration import FEATURE_NAMES, SCHEMA


class ImageFeatures(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return tuple(self.model.forward_image(image)['backbone_fpn'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model-config', default='sam2_hiera_t.yaml')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    import onnx
    import onnxruntime as ort
    from sam2.build_sam import build_sam2
    if not args.checkpoint.is_file() or args.threads < 1:
        parser.error('Provide an existing checkpoint and a positive thread count')
    torch.set_num_threads(args.threads)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = sam2_config_name(args.model_config)
    model = build_sam2(config, str(args.checkpoint), device='cpu').eval()
    shape = [1, 3, model.image_size, model.image_size]
    image = torch.from_numpy(np.random.default_rng(8).standard_normal(shape).astype(np.float32))
    manifest = dict(schema=SCHEMA, status='exporting', checkpoint_sha256=file_digest(args.checkpoint),
                    model_config=config, input_shape=shape, torch_version=torch.__version__,
                    onnx_version=onnx.__version__, onnxruntime_version=ort.__version__, opset=18)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    with torch.inference_mode():
        reference = model.forward_image(image)
        if len(reference['backbone_fpn']) != 3:
            raise ValueError('This exporter requires a three-level SAM2 image pyramid')
        np.savez_compressed(output/'positions.npz', **{f'position_{i}': value.numpy()
                            for i, value in enumerate(reference['vision_pos_enc'])})
        torch.onnx.export(ImageFeatures(model), (image,), str(output/'encoder.onnx'),
                          input_names=['image'], output_names=FEATURE_NAMES, opset_version=18,
                          dynamo=False, external_data=False)
    onnx.checker.check_model(str(output/'encoder.onnx'))
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(output/'encoder.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    values = session.run(FEATURE_NAMES, {'image': image.numpy()})
    validation = []
    for expected, actual in zip(reference['backbone_fpn'], values):
        expected = expected.numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
        validation.append({'shape': list(actual.shape), 'max_abs_error': float(np.abs(actual-expected).max()),
                           'rmse': float(np.sqrt(np.mean((actual-expected)**2)))})
    manifest.update(status='validated', validation=validation,
                    files={name: file_digest(output/name) for name in ('encoder.onnx', 'positions.npz')})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
