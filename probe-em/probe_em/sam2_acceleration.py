"""Optional ONNX image features and bounded CPU caching for real SAM2 memory.

Only forward_image is replaced. The upstream prompt decoder, memory encoder,
memory attention and propagation state machine remain in PyTorch.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = 'probe-em-sam2-encoder-v1'
FEATURE_NAMES = ['feature_0', 'feature_1', 'feature_2']
PROVIDERS = {'cpu': 'CPUExecutionProvider', 'coreml': 'CoreMLExecutionProvider',
             'cuda': 'CUDAExecutionProvider', 'directml': 'DmlExecutionProvider'}


def provider_settings(name, available):
    if name not in PROVIDERS:
        raise ValueError('ONNX provider must be cpu, coreml, cuda or directml')
    primary = PROVIDERS[name]
    if primary not in available:
        raise RuntimeError(f'{primary} is unavailable; installed providers: {list(available)}')
    options = {'ModelFormat': 'MLProgram', 'MLComputeUnits': 'ALL', 'RequireStaticInputShapes': '1'} if name == 'coreml' else {}
    providers = [(primary, options)]
    if name != 'cpu':
        providers.append(('CPUExecutionProvider', {}))
    return providers


class FeatureStore:
    """LRU of detached feature tensors on CPU; never stores temporal memory."""

    def __init__(self, megabytes):
        if not np.isfinite(megabytes) or megabytes < 0:
            raise ValueError('Feature cache size must be finite and nonnegative')
        self.limit = int(megabytes*1024**2)
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = self.evictions = 0

    def get(self, key, device):
        value = self.entries.pop(key, None)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        self.entries[key] = value
        # A caller must not mutate the stored features, including on CPU.
        return [tensor.to(device=device, copy=True) for tensor in value]

    def put(self, key, features):
        size = sum(tensor.numel()*tensor.element_size() for tensor in features)
        if size > self.limit or not self.limit:
            return
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.bytes -= sum(t.numel()*t.element_size() for t in previous)
        while self.entries and self.bytes+size > self.limit:
            _, previous = self.entries.popitem(last=False)
            self.bytes -= sum(t.numel()*t.element_size() for t in previous)
            self.evictions += 1
        self.entries[key] = [tensor.detach().to(device='cpu', copy=True) for tensor in features]
        self.bytes += size

    def info(self):
        return dict(limit_bytes=self.limit, bytes=self.bytes, entries=len(self.entries),
                    hits=self.hits, misses=self.misses, evictions=self.evictions)


class EncoderRuntime:
    def __init__(self, checkpoint, model_config, *, runtime='torch', onnx_dir=None,
                 provider='cpu', threads=0, feature_cache_mb=0):
        if runtime not in ('torch', 'onnx'):
            raise ValueError('SAM2 runtime must be torch or onnx')
        if runtime == 'onnx' and onnx_dir is None:
            raise ValueError('ONNX runtime requires a local exported model directory')
        if threads < 0:
            raise ValueError('ONNX thread count must be nonnegative')
        self.runtime, self.provider, self.threads = runtime, provider, int(threads)
        self.directory = Path(onnx_dir).resolve() if onnx_dir is not None else None
        self.checkpoint, self.model_config = checkpoint, model_config
        self.store = FeatureStore(feature_cache_mb)
        self.session = self.manifest = None
        self.positions = None
        self.positions_by_grid = {}
        self.calls = 0

    def _load_onnx(self):
        import onnxruntime as ort
        import torch
        from probe_em.device import sam2_config_name
        from probe_em.prompt_models import file_digest
        manifest = json.loads((self.directory/'manifest.json').read_text())
        if manifest.get('schema') != SCHEMA or manifest.get('status') != 'validated':
            raise ValueError('ONNX encoder manifest is incompatible or was not validated')
        if manifest['checkpoint_sha256'] != file_digest(self.checkpoint):
            raise ValueError('ONNX encoder checkpoint differs from the PyTorch decoder checkpoint')
        if manifest['model_config'] != sam2_config_name(self.model_config):
            raise ValueError('ONNX encoder config differs from the PyTorch decoder config')
        for name in ('encoder.onnx', 'positions.npz'):
            if manifest['files'][name] != file_digest(self.directory/name):
                raise ValueError(f'ONNX artifact checksum mismatch: {name}')
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_BASIC if self.provider == 'coreml'
                                             else ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        if self.provider == 'directml':
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers = provider_settings(self.provider, ort.get_available_providers())
        session = ort.InferenceSession(str(self.directory/'encoder.onnx'), sess_options=options, providers=providers)
        if PROVIDERS[self.provider] not in session.get_providers():
            raise RuntimeError('Requested ONNX provider failed to initialize; refusing a silent provider change')
        session.disable_fallback()
        with np.load(self.directory/'positions.npz', allow_pickle=False) as data:
            self.positions = [torch.from_numpy(data[f'position_{i}'].copy()) for i in range(3)]
        self.manifest = manifest
        self.session = session

    def install(self, model):
        if self.runtime == 'torch' and not self.store.limit:
            return
        if model.training:
            raise ValueError('Encoder acceleration is only supported for inference')
        original = model.forward_image

        def forward(image):
            import torch
            self.calls += 1
            if self.runtime == 'onnx' and self.session is None:
                self._load_onnx()
            host = image.detach().float().cpu().contiguous().numpy()
            if self.runtime == 'onnx' and list(host.shape) != self.manifest['input_shape']:
                raise ValueError('ONNX encoder requires the exported batch-one image geometry')
            precision = (str(image.dtype), torch.is_autocast_enabled(image.device.type),
                         str(torch.get_autocast_dtype(image.device.type)))
            key = (host.shape, str(image.device), precision, hashlib.sha256(host.tobytes()).digest())
            features = self.store.get(key, image.device) if self.store.limit else None
            position_key = (str(image.device), host.shape)
            if features is None:
                if self.runtime == 'onnx':
                    values = self.session.run(FEATURE_NAMES, {'image': host})
                    if any(not np.isfinite(value).all() for value in values):
                        raise ValueError('ONNX encoder returned nonfinite image features')
                    features = [torch.from_numpy(value).to(image.device) for value in values]
                else:
                    result = original(image)
                    features = result['backbone_fpn']
                    self.positions_by_grid[position_key] = result['vision_pos_enc']
                self.store.put(key, features)
            if position_key not in self.positions_by_grid:
                if self.runtime == 'onnx':
                    self.positions_by_grid[position_key] = [value.to(image.device) for value in self.positions]
                else:
                    raise RuntimeError('Cached features are missing their positional geometry')
            return {'backbone_fpn': features, 'vision_features': features[-1],
                    'vision_pos_enc': self.positions_by_grid[position_key]}

        model.forward_image = forward

    def info(self):
        return {'encoder_runtime': self.runtime, 'decoder_and_memory_runtime': 'torch',
                'requested_provider': self.provider if self.runtime == 'onnx' else None,
                'session_providers': self.session.get_providers() if self.session else [],
                'encoder_calls': self.calls, 'feature_cache': self.store.info()}
