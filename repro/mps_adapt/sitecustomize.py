"""MPS compatibility through transparent runtime redirection for legacy code.

Python imports this module at startup when its directory is first on PYTHONPATH.
Redirect PyTorch 0.4-era `.cuda()` calls to MPS, falling back to CUDA or CPU:
  - Tensor.cuda() / nn.Module.cuda() -> .to(selected_device)
  - Legacy types such as torch.cuda.FloatTensor -> torch.FloatTensor without CUDA
  - torch.cuda.empty_cache() -> tolerate errors when CUDA is unavailable
Use patch_mps.py for explicit `torch.device('cuda:0')` and 'No GPU available' checks.

Usage (without changing application code):
    PYTHONPATH=$PWD:/path/to/other/libraries python main_devoem_sparse_membrane_triplet_2.py -c xxx
Alternatively, use ./run_mps.sh.
"""
import torch

def _mps_ok():
    try:
        return torch.backends.mps.is_available()
    except Exception:
        return False

def _dev():
    if _mps_ok():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

_DEV = _dev()

_orig_tensor_cuda = torch.Tensor.cuda
_orig_module_cuda = torch.nn.Module.cuda


def _tensor_cuda(self, *args, **kwargs):
    return self.to(_DEV)


def _module_cuda(self, *args, **kwargs):
    return self.to(_DEV)


torch.Tensor.cuda = _tensor_cuda
torch.nn.Module.cuda = _module_cuda

# Legacy tensor constructors, such as torch.cuda.FloatTensor(...).
if not torch.cuda.is_available():
    _map = {
        'FloatTensor': torch.FloatTensor, 'DoubleTensor': torch.DoubleTensor,
        'LongTensor': torch.LongTensor, 'ByteTensor': torch.ByteTensor,
        'BoolTensor': torch.BoolTensor, 'HalfTensor': torch.HalfTensor,
        'IntTensor': torch.IntTensor, 'ShortTensor': torch.ShortTensor,
    }
    for _name, _fn in _map.items():
        if hasattr(torch.cuda, _name):
            setattr(torch.cuda, _name, _fn)

# Tolerate empty_cache failures in PyTorch builds without CUDA.
try:
    _orig_empty_cache = torch.cuda.empty_cache

    def _empty_cache():
        try:
            _orig_empty_cache()
        except Exception:
            pass

    torch.cuda.empty_cache = _empty_cache
except Exception:
    pass

print(f'[mps_adapt] CUDA calls redirected to {_DEV} (MPS={_mps_ok()})')
