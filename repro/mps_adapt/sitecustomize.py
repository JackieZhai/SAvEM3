"""MPS 适配 · sitecustomize.py —— 运行时透明重定向（零改动运行旧代码）。

原理：Python 启动时自动 import 本模块（只要其所在目录在 PYTHONPATH 首位），
把 PyTorch 0.4 时代代码里的 `.cuda()` 调用透明重定向到 MPS 设备：
  - Tensor.cuda() / nn.Module.cuda()  →  .to(mps)（无 MPS 时回退 cuda/cpu）
  - torch.cuda.FloatTensor 等旧类型  →  torch.FloatTensor
  - torch.cuda.empty_cache()          →  容错 no-op
显式写死的 `torch.device('cuda:0')` 与 'No GPU available' 检查由 patch_mps.py 定点修复。

用法（无需改任何业务代码）：
    PYTHONPATH=$PWD:/path/to/其余库  python main_devoem_sparse_membrane_triplet_2.py -c xxx
或直接用 ./run_mps.sh。
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

# 旧式类型构造（torch.cuda.FloatTensor(...) 等）
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

# empty_cache 容错（无 CUDA 编译的 torch 上调用不报错）
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

print(f'[mps_adapt] CUDA 调用已重定向到 {_DEV}（MPS={_mps_ok()}）')
