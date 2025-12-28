# GPU VM setup (py_boost GBDT) — Python 3.11 + CUDA 12.x

This fixes the common failure mode:
- `py_boost GPU kernels are not initialised (feature_grouper_kernel is None)`

Assumption: you are on a CUDA VM (Linux) and you have `conda`.

## 1) Create a clean Python 3.11 env

```bash
conda create -n cafa311 python=3.11 -y
conda activate cafa311
python -V
```

## 2) Install basic and GPU deps (CUDA 12.x)

```bash
python -m pip install -U pip
python -m pip install -U ipywidgets jupyter
python -m pip install -U cupy-cuda12x py-boost ipykernel
```

If your VM is CUDA 11.x, use `cupy-cuda11x` instead.

## 3) Register the kernel for notebooks

```bash
python -m ipykernel install --user --name cafa311 --display-name "CAFA (py311)"
```

## 4) Use this kernel in VS Code

- Command Palette → **Notebook: Select Notebook Kernel** → pick **CAFA (py311)**
- Then **Restart Kernel** (important after changing CuPy)

## 5) Quick verification

Run these in a notebook cell:

```python
import sys
print(sys.version)
print(sys.executable)

import cupy as cp
print("cupy:", cp.__version__)
_ = cp.zeros((1,), dtype=cp.float32)
cp.cuda.runtime.deviceSynchronize()
print("CuPy GPU alloc: OK")

import py_boost.gpu.utils as gpu_utils
if hasattr(gpu_utils, "init_kernels"):
    gpu_utils.init_kernels()
print("feature_grouper_kernel is None?", getattr(gpu_utils, "feature_grouper_kernel", None) is None)
```

Expected:
- Python is 3.11.x
- `CuPy GPU alloc: OK`
- `feature_grouper_kernel is None? False`
