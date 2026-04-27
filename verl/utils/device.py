# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Small accelerator compatibility helpers for CUDA, Ascend NPU, and CPU."""

from __future__ import annotations

import contextlib
import os
from typing import Optional

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None


def _normalize_device_type(device_type: Optional[str]) -> str:
    if device_type is None:
        return "auto"
    value = device_type.strip().lower()
    if value in {"", "auto"}:
        return "auto"
    if value in {"gpu"}:
        return "cuda"
    if value in {"ascend", "ascend_npu"}:
        return "npu"
    if value not in {"cuda", "npu", "cpu"}:
        raise ValueError(f"Unsupported SEARCH_R1_DEVICE={device_type!r}; use auto, cuda, npu, or cpu.")
    return value


def _npu_module():
    return getattr(torch, "npu", None)


def cuda_available() -> bool:
    return torch.cuda.is_available()


def npu_available() -> bool:
    npu = _npu_module()
    return npu is not None and hasattr(npu, "is_available") and npu.is_available()


def get_device_type() -> str:
    """Return the selected accelerator type.

    Override with SEARCH_R1_DEVICE or VERL_DEVICE. In auto mode we prefer NPU
    when ASCEND_RT_VISIBLE_DEVICES is present, otherwise CUDA, then NPU, then CPU.
    """

    requested = _normalize_device_type(os.getenv("SEARCH_R1_DEVICE") or os.getenv("VERL_DEVICE"))
    if requested == "cuda":
        if not cuda_available():
            raise RuntimeError("SEARCH_R1_DEVICE=cuda was requested but CUDA is not available.")
        return "cuda"
    if requested == "npu":
        if not npu_available():
            raise RuntimeError("SEARCH_R1_DEVICE=npu was requested but torch-npu/NPU is not available.")
        return "npu"
    if requested == "cpu":
        return "cpu"

    if os.getenv("ASCEND_RT_VISIBLE_DEVICES") and npu_available():
        return "npu"
    if cuda_available():
        return "cuda"
    if npu_available():
        return "npu"
    return "cpu"


def is_accelerator_available() -> bool:
    return get_device_type() in {"cuda", "npu"}


def get_accelerator_module():
    device_type = get_device_type()
    if device_type == "cuda":
        return torch.cuda
    if device_type == "npu":
        return _npu_module()
    return None


def device_count() -> int:
    module = get_accelerator_module()
    if module is None or not hasattr(module, "device_count"):
        return 0
    return module.device_count()


def current_device_index() -> int:
    module = get_accelerator_module()
    if module is None or not hasattr(module, "current_device"):
        return 0
    return int(module.current_device())


def current_device() -> torch.device:
    device_type = get_device_type()
    if device_type == "cpu":
        return torch.device("cpu")
    return torch.device(device_type, current_device_index())


def set_device(index: int) -> None:
    module = get_accelerator_module()
    if module is not None and hasattr(module, "set_device"):
        if get_device_type() == "npu":
            try:
                module.set_device(f"npu:{index}")
                return
            except TypeError:
                pass
        module.set_device(index)


def empty_cache() -> None:
    module = get_accelerator_module()
    if module is not None and hasattr(module, "empty_cache"):
        module.empty_cache()


def synchronize() -> None:
    module = get_accelerator_module()
    if module is not None and hasattr(module, "synchronize"):
        module.synchronize()


def get_rng_state():
    module = get_accelerator_module()
    if module is not None and hasattr(module, "get_rng_state"):
        return module.get_rng_state()
    return torch.get_rng_state()


def set_rng_state(state) -> None:
    module = get_accelerator_module()
    if module is not None and hasattr(module, "set_rng_state"):
        module.set_rng_state(state)
    else:
        torch.set_rng_state(state)


def manual_seed(seed: int) -> None:
    module = get_accelerator_module()
    if module is not None and hasattr(module, "manual_seed"):
        module.manual_seed(seed)
    torch.manual_seed(seed)


def memory_allocated() -> int:
    module = get_accelerator_module()
    if module is not None and hasattr(module, "memory_allocated"):
        return int(module.memory_allocated())
    return 0


def memory_reserved() -> int:
    module = get_accelerator_module()
    if module is not None and hasattr(module, "memory_reserved"):
        return int(module.memory_reserved())
    return 0


def get_device_name() -> str:
    module = get_accelerator_module()
    if module is not None and hasattr(module, "get_device_name"):
        try:
            return str(module.get_device_name(current_device_index()))
        except TypeError:
            return str(module.get_device_name())
    return "cpu"


def distributed_backend() -> str:
    device_type = get_device_type()
    if device_type == "cuda":
        return "nccl"
    if device_type == "npu":
        return "hccl"
    return "gloo"


def attention_implementation() -> str:
    override = os.getenv("SEARCH_R1_ATTENTION_IMPL")
    if override:
        return override
    return "flash_attention_2" if get_device_type() == "cuda" else "sdpa"


@contextlib.contextmanager
def autocast(dtype=torch.bfloat16, enabled: bool = True):
    device_type = get_device_type()
    if not enabled or device_type == "cpu":
        yield
        return

    if device_type == "npu" and torch_npu is not None:
        npu_amp = getattr(getattr(torch_npu, "npu", None), "amp", None)
        if npu_amp is not None and hasattr(npu_amp, "autocast"):
            with npu_amp.autocast(dtype=dtype):
                yield
            return

    with torch.autocast(device_type=device_type, dtype=dtype):
        yield


def maybe_compile(fn, **kwargs):
    """Avoid torch.compile by default on NPU unless explicitly requested."""

    if get_device_type() == "npu" and os.getenv("SEARCH_R1_ENABLE_TORCH_COMPILE", "0") != "1":
        return fn
    if hasattr(torch, "compile"):
        return torch.compile(fn, **kwargs)
    return fn
