# Analysis for Issue #175720: DISABLED test_custom_op_with_memory_format_arg_xpu (__main__.AOTInductorTestABICompatibleGpu)

**Issue**: https://github.com/pytorch/pytorch/issues/175720
**Category**: confirmed_bug
**Summary**: DISABLED test_custom_op_with_memory_format_arg_xpu fails because the custom op only registers CPU and CUDA implementations, not XPU

**CC**: @yushangdi, @desertfire

## Repro Code

```python
python -m pytest test/inductor/test_aot_inductor_custom_ops.py::AOTInductorTestABICompatibleGpu::test_custom_op_with_memory_format_arg_xpu -xvs
```

## Fix Description

The test `test_custom_op_with_memory_format_arg` uses a custom op `aoti_custom_ops::fn_with_memory_format_arg` that is defined in the test file at line 126. The op's implementations are only registered for CPU (line 135-137) and CUDA (line 138-140), but not for XPU. When the test runs on XPU, it fails because there is no XPU dispatch kernel for this custom op.

The fix is to add an XPU implementation registration, similar to the CPU and CUDA ones:
```python
_memory_format_test_lib.impl(
    "fn_with_memory_format_arg", _fn_with_memory_format_arg_impl, "XPU"
)
```

The same issue likely applies to the other custom ops in this file (`fn_with_layout_arg`, `fn_with_dtype_arg`, etc.) that were enabled for XPU in PR #166047 but only have CPU/CUDA implementations. The test was enabled for XPU by etaf in PR #166047, and the custom op was later added by Bin Bao in PR #173562 without XPU support.
