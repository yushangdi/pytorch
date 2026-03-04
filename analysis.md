# Analysis for Issue #176349: Exporting a module with dynamic shape an example input the shape shape as the min boundry causes a failure

**Issue**: https://github.com/pytorch/pytorch/issues/176349
**Category**: confirmed_bug
**Summary**: Export fails with ConstraintViolationError when example input shape equals the min boundary of a dynamic Dim (e.g., min=1 with input size 1), because size-1 dimensions are specialized during tracing due to broadcasting semantics.

**CC**: @pianpwk, @angelayi

## Repro Code

```python
import torch

class MLP(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x @ x.T)

model = MLP().eval()
inputs = [torch.randn(1, 128)]
dynamic_shapes = ({0: torch.export.Dim("batch", min=1, max=32)},)
exp_program = torch.export.export(model, tuple(inputs), dynamic_shapes=dynamic_shapes)
```

## Repro Output

```
torch._dynamo.exc.UserError: Constraints violated (batch)! For more information, run with TORCH_LOGS="+dynamic".
  - You marked batch as dynamic but your code specialized it to be a constant (1). If you're using mark_dynamic, either remove it or use maybe_mark_dynamic. If you're using Dim.DYNAMIC, replace it with either Dim.STATIC or Dim.AUTO.
Suggested fixes:
  batch = 1
```

## Fix Description

Root cause analysis:

The bug has two contributing factors:

1. **create_symbol specializes val=1**: In `symbolic_shapes.py:create_symbol()`, when `specialize_zero_one=True` (default) and `val in (0, 1)`, the function returns `sympy.S.One` instead of creating a symbolic variable, even when the user explicitly constrained the dimension via `Dim(min=1, max=32)`. Fix: when `constraint_dim is not None`, set `do_not_specialize_zero_one = True`.

2. **Broadcasting logic guards on dim==1**: Even if fix (1) is applied so the symbol is created correctly, the broadcasting code in `torch/_refs/__init__.py:_broadcast_shapes()` (line ~444) evaluates `guard_or_false(shape[idx] == common_shape[idx])` and `guard_or_false(shape[idx] == 1)`, which creates guards that narrow the symbol's range from [1, 32] to [1, 1], causing specialization. Additionally, `_extract_tensor_metadata` in `proxy_tensor.py` checks contiguity via `check_contiguous_sizes_strides` which also guards on `dim == 1`.

A complete fix requires changes in multiple locations:
- `symbolic_shapes.py:create_symbol()`: Don't specialize 0/1 when constraint_dim is set
- `_refs/__init__.py:_broadcast_shapes()`: Handle backed symbols with constraints differently when hint is 0/1 - defer to runtime assertions instead of specializing
- `proxy_tensor.py:set_meta()` / `shape_prop.py:_extract_tensor_metadata()`: Suppress guards during metadata extraction

This is a fundamental issue with tracing at a broadcasting boundary value. The framework needs to handle the case where a user-constrained dimension has a hint of 0 or 1 without specializing.

**Workaround**: Use an example input with a shape > 1 for the dynamic dimension (e.g., `torch.randn(2, 128)` instead of `torch.randn(1, 128)`).
