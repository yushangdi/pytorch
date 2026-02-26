# Analysis for Issue #175469: [DTensor] ExportedProgram.run_decompositions() fails with AssertionError: out is not NotImplemented

**Issue**: https://github.com/pytorch/pytorch/issues/175469
**Category**: confirmed_bug
**Summary**: ExportedProgram.run_decompositions() fails on DTensor models because aten.linear has no sharding strategy and its CIA decomposition is blocked during export

**CC**: @pianpwk, @tugsbayasgalan

## Repro Code

```python
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed._tensor import init_device_mesh
from torch._decomp import get_decompositions

if not dist.is_initialized():
    dist.init_process_group(backend="nccl")

device_mesh = init_device_mesh("cuda", (dist.get_world_size(),))
rank = dist.get_rank()

import torch.utils._pytree
import torch.distributed.tensor._dtensor_spec
torch.utils._pytree.register_constant(
    torch.distributed.tensor._dtensor_spec.DTensorSpec
)

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(10, 320)
        self.out_proj = nn.Linear(320, 160)

    def forward(self, x):
        return self.out_proj(torch.relu(self.in_proj(x)))

model = ToyModel().to("cuda")
parallelize_module(model.in_proj, device_mesh, ColwiseParallel())
parallelize_module(model.out_proj, device_mesh, RowwiseParallel())

inp = torch.rand(2, 10, device="cuda")
exported_program = torch.export.export(model, (inp,), strict=False)

decomp_table = get_decompositions([
    torch.ops.aten.embedding_dense_backward,
    torch.ops.aten.native_layer_norm_backward,
    torch.ops.aten.slice_backward,
    torch.ops.aten.select_backward,
    torch.ops.aten.norm.ScalarOpt_dim,
    torch.ops.aten.native_group_norm_backward,
    torch.ops.aten.upsample_bilinear2d.vec,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.split_with_sizes,
])

decomposed = exported_program.run_decompositions(decomp_table)

dist.destroy_process_group()
```

## Repro Output

```
Requires distributed NCCL setup (multi-GPU). Bug confirmed by multiple developers in issue comments. Comment #6 confirms it still fails on PyTorch 2.12.0.dev20260223+cu130 with: AssertionError: While executing %linear : [num_users=1] = call_function[target=torch.ops.aten.linear.default](args = (%flat_apply, %in_proj_weight, %in_proj_bias), kwargs = {})
```

## Fix Description

The root cause is a conflict between export's run_decompositions() and DTensor dispatch. When run_decompositions() is called:

1. aten.linear (a CompositeImplicitAutograd op) gets its CIA decomposition blocked by _override_composite_implicit_decomp context manager in torch/export/exported_program.py.
2. During retracing, aten.linear hits DTensor's __torch_dispatch__ (since model params are DTensors).
3. DTensor's sharding propagator has NO strategy for aten.linear.default (only for aten.mm, aten.addmm, aten.bmm in torch/distributed/tensor/_ops/_matrix_ops.py).
4. The CIA fallback in _dispatch.py:240-247 tries op_call.decompose(), but the decomposition is blocked/returns NotImplemented, triggering the assertion.

Two potential fixes:
(a) Register a sharding strategy for aten.linear.default in DTensor (PR #175591 is a proof-of-concept). This is most robust since it handles both export and eager inference paths.
(b) In _dispatch.py line 246, change the assertion to a conditional: if decompose() returns NotImplemented, re-raise the original NotImplementedError instead of asserting. This allows proper error propagation.

Option (a) is the preferred long-term fix. The sharding strategy for linear(input, weight, bias) mirrors addmm semantics: output = input @ weight.T + bias. PR #175194 and #175591 are already addressing this.
