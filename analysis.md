# Analysis for Issue #176482: [torch.export] ExportedProgram.run_decompositions breaks backward for self attention

**Issue**: https://github.com/pytorch/pytorch/issues/176482
**Category**: confirmed_bug
**Summary**: run_decompositions() bakes compute_log_sumexp=False into the graph because FakeTensors lack requires_grad during tracing, causing backward to fail with 'LSE is not correctly aligned (strideH)'

**CC**: @tugsbayasgalan

## Repro Code

```python
import torch
from torchvision.models import VisionTransformer


device = "cuda"

inner_model = VisionTransformer(
    image_size=224,
    patch_size=16,
    num_layers=12,
    num_heads=6,
    hidden_dim=384,
    mlp_dim=4 * 384,
    num_classes=1000,
).to(device)

sample_args = (torch.rand([10, 3, 224, 224], device=device),)

program = torch.export.export(inner_model, args=sample_args)
program = program.run_decompositions()
program.module()(*sample_args).sum().backward()
```

## Repro Output

```
RuntimeError: LSE is not correctly aligned (strideH)

Reproduced on nightly 2.12.0a0+gitcbc35eb. After run_decompositions(), the graph contains aten._scaled_dot_product_efficient_attention.default with compute_log_sumexp=False baked in. The FakeTensor metadata shows logsumexp shape [4, 6, 0] (empty). At runtime, the backward kernel receives this empty LSE and fails the alignment check.
```

## Fix Description

Root cause: run_decompositions() decomposes scaled_dot_product_attention via its CIA (Composite Implicit Autograd) implementation in attention.cpp. This calls should_compute_logsumexp() (attention.cpp:635-638), which checks requires_grad && GradMode. During export tracing with FakeTensors, requires_grad is False, so compute_log_sumexp=False gets baked into the decomposed graph. At runtime, _scaled_dot_product_efficient_attention produces an empty logsumexp tensor (shape [B, H, 0]), and the backward kernel in kernel_backward.h:1188 fails the lse_strideH % 8 == 0 check.

Suggested fix: In attention.cpp, modify should_compute_logsumexp() to also return true when inputs are tensor subclass-like (FakeTensors, proxy tensors), using the already-included isTensorSubclassLike() utility. This conservatively computes logsumexp during tracing to ensure backward correctness at runtime:

  bool should_compute_logsumexp(...) {
    ...
    const bool any_subclass_like = at::isTensorSubclassLike(query) ||
        at::isTensorSubclassLike(key) || at::isTensorSubclassLike(value);
    return (any_inputs_require_grad && gradmode_enabled) || any_subclass_like;
  }

User workaround: Force the math SDPA backend during export to avoid the efficient attention kernel entirely:

  with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
      program = torch.export.export(inner_model, args=sample_args)
      program = program.run_decompositions()
