# Analysis for Issue #175520: Question about torch._dynamo.export and nn.GRU support

**Issue**: https://github.com/pytorch/pytorch/issues/175520
**Category**: question
**Summary**: User asks if torch._dynamo.export not supporting nn.GRU is a known limitation or a bug; it is a known intentional limitation with experimental workaround available.

**CC**: @angelayi

## Answer

This is a known, intentional limitation of `torch._dynamo.export`. Dynamo does not support `nn.RNN`, `nn.GRU`, or `nn.LSTM` by default. The check is in `torch/_dynamo/variables/builder.py` in the `wrap_module()` method.

However, there are two workarounds:

1. **Experimental RNN support**: You can enable experimental support by setting `torch._dynamo.config.allow_rnn = True` before calling export:
   ```python
   torch._dynamo.config.allow_rnn = True
   exported = dynamo.export(model)(sample_input)
   ```

2. **Use `torch.export.export` (recommended)**: The newer `torch.export` API (not `torch._dynamo.export`) handles GRU/LSTM/RNN modules properly by decomposing them into primitive operations:
   ```python
   import torch
   exported = torch.export.export(model, (sample_input,))
   ```

Note: `torch._dynamo.export` is a legacy API. The recommended path for exporting PyTorch models is `torch.export.export`, which has broader module support including RNN variants.

## Repro Code

```python
import torch
import torch.nn as nn
import torch._dynamo as dynamo

class SimpleGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=80,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        out, h = self.gru(x)
        return out, h

model = SimpleGRU()
sample_input = torch.randn(2, 5, 80)

exported = dynamo.export(model)(sample_input)
```
