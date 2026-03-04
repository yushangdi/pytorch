# Analysis for Issue #176148: export runtime error: Tensor specified that the operator does not mutate the argument, but this seems to be empirically wrong.

**Issue**: https://github.com/pytorch/pytorch/issues/176148
**Category**: no_repro
**Summary**: FBGEMM op schema incorrectly declares indice_weights as non-mutating; no repro code provided, fix belongs in FBGEMM source

**CC**: @angelayi

## Fix Description

The error is not a PyTorch export bug. The FBGEMM operator fbgemm::int_nbit_split_embedding_codegen_lookup_function declares indice_weights as Tensor? (non-mutating) in its schema, but at runtime it mutates the tensor. The fix is to update the FBGEMM operator schema to change Tensor? indice_weights to Tensor(a!)? indice_weights. This change must be made in the FBGEMM source code where the operator is registered. As a workaround without modifying FBGEMM, the user could try wrapping the call with torch.no_grad() or cloning indice_weights before passing it to the operator, but the proper fix is the schema correction in FBGEMM.
