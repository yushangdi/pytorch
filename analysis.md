# Analysis for Issue #176428: torch.export.load() slow due to Python-side schema deserialization

**Issue**: https://github.com/pytorch/pytorch/issues/176428
**Category**: feature_request
**Summary**: Feature request to speed up torch.export.load() by exposing C++ schema types with pybind11 property bindings, eliminating slow Python-side JSON/dataclass deserialization

**CC**: @angelayi, @tolleybot
