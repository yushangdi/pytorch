# Analysis for Issue #175865: torch.export.load holds GIL during tensor deserialization, preventing parallel loading

**Issue**: https://github.com/pytorch/pytorch/issues/175865
**Category**: confirmed_bug
**Summary**: torch.export.load holds the GIL during tensor deserialization via get_record, preventing parallel loading. The get_record C++ binding lacks a gil_scoped_release and returns py::bytes (double copy), while get_storage_from_record already returns tensors directly.

**CC**: @angelayi

## Fix Description

Two changes: (1) C++ side (init.cpp): Release the GIL during the zip read in the get_record binding using py::gil_scoped_release, and add a new get_record_as_tensor binding that reads a record and returns a uint8 tensor directly (zero-copy from the C++ DataPtr, no py::bytes intermediate). (2) Python side (_package.py): Add a read_tensor method to PT2ArchiveReader that calls get_record_as_tensor and views the result as the target dtype. Update _build_file_map to use read_tensor instead of read_bytes, eliminating the double copy (file -> C++ buffer -> Python bytes -> tensor) and allowing the GIL to be released during I/O. This matches the torch.load path which achieves near-linear thread scaling.
