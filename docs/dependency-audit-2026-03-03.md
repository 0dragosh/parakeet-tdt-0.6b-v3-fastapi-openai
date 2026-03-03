# Dependency Audit (March 3, 2026)

This snapshot compares direct dependency constraints in `pyproject.toml` to the latest releases on PyPI as of **March 3, 2026**.

## Summary

- Dependency management was migrated to `uv` with a committed `uv.lock`.
- Minimum versions in `pyproject.toml` were raised to current stable releases.
- `numpy` was unblocked from `<2` and aligned to upstream requirements (`>=1.21.6`).

## Direct Dependencies

| Package | Constraint in repo | Latest on PyPI (2026-03-03) | Status |
|---|---:|---:|---|
| `onnx-asr` | `>=0.10.2` | `0.10.2` | Up to date |
| `waitress` | `>=3.0.2` | `3.0.2` | Up to date |
| `flask` | `>=3.1.3` | `3.1.3` | Up to date |
| `openai` | `>=2.24.0` | `2.24.0` | Up to date |
| `typing_extensions` | `>=4.15.0` | `4.15.0` | Up to date |
| `psutil` | `>=7.2.2` | `7.2.2` | Up to date |
| `requests` | `>=2.32.5` | `2.32.5` | Up to date |
| `numpy` | `>=1.21.6` | `2.4.2` | Allowed by constraint |
| `onnxruntime` (cpu extra) | `>=1.24.2` | `1.24.2` | Up to date |
| `onnxruntime-gpu` (gpu extra) | `>=1.24.2` | `1.24.2` | Up to date |
| `pytest` (dev extra) | `>=9.0.2` | `9.0.2` | Up to date |

## Notes

- `onnxruntime` / `onnxruntime-gpu` / `onnx-asr` currently declare `numpy>=1.21.6` on PyPI, so constraining `numpy<2` is no longer required.
- The project now relies on `uv.lock` for reproducible installs in CI and local development.

## Sources

- PyPI project pages:
  - https://pypi.org/project/onnx-asr/
  - https://pypi.org/project/waitress/
  - https://pypi.org/project/Flask/
  - https://pypi.org/project/openai/
  - https://pypi.org/project/typing-extensions/
  - https://pypi.org/project/psutil/
  - https://pypi.org/project/requests/
  - https://pypi.org/project/numpy/
  - https://pypi.org/project/onnxruntime/
  - https://pypi.org/project/onnxruntime-gpu/
  - https://pypi.org/project/pytest/
