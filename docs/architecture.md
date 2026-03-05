# Architecture

## Overview

The service is a Flask API that exposes an OpenAI-compatible transcription endpoint powered by `onnx-asr` and ONNX Runtime.

## Module Layout

- `src/parakeet_server/config.py`
  - Global constants, model catalog, filesystem paths, and environment wiring.
  - Handles dangling `models` symlink by switching to local fallback cache directory `.models/`.
- `src/parakeet_server/runtime.py`
  - ONNX Runtime provider selection, model initialization, lazy model cache, and runtime health state.
- `src/parakeet_server/audio_processing.py`
  - Audio duration probing, silence detection, chunk split-point selection, and text/token cleanup.
- `src/parakeet_server/transcript_formats.py`
  - SRT/VTT formatting helpers.
- `src/parakeet_server/server.py`
  - Flask routes, request/response handling, chunk orchestration, and API-compatible output formatting.
- `app.py`
  - Compatibility entrypoint that imports and launches `parakeet_server.server:main`.

## Runtime Flow

1. `server.py` configures Flask and request limits.
2. On first request (or startup), `runtime.ensure_runtime_initialized()`:
   - loads `onnxruntime`,
   - resolves providers from `PARAKEET_DEVICE`,
   - loads default model (`parakeet-tdt-0.6b-v3`),
   - stores provider and status metadata in `runtime_state`.
3. `/v1/audio/transcriptions`:
   - validates upload,
   - converts to mono 16k wav,
   - optionally splits by silence-aware chunking,
   - transcribes each chunk with selected model,
   - merges offsets and formats output (`json`, `text`, `srt`, `vtt`, `verbose_json`).

## Device/Provider Modes

`PARAKEET_DEVICE` controls provider behavior:

- `auto` (default): prefer CUDA, then CPU
- `cuda`: require CUDA provider (with CPU fallback if available)
- `tensorrt`: require TensorRT provider (with CUDA/CPU fallbacks if available). ~2-4x faster than CUDA but has a one-time engine build cost on first run.
- `cpu`: CPU only

TensorRT libraries are included in `Dockerfile.gpu` but the provider is **not active by default**. Set `PARAKEET_DEVICE=tensorrt` to opt in.

Provider status is exposed in `/health`.
