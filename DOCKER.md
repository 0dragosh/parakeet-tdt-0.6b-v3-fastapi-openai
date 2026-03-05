# Docker Deployment Guide

This document covers Docker deployment options for Parakeet TDT transcription service.

## Quick Start

### CPU Deployment (Recommended for most users)

```bash
# Build and run
docker compose up parakeet-cpu -d

# Or build manually
docker build -f Dockerfile.cpu -t ghcr.io/0dragosh/parakeet-tdt-0.6b-v3-fastapi-openai:cpu .
docker run -d --name parakeet -p 5092:5092 -v parakeet-models:/app/models ghcr.io/0dragosh/parakeet-tdt-0.6b-v3-fastapi-openai:cpu
```

### GPU Deployment (Requires NVIDIA GPU)

**Prerequisites:**
- NVIDIA GPU with CUDA support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Published images are pushed by GitHub Actions to GHCR:
- `ghcr.io/0dragosh/parakeet-tdt-0.6b-v3-fastapi-openai:latest` (GPU)
- `ghcr.io/0dragosh/parakeet-tdt-0.6b-v3-fastapi-openai:cpu` (CPU)

```bash
# Build and run with Docker Compose
docker compose up parakeet-gpu -d

# Or build manually
docker build -f Dockerfile.gpu -t ghcr.io/0dragosh/parakeet-tdt-0.6b-v3-fastapi-openai:latest .
docker run -d --name parakeet-gpu -p 5092:5092 --gpus all \
    -v parakeet-models:/app/models ghcr.io/0dragosh/parakeet-tdt-0.6b-v3-fastapi-openai:latest
```

### GPU with TensorRT (Maximum Performance)

TensorRT optimizes the model graph for your specific GPU, providing ~2-4x faster inference than CUDA alone. The first run takes a few minutes while TensorRT builds its optimized engine (cached afterward).

**Prerequisites:**
- Same as GPU deployment above
- NVIDIA GPU with compute capability 7.0+ (Volta or newer recommended)

```bash
# Build and run with Docker Compose
docker compose up parakeet-tensorrt -d

# Or run manually with the same GPU image + env var
docker run -d --name parakeet-tensorrt -p 5092:5092 --gpus all \
    -e PARAKEET_DEVICE=tensorrt \
    -v parakeet-models:/app/models ghcr.io/0dragosh/parakeet-tdt-0.6b-v3-fastapi-openai:latest
```

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `http://localhost:5092` | Web UI |
| `http://localhost:5092/health` | Health check |
| `http://localhost:5092/v1/audio/transcriptions` | OpenAI-compatible API |
| `http://localhost:5092/docs` | Swagger documentation |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_HOME` | `/app/models` | HuggingFace model cache |
| `HF_HUB_CACHE` | `/app/models` | HuggingFace hub cache |
| `PARAKEET_DEVICE` | `auto` | Runtime provider: `auto` (CUDA→CPU), `cuda`, `tensorrt`, or `cpu` |

### Persistent Model Cache

Models are cached in a Docker volume to avoid re-downloading:

```bash
# List volumes
docker volume ls | grep parakeet

# Inspect volume
docker volume inspect parakeet-models

# Remove volume (forces model re-download)
docker volume rm parakeet-models
```

## Files Created

| File | Description |
|------|-------------|
| `pyproject.toml` | uv dependency definitions (cpu/gpu extras) |
| `Dockerfile.cpu` | CPU-only image (Python 3.10 slim) |
| `Dockerfile.gpu` | NVIDIA CUDA 12.4.1 cuDNN runtime image with GPU + TensorRT support |
| `docker-compose.yml` | Orchestration for both variants |
| `.dockerignore` | Excludes unnecessary files from build |

## Testing

```bash
# Check health
curl http://localhost:5092/health

# Transcribe audio (OpenAI-compatible)
curl -X POST http://localhost:5092/v1/audio/transcriptions \
    -F "file=@audio.mp3" \
    -F "model=parakeet-tdt-0.6b-v3"
```

## Troubleshooting

**Container won't start:**
- Check logs: `docker logs parakeet-cpu`
- First startup takes ~60s to download the model

**GPU not detected:**
- Verify NVIDIA Container Toolkit: `nvidia-smi` should work inside container
- Run: `docker run --rm --gpus all nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 nvidia-smi`

**Out of memory:**
- CPU image requires ~2GB RAM
- GPU image requires ~4GB VRAM
