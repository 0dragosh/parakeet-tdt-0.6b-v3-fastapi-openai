#!/usr/bin/env bash
set -euo pipefail

buildDir="${1:-.}"

# Patch base image: upstream uses cudnn8 but onnxruntime-gpu requires cuDNN 9
sed -i 's|nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04|nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04|' "${buildDir}/Dockerfile.gpu"
