"""Application configuration constants and environment setup."""

from pathlib import Path
import os
import sys

host = "0.0.0.0"
port = 5092
threads = 8
CHUNK_MINUTE = 1.5

# Intelligent chunking configuration
SILENCE_THRESHOLD = "-40dB"
SILENCE_MIN_DURATION = 0.5
SILENCE_SEARCH_WINDOW = 30.0
SILENCE_DETECT_TIMEOUT = 300
MIN_SPLIT_GAP = 5.0

DEFAULT_MODEL_NAME = "parakeet-tdt-0.6b-v3"
PROVIDER_PRIORITY = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)
DEVICE_MODE = os.getenv("PARAKEET_DEVICE", "auto").strip().lower()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RAW_MODELS_DIR = Path(os.getenv("PARAKEET_MODELS_DIR", str(PROJECT_ROOT / "models")))
if _RAW_MODELS_DIR.is_symlink() and not _RAW_MODELS_DIR.exists():
    MODELS_DIR = PROJECT_ROOT / ".models"
    MODELS_DIR_FALLBACK_USED = True
else:
    MODELS_DIR = _RAW_MODELS_DIR
    MODELS_DIR_FALLBACK_USED = False
ASSETS_DIR = PROJECT_ROOT / "assets"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
UPLOAD_DIR = PROJECT_ROOT / "temp_uploads"

MODEL_CONFIGS = {
    "parakeet-tdt-0.6b-v3": {
        "hf_id": "nemo-parakeet-tdt-0.6b-v3",
        "quantization": "int8",
        "description": "INT8 (fastest)",
    },
    "istupakov/parakeet-tdt-0.6b-v3-onnx": {
        "hf_id": "istupakov/parakeet-tdt-0.6b-v3-onnx",
        "quantization": None,
        "description": "FP32",
    },
    "grikdotnet/parakeet-tdt-0.6b-fp16": {
        "hf_id": "grikdotnet/parakeet-tdt-0.6b-fp16",
        "quantization": "fp16",
        "description": "FP16",
    },
}


def configure_environment() -> None:
    """Set runtime environment variables expected by onnx_asr/huggingface."""
    if MODELS_DIR_FALLBACK_USED:
        print(
            f"⚠️ Detected dangling models symlink at '{_RAW_MODELS_DIR}'. "
            f"Using fallback models directory '{MODELS_DIR}'."
        )

    os.environ["HF_HOME"] = str(MODELS_DIR)
    os.environ["HF_HUB_CACHE"] = str(MODELS_DIR)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

    if sys.platform == "win32":
        os.environ["PATH"] = (
            f"{PROJECT_ROOT};{PROJECT_ROOT / 'ffmpeg'};{os.environ['PATH']}"
        )
