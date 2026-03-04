"""ONNX runtime and model loading management."""

from typing import Dict, List
import os
import threading
import traceback

from .config import (
    DEFAULT_MODEL_NAME,
    DEVICE_MODE,
    MODEL_CONFIGS,
    PROVIDER_PRIORITY,
)

model_cache: Dict[str, object] = {}
runtime_lock = threading.Lock()
runtime_state: Dict[str, object] = {
    "initialized": False,
    "device_mode": DEVICE_MODE,
    "available_providers": [],
    "requested_providers": [],
    "active_providers": [],
    "active_provider": None,
    "last_error": None,
}

onnx_asr = None
ort = None


def build_provider_priority(available_providers: List[str]) -> List[str]:
    available = set(available_providers)

    if DEVICE_MODE == "cpu":
        if "CPUExecutionProvider" not in available:
            raise RuntimeError(
                "CPUExecutionProvider is unavailable but PARAKEET_DEVICE=cpu was requested"
            )
        return ["CPUExecutionProvider"]

    if DEVICE_MODE == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable but PARAKEET_DEVICE=cuda was requested"
            )
        providers = ["CUDAExecutionProvider"]
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")
        return providers

    if DEVICE_MODE == "tensorrt":
        if "TensorrtExecutionProvider" not in available:
            raise RuntimeError(
                "TensorrtExecutionProvider is unavailable but PARAKEET_DEVICE=tensorrt was requested"
            )
        providers = ["TensorrtExecutionProvider"]
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")
        return providers

    if DEVICE_MODE != "auto":
        raise ValueError(f"Unsupported PARAKEET_DEVICE value: {DEVICE_MODE}")

    providers = [provider for provider in PROVIDER_PRIORITY if provider in available]
    if not providers:
        raise RuntimeError("No supported ONNX Runtime providers found")
    return providers


def build_session_options():
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return sess_options


def extract_active_providers(model) -> List[str]:
    session = getattr(model, "session", None)
    if session is None:
        return []
    if hasattr(session, "get_providers"):
        return list(session.get_providers())
    return []


def ensure_runtime_initialized() -> None:
    global onnx_asr, ort

    with runtime_lock:
        if runtime_state["initialized"]:
            return

        print("\nInitializing ONNX Runtime...")
        try:
            import onnx_asr as onnx_asr_module
            import onnxruntime as ort_module

            onnx_asr = onnx_asr_module
            ort = ort_module

            # Suppress C++ TensorRT probe errors on stderr (the native
            # dlopen of libnvinfer.so logs to fd 2 before Python can act).
            _devnull = os.open(os.devnull, os.O_WRONLY)
            _old_stderr = os.dup(2)
            os.dup2(_devnull, 2)
            try:
                available_providers = ort.get_available_providers()
            finally:
                os.dup2(_old_stderr, 2)
                os.close(_devnull)
                os.close(_old_stderr)
            providers_to_try = build_provider_priority(available_providers)

            runtime_state["available_providers"] = available_providers
            runtime_state["requested_providers"] = providers_to_try

            print(f"ONNX Runtime available providers: {available_providers}")
            print(f"ONNX Runtime requested provider order: {providers_to_try}")

            default_config = MODEL_CONFIGS[DEFAULT_MODEL_NAME]
            default_model = onnx_asr.load_model(
                default_config["hf_id"],
                quantization=default_config["quantization"],
                providers=providers_to_try,
                sess_options=build_session_options(),
            ).with_timestamps()

            model_cache[DEFAULT_MODEL_NAME] = default_model
            active_providers = extract_active_providers(default_model)
            runtime_state["active_providers"] = active_providers
            runtime_state["active_provider"] = (
                active_providers[0] if active_providers else providers_to_try[0]
            )
            runtime_state["initialized"] = True
            runtime_state["last_error"] = None

            print(
                "Default model loaded successfully "
                f"(active provider: {runtime_state['active_provider']}, device mode: {DEVICE_MODE})"
            )
            print("=" * 50)
        except Exception as exc:
            runtime_state["last_error"] = str(exc)
            print(f"❌ Runtime initialization failed: {exc}")
            traceback.print_exc()
            raise


def get_model(model_name: str):
    """Get or lazily load a model and return it from cache."""
    ensure_runtime_initialized()

    if model_name not in MODEL_CONFIGS:
        print(f"⚠️ Unknown model '{model_name}', falling back to default INT8 model")
        model_name = DEFAULT_MODEL_NAME

    if model_name in model_cache:
        print(f"Using cached model: {model_name}")
        return model_cache[model_name]

    print(f"Loading model: {model_name}")
    config = MODEL_CONFIGS[model_name]
    providers_to_try = list(runtime_state["requested_providers"])

    try:
        model = onnx_asr.load_model(
            config["hf_id"],
            quantization=config["quantization"],
            providers=providers_to_try,
            sess_options=build_session_options(),
        ).with_timestamps()

        model_cache[model_name] = model
        active_providers = extract_active_providers(model)
        if active_providers:
            runtime_state["active_providers"] = active_providers
            runtime_state["active_provider"] = active_providers[0]

        print(
            f"Model {model_name} loaded successfully "
            f"(active provider: {runtime_state['active_provider']})"
        )
        return model
    except Exception as exc:
        print(f"❌ Failed to load model {model_name}: {exc}")
        traceback.print_exc()
        if DEFAULT_MODEL_NAME in model_cache:
            print(
                "⚠️ Falling back to cached default model "
                f"(active provider: {runtime_state.get('active_provider')})"
            )
            return model_cache[DEFAULT_MODEL_NAME]
        raise RuntimeError(
            f"Failed to load model {model_name}. "
            f"Providers attempted: {providers_to_try}. "
            f"Device mode: {DEVICE_MODE}. No fallback model is available."
        ) from exc
