#!/usr/bin/env python3
"""Unit tests for model configuration and provider priority logic."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from parakeet_server import config, runtime  # noqa: E402


class TestModelSelection(unittest.TestCase):
    def test_model_configs_have_expected_variants(self):
        self.assertIn("parakeet-tdt-0.6b-v3", config.MODEL_CONFIGS)
        self.assertIn("istupakov/parakeet-tdt-0.6b-v3-onnx", config.MODEL_CONFIGS)
        self.assertIn("grikdotnet/parakeet-tdt-0.6b-fp16", config.MODEL_CONFIGS)

    def test_default_model_constant_exists(self):
        self.assertIn(config.DEFAULT_MODEL_NAME, config.MODEL_CONFIGS)

    def test_auto_provider_priority_prefers_gpu_then_cpu(self):
        old_device_mode = runtime.DEVICE_MODE
        runtime.DEVICE_MODE = "auto"
        try:
            selected = runtime.build_provider_priority(
                ["CPUExecutionProvider", "CUDAExecutionProvider"]
            )
            self.assertEqual(selected, ["CUDAExecutionProvider", "CPUExecutionProvider"])
        finally:
            runtime.DEVICE_MODE = old_device_mode

    def test_cpu_mode_requires_cpu_provider(self):
        old_device_mode = runtime.DEVICE_MODE
        runtime.DEVICE_MODE = "cpu"
        try:
            selected = runtime.build_provider_priority(["CPUExecutionProvider"])
            self.assertEqual(selected, ["CPUExecutionProvider"])
        finally:
            runtime.DEVICE_MODE = old_device_mode


if __name__ == "__main__":
    unittest.main()
