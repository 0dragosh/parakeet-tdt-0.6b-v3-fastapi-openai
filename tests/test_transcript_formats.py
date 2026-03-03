#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from parakeet_server.transcript_formats import (  # noqa: E402
    format_srt_time,
    segments_to_srt,
    segments_to_vtt,
)


class TestTranscriptFormats(unittest.TestCase):
    def test_format_srt_time(self):
        self.assertEqual(format_srt_time(1.234), "0:00:01,234")

    def test_segments_to_srt(self):
        segments = [{"start": 0.0, "end": 1.0, "segment": "hello"}]
        rendered = segments_to_srt(segments)
        self.assertIn("1", rendered)
        self.assertIn("0:00:00,000 --> 0:00:01,000", rendered)
        self.assertIn("hello", rendered)

    def test_segments_to_vtt(self):
        segments = [{"start": 0.0, "end": 1.0, "segment": "hello"}]
        rendered = segments_to_vtt(segments)
        self.assertTrue(rendered.startswith("WEBVTT"))
        self.assertIn("0:00:00.000 --> 0:00:01.000", rendered)


if __name__ == "__main__":
    unittest.main()
