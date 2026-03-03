#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from parakeet_server.audio_processing import (  # noqa: E402
    clean_token_text,
    clean_transcript_text,
    find_optimal_split_points,
)


class TestAudioProcessing(unittest.TestCase):
    def test_clean_transcript_text(self):
        raw = "\u2581hello   world  ' test"
        self.assertEqual(clean_transcript_text(raw), "hello world' test")

    def test_clean_token_text(self):
        self.assertEqual(clean_token_text("\u2581hello"), "hello")

    def test_find_optimal_split_points_uses_silence(self):
        points = find_optimal_split_points(
            total_duration=200.0,
            target_chunk_duration=90.0,
            silence_points=[(88.0, 92.0), (176.0, 179.0)],
            search_window=30.0,
            min_gap=5.0,
        )
        self.assertTrue(points)
        self.assertAlmostEqual(points[0], 90.0, places=1)

    def test_find_optimal_split_points_without_silence_falls_back(self):
        points = find_optimal_split_points(
            total_duration=200.0,
            target_chunk_duration=90.0,
            silence_points=[],
            search_window=30.0,
            min_gap=5.0,
        )
        self.assertEqual(points, [])


if __name__ == "__main__":
    unittest.main()
