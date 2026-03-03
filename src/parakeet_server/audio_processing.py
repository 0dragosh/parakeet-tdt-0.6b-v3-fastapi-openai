"""Audio duration, silence detection, and transcript text cleaning utilities."""

from typing import List, Optional, Tuple
import math
import os
import re
import subprocess

from .config import (
    MIN_SPLIT_GAP,
    SILENCE_DETECT_TIMEOUT,
    SILENCE_MIN_DURATION,
    SILENCE_SEARCH_WINDOW,
    SILENCE_THRESHOLD,
)


def get_audio_duration(file_path: str) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return float(result.stdout)
    except (subprocess.CalledProcessError, ValueError) as exc:
        print(f"Could not get duration of file '{file_path}': {exc}")
        return 0.0


def detect_silence_points(
    file_path: str,
    silence_thresh: str = SILENCE_THRESHOLD,
    silence_duration: float = SILENCE_MIN_DURATION,
    total_duration: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Detect silence periods using ffmpeg silencedetect filter."""
    if not os.path.exists(file_path):
        print(f"Error: Audio file '{file_path}' not found for silence detection")
        return []

    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        file_path,
        "-af",
        f"silencedetect=noise={silence_thresh}:d={silence_duration}",
        "-f",
        "null",
        "-",
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=SILENCE_DETECT_TIMEOUT,
        )

        silence_points: List[Tuple[float, float]] = []
        silence_start = None

        for line in result.stderr.splitlines():
            if "silence_start:" in line:
                try:
                    silence_start = float(line.split("silence_start:")[1].split()[0])
                except (ValueError, IndexError):
                    silence_start = None
            elif "silence_end:" in line and silence_start is not None:
                try:
                    silence_end = float(line.split("silence_end:")[1].split()[0])
                    silence_points.append((silence_start, silence_end))
                    silence_start = None
                except (ValueError, IndexError):
                    pass

        if silence_start is not None and total_duration is not None:
            silence_points.append((silence_start, total_duration))

        return silence_points
    except subprocess.TimeoutExpired:
        print(f"Timeout: Silence detection exceeded {SILENCE_DETECT_TIMEOUT}s timeout")
        return []
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"Error running FFmpeg for silence detection: {exc}")
        return []
    except Exception as exc:
        print(f"Unexpected error detecting silence: {exc}")
        return []


def find_optimal_split_points(
    total_duration: float,
    target_chunk_duration: float,
    silence_points: List[Tuple[float, float]],
    search_window: float = SILENCE_SEARCH_WINDOW,
    min_gap: float = MIN_SPLIT_GAP,
) -> List[float]:
    """Find chunk split points based on nearby silence intervals."""
    if not silence_points or total_duration <= target_chunk_duration:
        return []

    split_points = []
    prev = 0.0
    num_chunks = math.ceil(total_duration / target_chunk_duration)

    for i in range(1, num_chunks):
        target_time = i * target_chunk_duration
        search_start = max(0.0, target_time - search_window)
        search_end = min(total_duration, target_time + search_window)

        candidates = [
            (start, end)
            for (start, end) in silence_points
            if start <= search_end and end >= search_start
        ]

        chosen = None
        if candidates:
            candidates_sorted = sorted(
                candidates,
                key=lambda silence_range: abs(
                    ((silence_range[0] + silence_range[1]) / 2.0) - target_time
                ),
            )
            for start, end in candidates_sorted:
                split_point = (start + end) / 2.0
                if split_point > prev + min_gap and split_point <= total_duration - min_gap:
                    chosen = split_point
                    break

        if chosen is None:
            chosen = max(prev + min_gap, min(target_time, total_duration - min_gap))
            if chosen > total_duration:
                chosen = None

        split_points.append(chosen)
        prev = chosen if chosen is not None else prev

    return [point for point in split_points if point is not None]


def clean_transcript_text(text: str) -> str:
    """Normalize model output text and remove tokenization artifacts."""
    if not text:
        return ""

    text = text.replace("\u2581", " ")
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.replace(" '", "'")


def clean_token_text(token: str) -> str:
    """Normalize per-token output for word timing payloads."""
    return token.replace("\u2581", " ").strip()
