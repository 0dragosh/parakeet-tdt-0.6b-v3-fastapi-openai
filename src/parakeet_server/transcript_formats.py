"""Output formatting helpers (SRT/VTT)."""

import datetime


def format_srt_time(seconds: float) -> str:
    delta = datetime.timedelta(seconds=seconds)
    rendered = str(delta)
    if "." in rendered:
        integer_part, fractional_part = rendered.split(".", 1)
        fractional_part = fractional_part[:3]
    else:
        integer_part = rendered
        fractional_part = "000"

    if len(integer_part.split(":")) == 2:
        integer_part = "0:" + integer_part

    return f"{integer_part},{fractional_part}"


def segments_to_srt(segments: list) -> str:
    lines = []
    for index, segment in enumerate(segments):
        start_time = format_srt_time(segment["start"])
        end_time = format_srt_time(segment["end"])
        text = segment["segment"].strip()

        if text:
            lines.append(str(index + 1))
            lines.append(f"{start_time} --> {end_time}")
            lines.append(text)
            lines.append("")

    return "\n".join(lines)


def segments_to_vtt(segments: list) -> str:
    lines = ["WEBVTT", ""]
    for segment in segments:
        start_time = format_srt_time(segment["start"]).replace(",", ".")
        end_time = format_srt_time(segment["end"]).replace(",", ".")
        text = segment["segment"].strip()

        if text:
            lines.append(f"{start_time} --> {end_time}")
            lines.append(text)
            lines.append("")
    return "\n".join(lines)
