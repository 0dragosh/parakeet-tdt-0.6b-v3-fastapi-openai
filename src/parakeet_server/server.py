import json
import math
import os
import subprocess
import threading
import uuid

import flask
import psutil
from flask import Flask, Response, jsonify, render_template, request
from waitress import serve
from werkzeug.utils import secure_filename

from .audio_processing import (
    clean_token_text,
    clean_transcript_text,
    detect_silence_points,
    find_optimal_split_points,
    get_audio_duration,
)
from .config import (
    ASSETS_DIR,
    CHUNK_MINUTE,
    DEFAULT_MODEL_NAME,
    DEVICE_MODE,
    MODEL_CONFIGS,
    MODELS_DIR,
    SILENCE_SEARCH_WINDOW,
    TEMPLATES_DIR,
    UPLOAD_DIR,
    configure_environment,
    host,
    port,
    threads,
)
from .runtime import ensure_runtime_initialized, get_model, runtime_state
from .transcript_formats import segments_to_srt, segments_to_vtt

configure_environment()

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
if not MODELS_DIR.exists() and not MODELS_DIR.is_symlink():
    os.makedirs(MODELS_DIR, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 2000 * 1024 * 1024

progress_tracker = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/parakeet.png")
def serve_logo():
    return flask.send_file(ASSETS_DIR / "parakeet.png", mimetype="image/png")


@app.route("/health")
def health():
    runtime_error = None
    if not runtime_state["initialized"]:
        try:
            ensure_runtime_initialized()
        except Exception as exc:
            runtime_error = str(exc)

    payload = {
        "status": "healthy" if runtime_error is None else "degraded",
        "models": list(MODEL_CONFIGS.keys()),
        "default_model": DEFAULT_MODEL_NAME,
        "speedup": "20.7x",
        "runtime": {
            "device_mode": DEVICE_MODE,
            "available_providers": runtime_state.get("available_providers", []),
            "requested_providers": runtime_state.get("requested_providers", []),
            "active_provider": runtime_state.get("active_provider"),
            "active_providers": runtime_state.get("active_providers", []),
            "initialized": runtime_state.get("initialized", False),
        },
    }

    if runtime_error:
        payload["runtime"]["error"] = runtime_error
        return jsonify(payload), 503
    return jsonify(payload)


@app.route("/docs")
def swagger_ui():
    return render_template("swagger.html")


@app.route("/openapi.json")
def openapi_spec():
    base_url = request.host_url.rstrip("/")
    return jsonify(
        {
            "openapi": "3.0.0",
            "info": {
                "title": "Parakeet Transcription API",
                "description": "High-performance ONNX-optimized speech transcription API compatible with OpenAI.",
                "version": "1.0.0",
            },
            "servers": [{"url": base_url}],
            "paths": {
                "/v1/audio/transcriptions": {
                    "post": {
                        "summary": "Transcribe Audio",
                        "description": "Transcribes audio into the input language. Supports real-time streaming progress.",
                        "operationId": "transcribe_audio",
                        "requestBody": {
                            "content": {
                                "multipart/form-data": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "file": {
                                                "type": "string",
                                                "format": "binary",
                                                "description": "The audio file object (not file name) to transcribe.",
                                            },
                                            "model": {
                                                "type": "string",
                                                "default": DEFAULT_MODEL_NAME,
                                                "enum": list(MODEL_CONFIGS.keys()),
                                                "description": "Model variant to use.",
                                            },
                                            "response_format": {
                                                "type": "string",
                                                "default": "json",
                                                "enum": ["json", "text", "srt", "verbose_json", "vtt"],
                                                "description": "The format of the transcript output.",
                                            },
                                        },
                                        "required": ["file"],
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "Successful Response",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {"text": {"type": "string"}},
                                        }
                                    },
                                    "text/plain": {"schema": {"type": "string"}},
                                },
                            }
                        },
                    }
                }
            },
        }
    )


@app.route("/progress/<job_id>")
def get_progress(job_id):
    if job_id in progress_tracker:
        return jsonify(progress_tracker[job_id])
    return jsonify({"status": "not_found"}), 404


@app.route("/status")
def get_status():
    for job_id, progress in progress_tracker.items():
        if progress.get("status") == "processing":
            return jsonify({"job_id": job_id, **progress})
    return jsonify({"status": "idle"})


@app.route("/metrics")
def get_metrics():
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    return jsonify(
        {
            "cpu_percent": cpu_percent,
            "ram_percent": memory.percent,
            "ram_used_gb": round(memory.used / (1024**3), 2),
            "ram_total_gb": round(memory.total / (1024**3), 2),
        }
    )


@app.route("/v1/audio/transcriptions", methods=["POST"])
def transcribe_audio():
    try:
        ensure_runtime_initialized()
    except Exception as exc:
        return jsonify(
            {
                "error": "Runtime initialization failed",
                "details": str(exc),
                "provider_state": runtime_state,
            }
        ), 500

    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected"}), 400

    model_name = request.form.get("model", DEFAULT_MODEL_NAME).lower().strip()
    response_format = request.form.get("response_format", "json")
    print(f"Request Model: {model_name} | Format: {response_format}")

    if model_name not in MODEL_CONFIGS:
        print(f"⚠️ Unknown model '{model_name}' requested, using default")
        model_name = DEFAULT_MODEL_NAME

    model_to_use = get_model(model_name)

    original_filename = secure_filename(file.filename)
    unique_id = str(uuid.uuid4())
    temp_original_path = os.path.join(
        app.config["UPLOAD_FOLDER"], f"{unique_id}_{original_filename}"
    )
    target_wav_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{unique_id}.wav")
    temp_files_to_clean = []

    try:
        file.save(temp_original_path)
        temp_files_to_clean.append(temp_original_path)

        print(f"[{unique_id}] Converting '{original_filename}' to standard WAV format...")
        ffmpeg_command = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            temp_original_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            target_wav_path,
        ]

        result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return jsonify({"error": "File conversion failed", "details": result.stderr}), 500

        temp_files_to_clean.append(target_wav_path)

        chunk_duration_seconds = CHUNK_MINUTE * 60
        total_duration = get_audio_duration(target_wav_path)
        if total_duration == 0:
            return jsonify({"error": "Cannot process audio with 0 duration"}), 400

        chunk_paths = []
        split_points = []

        if total_duration > chunk_duration_seconds:
            print(f"[{unique_id}] Detecting silence points for intelligent chunking...")
            silence_points = detect_silence_points(
                target_wav_path,
                total_duration=total_duration,
            )

            if silence_points:
                print(f"[{unique_id}] Found {len(silence_points)} silence periods")
                split_points = find_optimal_split_points(
                    total_duration,
                    chunk_duration_seconds,
                    silence_points,
                    search_window=SILENCE_SEARCH_WINDOW,
                )
                print(
                    f"[{unique_id}] Optimal split points: "
                    f"{[f'{split_point:.2f}s' for split_point in split_points]}"
                )
            else:
                print(f"[{unique_id}] No silence detected, using time-based chunking")

        if split_points:
            chunk_boundaries = [0.0] + split_points + [total_duration]
            num_chunks = len(chunk_boundaries) - 1
        else:
            num_chunks = math.ceil(total_duration / chunk_duration_seconds)
            chunk_boundaries = [
                min(i * chunk_duration_seconds, total_duration)
                for i in range(num_chunks + 1)
            ]

        progress_tracker[unique_id] = {
            "status": "processing",
            "current_chunk": 0,
            "total_chunks": num_chunks,
            "progress_percent": 0,
            "partial_text": "",
        }

        print(
            f"[{unique_id}] Total duration: {total_duration:.2f}s. "
            f"Splitting into {num_chunks} chunks."
        )

        if num_chunks > 1:
            for i in range(num_chunks):
                start_time = chunk_boundaries[i]
                duration = chunk_boundaries[i + 1] - start_time
                chunk_path = os.path.join(
                    app.config["UPLOAD_FOLDER"], f"{unique_id}_chunk_{i}.wav"
                )
                chunk_paths.append(chunk_path)
                temp_files_to_clean.append(chunk_path)

                print(
                    f"[{unique_id}] Creating chunk {i + 1}/{num_chunks} "
                    f"({start_time:.2f}s - {chunk_boundaries[i + 1]:.2f}s)..."
                )
                chunk_command = [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-ss",
                    str(start_time),
                    "-t",
                    str(duration),
                    "-i",
                    target_wav_path,
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    chunk_path,
                ]
                result = subprocess.run(chunk_command, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Warning: Chunk extraction failed: {result.stderr}")
        else:
            chunk_paths.append(target_wav_path)

        all_segments = []
        all_words = []
        cumulative_time_offset = 0.0

        if num_chunks > 1:
            chunk_durations = [
                chunk_boundaries[i + 1] - chunk_boundaries[i] for i in range(num_chunks)
            ]
        else:
            chunk_durations = [total_duration]

        for i, chunk_path in enumerate(chunk_paths):
            progress_tracker[unique_id].update(
                {
                    "current_chunk": i + 1,
                    "progress_percent": int((i + 1) / num_chunks * 100),
                }
            )
            print(f"[{unique_id}] Transcribing chunk {i + 1}/{num_chunks}...")

            recognition_result = model_to_use.recognize(chunk_path)
            if recognition_result and recognition_result.text:
                start_time = recognition_result.timestamps[0] if recognition_result.timestamps else 0
                end_time = (
                    recognition_result.timestamps[-1]
                    if len(recognition_result.timestamps) > 1
                    else start_time + 0.1
                )

                cleaned_text = clean_transcript_text(recognition_result.text)
                all_segments.append(
                    {
                        "start": start_time + cumulative_time_offset,
                        "end": end_time + cumulative_time_offset,
                        "segment": cleaned_text,
                    }
                )

                progress_tracker[unique_id]["partial_text"] += cleaned_text + " "

                for j, (token, timestamp) in enumerate(
                    zip(recognition_result.tokens, recognition_result.timestamps)
                ):
                    if j < len(recognition_result.timestamps) - 1:
                        word_end = recognition_result.timestamps[j + 1]
                    else:
                        word_end = end_time

                    all_words.append(
                        {
                            "start": timestamp + cumulative_time_offset,
                            "end": word_end + cumulative_time_offset,
                            "word": clean_token_text(token),
                        }
                    )

            cumulative_time_offset += chunk_durations[i]

        print(f"[{unique_id}] All chunks transcribed, merging results.")
        progress_tracker[unique_id]["status"] = "complete"
        progress_tracker[unique_id]["progress_percent"] = 100

        full_text = " ".join([segment["segment"] for segment in all_segments])

        if response_format == "srt":
            return Response(segments_to_srt(all_segments), mimetype="text/plain")

        if response_format == "vtt":
            return Response(segments_to_vtt(all_segments), mimetype="text/plain")

        if response_format == "text":
            return Response(full_text, mimetype="text/plain")

        if response_format == "verbose_json":
            return jsonify(
                {
                    "task": "transcribe",
                    "language": "english",
                    "duration": total_duration,
                    "text": full_text,
                    "segments": [
                        {
                            "id": idx,
                            "seek": 0,
                            "start": segment["start"],
                            "end": segment["end"],
                            "text": segment["segment"],
                            "tokens": [],
                            "temperature": 0.0,
                            "avg_logprob": 0.0,
                            "compression_ratio": 0.0,
                            "no_speech_prob": 0.0,
                        }
                        for idx, segment in enumerate(all_segments)
                    ],
                }
            )

        response = jsonify({"text": full_text})
        response.headers["X-Job-ID"] = unique_id
        return response

    except Exception as exc:
        print(f"A serious error occurred during processing: {exc}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": str(exc)}), 500
    finally:
        print(f"[{unique_id}] Cleaning up temporary files...")
        for file_path in temp_files_to_clean:
            if os.path.exists(file_path):
                os.remove(file_path)
        print(f"[{unique_id}] Temporary files cleaned.")


def openweb():
    import time
    import webbrowser

    time.sleep(5)
    webbrowser.open_new_tab(f"http://127.0.0.1:{port}")


def main():
    print("Starting server...")
    print(f"Web interface: http://127.0.0.1:{port}")
    print(f"API Endpoint: POST http://{host}:{port}/v1/audio/transcriptions")
    print(f"Running with {threads} threads.")

    ensure_runtime_initialized()

    if os.getenv("PARAKEET_OPEN_BROWSER", "false").strip().lower() in {"1", "true", "yes"}:
        print("Starting web browser thread...")
        threading.Thread(target=openweb, daemon=True).start()

    print("Starting waitress server...")
    serve(app, host=host, port=port, threads=threads)


if __name__ == "__main__":
    main()
