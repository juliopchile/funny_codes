"""
Normalize Planview demo audio tracks with FFmpeg before YouTube upload.

Requirements
============

.. code-block:: bash

   ffmpeg -version
   ffprobe -version

Usage
=====

.. code-block:: bash

   # VS Code Play
   python codes/normalize_planview_demos_audio.py

   # CLI
   python codes/normalize_planview_demos_audio.py normalize
   python codes/normalize_planview_demos_audio.py reset-youtube
   python codes/normalize_planview_demos_audio.py all
   python codes/normalize_planview_demos_audio.py normalize --limit 3

Notes
=====

This module is intentionally specific to the Planview JSON files stored in
``out/scrapping``. Video streams are always copied without re-encoding.
Audio must be re-encoded because FFmpeg cannot apply filters while stream-
copying that same audio stream.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


LOGGER = logging.getLogger("normalize_planview_demos_audio")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "out" / "scrapping"
NORMALIZED_VIDEOS_ROOT = OUTPUT_ROOT / "videos_normalized"

RUN_ACTION = "all"  # "normalize", "reset-youtube" or "all"
RUN_LIMIT: int | None = None
RUN_MAX_WORKERS = max(1, min(4, os.cpu_count() or 1))

LOUDNORM_TARGET_I = -16.0
LOUDNORM_TARGET_LRA = 11.0
LOUDNORM_TARGET_TP = -1.5
LOUDNORM_DUAL_MONO = True

DEFAULT_FFPROBE_TIMEOUT_SECONDS = 60
DEFAULT_FFMPEG_TIMEOUT_SECONDS = 60 * 60

NORMALIZATION_FIELD_NAMES = (
    "normalized_video_path",
    "audio_normalized",
    "audio_normalized_at",
    "audio_normalization",
    "audio_normalization_error",
    "source_media_date",
    "source_media_date_source",
    "normalized_media_date",
    "normalized_media_date_source",
)


@dataclass(frozen=True)
class CategoryConfig:
    """Static configuration per JSON store."""

    name: str
    json_path: Path
    normalized_dir: Path


CATEGORY_CONFIGS = (
    CategoryConfig(
        name="product_demos",
        json_path=OUTPUT_ROOT / "product_demos.json",
        normalized_dir=NORMALIZED_VIDEOS_ROOT / "product_demos",
    ),
    CategoryConfig(
        name="solution_demos",
        json_path=OUTPUT_ROOT / "solution_demos.json",
        normalized_dir=NORMALIZED_VIDEOS_ROOT / "solution_demos",
    ),
)


def utc_now_iso() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def normalize_inline_text(value: str | None) -> str:
    """Collapse whitespace for single-line display."""

    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_iso_datetime(value: str | None) -> str | None:
    """Normalize a datetime-like string to ISO-8601 when possible."""

    if not value:
        return None

    text = value.strip()
    candidates = [text]
    if text.endswith("Z"):
        candidates.append(text[:-1] + "+00:00")

    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.isoformat()
        except ValueError:
            continue
    return None


def card_numeric_id(entry: dict[str, Any]) -> int:
    """Return the numeric part of ``card_id`` when possible."""

    match = re.search(r"(\d+)$", normalize_inline_text(entry.get("card_id")))
    return int(match.group(1)) if match else 10**12


def ensure_directories() -> None:
    """Create output directories if missing."""

    NORMALIZED_VIDEOS_ROOT.mkdir(parents=True, exist_ok=True)
    for config in CATEGORY_CONFIGS:
        config.normalized_dir.mkdir(parents=True, exist_ok=True)


def ensure_ffmpeg_binaries() -> None:
    """Ensure ffmpeg and ffprobe are available."""

    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            raise RuntimeError(f"No se encontró '{binary}' en PATH.")


def load_store(config: CategoryConfig) -> dict[str, Any]:
    """Load one JSON store."""

    if not config.json_path.exists():
        raise FileNotFoundError(f"No existe el JSON esperado: {config.json_path}")
    with config.json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_store(config: CategoryConfig, store: dict[str, Any]) -> None:
    """Save one JSON store atomically."""

    temp_path = config.json_path.with_suffix(config.json_path.suffix + ".tmp")
    store["updated_at"] = utc_now_iso()
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(store, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temp_path.replace(config.json_path)


def sort_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort entries by card id for stable processing."""

    return sorted(items, key=card_numeric_id)


def ensure_normalization_fields(entry: dict[str, Any]) -> None:
    """Ensure JSON fields used by the normalization stage exist."""

    entry.setdefault("normalized_video_path", None)
    entry.setdefault("audio_normalized", False)
    entry.setdefault("audio_normalized_at", None)
    entry.setdefault("audio_normalization", None)
    entry.setdefault("audio_normalization_error", None)
    entry.setdefault("source_media_date", None)
    entry.setdefault("source_media_date_source", None)
    entry.setdefault("normalized_media_date", None)
    entry.setdefault("normalized_media_date_source", None)


def clone_normalization_fields(source: dict[str, Any], target: dict[str, Any]) -> None:
    """Copy normalization-related fields between entries."""

    for field in NORMALIZATION_FIELD_NAMES:
        target[field] = source.get(field)


def source_video_key(entry: dict[str, Any]) -> str | None:
    """Return a stable key for the source video path."""

    video_path = entry.get("video_path")
    if not video_path:
        return None
    path = Path(str(video_path)).expanduser()
    if not path.exists():
        return None
    return str(path.resolve())


def clear_youtube_fields(entry: dict[str, Any]) -> bool:
    """Reset YouTube-related fields so uploads start from zero."""

    changed = False
    defaults = {
        "youtube_uploaded": False,
        "youtube_video_id": None,
        "youtube_video_url": None,
        "youtube_playlist_id": None,
        "youtube_playlist_item_id": None,
        "youtube_uploaded_at": None,
        "youtube_upload_error": None,
        "youtube_title_used": None,
        "youtube_description_used": None,
        "youtube_privacy_status": None,
    }
    for key, value in defaults.items():
        if entry.get(key) != value:
            entry[key] = value
            changed = True
    return changed


def run_subprocess(
    command: list[str],
    *,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and return the completed process."""

    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def ffprobe_json(video_path: Path) -> dict[str, Any]:
    """Probe one media file with ffprobe."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    result = run_subprocess(command, timeout_seconds=DEFAULT_FFPROBE_TIMEOUT_SECONDS)
    if result.returncode != 0:
        raise RuntimeError(normalize_inline_text(result.stderr or result.stdout or "ffprobe failed"))
    return json.loads(result.stdout)


def extract_media_date(probe: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extract an embedded media date from container or streams if available."""

    candidate_paths = [
        ("format.tags.creation_time", probe.get("format", {}).get("tags", {}).get("creation_time")),
        ("format.tags.date", probe.get("format", {}).get("tags", {}).get("date")),
        (
            "format.tags.com.apple.quicktime.creationdate",
            probe.get("format", {}).get("tags", {}).get("com.apple.quicktime.creationdate"),
        ),
        ("format.tags.encoded_date", probe.get("format", {}).get("tags", {}).get("encoded_date")),
    ]

    for index, stream in enumerate(probe.get("streams", []) or []):
        tags = stream.get("tags", {}) if isinstance(stream, dict) else {}
        candidate_paths.extend(
            [
                (f"streams[{index}].tags.creation_time", tags.get("creation_time")),
                (f"streams[{index}].tags.date", tags.get("date")),
                (f"streams[{index}].tags.encoded_date", tags.get("encoded_date")),
                (f"streams[{index}].tags.timecode", tags.get("timecode")),
            ]
        )

    for source, raw_value in candidate_paths:
        normalized = parse_iso_datetime(raw_value)
        if normalized:
            return normalized, source
    return None, None


def extract_audio_stream_info(probe: dict[str, Any]) -> dict[str, Any]:
    """Return the first audio stream information."""

    for stream in probe.get("streams", []) or []:
        if isinstance(stream, dict) and stream.get("codec_type") == "audio":
            return stream
    raise ValueError("El archivo no tiene stream de audio.")


def derive_audio_encode_options(audio_stream: dict[str, Any]) -> list[str]:
    """Choose practical AAC encoding options for normalized outputs."""

    bit_rate_text = normalize_inline_text(audio_stream.get("bit_rate"))
    bit_rate = int(bit_rate_text) if bit_rate_text.isdigit() else None
    if bit_rate is None:
        bit_rate = 192_000
    bit_rate = max(96_000, min(bit_rate, 320_000))
    return ["-c:a", "aac", "-b:a", str(bit_rate)]


def build_loudnorm_measure_filter() -> str:
    """Return the first-pass loudnorm filter."""

    parts = [
        f"I={LOUDNORM_TARGET_I}",
        f"LRA={LOUDNORM_TARGET_LRA}",
        f"TP={LOUDNORM_TARGET_TP}",
        "print_format=json",
    ]
    if LOUDNORM_DUAL_MONO:
        parts.append("dual_mono=true")
    return "loudnorm=" + ":".join(parts)


def extract_loudnorm_measurements(stderr_text: str) -> dict[str, str]:
    """Parse the JSON block printed by loudnorm in pass one."""

    matches = list(re.finditer(r"\{\s*\"input_i\".*?\}", stderr_text, flags=re.S))
    if not matches:
        raise ValueError("No se encontró la salida JSON de loudnorm en la primera pasada.")
    payload = json.loads(matches[-1].group(0))
    required = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Faltan medidas de loudnorm: {', '.join(missing)}")
    return {key: str(payload[key]) for key in payload}


def build_loudnorm_apply_filter(measurements: dict[str, str]) -> str:
    """Return the second-pass loudnorm filter."""

    parts = [
        f"I={LOUDNORM_TARGET_I}",
        f"LRA={LOUDNORM_TARGET_LRA}",
        f"TP={LOUDNORM_TARGET_TP}",
        f"measured_I={measurements['input_i']}",
        f"measured_LRA={measurements['input_lra']}",
        f"measured_TP={measurements['input_tp']}",
        f"measured_thresh={measurements['input_thresh']}",
        f"offset={measurements['target_offset']}",
        "linear=true",
        "print_format=summary",
    ]
    if LOUDNORM_DUAL_MONO:
        parts.append("dual_mono=true")
    return "loudnorm=" + ":".join(parts)


def normalize_output_path(config: CategoryConfig, source_path: Path) -> Path:
    """Return the normalized output path for one source video."""

    return config.normalized_dir / source_path.name


def should_skip_normalization(entry: dict[str, Any]) -> bool:
    """Return True when normalized output already exists."""

    normalized_path = entry.get("normalized_video_path")
    return bool(normalized_path and Path(normalized_path).exists())


def measure_loudness(input_path: Path) -> dict[str, str]:
    """Run the first loudnorm pass."""

    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(input_path),
        "-vn",
        "-sn",
        "-dn",
        "-af",
        build_loudnorm_measure_filter(),
        "-f",
        "null",
        "-",
    ]
    result = run_subprocess(command, timeout_seconds=DEFAULT_FFMPEG_TIMEOUT_SECONDS)
    if result.returncode != 0:
        raise RuntimeError(normalize_inline_text(result.stderr or result.stdout or "ffmpeg first pass failed"))
    return extract_loudnorm_measurements(result.stderr)


def normalize_single_video(config: CategoryConfig, entry: dict[str, Any]) -> Path:
    """Normalize one video audio track while copying video streams."""

    input_path = Path(str(entry["video_path"])).expanduser().resolve()
    output_path = normalize_output_path(config, input_path)
    temp_output_path = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)

    probe = ffprobe_json(input_path)
    source_media_date, source_media_date_source = extract_media_date(probe)
    audio_stream = extract_audio_stream_info(probe)
    measurements = measure_loudness(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if temp_output_path.exists():
        temp_output_path.unlink()

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-nostats",
        "-i",
        str(input_path),
        "-map",
        "0",
        "-map_metadata",
        "0",
        "-map_chapters",
        "0",
        "-c:v",
        "copy",
        "-c:s",
        "copy",
        "-movflags",
        "+faststart",
        "-af",
        build_loudnorm_apply_filter(measurements),
        *derive_audio_encode_options(audio_stream),
        str(temp_output_path),
    ]
    result = run_subprocess(command, timeout_seconds=DEFAULT_FFMPEG_TIMEOUT_SECONDS)
    if result.returncode != 0:
        raise RuntimeError(normalize_inline_text(result.stderr or result.stdout or "ffmpeg second pass failed"))

    temp_output_path.replace(output_path)

    normalized_probe = ffprobe_json(output_path)
    normalized_media_date, normalized_media_date_source = extract_media_date(normalized_probe)

    entry["normalized_video_path"] = str(output_path.resolve())
    entry["audio_normalized"] = True
    entry["audio_normalized_at"] = utc_now_iso()
    entry["audio_normalization"] = {
        "method": "ffmpeg_loudnorm_two_pass",
        "target_i": LOUDNORM_TARGET_I,
        "target_lra": LOUDNORM_TARGET_LRA,
        "target_tp": LOUDNORM_TARGET_TP,
        "dual_mono": LOUDNORM_DUAL_MONO,
        "source_audio_codec": audio_stream.get("codec_name"),
        "source_audio_bit_rate": audio_stream.get("bit_rate"),
        "measurements": measurements,
    }
    entry["audio_normalization_error"] = None
    entry["source_media_date"] = source_media_date
    entry["source_media_date_source"] = source_media_date_source
    entry["normalized_media_date"] = normalized_media_date or source_media_date
    entry["normalized_media_date_source"] = normalized_media_date_source or source_media_date_source
    return output_path


def normalize_entry_worker(config: CategoryConfig, entry_snapshot: dict[str, Any]) -> dict[str, Any]:
    """Normalize one entry in a worker thread and return updated fields."""

    ensure_normalization_fields(entry_snapshot)
    normalize_single_video(config, entry_snapshot)
    return entry_snapshot


def set_normalization_error(entry: dict[str, Any], exc: Exception) -> None:
    """Store a structured normalization error."""

    entry["audio_normalization_error"] = {
        "timestamp": utc_now_iso(),
        "type": exc.__class__.__name__,
        "message": normalize_inline_text(str(exc)),
    }


def normalize_category(config: CategoryConfig, *, limit: int | None, max_workers: int) -> None:
    """Normalize one category of videos."""

    store = load_store(config)
    items = [
        entry
        for entry in sort_items(store.get("items", []))
        if entry.get("video_path") and Path(str(entry["video_path"])).exists()
    ]
    if limit is not None:
        items = items[:limit]

    processed = len(items)
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in items:
        ensure_normalization_fields(entry)
        key = source_video_key(entry)
        if key is None:
            continue
        groups.setdefault(key, []).append(entry)

    if processed == 0:
        LOGGER.info("%s: no hay videos locales para normalizar.", config.name)
        return

    pending_futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for source_key, grouped_entries in groups.items():
            representative = grouped_entries[0]
            if should_skip_normalization(representative):
                for sibling in grouped_entries[1:]:
                    clone_normalization_fields(representative, sibling)
                save_store(config, store)
                continue

            future = executor.submit(normalize_entry_worker, config, dict(representative))
            pending_futures[future] = (source_key, grouped_entries)

        for future in as_completed(pending_futures):
            _, grouped_entries = pending_futures[future]
            representative = grouped_entries[0]
            try:
                normalized_entry = future.result()
                clone_normalization_fields(normalized_entry, representative)
                for sibling in grouped_entries[1:]:
                    clone_normalization_fields(normalized_entry, sibling)
                output_path = Path(str(normalized_entry["normalized_video_path"]))
                LOGGER.info("%s: audio normalizado para %s -> %s", config.name, representative.get("card_id"), output_path.name)
            except Exception as exc:
                set_normalization_error(representative, exc)
                for sibling in grouped_entries[1:]:
                    set_normalization_error(sibling, exc)
                LOGGER.warning("%s: fallo normalizando %s: %s", config.name, representative.get("card_id"), exc)
            finally:
                save_store(config, store)


def normalize_all_videos(*, limit: int | None = None, max_workers: int) -> None:
    """Normalize audio for all local videos found in the stores."""

    ensure_ffmpeg_binaries()
    ensure_directories()
    for config in CATEGORY_CONFIGS:
        normalize_category(config, limit=limit, max_workers=max_workers)


def reset_youtube_state() -> None:
    """Clear all YouTube upload markers in both JSON stores."""

    for config in CATEGORY_CONFIGS:
        store = load_store(config)
        changed = False
        for entry in store.get("items", []):
            ensure_normalization_fields(entry)
            changed = clear_youtube_fields(entry) or changed
        if isinstance(store.get("stats"), dict) and "youtube" in store["stats"]:
            store["stats"].pop("youtube", None)
            changed = True
        if changed:
            save_store(config, store)
        LOGGER.info("%s: estado de YouTube reiniciado.", config.name)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Normaliza audio de demos de Planview antes de subirlos a YouTube.")
    parser.add_argument(
        "action",
        nargs="?",
        choices=("normalize", "reset-youtube", "all"),
        default=None,
        help="Acción a ejecutar.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Procesa solo los primeros N items por categoría.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Cantidad máxima de procesos FFmpeg concurrentes por categoría.",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit debe ser un entero positivo.")
    if args.max_workers is not None and args.max_workers <= 0:
        parser.error("--max-workers debe ser un entero positivo.")
    return args


def main(argv: list[str] | None = None) -> int:
    """Program entry point."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    effective_argv = list(sys.argv[1:] if argv is None else argv)

    if effective_argv:
        args = parse_args(effective_argv)
        action = args.action or "all"
        limit = args.limit
        max_workers = args.max_workers or RUN_MAX_WORKERS
    else:
        if RUN_ACTION not in {"normalize", "reset-youtube", "all"}:
            raise ValueError("RUN_ACTION debe ser 'normalize', 'reset-youtube' o 'all'.")
        if RUN_LIMIT is not None and RUN_LIMIT <= 0:
            raise ValueError("RUN_LIMIT debe ser None o un entero positivo.")
        if RUN_MAX_WORKERS <= 0:
            raise ValueError("RUN_MAX_WORKERS debe ser un entero positivo.")
        action = RUN_ACTION
        limit = RUN_LIMIT
        max_workers = RUN_MAX_WORKERS

    LOGGER.info(
        "Ejecutando action=%s limit=%s max_workers=%s normalized_root=%s",
        action,
        limit,
        max_workers,
        NORMALIZED_VIDEOS_ROOT,
    )

    if action in {"normalize", "all"}:
        normalize_all_videos(limit=limit, max_workers=max_workers)
    if action in {"reset-youtube", "all"}:
        reset_youtube_state()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
