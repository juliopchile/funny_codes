#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ffmpeg_compatibility_tester.py

Este script automatiza la prueba de compatibilidad de diferentes combinaciones 
de códecs de video, códecs de audio y contenedores usando FFmpeg.

Propósito y funcionamiento:
--------------------------------
La finalidad de este código es ayudar a identificar qué combinaciones de 
códecs y contenedores funcionan correctamente para un archivo de entrada dado 
y cuáles fallan, guardando evidencia detallada de cada intento.

Esto es útil para:
- Pruebas de compatibilidad de flujos de trabajo audiovisuales.
- Verificación de soporte de formatos en FFmpeg instalado.
- Análisis rápido de rendimiento y éxito/fallo de transcodificaciones.
- Documentar qué formatos funcionan en un entorno concreto.

El proceso general es:
1. Se carga un archivo de video de entrada desde la carpeta `ffmpeg_test`.
2. Se generan todas las combinaciones posibles entre:
   - Códecs de video definidos.
   - Códecs de audio definidos.
   - Contenedores definidos.
3. Se ejecuta FFmpeg para cada combinación.
4. Se registran resultados como:
   - Código de retorno de FFmpeg.
   - Tamaño del archivo generado.
   - Estado (éxito/fallo).
   - Errores capturados.
5. Guardado JSON y CSV robusto (CSV con quoting=QUOTE_ALL)..
6. Si la combinación ya fue probada, se puede omitir para ahorrar tiempo.
7. Los logs y resultados se guardan organizados:
   - Videos generados: `ffmpeg_test/bin/out`
   - Logs, errores y CSV/JSON: `ffmpeg_test/bin/logs`
   - Reporte Markdown (opcional): `ffmpeg_test/results.md`

Estructura esperada (por defecto):
- work dir: ./ffmpeg_test
  - input video esperado: ./ffmpeg_test/video.mp4
  - out files: ./ffmpeg_test/bin/out/
- logs & results: ./ffmpeg_test_bin/logs/
- results.csv & results.json -> en logs dir
- Se ejecuta desde un IDE al presionar "Run" (usa SCRIPT_CONFIG para editar)

Autor: Asistente
Fecha: 2025-08-15
"""
from __future__ import annotations

import csv
import itertools
import json
import logging
import os
import subprocess
import sys
import time
import pandas as pd
import numpy as np
import numpy.typing as npt
from datetime import datetime
from pathlib import Path
from collections.abc import Iterable
from typing import TypedDict


# * -----------------------------
# * CLASES DE DICCIONARIOS
# * -----------------------------
class Container(TypedDict):
    muxer: str
    ext: str

class ScriptConfig(TypedDict):
    work_dir: str
    input_filename: str
    m_list: str
    duration: int
    timeout: int
    skip_done: bool
    ffmpeg: str | Path
    video_encoders: dict[str, str]
    audio_encoders: dict[str, str]
    containers: dict[str, Container]
    known_impossible: list[tuple[str, str, str]]
    video_codecs_to_test: list
    audio_codecs_to_test: list
    containers_to_test: list

class PathsDict(TypedDict):
    work_dir: Path
    input_file: Path
    out_dir: Path
    logs_dir: Path
    results_json: Path
    results_csv: Path
    results_md: Path
    experiment_log: Path

class OutDict(TypedDict):
    v_list: list[str]
    a_list: list[str]
    m_list: list[str]
    valid_tuples: list[tuple[str, str, str]]
    nested: dict[str, dict[str, list[str]]]

# ? -----------------------------
# ? CONFIGURACIÓN EMBEBIDA
# ? -----------------------------
SCRIPT_CONFIG: ScriptConfig = {
    "work_dir": "ffmpeg_test",          # carpeta de trabajo
    "input_filename": "video.mp4",      # video dentro de work_dir
    "m_list": "ffmpeg",                 # ruta al binario ffmpeg
    "duration": 0,     # largo del video a usar (0 = todo)
    "timeout": 10,     # timeout por proceso ffmpeg (s)
    "skip_done": True,  # si saltarse los experimentos ya realizados
    "ffmpeg": "ffmpeg",
    "video_encoders": {
        # H.264
        "libx264": "libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)",
        "h264_nvenc": "NVIDIA NVENC H.264 encoder (codec h264)",
        # H.265 o HEVC
        "libx265": "libx265 H.265 / HEVC (codec hevc)",
        "hevc_nvenc": "NVIDIA NVENC hevc encoder (codec hevc)",
        # AV1
        "av1_vaapi": "AV1 (VAAPI) (codec av1)",
        "libaom-av1": "libaom AV1 (codec av1)",
        "av1_nvenc": "NVIDIA NVENC av1 encoder (codec av1)",
        # Others
        "vp9_vaapi": "VP9 (VAAPI) (codec vp9)",
        "libvpx-vp9": "VP9 (libvpx-vp9)",
        "prores_ks": "Apple ProRes (prores_ks)",
        "dnxhd": "VC3/DNxHD",
        "mpeg4": "MPEG-4 part 2",
        "jpeg2000": "JPEG 2000",
        "libopenjpeg": "OpenJPEG JPEG 2000 (codec jpeg2000)"
    },
    "audio_encoders": {
        "aac": "AAC (Advanced Audio Coding)",
        "flac": "FLAC (Free Lossless Audio Codec)",
        "libopus": "libopus Opus (codec opus)",
        "libvorbis": "libvorbis (codec vorbis)",
        "pcm_s24le": "PCM signed 24-bit little-endian (used in wav)"
    },
    "containers": {
        "avi": {"muxer": "avi", "ext": "avi"},
        "matroska": {"muxer": "matroska", "ext": "mkv"},
        "mov": {"muxer": "mov", "ext": "mov"},
        "mp4": {"muxer": "mp4", "ext": "mp4"},
        "mxf": {"muxer": "mxf", "ext": "mxf"},
        # Incompatible with Davinci Resolve (ignorados)
        #"webm": {"muxer": "webm", "ext": "webm"},
        #"mpeg": {"muxer": "mpeg", "ext": "mpg"},
        #"ogg": {"muxer": "ogg", "ext": "ogv"}
    },
    # combos ya sabidos que no conviene probar (puedes añadir)
    # ej: [("libvpx-vp9","aac","webm")]
    "known_impossible": [],
    # TODO Optional: tests subset
    "video_codecs_to_test": [],
    "audio_codecs_to_test": [],
    "containers_to_test": []
}

# ? -----------------------------
# ? UTILIDADES
# ? -----------------------------
def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)


def _run_subprocess(cmd: list[str], timeout: int | None = None) -> dict[str, object]:
    """Ejecuta comando y captura stdout/stderr/rc/duración; maneja timeouts."""
    start = time.time_ns()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        rc = proc.returncode
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired as ex:
        rc = -1
        stdout = f"{ex.stdout or ""}"
        stderr = f"{ex.stderr or ""} + \n[timeout after {timeout}s]"
    except Exception as ex:
        rc = -2
        stdout = ""
        stderr = str(ex)
    end = time.time_ns()
    duration_ms = (end - start) / 1000
    return {"rc": rc, "stdout": stdout, "stderr": stderr, "duration_ms": duration_ms / 1000}


# ? -----------------------------
# ? RUTAS / PREPARACIÓN
# ? -----------------------------
def _prepare_paths(work_dir: str | Path) -> PathsDict:
    work = Path(work_dir)
    input_file = work / SCRIPT_CONFIG.get("input_filename", "video.mp4")
    out_dir = work / "bin" / "out"
    logs_dir = work / "bin" / "logs"
    results_json = logs_dir / "results.json"
    results_csv = logs_dir / "results.csv"
    results_md = work / "results.md"
    experiment_log = logs_dir / "experiment.log"
    return {
        "work_dir": work,
        "input_file": input_file,
        "out_dir": out_dir,
        "logs_dir": logs_dir,
        "results_json": results_json,
        "results_csv": results_csv,
        "results_md": results_md,
        "experiment_log": experiment_log
    }


def _ensure_dirs(paths: PathsDict | dict[str, Path]) -> None:
    paths["work_dir"].mkdir(parents=True, exist_ok=True)
    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    paths["logs_dir"].mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(paths["experiment_log"], mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Directorios preparados.")
    logging.info(f" Work dir : {paths['work_dir']}")
    logging.info(f" Input    : {paths['input_file']}")
    logging.info(f" Out dir  : {paths['out_dir']}")
    logging.info(f" Logs dir : {paths['logs_dir']}")


# ? -----------------------------
# ? Detección de experimentos existentes
# ? -----------------------------
def _parse_result_filename(filename: str, input_stem: str) -> tuple[str, str, str] | None:
    """
    Parsea nombres con patrón:
    {input_stem}__{vcodec}__{acodec}__{container_key}__{timestamp}.{ext}
    Devuelve (vcodec, acodec, container_key) o None si no concuerda.
    """
    stem = Path(filename).stem
    parts = stem.split("__")
    # Esperamos al menos 5 partes: inputstem, vcodec, acodec, container, ts
    if len(parts) < 5:
        return None
    if parts[0] != input_stem:
        return None
    vcodec = parts[1]
    acodec = parts[2]
    container_key = parts[3]
    return (vcodec, acodec, container_key)


def _scan_existing_outs(out_dir: Path, input_stem: str) -> set:
    """
    Escanea out_dir y devuelve set de tuplas (vcodec, acodec, container_key)
    correspondientes a experimentos ya realizados (archivo presente y tamaño > 0).
    """
    done = set()
    if not out_dir.exists():
        return done
    for p in out_dir.iterdir():
        if not p.is_file():
            continue
        if p.stat().st_size == 0:
            continue
        parsed = _parse_result_filename(p.name, input_stem)
        if parsed:
            done.add(parsed)
    logging.info(f"Detectados {len(done)} experimentos ya realizados en {out_dir}")
    return done


# ? -----------------------------
# ? Prueba de combinación
# ? -----------------------------
def _generate_output_path(
    out_dir: Path,
    input_stem: str,
    vcodec: str,
    acodec: str,
    container_key: str,
    ext: str) -> Path:
    name = f"{input_stem}__{vcodec}__{acodec}__{container_key}__{_now_ts()}.{ext}"
    return out_dir / _safe_filename(name)


def _test_combination(
    ffmpeg_bin: str,
    input_file: Path,
    outpath: Path,
    vcodec: str,
    acodec: str,
    muxer: str,
    test_duration: int = 0,
    timeout: int = 600) -> dict:
    cmd = [ffmpeg_bin, "-y", "-v", "error", "-i", str(input_file)]
    if test_duration and test_duration > 0:
        cmd += ["-t", str(test_duration)]
    cmd += ["-c:v", vcodec, "-c:a", acodec]
    cmd += ["-f", muxer, str(outpath)]

    logging.info(f"Ejecutando ffmpeg: {' '.join(cmd)}")
    res = _run_subprocess(cmd, timeout=timeout)
    out_exists = outpath.exists() and outpath.stat().st_size > 0
    out_size = outpath.stat().st_size if out_exists else 0

    return {
        "timestamp": datetime.now().isoformat(),
        "input": str(input_file),
        "output": str(outpath),
        "vcodec": vcodec,
        "acodec": acodec,
        "muxer": muxer,
        "rc": res["rc"],
        "stdout": res["stdout"],
        "stderr": res["stderr"],
        "duration_ms": round(res["duration_ms"]),
        "output_exists": out_exists,
        "output_size": out_size
    }


# ? -----------------------------
# ? Guardado robusto (CSV/JSON)
# ? -----------------------------
def _save_json(results: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def _save_csv(results: list[dict], path: Path) -> None:
    if not results:
        return
    # columnas base
    base_keys = ["timestamp", "input", "output", "vcodec", "acodec",
                 "muxer", "rc", "duration_ms", "output_exists",
                 "output_size", "stderr"]
    extra = [k for k in results[0].keys() if k not in base_keys]
    keys = base_keys + extra
    # Abrir con newline='' y quoting=QUOTE_ALL para seguridad con caracteres especiales
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for r in results:
            # asegurar que todas las claves existan
            row = {k: (r.get(k, "") if r.get(k, "") is not None else "") for k in keys}
            writer.writerow(row)


# ? -----------------------------
# ? Mostrar CSV (leer y mostrar) - en vez de crear MD
# ? -----------------------------
def _read_csv_and_print(path: Path, max_rows: int = 200) -> None:
    """
    Lee results.csv con csv.DictReader y muestra una tabla con columnas seleccionadas.
    Trunca/sanitiza campos largos (stderr).
    """
    if not path.exists():
        logging.warning(f"CSV no encontrado: {path}")
        return

    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(r)

    if not rows:
        print("No hay filas en CSV para mostrar.")
        return

    # Columnas a mostrar y anchos
    cols = ["vcodec", "acodec", "muxer", "rc", "output_exists", "output_size", "stderr"]
    col_widths: dict[str, int] = {}
    for c in cols:
        max_len = max(len(c), *(len(str(r.get(c, ""))) for r in rows))
        col_widths[c] = min(max_len, 40)  # limitar ancho máximo para consola

    # Cabecera
    header_line = " | ".join(c.ljust(col_widths[c]) for c in cols)
    sep_line = "-+-".join("-" * col_widths[c] for c in cols)
    print(header_line)
    print(sep_line)

    # Filas
    for r in rows:
        parts = []
        for c in cols:
            v = r.get(c, "")
            if c == "stderr":
                # mostrar sólo las primeras 2 líneas y truncar
                v = (v or "").splitlines()[:2]
                v = " / ".join(line.strip() for line in v)
                if len(v) > col_widths[c]:
                    v = v[: col_widths[c] - 3] + "..."
            else:
                s = str(v)
                if len(s) > col_widths[c]:
                    s = s[: col_widths[c] - 3] + "..."
                v = s
            parts.append(v.ljust(col_widths[c]))
        print(" | ".join(parts))


# ? -----------------------------
# ? Normalizar known_impossible
# ? -----------------------------
def _normalize_known_impossible(raw: dict[str | tuple | list, str]) -> dict[tuple[str, ...], str]:
    out = {}
    for k, v in (raw or {}).items():
        if isinstance(k, (list, tuple)):
            keyt = tuple(str(x) for x in k)
        elif isinstance(k, str):
            parts = [p.strip() for p in k.split(",")]
            keyt = tuple(parts)
        else:
            keyt = (str(k),)
        out[keyt] = v
    return out


# ? -----------------------------
# ? Crear Markdown los resultados
# ? -----------------------------
def _create_csv_markdown(paths: PathsDict | dict[str, Path]):
    """
    Crea un archivo markdown con tabla pura de markdown desde un CSV usando pandas.
    
    Args:
        paths (dict[str, Path]): Diccionario con los paths necesarios
        
    Raises:
        FileNotFoundError: Si el archivo CSV no existe
        ValueError: Si el archivo CSV está vacío
        IOError: Si hay problemas de lectura/escritura
    """
    csv_path, results_md = paths["results_csv"], paths["results_md"]
    df = load_csv_to_pandas(csv_path)
    try:
        # Generar el contenido markdown
        markdown_content = generate_markdown_table(df, paths)
        # Escribir el archivo markdown
        with open(results_md, 'w', encoding='utf-8') as mdfile:
            mdfile.write(markdown_content)
        logging.info(f"✅ Archivo markdown creado exitosamente: {results_md}")
        logging.info(f"📊 Se procesaron {len(df)} filas de datos")
    except Exception as e:
        logging.error(f"❌ Error al procesar el archivo: {e}")
        raise

def generate_markdown_table(df: pd.DataFrame, paths: PathsDict | dict[str, Path]) -> str:
    """Genera una tabla de markdown pura con las columnas seleccionadas y renombradas.
       Envuelve la columna stderr en <div class="cell-truncate">...</div> para truncado por CSS.
    """
    if df.empty:
        return "# CSV Viewer\n\nNo hay datos para mostrar."

    # Definir las columnas a mostrar y sus nuevos nombres
    columns_mapping = {
        "vcodec": "V-Codec",
        "acodec": "A-Codec", 
        "muxer": "Muxer",
        "rc": "rc",
        "duration_ms": "Time (ms)",
        "output_exists": "Output",
        "output_size": "Bytes",
        "stderr": "stderr"
    }
    
    # Filtrar y renombrar columnas
    available_columns = [col for col in columns_mapping.keys() if col in df.columns]
    if not available_columns:
        return "# Tabla de compatibilidades\n\nNo se encontraron las columnas esperadas en el CSV."
    
    # Seleccionar columnas disponibles y renombrarlas
    df_filtered = df[available_columns].copy()
    df_filtered = df_filtered.rename(columns={col: columns_mapping[col] for col in available_columns})
    
    # Comenzar el markdown
    content = "# Tabla de compatibilidades\n\n"
    
    # Crear tabla de markdown
    if len(df_filtered) > 0:
        # Headers
        headers = df_filtered.columns.tolist()
        content += "| " + " | ".join(escape_markdown(str(header)) for header in headers) + " |\n"
        # Separador
        content += "|" + "|".join(" --- " for _ in headers) + "|\n"
        # Filas de datos
        for _, row in df_filtered.iterrows():
            cells = []
            for col, cell in row.items():
                cell_text = escape_markdown(str(cell))
                if col == "stderr":
                    # envolver en <div class="cell-truncate">...</div>
                    cell_text = f'<div class="cell-truncate">{cell_text}</div>'
                cells.append(cell_text)
            content += "| " + " | ".join(cells) + " |\n"
    
    # Footer con rutas
    content += "\n---\n\nRutas importantes:\n"
    content += f" - Work dir : {paths['work_dir']}\n"
    content += f" - Input    : {paths['input_file']}\n"
    content += f" - Out dir  : {paths['out_dir']}\n"
    content += f" - Logs dir : {paths['logs_dir']}\n"
    content += f" - CSV      : {paths['results_csv']}\n"
    content += f" - JSON     : {paths['results_json']}\n"
    return content


def escape_markdown(text):
    """Escapa caracteres especiales de markdown"""
    if pd.isna(text) or text == '':
        return ""
    text = str(text)
    # Escapar caracteres especiales de markdown
    special_chars = ['|', '\\', '`', '*', '_', '{', '}', '[', ']', '(', ')', '#', '+', '-', '.', '!']
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    # Reemplazar saltos de línea con <br> para permitirlos en tablas
    text = text.replace('\n', '<br>').replace('\r', '<br>')
    return text


def load_csv_to_pandas(csv_path: Path) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        error_msg = f"No se encontró el archivo CSV en {csv_path}"
        logging.error(f"❌ Error: {error_msg}")
        raise FileNotFoundError(error_msg)
    # Cargar el archivo CSV
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except Exception as e:
        logging.error(f"❌ Error al procesar el archivo: {e}")
        raise
    # Sanity check
    if df.empty:
        error_msg = "El archivo CSV está vacío"
        logging.error(f"❌ Error: {error_msg}")
        raise ValueError(error_msg)
    return df


# ? -----------------------------
# ? GUARDAR TABLA DE COMPATIBILIDAD
# ? -----------------------------
def _save_compatibility_results(csv_path: Path | str, out_path: Path | str):
    """
    Guarda los resultados compatibilidad entre encoders (v/a) y muxers
    para ffmpeg. Se cargan los datos de un archivo CSV y se guarda una
    lookup table en forma de arreglo Numpy y archivo JSON.

    :param Path | str csv_path: CSV con resultados del experimento.
    :param Path | str out_path: Directorio donde guardar las tablas.
    """
    # Manejar el pathing de los archivos
    csv_path = Path(csv_path).absolute()
    out_path = Path(out_path).absolute()
    npy_file = out_path / "ffmpeg_compatibility.npy"
    json_file = out_path / "ffmpeg_compatibility.json"

    # Cagar datagrama y añadir columna de validez
    df = load_csv_to_pandas(csv_path)
    df = create_valid_columns(df)

    # Obtener lookup table y diccionario de relaciones para guardarlo
    mask, out = make_lookup_tables(df)

    # Guardar los datos para su uso futuro
    np.save(npy_file, mask)
    json_file.write_text(json.dumps(out, indent=2, ensure_ascii=False))


def create_valid_columns(df: pd.DataFrame, min_size: int = 10000) -> pd.DataFrame:
    df = df.copy()
    exists = df["output_exists"].fillna(False).astype(bool)
    size = pd.to_numeric(df["output_size"], errors="coerce").fillna(0)
    rc = pd.to_numeric(df["rc"], errors="coerce").fillna(1)
    stderr = df["stderr"].fillna("").astype(str).str.strip()

    mask_no_output = ~exists
    mask_small_size = size < min_size
    mask_nonzero_rc = rc != 0
    mask_stderr_nonem = stderr != ""

    df["valid"] = ~(
        mask_no_output | mask_small_size | mask_nonzero_rc | mask_stderr_nonem
    )
    df["invalid_reason"] = np.select(
        [mask_no_output, mask_small_size, mask_nonzero_rc, mask_stderr_nonem],
        ["no_output", "small_size", "nonzero_rc", "stderr_nonempty"],
        default="ok",
    )
    return df

def make_lookup_tables(df: pd.DataFrame,
    ) -> tuple[npt.NDArray[np.bool_], OutDict]:
    
    # IDs y tensor numpy boolean para lookup ultra-rápido / vectorizado
    v_list = sorted(df['vcodec'].unique())
    a_list = sorted(df['acodec'].unique())
    m_list = sorted(df['muxer'].unique())
    v2i = {v:i for i,v in enumerate(v_list)}
    a2i = {a:i for i,a in enumerate(a_list)}
    m2i = {m:i for i,m in enumerate(m_list)}
    # Tensor que sirve de lookup table
    mask = np.zeros((len(v_list), len(a_list), len(m_list)), dtype=np.bool_)
    for v,a,m,ok in df[['vcodec','acodec','muxer','valid']].itertuples(index=False):
        mask[v2i[v], a2i[a], m2i[m]] = bool(ok)

    # Estructura set de tuplas (vcodec, acodec, muxer) válidas
    valid_set = set(tuple(x) for x in df[df['valid']][['vcodec','acodec','muxer']].values)
    # Nested dict: vcodec -> acodec -> set(muxers válidos)
    nested = {}
    for v,a,m,ok in df[['vcodec','acodec','muxer','valid']].itertuples(index=False):
        nested.setdefault(v, {}).setdefault(a, set())
        if ok:
            nested[v][a].add(m)
    # Estructura resumida para uso posterior
    out: OutDict = {
        "v_list": v_list,
        "a_list": a_list,
        "m_list": m_list,
        "valid_tuples": sorted(list(valid_set)),
        "nested": {v: {a: sorted(list(ms)) for a,ms in nested[v].items()} for v in nested},
    }

    return mask, out


# ? -----------------------------
# ? MAIN
# ? -----------------------------
def main(argv: Iterable[str] | None = None) -> None:
    # No es obligatorio pasar args; script usa SCRIPT_CONFIG por defecto (útil para IDE)
    import argparse
    parser = argparse.ArgumentParser(description="FFmpeg compatibility tester (ffmpeg_test).")
    parser.add_argument("--ffmpeg", help="ruta al binario ffmpeg (opcional)")
    parser.add_argument("--duration", type=int, help="duración (s) para pruebas rápidas (opcional)")
    parser.add_argument("--timeout", type=int, help="timeout (s) por proceso (opcional)")
    parser.add_argument("--skip_done", type=bool, help="saltar experimentos realizados (opcional)")
    parser.add_argument("--config", help="archivo JSON para sobrescribir SCRIPT_CONFIG completo (opcional)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = SCRIPT_CONFIG
    if args.config:
        cfg_path = Path(args.config)
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                newcfg = json.load(f)
            cfg.update(newcfg)

    if args.ffmpeg:
        cfg["ffmpeg"] = args.ffmpeg
    if args.duration is not None:
        cfg["duration"] = args.duration
    if args.timeout is not None:
        cfg["timeout"] = args.timeout

    paths = _prepare_paths(cfg["work_dir"])
    _ensure_dirs(paths)

    input_file = paths["input_file"]
    if not input_file.exists():
        logging.error(f"Archivo de entrada NO encontrado: {input_file}")
        print(f"ERROR: archivo de entrada no encontrado: {input_file}")
        sys.exit(1)

    # Inferir keys a probar
    video_map = cfg.get("video_encoders", {}) or {}
    audio_map = cfg.get("audio_encoders", {}) or {}
    cont_map = cfg.get("containers", {}) or {}

    video_keys = cfg.get("video_encoders_to_test") or list(video_map.keys())
    audio_keys = cfg.get("audio_encoders_to_test") or list(audio_map.keys())
    container_keys = cfg.get("containers_to_test") or list(cont_map.keys())

    known_impossible = _normalize_known_impossible(cfg.get("known_impossible", {}))

    # Detectar exper. ya realizados en out_dir y saltarlos
    input_stem = input_file.stem
    done_set = _scan_existing_outs(paths["out_dir"], input_stem)

    combos = list(itertools.product(video_keys, audio_keys, container_keys))
    logging.info(f"Total combos a intentar: {len(combos)}")

    results: list[dict] = []

    for vcodec, acodec, container_key in combos:
        key = (vcodec, acodec, container_key)
        if key in known_impossible:
            logging.info(f"Skipping known_impossible {key}: {known_impossible[key]}")
            results.append({
                "timestamp": datetime.now().isoformat(),
                "input": str(input_file),
                "output": "",
                "vcodec": vcodec,
                "acodec": acodec,
                "muxer": container_key,
                "rc": None,
                "stdout": "",
                "stderr": f"SKIPPED: {known_impossible[key]}",
                "duration_ms": 0,
                "output_exists": False,
                "output_size": 0
            })
            continue

        if key in done_set and args.skip_done:
            logging.info(f"Skipping already-done experiment {key}")
            results.append({
                "timestamp": datetime.now().isoformat(),
                "input": str(input_file),
                "output": "(skipped - already exists)",
                "vcodec": vcodec,
                "acodec": acodec,
                "muxer": container_key,
                "rc": 0,
                "stdout": "",
                "stderr": "SKIPPED: already exists in out folder",
                "duration_ms": 0,
                "output_exists": True,
                "output_size": -1
            })
            continue

        if container_key not in cont_map:
            logging.warning(f"Contenedor desconocido (skip): {container_key}")
            continue

        muxer = cont_map[container_key]["muxer"]
        ext = cont_map[container_key]["ext"]
        outpath = _generate_output_path(paths["out_dir"], input_stem, vcodec, acodec, container_key, ext)

        try:
            res = _test_combination(
                ffmpeg_bin=str(cfg["ffmpeg"]),
                input_file=input_file,
                outpath=outpath,
                vcodec=vcodec,
                acodec=acodec,
                muxer=muxer,
                test_duration=cfg.get("duration", 0),
                timeout=cfg.get("timeout", 600)
            )
            results.append(res)
            logging.info(f"Result: {vcodec}/{acodec}/{container_key} rc={res['rc']} exists={res['output_exists']} size={res['output_size']}")
            if res["rc"] != 0:
                snippet_file = paths["logs_dir"] / f"err_{_safe_filename(vcodec + '_' + acodec + '_' + container_key)}.log"
                with snippet_file.open("w", encoding="utf-8") as f:
                    f.write(res.get("stderr") or "")
        except Exception as ex:
            logging.exception(f"Exception testing {key}: {ex}")
            results.append({
                "timestamp": datetime.now().isoformat(),
                "input": str(input_file),
                "output": str(outpath),
                "vcodec": vcodec,
                "acodec": acodec,
                "muxer": container_key,
                "rc": -99,
                "stdout": "",
                "stderr": str(ex),
                "duration_ms": 0,
                "output_exists": outpath.exists() and outpath.stat().st_size > 0,
                "output_size": outpath.stat().st_size if outpath.exists() else 0
            })

    # Guardar CSV y JSON con manejo seguro
    try:
        _save_json(results, paths["results_json"])
        logging.info(f"Results JSON guardado en {paths['results_json']}")
    except Exception:
        logging.exception("Error guardando results.json")
    try:
        _save_csv(results, paths["results_csv"])
        logging.info(f"Results CSV guardado en {paths['results_csv']}")
    except Exception:
        logging.exception("Error guardando results.csv")
    # Crear la tabla con resultados.
    try:
        _create_csv_markdown(paths)
        logging.info(f"Results CSV guardado en {paths['results_md']}")
    except FileNotFoundError:
        logging.error("No se pudo crear el markdown: archivo CSV no encontrado")
    except ValueError:
        logging.error("No se pudo crear el markdown: archivo CSV vacío")
    except Exception:
        logging.exception("Error creando results.md")
    # Crear lookup tables
    try:
        _save_compatibility_results(paths["results_csv"], paths["work_dir"])
        logging.info(f"Lookup tables creadas exitosamente!!")
    except Exception as e:
        logging.exception("Error creando las lookup tables")

    # En vez de crear MD, leemos CSV y mostramos
    print("\n=== Vista rápida de results.csv ===\n")
    _read_csv_and_print(paths["results_csv"], max_rows=500)
    print("\n=== Fin de tabla ===\n")

    logging.info("Ejecución finalizada.")
    print("Rutas importantes:")
    print(f" - Work dir : {paths['work_dir']}")
    print(f" - Input    : {paths['input_file']}")
    print(f" - Out dir  : {paths['out_dir']}")
    print(f" - Logs dir : {paths['logs_dir']}")
    print(f" - CSV      : {paths['results_csv']}")
    print(f" - JSON     : {paths['results_json']}")
    print(f" - Markdown : {paths['results_md']}")


if __name__ == "__main__":
    # main()
    pass
