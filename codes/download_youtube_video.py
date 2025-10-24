"""
Descarga de videos de YouTube con yt_dlp priorizando:
- Video: Premium > Resolución > FPS > VBR (orden absoluto).
- Audio: evita siempre OPUS; orden por nota de calidad > ABR > TBR > ASR > preferencia de códec.
- Contenedor: candidatos válidos según compatibilidad FFmpeg (JSON) y preferencia tipo DaVinci Resolve por SO.
- Fallbacks: genera múltiples pares v+a y varios contenedores candidatos; prueba en orden hasta que funcione.
- No recodifica: solo remux (cambio de contenedor) cuando hace falta.

Notas:
- Para streams separados (video+audio): usa merge_output_format.
- Para streams combinados: usa FFmpegVideoRemuxer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pathlib import Path
import os
import json
import yt_dlp
import platform
import unicodedata

# ---------------------------
# Constantes mínimas
# ---------------------------

VIDEOS_PATH = "out/youtube_videos"  # carpeta de salida por defecto


# ---------------------------
# Descubrimiento de metadata
# ---------------------------

def _yt_sanitize_candidates(name: str) -> list[str]:
    """
    Genera variantes del nombre similares a las que produce yt_dlp al sanear
    nombres de archivo:
      - sanitize_filename(restricted=False/True) si está disponible.
      - Reemplazos manuales comunes (/, :, ?, *, <, >, |, comillas).
      - Normalización Unicode (NFC y NFKC).
    Devuelve una lista en orden estable, sin duplicados.
    """
    variants: list[str] = [name]
    seen = {name}

    # Variantes usando la utilidad interna de yt_dlp (si existe)
    try:
        from yt_dlp.utils import sanitize_filename
        for restricted in (False, True):
            v = sanitize_filename(name, restricted=restricted)
            if v not in seen:
                variants.append(v)
                seen.add(v)
    except Exception:
        pass

    # Reemplazos manuales típicos (yt_dlp usa lookalikes Unicode)
    manual_map = {
        "/": "⧸",
        "\\": "＼",
        ":": "꞉",
        "?": "？",
        "*": "∗",
        "<": "＜",
        ">": "＞",
        "|": "ǀ",
        '"': "”",
        "'": "’",
    }
    base_list = list(variants)  # congelar actuales para derivar
    for base in base_list:
        s = base
        for bad, repl in manual_map.items():
            s = s.replace(bad, repl)
        if s not in seen:
            variants.append(s)
            seen.add(s)

    # Normalización Unicode NFC y NFKC
    norm_added: list[str] = []
    for s in variants:
        nfc = unicodedata.normalize("NFC", s)
        nfkc = unicodedata.normalize("NFKC", s)
        if nfc not in seen:
            norm_added.append(nfc)
            seen.add(nfc)
        if nfkc not in seen:
            norm_added.append(nfkc)
            seen.add(nfkc)
    variants.extend(norm_added)

    # Deduplicar preservando orden
    variants = list(dict.fromkeys(variants))
    return variants


def _check_predownloaded(video_title: str | Path,
                         directory: str | Path = VIDEOS_PATH,
                         exts: list[str] | None = None) -> bool:
    """
    Devuelve True si ya existe un archivo con ese título (cualquier extensión o
    alguna de 'exts') en el directorio dado. Considera la sanitización de nombres
    que aplica yt_dlp (caracteres inválidos, normalización Unicode, etc.).
    """
    directory = Path(directory)
    if not directory.exists():
        return False
    # Tomar solo el nombre base; yt_dlp genera "%(title)s.%(ext)s"
    stem_raw = Path(str(video_title)).name
    stems = _yt_sanitize_candidates(stem_raw)
    allow = {e.lower().lstrip(".") for e in (exts or [])}
    for stem in stems:
        # Buscar archivos con ese stem y cualquier extensión
        for p in directory.glob(f"{stem}.*"):
            if not exts or p.suffix.lower().lstrip(".") in allow:
                return True
    return False


def _retrieve_metadata_from_playlist(info: dict, skip: bool) -> list[dict]:
    """
    Dado el dict de una playlist (extract_flat=True), extrae la metadata completa
    de cada video (extract_flat=False). Si skip=True, salta los ya descargados.
    """
    ydl_opts = {"quiet": True, "skip_download": True, "extract_flat": False}
    ydl = yt_dlp.YoutubeDL(params=ydl_opts)
    list_of_metadata: list[dict] = []

    for entry in info.get("entries", []):
        title = entry.get("title") or entry.get("id")
        if skip and title and _check_predownloaded(title):
            print(f"Video '{title}' ya descargado.")
            continue
        url = entry.get("url") or entry.get("webpage_url")
        if not url:
            continue
        md = ydl.extract_info(url, download=False)
        if md:
            list_of_metadata.append(md)

    return list_of_metadata


def get_videos_metadata(urls: list[str], skip: bool = False) -> list[dict]:
    """
    Acepta URLs de videos o playlists y devuelve lista plana de metadatas de video.
    """
    ydl_opts = {"quiet": True, "skip_download": True, "extract_flat": True}
    list_of_metadata: list[dict] = []

    for url in urls:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info: dict[str, Any] | None = ydl.extract_info(url, download=False)
        if not info:
            continue

        if info.get("_type") == "playlist":
            list_of_metadata.extend(_retrieve_metadata_from_playlist(info, skip))
        else:
            title = info.get("title") or info.get("id")
            if skip and title and _check_predownloaded(title):
                print(f"Video '{title}' ya descargado.")
                continue
            list_of_metadata.append(info)

    return list_of_metadata


def get_videos_info(urls: str | list[str]) -> list[dict]:
    urls = [urls] if isinstance(urls, str) else urls
    return get_videos_metadata(urls)


# ---------------------------
# CARGA COMPATIBILIDAD FFMPEG
# ---------------------------

def load_ffmpeg_compat(path: str | Path) -> dict:
    """
    Carga ffmpeg_compatibility.json y añade estructuras útiles:
      - _valid_set: set de (vcodec, acodec, muxer)
      - _ext_by_muxer: map de muxer a extensión preferida
      - _container_pref: orden global de preferencia de contenedores
    Espera claves en el JSON:
      - "valid_tuples": lista de ternas [vcodec, acodec, muxer]
      - "nested": dict anidado nested[vcodec][acodec] -> [muxers válidos]
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    data["_valid_set"] = set(tuple(x) for x in data.get("valid_tuples", []))
    data["_ext_by_muxer"] = {
        "matroska": "mkv",
        "mp4": "mp4",
        "mov": "mov",
        "avi": "avi",
        "mxf": "mxf",
    }
    data["_container_pref"] = data.get("_container_pref") or ["mp4", "mov", "matroska", "avi", "mxf"]
    return data


# ---------------------------
# NORMALIZACIÓN DE CODECS (YouTube -> familias FFmpeg usadas en compat JSON)
# ---------------------------

def _norm_vcodec(vcodec_raw: str | None) -> str | None:
    c = (vcodec_raw or "").lower()
    if not c or c == "none":
        return None
    if "vp9" in c or "vp09" in c:
        return "libvpx-vp9"
    if "av01" in c or "av1" in c:
        return "libaom-av1"
    if "avc1" in c or "h264" in c:
        return "libx264"
    if "hev1" in c or "hvc1" in c or "hevc" in c:
        return "libx265"
    if "prores" in c:
        return "prores_ks"
    if "mpeg4" in c or c == "mp4v":
        return "mpeg4"
    if "jpeg2000" in c or "j2k" in c:
        return "jpeg2000"
    return None


def _norm_acodec(acodec_raw: str | None) -> str | None:
    c = (acodec_raw or "").lower()
    if not c or c == "none":
        return None
    if "opus" in c:
        return "libopus"
    if "vorbis" in c:
        return "libvorbis"
    if "aac" in c or "mp4a" in c:
        return "aac"
    if "flac" in c:
        return "flac"
    if c.startswith("pcm"):
        return "pcm_s24le"
    return None


# ---------------------------
# SELECCIÓN DE FORMATOS
# ---------------------------

def _is_premium(f: dict) -> bool:
    return "premium" in (f.get("format_note") or "").lower()


def _quality_from_note(note: str | None) -> int:
    n = (note or "").lower()
    # high > medium > low > default/unknown
    if "high" in n:
        return 3
    if "medium" in n:
        return 2
    if "low" in n:
        return 1
    return 0


def _video_sort_key(f: dict) -> tuple[int, int, float, float]:
    # Orden absoluto: premium > height > fps > vbr/tbr
    premium = 1 if _is_premium(f) else 0
    height = int(f.get("height") or 0)
    fps = float(f.get("fps") or 0)
    vbr = float(f.get("vbr") or f.get("tbr") or 0)
    return (premium, height, fps, vbr)


def _audio_score(f: dict) -> tuple[int, float, float, int, int]:
    # quality(high>medium>low) > abr > tbr > asr > codec_pref
    q = _quality_from_note(f.get("format_note"))
    abr = float(f.get("abr") or 0)
    tbr = float(f.get("tbr") or 0)
    asr = int(f.get("asr") or 0)
    ac = (f.get("acodec") or "").lower()
    codec_pref = 0
    # Preferir AAC sobre otros; no favorecer OPUS (se filtra igual, pero no sumar puntos)
    if "aac" in ac or "mp4a" in ac:
        codec_pref = 2
    elif "opus" in ac:
        codec_pref = 0
    return (q, abr or tbr, tbr, asr, codec_pref)


def _pick_best_video_format(info: dict) -> tuple[dict | None, bool]:
    """
    Devuelve (video_format, is_combined). Orden absoluto:
    premium > resolución > FPS > bitrate.
    Busca primero video-only; si no hay, combinado.
    """
    formats = info.get("formats") or []
    video_only = [f for f in formats if (f.get("vcodec") or "").lower() != "none" and (f.get("acodec") or "").lower() == "none"]
    combined = [f for f in formats if (f.get("vcodec") or "").lower() != "none" and (f.get("acodec") or "").lower() != "none"]

    if video_only:
        video_only.sort(key=_video_sort_key, reverse=True)
        return video_only[0], False

    if combined:
        combined.sort(key=_video_sort_key, reverse=True)
        return combined[0], True

    return None, False


def _pick_best_audio_format(info: dict, v_ff: str | None = None, compat: dict | None = None) -> dict | None:
    """Mejor audio-only evitando OPUS; orden por quality>abr>tbr>asr y prioriza códecs compatibles con v_ff."""
    formats = info.get("formats") or []
    audio_only = [f for f in formats if (f.get("vcodec") or "").lower() == "none" and (f.get("acodec") or "").lower() != "none"]
    if not audio_only:
        # incluir audios con acodec desconocido (p.ej. HLS) si existen
        audio_only = [f for f in formats if (f.get("vcodec") or "").lower() == "none"]

    if not audio_only:
        return None

    # Evitar OPUS siempre
    audio_only = [f for f in audio_only if "opus" not in (f.get("acodec") or "").lower()]
    if not audio_only:
        return None

    # Prioriza acodecs que tengan al menos un muxer válido con el vcodec elegido
    if v_ff and compat and isinstance(compat.get("nested"), dict):
        allowed_acodecs = set((compat["nested"].get(v_ff) or {}).keys())
        preferred: list[dict] = []
        others: list[dict] = []
        for f in audio_only:
            a_ff = _norm_acodec(f.get("acodec"))
            (preferred if (a_ff and a_ff in allowed_acodecs) else others).append(f)
        pool = preferred or others
    else:
        pool = audio_only

    pool.sort(key=_audio_score, reverse=True)
    return pool[0]


# ---------------------------
# Ayudas: ordenar y generar fallback de formatos
# ---------------------------

def _sorted_video_only_formats(info: dict) -> list[dict]:
    formats = info.get("formats") or []
    video_only = [f for f in formats if (f.get("vcodec") or "").lower() != "none" and (f.get("acodec") or "").lower() == "none"]
    video_only.sort(key=_video_sort_key, reverse=True)
    return video_only


def _sorted_combined_formats(info: dict) -> list[dict]:
    formats = info.get("formats") or []
    combined = [f for f in formats if (f.get("vcodec") or "").lower() != "none" and (f.get("acodec") or "").lower() != "none"]
    combined.sort(key=_video_sort_key, reverse=True)
    return combined


def _sorted_audio_only_formats(info: dict, v_ff: str | None = None, compat: dict | None = None) -> list[dict]:
    """
    Devuelve audios ordenados por calidad, evitando OPUS siempre.
    Prioriza acodecs compatibles con v_ff si se provee compat['nested'].
    """
    formats = info.get("formats") or []
    audio_only = [f for f in formats if (f.get("vcodec") or "").lower() == "none" and (f.get("acodec") or "").lower() != "none"]
    if not audio_only:
        audio_only = [f for f in formats if (f.get("vcodec") or "").lower() == "none"]
    if not audio_only:
        return []

    # Evitar OPUS siempre (no devolver audios si solo hay OPUS)
    audio_only = [f for f in audio_only if "opus" not in (f.get("acodec") or "").lower()]
    if not audio_only:
        return []

    if v_ff and compat and isinstance(compat.get("nested"), dict):
        allowed_acodecs = set((compat["nested"].get(v_ff) or {}).keys())
        preferred, others = [], []
        for f in audio_only:
            a_ff = _norm_acodec(f.get("acodec"))
            (preferred if (a_ff and a_ff in allowed_acodecs) else others).append(f)
        pool = preferred or others
    else:
        pool = audio_only
    pool.sort(key=_audio_score, reverse=True)
    return pool


def _build_fallback_fmt(video_ids: list[str], audio_ids: list[str], max_pairs: int = 12) -> str:
    """
    Devuelve un format string con múltiples pares v+a en orden de preferencia.
    Evita OPUS intentando primero bestaudio[acodec!=opus], luego bestaudio y por último best.
    """
    pairs: list[str] = []
    count = 0
    for v in video_ids:
        for a in audio_ids:
            pairs.append(f"{v}+{a}")
            count += 1
            if count >= max_pairs:
                break
        if count >= max_pairs:
            break
    tail = "bestvideo*+bestaudio[acodec!=opus]/bestvideo*+bestaudio/best"
    return "/".join(pairs + [tail]) if pairs else tail


# ---------------------------
# Política Resolve y candidatos de contenedor
# ---------------------------

def get_resolve_policy(os_name: str | None = None) -> dict[str, list[str]]:
    """
    Política de contenedor preferido por vcodec normalizado para DaVinci Resolve.
    Devuelve listas de contenedores en orden de preferencia. La compat FFmpeg decide lo final.
    """
    return {
        "libvpx-vp9": ["matroska", "mp4", "mov"],
        "libaom-av1": ["matroska", "mp4", "mov"],
        "libx264":    ["mp4", "mov", "matroska"],
        "libx265":    ["mp4", "matroska", "mov"],
        "prores_ks":  ["mov", "mxf", "matroska"],
        "dnxhd":      ["mxf", "mov", "matroska"],
        "mpeg4":      ["mov", "mp4", "matroska"],
    }


def _choose_container_candidates(
    v_ff: str | None,
    a_ff: str | None,
    compat: dict,
    resolve_policy: dict[str, list[str]] | None = None
) -> list[str]:
    """
    Devuelve lista de muxers candidatos válidos según compat['nested'],
    ordenados por política Resolve (si hay) y preferencia global.
    """
    nested = compat.get("nested", {})
    global_pref = compat.get("_container_pref", ["mp4", "mov", "matroska", "avi", "mxf"])
    allowed = []
    if v_ff and a_ff:
        allowed = list(nested.get(v_ff, {}).get(a_ff, []))
    if not allowed:
        # si no hay info suficiente, cae a preferencia global pero ordenado por política
        policy = resolve_policy or {}
        policy_list = policy.get(v_ff or "", [])
        return _order_by_policy(list(global_pref), policy_list, global_pref)

    # reordenar allowed por política si aplica
    policy = resolve_policy or {}
    policy_list = policy.get(v_ff or "", [])
    if policy_list:
        in_policy = [m for m in policy_list if m in allowed]
        rest = [m for m in allowed if m not in in_policy]
        rest_sorted = sorted(rest, key=lambda x: global_pref.index(x) if x in global_pref else 9999)
        return in_policy + rest_sorted

    return sorted(allowed, key=lambda x: global_pref.index(x) if x in global_pref else 9999)


def _order_by_policy(muxers: list[str], policy_list: list[str], global_pref: list[str]) -> list[str]:
    in_policy = [m for m in policy_list if m in muxers]
    rest = [m for m in muxers if m not in in_policy]
    rest_sorted = sorted(rest, key=lambda x: global_pref.index(x) if x in global_pref else 9999)
    return in_policy + rest_sorted


def _container_candidates_for(v_ff: str | None, a_families: list[str], compat: dict) -> list[str]:
    """
    Une todos los muxers válidos para v_ff con cada familia de audio en a_families.
    """
    nested = compat.get("nested", {})
    allowed: list[str] = []
    if v_ff:
        for a_ff in a_families:
            allowed.extend(nested.get(v_ff, {}).get(a_ff, []))
    # único y en orden de aparición
    return list(dict.fromkeys(allowed))


def _format_id(f: dict) -> str | None:
    return f.get("format_id") or f.get("format") or f.get("id")


# Helpers de compatibilidad (lookup rápido y listado)
def is_valid_tuple(vcodec: str, acodec: str, muxer: str, valid_set: set[tuple[str, str, str]]) -> bool:
    """Búsqueda rápida usando el set"""
    return (vcodec, acodec, muxer) in valid_set


def list_valid_muxers(vcodec: str, acodec: str, nested: dict) -> list[str]:
    """Devuelve lista de muxers válidos para vcodec+acodec"""
    return sorted(nested.get(vcodec, {}).get(acodec, []))


# ---------------------------
# Plan por video
# ---------------------------

@dataclass
class SelectedFormats:
    fmt_str: str                 # e.g. "137+140/136+140/bestvideo*+bestaudio/best"
    v_fmt: dict | None
    a_fmt: dict | None
    combined: bool
    desired_container_key: str
    desired_ext: str
    need_remux: bool
    container_candidates: list[str]     # los contenedores válidos, ordenados
    candidate_exts: list[str]           # extensiones correspondientes


def plan_for_video(info: dict, compat: dict, container_policy: dict[str, str] | None = None) -> SelectedFormats | None:
    """
    Construye un plan con múltiples fallbacks de formato y múltiples contenedores candidatos.
    Evita siempre OPUS en audios separados y prefiere combinados sin OPUS.
    """
    # Ordenar candidatos de video
    video_only_sorted = _sorted_video_only_formats(info)
    combined_sorted = _sorted_combined_formats(info)

    # Si hay video-only, trabajamos con pares v+a; si no, combinados
    if video_only_sorted:
        v_top = video_only_sorted[0]
        v_ff = _norm_vcodec(v_top.get("vcodec"))
        # Ordenar audios (evitando OPUS)
        audio_sorted = _sorted_audio_only_formats(info, v_ff=v_ff, compat=compat)
        a_top = audio_sorted[0] if audio_sorted else None

        # Familias de audio (para contenedores válidos), sin libopus
        a_families: list[str] = []
        for a in audio_sorted[:5]:  # top-N familias
            fam = _norm_acodec(a.get("acodec"))
            if fam and fam != "libopus":
                a_families.append(fam)
        if not a_families and a_top:
            fam = _norm_acodec(a_top.get("acodec"))
            if fam and fam != "libopus":
                a_families = [fam]
        a_families = list(dict.fromkeys(a_families))  # único

        # IDs para fallback (filtrar None explícitamente para tipos)
        v_ids: list[str] = []
        for _f in video_only_sorted[:6]:
            _id = _format_id(_f)
            if _id:
                v_ids.append(_id)
        a_ids: list[str] = []
        for _f in audio_sorted[:6]:
            _id = _format_id(_f)
            if _id:
                a_ids.append(_id)
        fmt_str = _build_fallback_fmt(v_ids, a_ids, max_pairs=12)

        # Candidatos de contenedor válidos por FFmpeg, ordenados por política Resolve
        resolve_pref = get_resolve_policy()
        global_pref = compat.get("_container_pref", ["mp4", "mov", "matroska", "avi", "mxf"])
        muxers_valid = _container_candidates_for(v_ff, a_families or [], compat=compat)

        # Permitir override simple: empujar contenedor al frente si es válido
        if container_policy and v_ff:
            override = None
            for key, cont in container_policy.items():
                if key and key.lower() in v_ff.lower():
                    override = cont
                    break
            if override and override in muxers_valid:
                muxers_valid = [override] + [m for m in muxers_valid if m != override]

        policy_list = resolve_pref.get(v_ff or "", [])
        if muxers_valid:
            muxers_ordered = _order_by_policy(muxers_valid, policy_list, global_pref)
        else:
            # Sin info de compat: ordena la preferencia global según la política
            muxers_ordered = _order_by_policy(list(global_pref), policy_list, global_pref)
        ext_map = compat.get("_ext_by_muxer", {})
        candidate_exts = [ext_map.get(m, m) for m in muxers_ordered]
        desired_container = muxers_ordered[0]
        desired_ext = candidate_exts[0]

        return SelectedFormats(
            fmt_str=fmt_str,
            v_fmt=v_top,
            a_fmt=a_top,
            combined=False,
            desired_container_key=desired_container,
            desired_ext=desired_ext,
            need_remux=False,
            container_candidates=muxers_ordered,
            candidate_exts=candidate_exts,
        )

    if combined_sorted:
        # Preferir combinados sin OPUS en el fallback
        combined_no_opus = [f for f in combined_sorted if "opus" not in (f.get("acodec") or "").lower()]
        c_top = (combined_no_opus or combined_sorted)[0]
        v_ff = _norm_vcodec(c_top.get("vcodec"))
        a_ff = _norm_acodec(c_top.get("acodec"))

        # Fallback de combinados por id (no-OPUS primero)
        c_ids_no_opus: list[str] = []
        for _f in combined_no_opus[:6]:
            _id = _format_id(_f)
            if _id:
                c_ids_no_opus.append(_id)
        c_ids_all: list[str] = []
        for _f in combined_sorted[:6]:
            _id = _format_id(_f)
            if _id:
                c_ids_all.append(_id)
        c_ids: list[str] = list(dict.fromkeys(c_ids_no_opus + c_ids_all))
        fmt_str = "/".join(c_ids + ["best"]) if c_ids else "best"

        # Candidatos de contenedor para combinado (usar familia del combinado)
        resolve_pref = get_resolve_policy()
        global_pref = compat.get("_container_pref", ["mp4", "mov", "matroska", "avi", "mxf"])
        muxers_valid = list(compat.get("nested", {}).get(v_ff or "", {}).get(a_ff or "", []))
        policy_list = resolve_pref.get(v_ff or "", [])
        if muxers_valid:
            muxers_ordered = _order_by_policy(muxers_valid, policy_list, global_pref)
        else:
            muxers_ordered = _order_by_policy(list(global_pref), policy_list, global_pref)
        ext_map = compat.get("_ext_by_muxer", {})
        candidate_exts = [ext_map.get(m, m) for m in muxers_ordered]
        desired_container = muxers_ordered[0]
        desired_ext = candidate_exts[0]
        need_remux = (c_top.get("ext") or "").lower() != desired_ext.lower()

        return SelectedFormats(
            fmt_str=fmt_str,
            v_fmt=c_top,
            a_fmt=None,
            combined=True,
            desired_container_key=desired_container,
            desired_ext=desired_ext,
            need_remux=need_remux,
            container_candidates=muxers_ordered,
            candidate_exts=candidate_exts,
        )

    return None


# ---------------------------
# Resolver configuración para yt_dlp
# ---------------------------

def resolve_installation_config(
    metadata_list: list[dict],
    compat_path: str | Path = "ffmpeg_test/ffmpeg_compatibility.json",
    container_policy: dict[str, str] | None = None,
    outdir: str | Path = VIDEOS_PATH
) -> list[dict]:
    compat = load_ffmpeg_compat(compat_path)
    outdir = str(outdir)
    os.makedirs(outdir, exist_ok=True)
    plans: list[dict] = []

    for info in metadata_list:
        plan = plan_for_video(info, compat, container_policy)
        if not plan:
            plans.append({
                "url": info.get("webpage_url") or info.get("url"),
                "error": "no_video_format_found",
            })
            continue

        title = info.get("title") or info.get("id") or "video"
        outtmpl = os.path.join(outdir, "%(title)s.%(ext)s")

        # ydl_opts base (se ajusta por intento/contendor)
        base_opts = {
            "quiet": False,
            "no_warnings": True,
            "format": plan.fmt_str,  # incluye múltiples fallbacks
            "outtmpl": outtmpl,
            "noplaylist": True,
        }

        # Archivos esperados para cada candidato
        expected_files = [os.path.join(outdir, f"{title}.{ext}") for ext in plan.candidate_exts]
        already = any(os.path.exists(f) for f in expected_files)

        plans.append({
            "url": info.get("webpage_url") or info.get("url"),
            "title": title,
            "vcodec": (plan.v_fmt or {}).get("vcodec"),
            "acodec": (plan.a_fmt or plan.v_fmt or {}).get("acodec"),
            "container": plan.desired_container_key,
            "ext": plan.desired_ext,
            "format_str": plan.fmt_str,
            "combined": plan.combined,
            "need_remux": plan.need_remux,
            "container_candidates": plan.container_candidates,
            "candidate_exts": plan.candidate_exts,
            "expected_files": expected_files,
            "already_downloaded": already,
            "ydl_opts_base": base_opts,
        })

    return plans


# -------------------------------
# Entrypoint
# -------------------------------

def main(playlist_url: str):
    # 1) Obtener metadata
    metadata_list = get_videos_info(playlist_url)

    # 2) Generar planes (policy se deriva de Resolve internamente; puedes pasar overrides)
    plans = resolve_installation_config(
        metadata_list,
        compat_path="ffmpeg_test/ffmpeg_compatibility.json",
        container_policy=None,  # puedes pasar overrides tipo {"libvpx-vp9": "matroska"}
        outdir=VIDEOS_PATH
    )

    # 3) Mostrar plan resumido
    for p in plans:
        cands = ", ".join(p.get("container_candidates") or [])
        print(f"- {p.get('title')}: {p.get('format_str')} -> .{p.get('ext')} (candidatos=[{cands}])")

    # 4) Descargar faltantes probando contenedores candidatos
    to_download = [p for p in plans if not p.get("already_downloaded") and p.get("url")]
    done, failed, skipped = 0, 0, len(plans) - len(to_download)

    for p in to_download:
        url = p["url"]
        title = p.get("title") or url
        combined = bool(p.get("combined"))
        candidates = p.get("container_candidates") or ["mp4"]
        exts = p.get("candidate_exts") or ["mp4"]

        success = False
        for muxer, ext in zip(candidates, exts):
            # Construir ydl_opts para este intento
            ydl_opts = dict(p["ydl_opts_base"])
            # Siempre remux al contenedor objetivo (cubre tanto combinados como fallback a 'best')
            ydl_opts["postprocessors"] = [{
                "key": "FFmpegVideoRemuxer",
                "preferedformat": ext,
            }]
            if not combined:
                # Para streams separados, además fija el formato de merge del contenedor
                ydl_opts["merge_output_format"] = ext

            print(f"[DOWN] {title} -> {p.get('format_str')} (.{ext}) try({muxer})")
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                success = True
                break
            except Exception as e:
                print(f"[WARN] {title} falló con {muxer} (. {ext}): {e}")
                continue

        if success:
            done += 1
        else:
            failed += 1
            print(f"[ERROR] {title}: sin contenedor válido (intentados: {candidates})")

    print(f"Resumen: descargados={done}, fallidos={failed}, ya_existían={skipped}")

def show_video_metada(url):
    ydl = yt_dlp.YoutubeDL(params={"quiet": True, "skip_download": True, "extract_flat": False})
    info = ydl.extract_info(url, download=False)
    ydl.list_formats(info)


if __name__ == "__main__":
    # Ejemplo: playlist; puedes reemplazar por URL de video o lista
    #url = "https://www.youtube.com/playlist?list=PLaAjsJBsA0UR77l10qnatZhgK7pwrdRHp"
    url = "https://youtu.be/2uXClcfciVI"
    #show_video_metada(url)

    main(playlist_url=url)

#? Title
#* YouTube behaviour
#! yt_dlp behaviour

#? For normal videos:
#*  - `share_url` and `copy_url` are the same.
#*  - `shorts` variants all redirect to `normal_url` with no timestamp.
#!  - Time to take info: ~2.21 seconds

#? For shorts:
#*  - `share_url` and copy_url are different.
#*  - There is no option for sharing videos in a specific timestamp,
#*  only using the "copy url at the current moment".
#*  - When watching as a "normal" video, there is the option to share
#*  at a specific timestamp.
#!  - Time to take info: ~22.8 seconds

#? For playlists:
#*  - `normal_url` starts the playlist.
#*  - `share_url` returns the playlist "menu".
#!  - Time to take info: ~13.1 seconds

#! == Method 1: Get videos URL and then retrieve each video's information separatedly. ==
#! Time to get URLS:   1.312929429 (s)
#!          Average:   0.008154841173913043 (s)
#! To retrieve data:   356.92526109000005 (s)
#!          Average:   2.2169270875155282 (s)
#! Total time taken:   358.23819051900006 (s)
#!          Average:   2.225081928689441 (s)

#! == Method 2 - Extract all videos information at once. ==
#! Total time taken:   321.541957976 (s)
#!          Average:   1.997155018484472 (s)