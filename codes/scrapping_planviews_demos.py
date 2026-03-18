"""
Scraper puntual para demos de Planview.

Requirements
============

.. code-block:: bash

   pip install playwright yt-dlp
   playwright install chromium

Uso
===

.. code-block:: bash

   # Opción 1: desde VS Code con el botón "Run Python File"
   # Edita RUN_MODE / RUN_HEADLESS / RUN_LIMIT más abajo y luego presiona Play.
   python codes/scrapping_planviews_demos.py

   # Opción 2: por CLI
   python codes/scrapping_planviews_demos.py scrape
   python codes/scrapping_planviews_demos.py download
   python codes/scrapping_planviews_demos.py all

Descripción
===========

Este módulo hace dos tareas, separadas en dos funciones principales:

``scrape_all_demos()``
    Descubre los cards de demos desde las dos páginas principales,
    entra a las páginas ``*_reg.html`` y ``*_confirm.html``, y guarda
    título, descripción, URL de confirmación y URL de Vidyard en JSON.

``download_all_videos()``
    Lee los JSON generados por el scraping, descarga los videos con
    ``yt_dlp`` y actualiza cada entrada con el path absoluto del video.

El script es deliberadamente específico para estas URLs y este flujo.
No depende de otros módulos del repositorio.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from html import unescape
from html.parser import HTMLParser
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any
from urllib.parse import SplitResult, urlencode, urlsplit, urlunsplit


LOGGER = logging.getLogger("scrapping_planviews_demos")

OUTPUT_ROOT = Path("out/scrapping")
VIDEOS_ROOT = OUTPUT_ROOT / "videos"

PRODUCT_DEMOS_URL = (
    "https://www.planview.com/resource-center/"
    "?resource_language%5b%5d=english&resource_category%5b%5d=product-demos"
)
SOLUTION_DEMOS_URL = (
    "https://www.planview.com/resource-center/"
    "?resource_language%5b%5d=english&resource_category%5b%5d=solution-demos"
)

RESOURCE_API_PATH = "/wp-json/resource_center/v1/post"

DEFAULT_NAVIGATION_TIMEOUT_MS = 45_000
DEFAULT_WAIT_AFTER_NAVIGATION_MS = 1_200
DEFAULT_NETWORK_IDLE_TIMEOUT_MS = 5_000
MAX_NAVIGATION_ATTEMPTS = 3
SYSTEM_CHROMIUM_CANDIDATES = (
    Path("/usr/bin/chromium"),
    Path("/usr/bin/chromium-browser"),
    Path("/usr/bin/google-chrome"),
    Path("/usr/bin/google-chrome-stable"),
)

# Configuracion para ejecutar con el boton "Play" de VS Code sin argumentos.
# Cambia estas variables y luego ejecuta este archivo directamente.
RUN_MODE = "download" # "download", "scrape" o "all"
RUN_HEADLESS = True
RUN_LIMIT: int | None = None


class WorkflowStage(str, Enum):
    """Etapas persistidas en el JSON por demo."""

    DISCOVERED = "discovered"
    METADATA_SCRAPED = "metadata_scraped"
    VIDEO_URL_SCRAPED = "video_url_scraped"
    DOWNLOADED = "downloaded"


@dataclass(frozen=True)
class CategoryConfig:
    """Configuración fija por categoría."""

    name: str
    list_url: str
    json_path: Path
    video_dir: Path


CATEGORY_CONFIGS = (
    CategoryConfig(
        name="product_demos",
        list_url=PRODUCT_DEMOS_URL,
        json_path=OUTPUT_ROOT / "product_demos.json",
        video_dir=VIDEOS_ROOT / "product_demos",
    ),
    CategoryConfig(
        name="solution_demos",
        list_url=SOLUTION_DEMOS_URL,
        json_path=OUTPUT_ROOT / "solution_demos.json",
        video_dir=VIDEOS_ROOT / "solution_demos",
    ),
)


class HTMLTextExtractor(HTMLParser):
    """Convierte fragmentos HTML simples a texto legible."""

    _BLOCK_TAGS = {
        "article",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "ol",
        "p",
        "section",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "br":
            self._parts.append("\n")
        elif tag == "li":
            if self._parts and not self._parts[-1].endswith("\n"):
                self._parts.append("\n")
            self._parts.append("- ")
        elif tag in self._BLOCK_TAGS and self._parts and not self._parts[-1].endswith("\n"):
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return normalize_multiline_text("".join(self._parts))


def utc_now_iso() -> str:
    """Devuelve el timestamp actual en UTC."""

    return datetime.now(timezone.utc).isoformat()


def ensure_directories() -> None:
    """Crea los directorios de salida necesarios."""

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    VIDEOS_ROOT.mkdir(parents=True, exist_ok=True)
    for config in CATEGORY_CONFIGS:
        config.video_dir.mkdir(parents=True, exist_ok=True)


def normalize_inline_text(value: str | None) -> str:
    """Compacta espacios para texto en una sola línea."""

    if value is None:
        return ""
    value = unescape(str(value)).replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_multiline_text(value: str | None) -> str:
    """Normaliza texto multilínea preservando saltos relevantes."""

    if value is None:
        return ""

    value = unescape(str(value)).replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    value = value.replace("\u200b", "")
    lines = []
    previous_blank = True
    for raw_line in value.split("\n"):
        line = re.sub(r"[ \t\f\v]+", " ", raw_line).strip()
        if line:
            lines.append(line)
            previous_blank = False
        elif not previous_blank and lines:
            lines.append("")
            previous_blank = True
    return "\n".join(lines).strip()


def html_fragment_to_text(fragment: str | None) -> str:
    """Limpia un fragmento HTML pequeño y lo convierte a texto."""

    if not fragment:
        return ""
    parser = HTMLTextExtractor()
    parser.feed(fragment)
    parser.close()
    return parser.get_text()


def parse_url(value: str) -> SplitResult:
    """Parsea una URL absoluta o lanza una excepción clara."""

    parsed = urlsplit(value.strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"URL inválida: {value!r}")
    return parsed


def normalize_url(value: str | None) -> str:
    """Normaliza una URL y elimina ruido mínimo."""

    if not value:
        return ""
    value = unescape(str(value)).strip()
    if value.startswith("//"):
        value = f"https:{value}"
    parsed = parse_url(value)
    normalized = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, parsed.fragment))
    return normalized


def build_confirm_url(reg_url: str) -> str:
    """Convierte una URL ``*_reg.html`` en ``*_confirm.html``."""

    parsed = parse_url(reg_url)
    new_path, replacements = re.subn(r"reg(?=\.html$)", "confirm", parsed.path)
    if replacements != 1:
        raise ValueError(f"No fue posible construir confirm_url desde: {reg_url}")
    return urlunsplit((parsed.scheme, parsed.netloc, new_path, parsed.query, parsed.fragment))


def entry_has_metadata(entry: dict[str, Any]) -> bool:
    """Indica si la entrada ya tiene la etapa de metadata lista."""

    return bool(
        entry.get("confirm_url")
        and entry.get("demo_title") is not None
        and entry.get("demo_description") is not None
    )


def entry_has_video_url(entry: dict[str, Any]) -> bool:
    """Indica si la entrada ya tiene la URL final de Vidyard."""

    return bool(entry.get("vidyard_url"))


def entry_has_download(entry: dict[str, Any]) -> bool:
    """Indica si la entrada ya tiene un video descargado en disco."""

    video_path = entry.get("video_path")
    return bool(video_path and Path(video_path).exists())


def infer_stage(entry: dict[str, Any]) -> str:
    """Infiera la etapa actual de la entrada según su contenido."""

    if entry_has_download(entry):
        return WorkflowStage.DOWNLOADED.value
    if entry_has_video_url(entry):
        return WorkflowStage.VIDEO_URL_SCRAPED.value
    if entry_has_metadata(entry):
        return WorkflowStage.METADATA_SCRAPED.value
    return WorkflowStage.DISCOVERED.value


def append_error(
    entry: dict[str, Any],
    stage: str,
    error: Exception | str,
    *,
    url: str | None = None,
) -> None:
    """Añade un error estructurado a una entrada."""

    if isinstance(error, Exception):
        error_type = error.__class__.__name__
        message = str(error)
    else:
        error_type = "Error"
        message = str(error)

    payload = {
        "timestamp": utc_now_iso(),
        "stage": stage,
        "type": error_type,
        "message": normalize_inline_text(message),
    }
    if url:
        payload["url"] = url

    entry.setdefault("errors", []).append(payload)
    entry["last_error"] = payload
    entry["updated_at"] = utc_now_iso()


def initialize_store(config: CategoryConfig) -> dict[str, Any]:
    """Crea la estructura base del JSON de una categoría."""

    return {
        "category": config.name,
        "source_list_url": config.list_url,
        "items": [],
        "listing_total_found_posts": None,
        "listing_total_pages": None,
        "updated_at": utc_now_iso(),
        "stats": {
            "total_items": 0,
            "items_with_errors": 0,
            "by_stage": {},
        },
    }


def load_store(config: CategoryConfig) -> dict[str, Any]:
    """Carga o inicializa el JSON de una categoría."""

    if not config.json_path.exists():
        return initialize_store(config)

    with config.json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"JSON inválido en {config.json_path}")

    data.setdefault("category", config.name)
    data.setdefault("source_list_url", config.list_url)
    data.setdefault("items", [])
    data.setdefault("stats", {})

    for entry in data["items"]:
        entry.setdefault("errors", [])
        entry.setdefault("workflow_stage", infer_stage(entry))

    return data


def summarize_store(store: dict[str, Any]) -> dict[str, Any]:
    """Calcula estadísticas derivadas para el JSON."""

    items = store.get("items", [])
    counter = Counter(infer_stage(item) for item in items)
    error_items = sum(1 for item in items if item.get("errors"))
    return {
        "total_items": len(items),
        "items_with_errors": error_items,
        "by_stage": dict(sorted(counter.items())),
    }


def summarize_download_status(store: dict[str, Any]) -> dict[str, int]:
    """Resume el estado de descarga de un store."""

    items = store.get("items", [])
    with_vidyard = sum(1 for item in items if item.get("vidyard_url"))
    downloaded = sum(1 for item in items if item.get("video_path"))
    pending = sum(1 for item in items if item.get("vidyard_url") and not item.get("video_path"))
    return {
        "total": len(items),
        "with_vidyard": with_vidyard,
        "downloaded": downloaded,
        "pending_download": pending,
    }


def sort_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ordena las entradas de forma estable para escritura."""

    return sorted(
        items,
        key=lambda item: (
            int(item.get("listing_position") or 10**9),
            normalize_inline_text(item.get("card_id")),
        ),
    )


def save_store(config: CategoryConfig, store: dict[str, Any]) -> None:
    """Escribe el JSON de una categoría de forma atómica."""

    store["category"] = config.name
    store["source_list_url"] = config.list_url
    store["items"] = sort_items(store.get("items", []))
    for entry in store["items"]:
        entry["workflow_stage"] = infer_stage(entry)
    store["stats"] = summarize_store(store)
    store["updated_at"] = utc_now_iso()

    temp_path = config.json_path.with_suffix(config.json_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(store, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temp_path.replace(config.json_path)


def build_entry_index(store: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Indexa las entradas por ``card_id``."""

    return {
        normalize_inline_text(entry.get("card_id")): entry
        for entry in store.get("items", [])
        if entry.get("card_id")
    }


def make_discovered_entry(
    config: CategoryConfig,
    raw_card: dict[str, Any],
    *,
    listing_position: int,
) -> dict[str, Any]:
    """Normaliza un card del listado al formato persistido."""

    resource_id = raw_card.get("ID")
    if resource_id is None:
        raise ValueError(f"Card sin ID válido: {raw_card}")

    reg_url = normalize_url(raw_card.get("url"))
    entry: dict[str, Any] = {
        "card_id": f"card_{resource_id}",
        "resource_id": int(resource_id),
        "category": config.name,
        "source_list_url": config.list_url,
        "listing_position": listing_position,
        "card_title": normalize_inline_text(raw_card.get("title")),
        "card_subtitle": normalize_inline_text(raw_card.get("subtitle")),
        "card_summary": html_fragment_to_text(raw_card.get("description") or raw_card.get("excerpt")),
        "card_type": normalize_inline_text(raw_card.get("type")),
        "resource_type": normalize_inline_text(raw_card.get("resource_type")),
        "reg_url": reg_url,
        "confirm_url": None,
        "demo_title": None,
        "demo_description": None,
        "vidyard_url": None,
        "video_path": None,
        "permalink": normalize_url(raw_card.get("permalink")) if raw_card.get("permalink") else "",
        "raw_card_date": normalize_inline_text(raw_card.get("date")),
        "target_blank": "_blank" in normalize_inline_text(raw_card.get("target")),
        "workflow_stage": WorkflowStage.DISCOVERED.value,
        "errors": [],
        "last_error": None,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
    }
    return entry


def merge_discovered_entry(existing: dict[str, Any] | None, discovered: dict[str, Any]) -> dict[str, Any]:
    """Fusiona una entrada recién descubierta con una existente."""

    if existing is None:
        return discovered

    original_reg_url = existing.get("reg_url")
    downstream_invalidated = original_reg_url and original_reg_url != discovered.get("reg_url")

    for field in (
        "resource_id",
        "category",
        "source_list_url",
        "listing_position",
        "card_title",
        "card_subtitle",
        "card_summary",
        "card_type",
        "resource_type",
        "reg_url",
        "permalink",
        "raw_card_date",
        "target_blank",
    ):
        existing[field] = discovered.get(field)

    if downstream_invalidated:
        existing["confirm_url"] = None
        existing["demo_title"] = None
        existing["demo_description"] = None
        existing["vidyard_url"] = None
        existing["video_path"] = None
        append_error(
            existing,
            "discovery",
            "La URL de registro cambió respecto de una corrida anterior; "
            "se limpiaron etapas posteriores para reprocesar.",
            url=discovered.get("reg_url"),
        )

    existing.setdefault("errors", [])
    existing.setdefault("created_at", utc_now_iso())
    existing["workflow_stage"] = infer_stage(existing)
    existing["updated_at"] = utc_now_iso()
    return existing


def prune_limit(items: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """Aplica un límite opcional de trabajo por categoría."""

    if limit is None:
        return items
    return items[:limit]


async def import_playwright_components() -> tuple[Any, Any]:
    """Importa Playwright de forma diferida."""

    try:
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright no está instalado. Ejecuta: "
            "'pip install playwright' y luego 'playwright install chromium'."
        ) from exc
    return async_playwright, PlaywrightTimeoutError


def import_yt_dlp_module() -> Any:
    """Importa yt_dlp de forma diferida."""

    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt_dlp no está instalado. Ejecuta: 'pip install yt-dlp'.") from exc
    return yt_dlp


async def navigate_with_retries(page: Any, url: str, timeout_ms: int = DEFAULT_NAVIGATION_TIMEOUT_MS) -> None:
    """Navega a una URL con reintentos simples."""

    last_error: Exception | None = None
    for attempt in range(1, MAX_NAVIGATION_ATTEMPTS + 1):
        try:
            response = await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            if response and response.status >= 400:
                raise RuntimeError(f"HTTP {response.status} al abrir {url}")
            await settle_page(page)
            return
        except Exception as exc:  # pragma: no cover - depende de red
            last_error = exc
            if attempt >= MAX_NAVIGATION_ATTEMPTS:
                break
            await page.wait_for_timeout(700 * attempt)
    assert last_error is not None
    raise last_error


async def settle_page(page: Any, extra_wait_ms: int = DEFAULT_WAIT_AFTER_NAVIGATION_MS) -> None:
    """Da tiempo al navegador para completar render y requests secundarios."""

    try:
        await page.wait_for_load_state("load", timeout=DEFAULT_NETWORK_IDLE_TIMEOUT_MS)
    except Exception:
        pass
    try:
        await page.wait_for_load_state("networkidle", timeout=DEFAULT_NETWORK_IDLE_TIMEOUT_MS)
    except Exception:
        pass
    await page.wait_for_timeout(extra_wait_ms)


def build_chromium_launch_options(*, headless: bool) -> dict[str, Any]:
    """Construye opciones de lanzamiento para Chromium.

    Si hay un Chromium del sistema disponible, se usa explícitamente para
    evitar depender del navegador descargado por Playwright.
    """

    options: dict[str, Any] = {"headless": headless}
    for candidate in SYSTEM_CHROMIUM_CANDIDATES:
        if candidate.exists():
            options["executable_path"] = str(candidate)
            return options
    return options


async def extract_init_cards_payload(page: Any) -> dict[str, Any] | None:
    """Extrae ``init_cards`` desde la página de listado si existe."""

    payload = await page.evaluate(
        """
        () => {
          if (typeof window.init_cards === "undefined" || window.init_cards === null) {
            return null;
          }
          return JSON.parse(JSON.stringify(window.init_cards));
        }
        """
    )
    if isinstance(payload, dict) and payload.get("card") is not None:
        return payload

    html = await page.content()
    match = re.search(r"var\s+init_cards\s*=\s*(\{.*?\})\s*;</script>", html, flags=re.S)
    if not match:
        return None
    return json.loads(match.group(1))


def build_resource_api_url(list_url: str, page_number: int | None = None) -> str:
    """Construye la URL del endpoint paginado manteniendo el query string."""

    parsed = parse_url(list_url)
    query = parsed.query
    if page_number is not None:
        query = f"{query}&page={page_number}" if query else f"page={page_number}"
    return urlunsplit((parsed.scheme, parsed.netloc, RESOURCE_API_PATH, query, ""))


async def post_resource_api(
    request_context: Any,
    list_url: str,
    *,
    action: str,
    page_number: int | None = None,
    exclude_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Consulta el endpoint interno del Resource Center."""

    api_url = build_resource_api_url(list_url, page_number=page_number)
    form_items: list[tuple[str, str]] = [
        ("action", action),
        ("is_review", "0"),
    ]
    for exclude_id in exclude_ids or []:
        form_items.append(("exclude[]", str(exclude_id)))

    payload = urlencode(form_items, doseq=True)
    response = await request_context.post(
        api_url,
        data=payload,
        headers={"content-type": "application/x-www-form-urlencoded; charset=UTF-8"},
    )
    if not response.ok:
        raise RuntimeError(f"Error {response.status} consultando {api_url}")
    return await response.json()


def unique_cards_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extrae y deduplica cards de una respuesta del sitio."""

    ordered_cards: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for group_name in ("featuredCard", "card"):
        group = payload.get(group_name) or []
        if not isinstance(group, list):
            continue
        for raw_card in group:
            try:
                card_id = int(raw_card.get("ID"))
            except Exception:
                continue
            if card_id in seen_ids:
                continue
            seen_ids.add(card_id)
            ordered_cards.append(raw_card)
    return ordered_cards


async def discover_all_cards(
    page: Any,
    request_context: Any,
    config: CategoryConfig,
) -> tuple[list[dict[str, Any]], int | None, int | None]:
    """Descubre todos los cards de una categoría.

    Primero intenta usar ``init_cards`` desde el HTML real renderizado.
    Si eso falla, cae al endpoint interno que la página misma usa para
    cargar cards adicionales.
    """

    await navigate_with_retries(page, config.list_url)
    payload = await extract_init_cards_payload(page)
    if payload is None:
        LOGGER.info("%s: init_cards no apareció; usando fallback API load.", config.name)
        payload = await post_resource_api(request_context, config.list_url, action="load")

    total_found_posts = payload.get("total_found_posts")
    total_pages = payload.get("count")

    seen_ids: set[int] = set()
    all_cards: list[dict[str, Any]] = []
    for raw_card in unique_cards_from_payload(payload):
        seen_ids.add(int(raw_card["ID"]))
        all_cards.append(raw_card)

    exclude_ids = [int(value) for value in payload.get("exclude") or [] if str(value).isdigit()]
    if total_pages:
        for page_number in range(2, int(total_pages) + 1):
            page_payload = await post_resource_api(
                request_context,
                config.list_url,
                action="paging",
                page_number=page_number,
                exclude_ids=exclude_ids,
            )
            for raw_card in unique_cards_from_payload(page_payload):
                card_id = int(raw_card["ID"])
                if card_id in seen_ids:
                    continue
                seen_ids.add(card_id)
                all_cards.append(raw_card)

    LOGGER.info(
        "%s: descubiertos %s cards (total reportado por el sitio: %s).",
        config.name,
        len(all_cards),
        total_found_posts,
    )
    return all_cards, total_found_posts, total_pages


async def extract_text_via_dom_clone(page: Any, selector: str) -> str:
    """Obtiene texto legible incluso si el nodo está oculto por CSS."""

    raw_text = await page.evaluate(
        """
        (selector) => {
          const source = document.querySelector(selector);
          if (!source) {
            return "";
          }

          const clone = source.cloneNode(true);
          const container = document.createElement("div");
          container.style.position = "fixed";
          container.style.left = "-10000px";
          container.style.top = "0";
          container.style.visibility = "hidden";
          container.style.display = "block";
          container.style.width = "1200px";
          container.appendChild(clone);
          document.body.appendChild(container);
          const text = container.innerText || container.textContent || "";
          container.remove();
          return text;
        }
        """,
        selector,
    )
    return normalize_multiline_text(raw_text)


async def extract_demo_metadata(page: Any, reg_url: str) -> tuple[str, str, str]:
    """Extrae título, descripción y confirm_url desde una página ``reg``."""

    await navigate_with_retries(page, reg_url)

    title = ""
    for selector in ("#bannerHeadline", "#bannerHeadlineLine2", "h1"):
        title = await extract_text_via_dom_clone(page, selector)
        if title:
            break
    if not title:
        raise ValueError(f"No se encontró bannerHeadline en {reg_url}")

    subtitle = await extract_text_via_dom_clone(page, "#bannerSubHeadline")
    main_body = await extract_text_via_dom_clone(page, "#mainBodyContent")
    if not main_body:
        main_body = await extract_text_via_dom_clone(page, "#mainBodyContentHidden")
    disclaimer = await extract_text_via_dom_clone(page, ".disclaimer")

    description_parts = [part for part in (subtitle, main_body, disclaimer) if part]
    description = "\n\n".join(description_parts)

    return title, description, build_confirm_url(reg_url)


def normalize_vidyard_candidate(candidate: str) -> str | None:
    """Convierte distintos hints del DOM a una URL utilizable de Vidyard."""

    if not candidate:
        return None

    candidate = normalize_url(candidate) if "://" in candidate or candidate.startswith("//") else candidate.strip()
    if not candidate:
        return None

    if candidate.startswith("https://") or candidate.startswith("http://"):
        parsed = urlsplit(candidate)
        if "vidyard.com" not in parsed.netloc:
            return None

        path = parsed.path or ""
        suffix = Path(path).suffix.lower()
        if "/embed/" in path or suffix in {".js", ".css", ".svg", ".webp"}:
            return None
        if suffix in {".jpg", ".jpeg", ".png", ".gif"}:
            stem = Path(path).stem
            if re.fullmatch(r"[A-Za-z0-9_-]{8,}", stem):
                return f"https://play.vidyard.com/{stem}"
            return None
        return candidate

    if re.fullmatch(r"[A-Za-z0-9_-]{8,}", candidate):
        return f"https://play.vidyard.com/{candidate}"

    return None


def score_vidyard_url(url: str) -> tuple[int, int]:
    """Da prioridad a URLs de reproducción sobre assets auxiliares."""

    parsed = urlsplit(url)
    score = 0
    if parsed.netloc == "play.vidyard.com":
        score += 10
    if "watch" in parsed.path:
        score += 5
    if parsed.query:
        score += 4
    if "type=inline" in parsed.query:
        score += 1
    if Path(parsed.path).suffix == "":
        score += 2
    return score, len(url)


async def extract_vidyard_url(page: Any, confirm_url: str) -> str:
    """Extrae la mejor URL de Vidyard desde una página ``confirm``."""

    await navigate_with_retries(page, confirm_url)

    raw_candidates = await page.evaluate(
        """
        () => {
          const candidates = new Set();
          const push = (value) => {
            if (value && typeof value === "string") {
              candidates.add(value.trim());
            }
          };

          document.querySelectorAll("[href], [src]").forEach((node) => {
            const href = node.getAttribute("href");
            const src = node.getAttribute("src");
            if (href && href.includes("vidyard")) {
              push(href);
            }
            if (src && src.includes("vidyard")) {
              push(src);
            }
          });

          document.querySelectorAll("[data-uuid]").forEach((node) => {
            const uuid = node.getAttribute("data-uuid");
            const type = node.getAttribute("data-type");
            const version = node.getAttribute("data-v");
            if (!uuid) {
              return;
            }
            const url = new URL(`https://play.vidyard.com/${uuid}`);
            if (type) {
              url.searchParams.set("type", type);
            }
            if (version) {
              url.searchParams.set("v", version);
            }
            url.searchParams.set("disable_popouts", "1");
            push(url.toString());
            push(uuid);
          });

          if (window.performance && window.performance.getEntriesByType) {
            window.performance.getEntriesByType("resource").forEach((entry) => {
              if (entry.name && entry.name.includes("vidyard")) {
                push(entry.name);
              }
            });
          }

          push(document.documentElement.outerHTML);
          return Array.from(candidates);
        }
        """
    )

    candidates: list[str] = []
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, str):
            continue
        if "play.vidyard.com" in raw_candidate or "vidyard.com" in raw_candidate:
            for match in re.findall(r'https?://[^"\'\s<>]+', raw_candidate):
                normalized = normalize_vidyard_candidate(match)
                if normalized:
                    candidates.append(normalized)
        else:
            normalized = normalize_vidyard_candidate(raw_candidate)
            if normalized:
                candidates.append(normalized)

    unique_candidates = sorted(set(candidates), key=score_vidyard_url, reverse=True)
    if not unique_candidates:
        raise ValueError(f"No se encontró una URL de Vidyard en {confirm_url}")

    return unique_candidates[0]


async def scrape_category(
    playwright: Any,
    PlaywrightTimeoutError: type[Exception],
    config: CategoryConfig,
    *,
    headless: bool,
    limit_per_category: int | None,
) -> None:
    """Scrapea una categoría completa en su propia instancia Chromium."""

    del PlaywrightTimeoutError  # El valor se importa por claridad y compatibilidad futura.

    store = load_store(config)
    entry_index = build_entry_index(store)

    browser = await playwright.chromium.launch(**build_chromium_launch_options(headless=headless))
    request_context = await playwright.request.new_context()
    try:
        list_page = await browser.new_page()
        detail_page = await browser.new_page()

        raw_cards, total_found_posts, total_pages = await discover_all_cards(list_page, request_context, config)
        store["listing_total_found_posts"] = total_found_posts
        store["listing_total_pages"] = total_pages

        for position, raw_card in enumerate(raw_cards, start=1):
            discovered = make_discovered_entry(config, raw_card, listing_position=position)
            existing = entry_index.get(discovered["card_id"])
            merged = merge_discovered_entry(existing, discovered)
            if existing is None:
                store["items"].append(merged)
                entry_index[merged["card_id"]] = merged

        save_store(config, store)

        metadata_cache: dict[str, tuple[str, str, str]] = {}
        vidyard_cache: dict[str, str] = {}

        items = prune_limit(sort_items(store["items"]), limit_per_category)
        for item in items:
            reg_url = item.get("reg_url") or ""
            if not reg_url:
                append_error(item, "discovery", "La entrada no tiene reg_url.")
                save_store(config, store)
                continue

            if not entry_has_metadata(item):
                try:
                    if reg_url in metadata_cache:
                        title, description, confirm_url = metadata_cache[reg_url]
                    else:
                        title, description, confirm_url = await extract_demo_metadata(detail_page, reg_url)
                        metadata_cache[reg_url] = (title, description, confirm_url)

                    item["demo_title"] = title
                    item["demo_description"] = description
                    item["confirm_url"] = confirm_url
                    item["workflow_stage"] = WorkflowStage.METADATA_SCRAPED.value
                    item["last_error"] = None
                    item["updated_at"] = utc_now_iso()
                except Exception as exc:  # pragma: no cover - depende de red
                    append_error(item, "metadata", exc, url=reg_url)
                    save_store(config, store)
                    continue

                save_store(config, store)

            confirm_url = item.get("confirm_url")
            if not confirm_url:
                try:
                    item["confirm_url"] = build_confirm_url(reg_url)
                    confirm_url = item["confirm_url"]
                    item["last_error"] = None
                    item["updated_at"] = utc_now_iso()
                    save_store(config, store)
                except Exception as exc:
                    append_error(item, "confirm_url", exc, url=reg_url)
                    save_store(config, store)
                    continue

            if not entry_has_video_url(item):
                try:
                    if confirm_url in vidyard_cache:
                        vidyard_url = vidyard_cache[confirm_url]
                    else:
                        vidyard_url = await extract_vidyard_url(detail_page, confirm_url)
                        vidyard_cache[confirm_url] = vidyard_url

                    item["vidyard_url"] = vidyard_url
                    item["workflow_stage"] = WorkflowStage.VIDEO_URL_SCRAPED.value
                    item["last_error"] = None
                    item["updated_at"] = utc_now_iso()
                except Exception as exc:  # pragma: no cover - depende de red
                    append_error(item, "vidyard", exc, url=confirm_url)
                    save_store(config, store)
                    continue

                save_store(config, store)

    finally:
        try:
            await request_context.dispose()
        except Exception:
            pass
        try:
            await browser.close()
        except Exception:
            pass


async def _scrape_all_demos_async(*, headless: bool, limit_per_category: int | None) -> None:
    """Implementación asíncrona del scraping completo."""

    ensure_directories()
    async_playwright, PlaywrightTimeoutError = await import_playwright_components()
    async with async_playwright() as playwright:
        await asyncio.gather(
            *[
                scrape_category(
                    playwright,
                    PlaywrightTimeoutError,
                    config,
                    headless=headless,
                    limit_per_category=limit_per_category,
                )
                for config in CATEGORY_CONFIGS
            ]
        )


def scrape_all_demos(*, headless: bool = True, limit_per_category: int | None = None) -> None:
    """Scrapea los demos de producto y solución.

    :param headless: Si es ``False``, Chromium se abrirá en modo visible.
    :param limit_per_category: Limita la cantidad de demos procesados por categoría.
    """

    asyncio.run(_scrape_all_demos_async(headless=headless, limit_per_category=limit_per_category))


def resolve_downloaded_path(
    info: dict[str, Any],
    output_dir: Path,
    files_before: set[Path],
) -> Path:
    """Resuelve el archivo final descargado por ``yt_dlp``."""

    candidates: list[Path] = []

    def add_candidate(value: Any) -> None:
        if isinstance(value, str):
            path = Path(value).expanduser()
            if path.exists():
                candidates.append(path.resolve())

    add_candidate(info.get("_filename"))
    add_candidate(info.get("filepath"))

    for key in ("requested_downloads", "requested_formats"):
        for item in info.get(key) or []:
            if isinstance(item, dict):
                add_candidate(item.get("filepath"))
                add_candidate(item.get("_filename"))

    for path in candidates:
        if path.exists():
            return path

    files_after = {path.resolve() for path in output_dir.glob("*") if path.is_file()}
    new_files = sorted(files_after - files_before, key=lambda path: path.stat().st_mtime, reverse=True)
    if new_files:
        return new_files[0]

    raise FileNotFoundError(f"No fue posible determinar el archivo descargado en {output_dir}")


def download_single_video(
    yt_dlp: Any,
    entry: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Descarga un video de Vidyard con ``yt_dlp``."""

    vidyard_url = entry.get("vidyard_url")
    if not vidyard_url:
        raise ValueError("La entrada no tiene vidyard_url.")

    files_before = {path.resolve() for path in output_dir.glob("*") if path.is_file()}

    ydl_opts = {
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as downloader:
        info = downloader.extract_info(vidyard_url, download=True)
    return resolve_downloaded_path(info, output_dir, files_before)


def download_category_videos(config: CategoryConfig, *, limit_per_category: int | None) -> None:
    """Descarga los videos pendientes de una categoría."""

    if not config.json_path.exists():
        LOGGER.warning("%s: no existe %s, se omite.", config.name, config.json_path)
        return

    yt_dlp = import_yt_dlp_module()
    store = load_store(config)
    initial_summary = summarize_download_status(store)
    LOGGER.info("%s: estado inicial de descarga %s", config.name, initial_summary)

    known_downloads: dict[str, str] = {}
    for item in store["items"]:
        if entry_has_download(item) and item.get("vidyard_url"):
            known_downloads[item["vidyard_url"]] = item["video_path"]

    items = prune_limit(sort_items(store["items"]), limit_per_category)
    processed_any = False
    for item in items:
        vidyard_url = item.get("vidyard_url")
        if not vidyard_url:
            continue

        processed_any = True
        if entry_has_download(item):
            item["workflow_stage"] = WorkflowStage.DOWNLOADED.value
            item["last_error"] = None
            item["updated_at"] = utc_now_iso()
            save_store(config, store)
            continue

        if vidyard_url in known_downloads and Path(known_downloads[vidyard_url]).exists():
            item["video_path"] = known_downloads[vidyard_url]
            item["workflow_stage"] = WorkflowStage.DOWNLOADED.value
            item["last_error"] = None
            item["updated_at"] = utc_now_iso()
            save_store(config, store)
            continue

        try:
            downloaded_path = download_single_video(yt_dlp, item, config.video_dir)
            item["video_path"] = str(downloaded_path.resolve())
            item["workflow_stage"] = WorkflowStage.DOWNLOADED.value
            item["last_error"] = None
            item["updated_at"] = utc_now_iso()
            known_downloads[vidyard_url] = item["video_path"]
        except Exception as exc:  # pragma: no cover - depende de red/yt_dlp
            append_error(item, "download", exc, url=vidyard_url)
        finally:
            save_store(config, store)

    final_summary = summarize_download_status(store)
    if not processed_any:
        LOGGER.info("%s: no hay entradas con vidyard_url para descargar todavía.", config.name)
    elif final_summary["pending_download"] == 0:
        LOGGER.info("%s: no quedaron descargas pendientes.", config.name)
    LOGGER.info("%s: estado final de descarga %s", config.name, final_summary)


def download_all_videos(*, limit_per_category: int | None = None) -> None:
    """Descarga los videos descritos en los JSON de salida.

    :param limit_per_category: Limita la cantidad de descargas por categoría.
    """

    ensure_directories()
    for config in CATEGORY_CONFIGS:
        download_category_videos(config, limit_per_category=limit_per_category)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parsea los argumentos del CLI."""

    parser = argparse.ArgumentParser(description="Scraper puntual de demos de Planview.")
    parser.add_argument(
        "action",
        choices=("scrape", "download", "all"),
        help="Etapa a ejecutar.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Abre Chromium en modo visible durante el scraping.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Procesa solo los primeros N demos por categoría. Útil para smoke tests.",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit debe ser un entero positivo.")
    return args


def validate_run_configuration() -> None:
    """Valida la configuracion usada al ejecutar el archivo sin argumentos."""

    if RUN_MODE not in {"scrape", "download", "all"}:
        raise ValueError("RUN_MODE debe ser 'scrape', 'download' o 'all'.")
    if RUN_LIMIT is not None and RUN_LIMIT <= 0:
        raise ValueError("RUN_LIMIT debe ser None o un entero positivo.")


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada del script."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if not effective_argv:
        validate_run_configuration()
        action = RUN_MODE
        headed = not RUN_HEADLESS
        limit = RUN_LIMIT
    else:
        args = parse_args(effective_argv)
        action = args.action
        headed = args.headed
        limit = args.limit

    LOGGER.info(
        "Ejecutando modo=%s headed=%s limit=%s output_root=%s",
        action,
        headed,
        limit,
        OUTPUT_ROOT,
    )

    if action == "scrape":
        scrape_all_demos(headless=not headed, limit_per_category=limit)
    elif action == "download":
        download_all_videos(limit_per_category=limit)
    else:
        scrape_all_demos(headless=not headed, limit_per_category=limit)
        download_all_videos(limit_per_category=limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
