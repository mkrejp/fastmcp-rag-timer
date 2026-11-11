"""Detect, render, and enrich large PDF figures (``pdf_figure`` source_type)."""

from __future__ import annotations

import hashlib
import io
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

from rag.config import (
    FIGURE_INDEX_VERSION,
    figure_index_enabled,
    resolve_figure_auto_check_min_confidence,
    resolve_figure_caption_max_gap,
    resolve_figure_context_max_chars,
    resolve_figure_max_per_doc,
    resolve_figure_min_area_ratio,
    resolve_figure_ocr_accept_confidence,
    resolve_figure_ocr_dpi,
    resolve_figure_render_dpi,
)
from rag.figure_summary import FigureLlmBudget, build_figure_summary
from rag.logging_setup import get_logger
from rag.ocr import ocr_pil_image
from rag.payloads import build_figure_payload

logger = get_logger("rag.pdf_figures")

CAPTION_RE = re.compile(
    r"^(Obr\.?|Obrázek|Fig\.?|Figure|Foto|Fotografie|Tab\.?|Tabulka|Mapa|Plán|Příloha)\s*"
    r"(\d+[a-z]?(?:\.\d+)?)",
    re.IGNORECASE,
)
EXPLICIT_REF_RE = re.compile(
    r"(?:viz|srov\.|na)?\s*(obr\.|obrázk\w*|fig\.|foto\w*|tab\.)\s*(\d+[a-z]?(?:\.\d+)?)",
    re.IGNORECASE,
)
IMPLICIT_REF_RE = re.compile(
    r"(na obrázku|na fotografii|vyobrazen|zachycuje)",
    re.IGNORECASE,
)

CONF_WEIGHT_CAPTION = 0.30
CONF_WEIGHT_EXPLICIT = 0.25
CONF_WEIGHT_GROUNDED = 0.20
CONF_WEIGHT_OCR = 0.15
CONF_WEIGHT_CHUNK_MAP = 0.10


def _sha256(data: bytes) -> str:
    """Return hex SHA-256."""
    return hashlib.sha256(data).hexdigest()


def _normalize_space(text: str) -> str:
    """Collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip())


def _clip_bbox(
    bbox: Sequence[float], page_rect: Any
) -> tuple[float, float, float, float]:
    """Intersect bbox with page rectangle."""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    return (
        max(page_rect.x0, x0),
        max(page_rect.y0, y0),
        min(page_rect.x1, x1),
        min(page_rect.y1, y1),
    )


def _bbox_area(bbox: Sequence[float]) -> float:
    """Return area of axis-aligned bbox."""
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _union_bbox(
    a: Sequence[float], b: Sequence[float]
) -> tuple[float, float, float, float]:
    """Return the union of two bboxes."""
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def _bboxes_merge(a: Sequence[float], b: Sequence[float], *, gap: float) -> bool:
    """Return whether bboxes overlap or are within ``gap``."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    if ax0 <= bx1 + gap and bx0 <= ax1 + gap and ay0 <= by1 + gap and by0 <= ay1 + gap:
        return True
    return False


@dataclass
class RawPlacement:
    """One image placement on a PDF page before tile merge."""

    page_index: int
    placement_index: int
    xref: int
    bbox: tuple[float, float, float, float]
    area_ratio: float
    digest: bytes | None = None


@dataclass
class FigurePlacement:
    """Merged figure candidate on one page."""

    page_index: int
    placement_index: int
    xrefs: list[int] = field(default_factory=list)
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    area_ratio: float = 0.0
    tiled: bool = False
    decorative: bool = False
    page_scan: bool = False


@dataclass
class BuiltFigure:
    """Fully enriched figure ready for Qdrant upsert."""

    figure_id: str
    payload: dict[str, Any]
    png_bytes: bytes
    skip_pdf_image_keys: set[tuple[int, int]]


def collect_raw_placements(
    pdf_bytes: bytes,
    *,
    min_area_ratio: float | None = None,
) -> list[RawPlacement]:
    """Scan all pages for image placements via PyMuPDF ``get_image_info``.

    When ``min_area_ratio`` is ``None``, every visible placement is returned so
    tile merging can run before the figure size threshold is applied.
    """
    try:
        import pymupdf as fitz
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PyMuPDF (pymupdf) is required for figure extraction"
        ) from exc

    if min_area_ratio is not None:
        apply_min = True
        min_ratio = float(min_area_ratio)
    else:
        apply_min = False
        min_ratio = resolve_figure_min_area_ratio()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    placements: list[RawPlacement] = []
    try:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            page_area = float(page.rect.get_area())
            if page_area <= 0:
                continue
            try:
                infos = page.get_image_info(xrefs=True)
            except Exception:
                infos = []
            for placement_index, info in enumerate(infos):
                bbox = _clip_bbox(info.get("bbox", (0, 0, 0, 0)), page.rect)
                if _bbox_area(bbox) <= 0:
                    continue
                ratio = _bbox_area(bbox) / page_area
                if apply_min and ratio < min_ratio:
                    continue
                xref = int(info.get("xref") or 0)
                placements.append(
                    RawPlacement(
                        page_index=page_index,
                        placement_index=placement_index,
                        xref=xref,
                        bbox=bbox,
                        area_ratio=ratio,
                        digest=info.get("digest"),
                    )
                )
    finally:
        doc.close()
    return placements


def merge_placements(
    raw: Sequence[RawPlacement], *, page_count: int
) -> list[FigurePlacement]:
    """Merge tile placements on each page; flag decorative xrefs."""
    xref_pages: dict[int, set[int]] = defaultdict(set)
    for item in raw:
        if item.xref:
            xref_pages[item.xref].add(item.page_index)

    threshold_pages = max(3, int(0.30 * page_count + 0.999))
    decorative_xrefs = {
        xref for xref, pages in xref_pages.items() if len(pages) >= threshold_pages
    }

    by_page: dict[int, list[RawPlacement]] = defaultdict(list)
    for item in raw:
        by_page[item.page_index].append(item)

    merged: list[FigurePlacement] = []
    per_page_counter: dict[int, int] = defaultdict(int)
    for page_index, items in sorted(by_page.items()):
        gap = 0.01  # fraction of page — applied per caller with page size later
        used = [False] * len(items)
        for i, first in enumerate(items):
            if used[i]:
                continue
            group = [first]
            used[i] = True
            bbox = first.bbox
            xrefs = [first.xref]
            for j, other in enumerate(items):
                if used[j] or i == j:
                    continue
                if _bboxes_merge(
                    bbox,
                    other.bbox,
                    gap=max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 0.01,
                ):
                    group.append(other)
                    used[j] = True
                    bbox = _union_bbox(bbox, other.bbox)
                    xrefs.append(other.xref)
            decorative = any(x in decorative_xrefs for x in xrefs)
            placement_index = per_page_counter[page_index]
            per_page_counter[page_index] += 1
            merged.append(
                FigurePlacement(
                    page_index=page_index,
                    placement_index=placement_index,
                    xrefs=xrefs,
                    bbox=bbox,
                    area_ratio=sum(g.area_ratio for g in group) / len(group),
                    tiled=len(group) > 1,
                    decorative=decorative,
                )
            )
    return merged


def _layout_blocks(page: Any) -> list[dict[str, Any]]:
    """Return text blocks from ``page.get_text('dict')``."""
    try:
        data = page.get_text("dict")
    except Exception:
        return []
    blocks = data.get("blocks") or []
    lines: list[dict[str, Any]] = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines") or []:
            spans = line.get("spans") or []
            text = "".join(str(s.get("text") or "") for s in spans)
            if not text.strip():
                continue
            bbox = line.get("bbox") or block.get("bbox")
            if not bbox:
                continue
            lines.append({"text": text.strip(), "bbox": tuple(bbox)})
    return lines


def _horizontal_overlap(a: Sequence[float], b: Sequence[float]) -> float:
    """Return horizontal overlap ratio relative to the narrower box."""
    ax0, _, ax1, _ = a
    bx0, _, bx1, _ = b
    overlap = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    width_a = max(1.0, ax1 - ax0)
    width_b = max(1.0, bx1 - bx0)
    return overlap / min(width_a, width_b)


def detect_caption(
    lines: Sequence[Mapping[str, Any]],
    figure_bbox: Sequence[float],
    *,
    page_height: float,
) -> tuple[str | None, str | None, str | None, str]:
    """Return caption text, label, number, and caption_source."""
    max_gap = resolve_figure_caption_max_gap() * page_height
    fx0, fy0, fx1, fy1 = figure_bbox
    best: tuple[float, str, str, str, str] | None = None
    for line in lines:
        text = str(line.get("text") or "")
        bbox = line.get("bbox")
        if not bbox:
            continue
        lx0, ly0, lx1, ly1 = bbox
        if _horizontal_overlap(figure_bbox, bbox) < 0.5:
            continue
        source = "none"
        gap = 0.0
        if ly0 >= fy1 and ly0 - fy1 <= max_gap:
            source = "below"
            gap = ly0 - fy1
        elif ly1 <= fy0 and fy0 - ly1 <= max_gap:
            source = "above"
            gap = fy0 - ly1
        elif lx0 >= fx1 and lx0 - fx1 <= max_gap:
            source = "side"
            gap = lx0 - fx1
        elif lx1 <= fx0 and fx0 - lx1 <= max_gap:
            source = "side"
            gap = fx0 - lx1
        else:
            continue
        if not CAPTION_RE.search(text):
            continue
        if best is None or gap < best[0]:
            match = CAPTION_RE.search(text)
            label = match.group(0) if match else None
            number = match.group(2) if match else None
            best = (gap, text, label or "", number or "", source)
    if best is None:
        return None, None, None, "none"
    _gap, caption, label, number, source = best
    return caption, label or None, number or None, source


def _page_texts_from_chunks(chunks: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    """Aggregate chunk text per page number."""
    pages: dict[int, list[str]] = defaultdict(list)
    for chunk in chunks:
        page = int(chunk.get("page", 0))
        text = str(chunk.get("text") or "")
        if text:
            pages[page].append(text)
    return {page: "\n".join(parts) for page, parts in pages.items()}


def _chunks_for_page(
    chunks: Sequence[Mapping[str, Any]], page: int
) -> list[Mapping[str, Any]]:
    """Return chunks whose page hint matches ``page``."""
    return [c for c in chunks if int(c.get("page", -1)) == page]


def _map_span_to_chunk(
    span: str,
    chunks: Sequence[Mapping[str, Any]],
    page: int,
) -> dict[str, Any] | None:
    """Find chunk index and char span for ``span`` on ``page``."""
    norm_span = _normalize_space(span)
    if not norm_span:
        return None
    for chunk in chunks:
        if int(chunk.get("page", -1)) != page:
            continue
        text = str(chunk.get("text") or "")
        norm_text = _normalize_space(text)
        idx = norm_text.find(norm_span)
        if idx < 0:
            compact_span = norm_span.replace(" ", "")
            compact_text = norm_text.replace(" ", "")
            idx = compact_text.find(compact_span)
            if idx >= 0:
                return {
                    "page": page,
                    "chunk_index": int(chunk.get("index", 0)),
                    "char_start": idx,
                    "char_end": idx + len(compact_span),
                    "score": 0.7,
                }
        else:
            return {
                "page": page,
                "chunk_index": int(chunk.get("index", 0)),
                "char_start": idx,
                "char_end": idx + len(norm_span),
                "score": 0.85,
            }
    return {"page": page, "score": 0.4}


def gather_related_refs(
    *,
    doc_page_texts: dict[int, str],
    chunks: Sequence[Mapping[str, Any]],
    figure_page: int,
    figure_bbox: Sequence[float],
    lines: Sequence[Mapping[str, Any]],
    caption: str | None,
    figure_number: str | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Collect related_text_refs and parallel span texts for summarization."""
    refs: list[dict[str, Any]] = []
    span_texts: list[str] = []
    max_chars = resolve_figure_context_max_chars()

    if caption:
        mapped = _map_span_to_chunk(caption, chunks, figure_page)
        ref = {
            **(mapped or {"page": figure_page}),
            "kind": "caption",
            "score": 0.9,
        }
        refs.append(ref)
        span_texts.append(caption)

    if figure_number:
        for page, text in doc_page_texts.items():
            for match in EXPLICIT_REF_RE.finditer(text):
                if match.group(2) != figure_number:
                    continue
                start = max(0, match.start() - 80)
                end = min(len(text), match.end() + 120)
                snippet = text[start:end].strip()
                mapped = _map_span_to_chunk(snippet, chunks, page)
                refs.append(
                    {
                        **(mapped or {"page": page}),
                        "kind": "explicit_ref",
                        "score": 0.85,
                    }
                )
                span_texts.append(snippet)

    for page in (figure_page, figure_page - 1, figure_page + 1):
        if page < 0 or page not in doc_page_texts:
            continue
        text = doc_page_texts[page]
        if IMPLICIT_REF_RE.search(text) and page != figure_page:
            mapped = _map_span_to_chunk(text[:400], chunks, page)
            refs.append(
                {
                    **(mapped or {"page": page}),
                    "kind": "adjacent_page",
                    "score": 0.5,
                }
            )
            span_texts.append(text[:400])
        elif page == figure_page:
            ranked = sorted(
                lines,
                key=lambda line: abs(float(line["bbox"][1]) - float(figure_bbox[1])),
            )
            context_parts: list[str] = []
            total = 0
            for line in ranked:
                piece = str(line.get("text") or "")
                if not piece or (caption and piece == caption):
                    continue
                if total + len(piece) > max_chars:
                    break
                context_parts.append(piece)
                total += len(piece)
            if context_parts:
                snippet = " ".join(context_parts)
                mapped = _map_span_to_chunk(snippet, chunks, page)
                refs.append(
                    {
                        **(mapped or {"page": page}),
                        "kind": "same_page",
                        "score": 0.6,
                    }
                )
                span_texts.append(snippet)

    deduped_refs: list[dict[str, Any]] = []
    deduped_spans: list[str] = []
    seen: set[str] = set()
    for ref, span in zip(refs, span_texts, strict=False):
        key = _normalize_space(span)[:80]
        if key in seen:
            continue
        seen.add(key)
        deduped_refs.append(ref)
        deduped_spans.append(span)
    return deduped_refs, deduped_spans


def _ocr_from_text_layer(
    page: Any, bbox: Sequence[float]
) -> tuple[str, str, float | None, int, str]:
    """Extract text inside bbox via PDF text layer."""
    try:
        text = page.get_textbox(bbox) or ""
    except Exception:
        text = ""
    text = _normalize_space(text)
    if not text:
        return "", "", None, 0, "none"
    words = text.split()
    return text, text, 100.0, len(words), "text_layer"


def compute_verification(
    *,
    area_ratio: float,
    caption: str | None,
    explicit_ref_count: int,
    ocr_text: str,
    ocr_confidence: float | None,
    ocr_source: str,
    summary_sentences: Sequence[Mapping[str, Any]],
    related_text_refs: Sequence[Mapping[str, Any]],
    span_texts: Sequence[str],
    decorative: bool,
    min_ratio: float,
) -> tuple[dict[str, Any], float, str, bool]:
    """Return checks dict, confidence, status, verified flag."""
    accept_conf = resolve_figure_ocr_accept_confidence()
    area_ok = area_ratio >= min_ratio
    caption_found = bool(caption)
    has_ocr = bool(ocr_text) or ocr_source == "text_layer"
    ocr_ok = not has_ocr or (
        ocr_confidence is not None
        and ocr_confidence >= accept_conf
        and len(ocr_text.split()) >= 1
    )
    grounded = True
    for sentence in summary_sentences:
        ref_i = int(sentence.get("ref", -1))
        if ref_i < 0 or ref_i >= len(related_text_refs):
            grounded = False
            break
        source = span_texts[ref_i] if ref_i < len(span_texts) else ""
        sent_text = str(sentence.get("text") or "")
        if source and _normalize_space(sent_text) not in _normalize_space(source):
            overlap = len(
                set(sent_text.lower().split()) & set(source.lower().split())
            ) / max(1, len(sent_text.split()))
            if overlap < 0.5:
                grounded = False
                break
    mapped = sum(1 for ref in related_text_refs if ref.get("chunk_index") is not None)
    map_ratio = mapped / max(1, len(related_text_refs))

    conf = 0.0
    if caption_found:
        conf += CONF_WEIGHT_CAPTION
    conf += CONF_WEIGHT_EXPLICIT * min(1.0, explicit_ref_count / 2.0)
    if grounded:
        conf += CONF_WEIGHT_GROUNDED
    if has_ocr and ocr_ok:
        conf += CONF_WEIGHT_OCR
    elif not has_ocr:
        conf += CONF_WEIGHT_OCR
    conf += CONF_WEIGHT_CHUNK_MAP * map_ratio

    checks = {
        "area_ok": area_ok,
        "caption_found": caption_found,
        "explicit_refs": explicit_ref_count,
        "ocr_ok": ocr_ok,
        "summary_grounded": grounded,
        "decorative": decorative,
    }
    auto_min = resolve_figure_auto_check_min_confidence()
    if (
        not decorative
        and area_ok
        and grounded
        and (caption_found or explicit_ref_count >= 1)
        and (not has_ocr or ocr_ok)
        and conf >= auto_min
    ):
        status = "auto_checked"
    else:
        status = "unverified"
    return (
        checks,
        conf,
        status,
        False,
    )


def iterate_figure_records(
    pdf_bytes: bytes,
    *,
    doc_id: str,
    parent_document: str,
    doc_sha256: str,
    chunks: Sequence[Mapping[str, Any]],
    llm_budget: FigureLlmBudget | None = None,
) -> Iterator[BuiltFigure]:
    """Yield enriched figures page-by-page (streaming friendly)."""
    if not figure_index_enabled():
        return

    try:
        import pymupdf as fitz
        from PIL import Image
    except ImportError:
        logger.warning("Figure pipeline skipped: missing pymupdf/PIL")
        return

    raw = collect_raw_placements(pdf_bytes, min_area_ratio=None)
    if not raw:
        return

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page_count = doc.page_count
        placements = merge_placements(raw, page_count=page_count)
        doc_page_texts = _page_texts_from_chunks(chunks)
        render_dpi = resolve_figure_render_dpi()
        ocr_dpi = resolve_figure_ocr_dpi()
        min_ratio = resolve_figure_min_area_ratio()
        max_figures = resolve_figure_max_per_doc()
        produced = 0

        for placement in placements:
            if produced >= max_figures:
                break
            if placement.decorative:
                continue
            page = doc.load_page(placement.page_index)
            page_area = float(page.rect.get_area())
            clipped = _clip_bbox(placement.bbox, page.rect)
            area_ratio = _bbox_area(clipped) / page_area if page_area else 0.0
            placement.area_ratio = area_ratio
            if area_ratio < min_ratio:
                continue

            plain = (page.get_text() or "").strip()
            if area_ratio >= 0.9 and not plain:
                placement.page_scan = True
                continue

            lines = _layout_blocks(page)
            caption, label, figure_number, caption_source = detect_caption(
                lines,
                clipped,
                page_height=float(page.rect.height),
            )

            related_refs, span_texts = gather_related_refs(
                doc_page_texts=doc_page_texts,
                chunks=chunks,
                figure_page=placement.page_index,
                figure_bbox=clipped,
                lines=lines,
                caption=caption,
                figure_number=figure_number,
            )
            explicit_count = sum(
                1 for ref in related_refs if ref.get("kind") == "explicit_ref"
            )

            render_pix = page.get_pixmap(clip=fitz.Rect(clipped), dpi=render_dpi)
            png_bytes = render_pix.tobytes("png")
            image_sha256 = _sha256(png_bytes)
            image_id = image_sha256[:16]

            ocr_text, ocr_raw, ocr_conf, ocr_words, ocr_source = _ocr_from_text_layer(
                page, clipped
            )
            if not ocr_text:
                ocr_pix = page.get_pixmap(clip=fitz.Rect(clipped), dpi=ocr_dpi)
                ocr_png = ocr_pix.tobytes("png")
                with Image.open(io.BytesIO(ocr_png)) as ocr_img:
                    ocr_text, ocr_raw, ocr_conf, ocr_words, ocr_source = ocr_pil_image(
                        ocr_img,
                        accept_min_confidence=resolve_figure_ocr_accept_confidence(),
                    )

            summary, summary_sentences, summary_method, summary_model = (
                build_figure_summary(
                    caption=caption,
                    related_text_refs=related_refs,
                    span_texts=span_texts or ([caption] if caption else [""]),
                    ocr_text=ocr_text or None,
                    image_sha256=image_sha256,
                    llm_budget=llm_budget,
                )
            )

            checks, confidence, status, verified = compute_verification(
                area_ratio=area_ratio,
                caption=caption,
                explicit_ref_count=explicit_count,
                ocr_text=ocr_text,
                ocr_confidence=ocr_conf,
                ocr_source=ocr_source,
                summary_sentences=summary_sentences,
                related_text_refs=related_refs,
                span_texts=span_texts,
                decorative=placement.decorative,
                min_ratio=min_ratio,
            )

            figure_id = f"f_{doc_sha256[:12]}_p{placement.page_index + 1}_{placement.placement_index}"
            payload = build_figure_payload(
                doc_id=doc_id,
                parent_document=parent_document,
                doc_sha256=doc_sha256,
                figure_id=figure_id,
                page_number=placement.page_index,
                bbox=clipped,
                page_size=(float(page.rect.width), float(page.rect.height)),
                area_ratio=area_ratio,
                placements=len(placement.xrefs),
                tiled=placement.tiled,
                image_id=image_id,
                image_sha256=image_sha256,
                file_path=f"processed/{doc_id}/images/{image_id}.png",
                render={"dpi": render_dpi, "clip": True},
                figure_label=label,
                figure_number=figure_number,
                caption=caption,
                caption_source=caption_source,
                ocr_text=ocr_text,
                ocr_text_raw=ocr_raw,
                ocr_confidence=ocr_conf,
                ocr_word_count=ocr_words,
                ocr_source=ocr_source,
                related_text_refs=related_refs,
                summary=summary,
                summary_sentences=summary_sentences,
                summary_method=summary_method,
                summary_model=summary_model,
                verified=verified,
                verification_status=status,
                confidence=confidence,
                verification={
                    "checks": checks,
                    "notes": None,
                    "verified_by": None,
                    "verified_at": None,
                    "human_overrides": {"caption": None, "summary": None},
                },
                figure_index_version=FIGURE_INDEX_VERSION,
            )
            skip_keys = {(placement.page_index, xref) for xref in placement.xrefs}
            produced += 1
            yield BuiltFigure(
                figure_id=figure_id,
                payload=payload,
                png_bytes=png_bytes,
                skip_pdf_image_keys=skip_keys,
            )
    finally:
        doc.close()


def stable_figure_point_id(doc_id: str, figure_id: str) -> str:
    """Stable Qdrant id for a figure (delegates to payloads helper)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:fig:{figure_id}"))
