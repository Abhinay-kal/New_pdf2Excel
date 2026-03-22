#!/usr/bin/env python3
"""
Electra-Core — Forensic-Adaptive Voter Roll PDF Extractor
==========================================================

CLI entry point.

Usage
-----
    python main.py <pdf_path> [options]

Options
-------
    --dpi         Rendering DPI for pdf2image  (default: 300)
    --output      Directory for the output .xlsx file
    --log-level   Verbosity: DEBUG | INFO | WARNING | ERROR  (default: INFO)

Exit codes
----------
    0  All pages processed successfully.
    1  Startup failure (PDF not found, render error).
    2  Partial success — at least one page is in the human-review queue.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd
from pdf2image import convert_from_path

# ── Bootstrap logging before any other local import ───────────────────────────
# (logging_config imports config.settings which has no heavy deps)
from config.logging_config import setup_logging
from config.settings import DEFAULT_DPI, OUTPUT_DIR, POPPLER_PATH
from infrastructure.ocr.engine import OcrEngine
from infrastructure.strategies import (
    BlobClusteringStrategy,
    CvGridChopStrategy,
    GridProjectionStrategy,
)
from pipeline.orchestrator import PageProcessor, PageResult, SkippedPageResult
from pipeline.validator import LayoutValidator

log = logging.getLogger(__name__)


# ── Wiring ────────────────────────────────────────────────────────────────────

def _build_processor() -> PageProcessor:
    """
    Assemble the strategy chain in priority order and return a
    configured PageProcessor.

    Priority (most robust for noisy scans first, per quality-gate upgrade):
      1. CvGridChopStrategy      — adaptive threshold morphology, Project A
      2. GridProjectionStrategy  — pure NumPy projection,         Project B
      3. BlobClusteringStrategy  — dilation blobs fallback,       Project A
    """
    strategies = [
        CvGridChopStrategy(),
        GridProjectionStrategy(),
        BlobClusteringStrategy(),
    ]
    return PageProcessor(
        strategies=strategies,
        ocr_engine=OcrEngine(),
        validator=LayoutValidator(),
    )


# ── Export ────────────────────────────────────────────────────────────────────

def _results_to_dataframe(results: List[PageResult]) -> pd.DataFrame:
    """Flatten all VoterCard objects from processed pages into one DataFrame."""
    rows = []
    for result in results:
        for card in result.cards:
            rows.append(
                {
                    "Page": result.page_no,
                    "CardIndex": card.card_index,
                    "StrategyUsed": result.strategy_used,
                    "ValidityRatio": f"{result.validity_ratio:.0%}",
                    "EPIC_ID": card.epic_id or "",
                    "SerialNo": card.serial_no or "",
                    "Name": card.name or "",
                    "RelationType": card.relation_type or "",
                    "RelationName": card.relation_name or "",
                    "HouseNo": card.house_no or "",
                    "Age": card.age if card.age is not None else "",
                    "Gender": card.gender or "",
                    "ParseStatus": ",".join(card.parse_status),
                }
            )
    return pd.DataFrame(rows)


def _build_page_qa_dataframe(
    processor: PageProcessor,
    results: List[PageResult],
    total_pages: int,
) -> pd.DataFrame:
    """
    Build a per-page QA table to make extraction regressions visible.

    Columns include:
      - missing_epic_ratio: fraction of rows without EPIC among extracted cards.
      - ok_row_count: number of cards with no parse-status flags.
      - skip_reason: classification skip cause or strategy failure summary.
    """
    accepted_by_page = {r.page_no: r for r in results}
    skipped_by_page = {s.page_no: s for s in processor.skipped_pages}
    review_by_page = {h.page_no: h for h in processor.human_review_queue}

    rows = []
    for page_no in range(1, total_pages + 1):
        row = {
            "Page": page_no,
            "PageStatus": "unknown",
            "PageType": "",
            "StrategyUsed": "",
            "CardCount": "",
            "MissingEPICCount": "",
            "MissingEPICRatio": "",
            "OKRowCount": "",
            "SkipReason": "",
        }

        if page_no in accepted_by_page:
            result = accepted_by_page[page_no]
            cards = result.cards
            card_count = len(cards)
            missing_epic = sum(1 for c in cards if not c.epic_id)
            ok_rows = sum(1 for c in cards if not c.parse_status)
            ratio = (missing_epic / card_count) if card_count else 0.0

            row.update(
                {
                    "PageStatus": "accepted",
                    "PageType": result.page_type.value,
                    "StrategyUsed": result.strategy_used,
                    "CardCount": card_count,
                    "MissingEPICCount": missing_epic,
                    "MissingEPICRatio": f"{ratio:.2%}",
                    "OKRowCount": ok_rows,
                }
            )

        elif page_no in skipped_by_page:
            skipped = skipped_by_page[page_no]
            row.update(
                {
                    "PageStatus": "skipped",
                    "PageType": skipped.page_type.value,
                    "SkipReason": f"classified_as_{skipped.page_type.value}",
                }
            )

        elif page_no in review_by_page:
            review_item = review_by_page[page_no]
            cards = review_item.best_cards
            card_count = len(cards)
            missing_epic = sum(1 for c in cards if not c.epic_id) if cards else ""
            ok_rows = sum(1 for c in cards if not c.parse_status) if cards else ""
            ratio = (missing_epic / card_count) if card_count else None

            row.update(
                {
                    "PageStatus": "review_queue",
                    "PageType": "VOTER_LIST",
                    "StrategyUsed": review_item.best_strategy or "",
                    "CardCount": card_count if card_count else "",
                    "MissingEPICCount": missing_epic,
                    "MissingEPICRatio": f"{ratio:.2%}" if ratio is not None else "",
                    "OKRowCount": ok_rows,
                    "SkipReason": review_item.last_error,
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def _write_page_qa_report(
    processor: PageProcessor,
    results: List[PageResult],
    total_pages: int,
    output_dir: Path,
) -> None:
    """Persist page-level QA metrics as a CSV file."""
    df = _build_page_qa_dataframe(processor, results, total_pages)
    out_csv = output_dir / "page_qa_report.csv"
    df.to_csv(out_csv, index=False)
    log.info("Page QA report saved → %s", out_csv)


def _write_review_report(processor: PageProcessor, output_dir: Path) -> None:
    """Write a plain-text triage report for all pages in the review queue."""
    if not processor.human_review_queue:
        return
    report_path = output_dir / "human_review_queue.txt"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("Electra-Core — Human Review Queue\n")
        fh.write("=" * 50 + "\n\n")
        for item in processor.human_review_queue:
            fh.write(f"Page {item.page_no}\n")
            fh.write(f"  Last error : {item.last_error}\n")
            for attempt in item.attempts:
                fh.write(f"    - {attempt}\n")
            fh.write("\n")
    log.info("Human-review report saved → %s", report_path)

def _write_review_json(
    processor: PageProcessor,
    images: List,
    output_dir: Path,
) -> None:
    """
    Save page-image crops (JPEG) and a structured JSON file for review_app.py.

    Output layout inside *output_dir*:
        human_review_queue.json   -- structured queue consumed by review_app.py
        crops/page_<N>.jpg        -- full rendered page image for each review item
    """
    if not processor.human_review_queue:
        return

    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for item in processor.human_review_queue:
        # Persist the full page image so the operator can see the raw scan.
        page_idx = item.page_no - 1          # images list is 0-based
        crop_path_str: str = ""
        if 0 <= page_idx < len(images):
            crop_file = crops_dir / f"page_{item.page_no}.jpg"
            images[page_idx].save(str(crop_file), format="JPEG", quality=85)
            crop_path_str = str(crop_file)

        # Serialize the best-guess OCR cards (empty list for gate-1 failures).
        cards_data = [
            {
                "card_index":    c.card_index,
                "epic_id":       c.epic_id,
                "serial_no":     c.serial_no,
                "name":          c.name,
                "relation_type": c.relation_type,
                "relation_name": c.relation_name,
                "house_no":      c.house_no,
                "age":           c.age,
                "gender":        c.gender,
                "parse_status":  c.parse_status,
            }
            for c in item.best_cards
        ]

        records.append({
            "id":            f"page_{item.page_no}",
            "page_no":       item.page_no,
            "crop_path":     crop_path_str,
            "last_error":    item.last_error,
            "attempts":      item.attempts,
            "best_strategy": item.best_strategy,
            "best_ratio":    item.best_ratio,
            "extracted_data": {"cards": cards_data},
        })

    json_path = output_dir / "human_review_queue.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
    log.info(
        "Review assets saved → %s  (%d items, crops in %s)",
        json_path,
        len(records),
        crops_dir,
    )

# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="electra-core",
        description="Forensic-Adaptive Voter Roll PDF Extractor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the source voter-roll PDF.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Rendering resolution for pdf2image.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for output files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Root logger verbosity.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging(args.log_level)

    pdf_path: Path = args.pdf_path.resolve()
    if not pdf_path.is_file():
        log.error("PDF not found: %s", pdf_path)
        return 1

    log.info(
        "Electra-Core starting | pdf=%s dpi=%d output=%s",
        pdf_path,
        args.dpi,
        args.output,
    )

    # ── Render PDF pages ───────────────────────────────────────────────────────
    try:
        images = convert_from_path(
            str(pdf_path),
            dpi=args.dpi,
            poppler_path=POPPLER_PATH,
        )
    except Exception as exc:
        log.exception("Failed to render PDF: %s", exc)
        return 1

    log.info("Rendered %d page(s) at %d DPI", len(images), args.dpi)

    # ── Process ────────────────────────────────────────────────────────────────
    processor = _build_processor()
    results = processor.process_pdf(images)

    # ── Summary ────────────────────────────────────────────────────────────────
    review_count = len(processor.human_review_queue)
    skipped_count = len(processor.skipped_pages)
    log.info(
        "Results: voter_list=%d  skipped=%d  review_queue=%d  total_pages=%d",
        len(results),
        skipped_count,
        review_count,
        len(images),
    )
    if skipped_count:
        from collections import Counter
        type_counts = Counter(s.page_type.name for s in processor.skipped_pages)
        log.info("Skipped page breakdown: %s", dict(type_counts))
    if review_count:
        log.warning(
            "Pages requiring human review: %s",
            [item.page_no for item in processor.human_review_queue],
        )

    # ── Export ─────────────────────────────────────────────────────────────────
    args.output.mkdir(parents=True, exist_ok=True)

    if results:
        df = _results_to_dataframe(results)
        out_xlsx = args.output / f"{pdf_path.stem}_extracted.xlsx"
        df.to_excel(out_xlsx, index=False)
        log.info("Saved %d voter records → %s", len(df), out_xlsx)

    _write_review_report(processor, args.output)
    _write_review_json(processor, images, args.output)
    _write_page_qa_report(
        processor=processor,
        results=results,
        total_pages=len(images),
        output_dir=args.output,
    )

    return 0 if review_count == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
