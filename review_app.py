"""
Electra-Core — Human Review App  (Multiple Choice Validation)
=============================================================
For every extracted field on every voter card the operator must explicitly
choose one of two options:

  ✅  Accept OCR        — trust the machine's extracted value
                          (pre-selected when OCR produced a non-empty result)
  ✏️  Correct manually  — type the correct value
                          (pre-selected when OCR is blank / missing)

All choices are collected per page; clicking "Submit Validated Page"
finalizes the record immediately and advances to the next queue item.

How to run
----------
    source .venv/bin/activate
    streamlit run review_app.py

    # Custom paths
    streamlit run review_app.py -- \\
        --queue  output/excel/human_review_queue.json \\
        --output output/excel/finalized_data.json
"""
from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

# ── Page config (must be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="Electra-Core Review",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CLI args ──────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--queue",
        default="output/human_review_queue.json",
        help="Path to human_review_queue.json produced by main.py",
    )
    parser.add_argument(
        "--output",
        default="output/finalized_data.json",
        help="Path for the finalized records JSON (created / appended)",
    )
    try:
        idx = sys.argv.index("--")
        args, _ = parser.parse_known_args(sys.argv[idx + 1 :])
    except ValueError:
        args, _ = parser.parse_known_args([])
    return args


_ARGS = _parse_args()
QUEUE_PATH = Path(_ARGS.queue)
OUTPUT_PATH = Path(_ARGS.output)

# ── Ordered field spec (key, display_label) ───────────────────────────────────
_CARD_FIELDS: List[tuple[str, str]] = [
    ("epic_id",       "EPIC ID"),
    ("serial_no",     "Serial No."),
    ("name",          "Name"),
    ("relation_type", "Relation Type"),
    ("relation_name", "Relation Name"),
    ("house_no",      "House No."),
    ("age",           "Age"),
    ("gender",        "Gender"),
]

# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data
def load_queue(path: str) -> List[Dict[str, Any]]:
    """Load the review queue JSON once; cached so widget interactions don't re-read."""
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else []


def _load_finalized() -> List[Dict]:
    if not OUTPUT_PATH.exists():
        return []
    try:
        with OUTPUT_PATH.open(encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        return []


def _append_finalized(record: Dict[str, Any]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_finalized()
    existing.append(record)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2, ensure_ascii=False)


def _image_or_none(path: str) -> Optional[Image.Image]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return Image.open(p)
    except (UnidentifiedImageError, OSError):
        return None


def _build_excel_bytes() -> bytes:
    """Generate deterministic Excel export from finalized voter data."""
    records = _load_finalized()
    df = export_to_excel(records)

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="FinalizedVoters")
    return buf.getvalue()


def _is_blank_card_payload(card: dict[str, Any]) -> bool:
    """Return True only when all useful identity signals are blank."""
    epic_clean = "".join(str(card.get("epic_id", "") or "").split())
    name_clean = "".join(str(card.get("name", "") or "").split())
    serial_clean = "".join(str(card.get("serial_no", "") or "").split())
    relation_clean = "".join(str(card.get("relation_name", "") or "").split())
    house_clean = "".join(str(card.get("house_no", "") or "").split())
    gender_clean = "".join(str(card.get("gender", "") or "").split())
    age_value = card.get("age", None)

    return not (
        epic_clean
        or name_clean
        or serial_clean
        or relation_clean
        or house_clean
        or gender_clean
        or age_value is not None
    )


def export_to_excel(json_data: list[dict]) -> pd.DataFrame:
    """Flatten voter JSON payload into a strict 1-to-1 Excel DataFrame.

    Args:
        json_data: List of page dictionaries containing card lists under keys
            such as ``cards``, ``records``, ``finalized_cards``, or
            ``extracted_data.cards``.

    Returns:
        A deduplicated DataFrame with schema:
        ``Page, Voter ID, Name, Relation, House No, Age, Gender``.
    """
    flat_rows: list[dict[str, Any]] = []

    for page in json_data:
        page_num = page.get("page_no", page.get("id", ""))

        cards: list[dict[str, Any]] = []

        page_cards = page.get("cards")
        if isinstance(page_cards, list):
            cards.extend(c for c in page_cards if isinstance(c, dict))

        page_records = page.get("records")
        if isinstance(page_records, list):
            cards.extend(c for c in page_records if isinstance(c, dict))

        finalized_cards = page.get("finalized_cards")
        if isinstance(finalized_cards, list):
            cards.extend(c for c in finalized_cards if isinstance(c, dict))

        extracted_data = page.get("extracted_data")
        if isinstance(extracted_data, dict):
            extracted_cards = extracted_data.get("cards")
            if isinstance(extracted_cards, list):
                cards.extend(c for c in extracted_cards if isinstance(c, dict))

        finalized_data = page.get("finalized_data")
        if isinstance(finalized_data, dict) and finalized_data:
            cards.append(finalized_data)

        for card in cards:
            epic_id = "".join(str(card.get("epic_id", "") or "").split())
            name = "".join(str(card.get("name", "") or "").split())
            if _is_blank_card_payload(card):
                continue  # Drop ghost cards (blank grid slots)

            relation = (
                str(card.get("relation_name", "") or "").strip()
                or str(card.get("relation_type", "") or "").strip()
            )

            flat_rows.append(
                {
                    "Page": page_num,
                    "Voter ID": epic_id,
                    "Name": name,
                    "Relation": relation,
                    "House No": str(card.get("house_no", "") or "").strip(),
                    "Age": card.get("age", ""),
                    "Gender": str(card.get("gender", "") or "").strip(),
                }
            )

    df = pd.DataFrame(
        flat_rows,
        columns=["Page", "Voter ID", "Name", "Relation", "House No", "Age", "Gender"],
    )

    # Safety net: remove rows where both identity fields are blank.
    df.dropna(subset=["Voter ID", "Name"], how="all", inplace=True)

    # Safe deduplication: never collapse rows missing Voter ID.
    id_mask = df["Voter ID"].astype(str).str.strip() != ""
    df_with_id = df[id_mask]
    df_missing_id = df[~id_mask]
    df_with_id = df_with_id.drop_duplicates(subset=["Voter ID", "Name"], keep="first")
    df = pd.concat([df_with_id, df_missing_id], ignore_index=True)

    # Excel-friendly cleanup.
    df.fillna("", inplace=True)

    return df


# ── Session-state helpers ─────────────────────────────────────────────────────

def _init_state() -> None:
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "finalized_count" not in st.session_state:
        st.session_state.finalized_count = 0


def _advance() -> None:
    st.session_state.current_index += 1
    st.session_state.finalized_count += 1


# ── Per-field multiple-choice row ─────────────────────────────────────────────

def _field_choice_row(
    field_key: str,
    label: str,
    ocr_value: Optional[Any],
    key_prefix: str,
) -> Optional[str]:
    """
    Render one field as a multiple-choice validation row.

    Layout
    ------
    **Label**
    ( ) ✅ Accept OCR: «extracted value»   ( ) ✏️ Correct manually
    [ text input — shown only when "Correct" is selected ]

    The radio defaults to "accept" when *ocr_value* is non-empty,
    and to "correct" when it is blank/missing.

    Returns the final resolved string value, or None if empty.
    """
    ocr_str = str(ocr_value).strip() if ocr_value is not None else ""
    has_ocr = bool(ocr_str)

    radio_key   = f"{key_prefix}_{field_key}_radio"
    correct_key = f"{key_prefix}_{field_key}_fix"

    default_idx = 0 if has_ocr else 1

    # Build the option labels once so the lambda closure is stable
    accept_label  = f"✅  Accept OCR:  «{ocr_str}»" if has_ocr else "✅  Accept OCR  (no value extracted)"
    correct_label = "✏️  Correct manually"

    st.markdown(f"**{label}**")
    choice = st.radio(
        label=label,
        options=["accept", "correct"],
        index=default_idx,
        key=radio_key,
        horizontal=True,
        format_func=lambda x, _a=accept_label, _c=correct_label: (
            _a if x == "accept" else _c
        ),
        label_visibility="collapsed",
    )

    if choice == "correct":
        correction = st.text_input(
            f"Correct value for {label}",
            value=ocr_str,          # pre-fill so operator only fixes typos
            key=correct_key,
            label_visibility="collapsed",
            placeholder=f"Type correct {label}…",
        )
        return correction.strip() or None

    # "accept" branch
    return ocr_str or None


# ── Single-card validation panel ─────────────────────────────────────────────

def _render_card_panel(
    card: Dict[str, Any],
    key_prefix: str,
    auto_expand: bool,
) -> Dict[str, Any]:
    """
    Render all fields for one card inside a collapsible expander.
    Returns a dict of {field_key: resolved_value, ...}.
    """
    card_idx = card.get("card_index", "?")
    epic     = card.get("epic_id") or "⚠ missing"
    flags    = card.get("parse_status") or []

    resolved: Dict[str, Any] = {"card_index": card_idx}

    with st.expander(f"Card {card_idx}  —  EPIC: {epic}", expanded=auto_expand):
        if flags:
            st.warning(f"OCR flags: {', '.join(flags)}", icon="⚠️")

        for field_key, label in _CARD_FIELDS:
            st.markdown("---")
            resolved[field_key] = _field_choice_row(
                field_key=field_key,
                label=label,
                ocr_value=card.get(field_key),
                key_prefix=f"{key_prefix}_card{card_idx}",
            )

    return resolved


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    _init_state()

    queue: List[Dict[str, Any]] = load_queue(str(QUEUE_PATH))
    total = len(queue)
    idx: int = st.session_state.current_index

    # ── App header ────────────────────────────────────────────────────────────
    st.title("🗳️ Electra-Core — Human Review Queue")

    # ── Empty queue guard ─────────────────────────────────────────────────────
    if total == 0:
        st.warning(
            f"No items found at `{QUEUE_PATH}`.\n\n"
            "Run `python main.py <pdf_file>` first to populate the queue, "
            "or check the `--queue` path."
        )
        return

    # ── Batch complete screen ─────────────────────────────────────────────────
    if idx >= total:
        st.balloons()
        st.success(
            f"🎉 **Batch Complete!**  All {total} items reviewed "
            f"({st.session_state.finalized_count} finalized)."
        )
        col_dl, col_restart, _ = st.columns([1, 1, 2])
        with col_dl:
            try:
                xlsx_bytes = _build_excel_bytes()
                st.download_button(
                    label="📥 Download Final Excel",
                    data=xlsx_bytes,
                    file_name="finalized_voter_data.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument"
                        ".spreadsheetml.sheet"
                    ),
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Could not build Excel: {exc}")
        with col_restart:
            if st.button("🔄 Review Again from Start", use_container_width=True):
                st.session_state.current_index = 0
                st.session_state.finalized_count = 0
                st.rerun()
        return

    # ── Progress ──────────────────────────────────────────────────────────────
    item = queue[idx]
    st.progress(
        idx / total,
        text=f"Reviewing **{idx + 1}** of **{total}** — Page {item.get('page_no', '?')}",
    )

    # ── Failure diagnosis strip (collapsed by default) ────────────────────────
    with st.expander("ℹ️ Strategy failure log", expanded=False):
        st.caption(f"**Last error:** {item.get('last_error', 'N/A')}")
        for attempt in item.get("attempts", []):
            st.caption(f"• {attempt}")
        st.caption(
            f"**Best strategy:** `{item.get('best_strategy') or '—'}`  "
            f"| **Quality ratio:** {item.get('best_ratio', 0):.0%}"
        )

    # ── Prepare card list ─────────────────────────────────────────────────────
    cards: List[Dict[str, Any]] = (
        item.get("extracted_data", {}).get("cards") or []
    )
    if not cards:
        # Gate-1 failure: no OCR was attempted — give the operator one blank card
        cards = [{"card_index": 1}]

    n_cards    = len(cards)
    n_with_epic = sum(1 for c in cards if c.get("epic_id"))

    # ── Split layout: image (left) | validation (right) ──────────────────────
    img_col, form_col = st.columns([1, 1], gap="large")

    with img_col:
        st.subheader("📄 Page Image  *(The Truth)*")
        img = _image_or_none(item.get("crop_path", ""))
        if img is not None:
            st.image(
                img,
                use_container_width=True,
                caption=f"Page {item.get('page_no', '?')}",
            )
        else:
            st.warning(
                "⚠ Page image unavailable — crop file not found or not yet generated."
            )
            st.code(item.get("crop_path") or "path not set", language=None)

    # final_cards is populated below during widget rendering; it must be
    # declared before `with form_col` so it survives outside that scope.
    final_cards: List[Dict[str, Any]] = []

    with form_col:
        st.subheader("✏️ Multiple Choice Validation")
        st.caption(
            "For **every field** choose  ✅ Accept OCR  or  ✏️ Correct manually.  "
            "Fields with no OCR result are pre-set to 'Correct'."
        )
        st.caption(
            f"{n_cards} card(s) — {n_with_epic} with EPIC ID  "
            f"({'⚠ all empty — manual entry required' if n_with_epic == 0 else '✅'})"
        )

        auto_expand = n_cards <= 4

        for card in cards:
            resolved = _render_card_panel(
                card=card,
                key_prefix=f"item{idx}",
                auto_expand=auto_expand,
            )
            final_cards.append(resolved)

        st.markdown("---")

        # ── Action buttons ────────────────────────────────────────────────────
        submit_col, skip_col, _ = st.columns([2, 1, 1])
        with submit_col:
            submit = st.button(
                "💾  Submit Validated Page",
                key=f"submit_{idx}",
                type="primary",
                use_container_width=True,
            )
        with skip_col:
            skip = st.button(
                "⏭️  Skip",
                key=f"skip_{idx}",
                use_container_width=True,
            )

    # ── Handle Submit ─────────────────────────────────────────────────────────
    if submit:
        persisted_cards = [
            c
            for c in final_cards
            if isinstance(c, dict) and not _is_blank_card_payload(c)
        ]
        _append_finalized({
            **item,
            "review_action":   "validated",
            "finalized_cards": persisted_cards,
            "finalized_at":    pd.Timestamp.now().isoformat(),
        })
        _advance()
        st.rerun()

    # ── Handle Skip ───────────────────────────────────────────────────────────
    elif skip:
        _append_finalized({
            **item,
            "review_action":   "skipped",
            "finalized_cards": [],
            "finalized_at":    pd.Timestamp.now().isoformat(),
        })
        _advance()
        st.rerun()

    # ── Sidebar: session stats ────────────────────────────────────────────────
    with st.sidebar:
        st.header("📊 Session Stats")
        st.metric("Reviewed",         st.session_state.finalized_count)
        st.metric("Remaining",        total - idx)
        st.metric("Total in queue",   total)
        st.divider()
        st.caption(f"**Queue:** `{QUEUE_PATH}`")
        st.caption(f"**Output:** `{OUTPUT_PATH}`")


if __name__ == "__main__":
    main()
