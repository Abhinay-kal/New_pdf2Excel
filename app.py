from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pandas as pd
import streamlit as st

from config.settings import OUTPUT_DIR
from pipeline.orchestrator import PageProcessor

st.set_page_config(page_title="Electra-Core", layout="wide")

QUEUE_PATH: Path = OUTPUT_DIR / "human_review_queue.json"
FINALIZED_PATH: Path = OUTPUT_DIR / "finalized_data.json"
EXTRACT_XLSX_PATH: Path = OUTPUT_DIR / "voter_roll.xlsx"


def _read_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def _append_json_record(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = _read_json_list(path)
    current.append(record)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(current, fh, indent=2, ensure_ascii=False)


def _results_to_dataframe(results: List[Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for result in results:
        for card in result.cards:
            rows.append(
                {
                    "Page": result.page_no,
                    "CardIndex": card.card_index,
                    "StrategyUsed": result.strategy_used,
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


def _xlsx_bytes_from_df(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Voters")
    return buffer.getvalue()


def _reset_review_state() -> None:
    for key in list(st.session_state.keys()):
        if key == "review_index" or key.startswith("radio_") or key.startswith("input_"):
            st.session_state.pop(key, None)


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _build_finalized_export_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build a clean end-user export DataFrame from review records.

    Drops system metadata and standardizes column names.
    Supports multiple stored schemas (finalized_data / extracted_data / data).
    """
    rows: List[Dict[str, Any]] = []

    for rec in records:
        if not isinstance(rec, dict):
            continue

        payload: Dict[str, Any] = {}
        for key in ("finalized_data", "extracted_data", "data"):
            maybe = rec.get(key)
            if isinstance(maybe, dict):
                payload = maybe
                break
        if not payload:
            payload = rec

        page_number = (
            payload.get("page_number")
            or payload.get("page_no")
            or rec.get("page_no")
        )

        row = {
            "Page Number": _as_int(page_number),
            "Voter ID": payload.get("id") or payload.get("epic_id") or "",
            "Name": payload.get("name") or "",
            "Relation": payload.get("relation") or "",
            "House No": payload.get("house_no") or "",
            "Age": payload.get("age") or "",
            "Gender": payload.get("gender") or "",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Page Number",
                "Voter ID",
                "Name",
                "Relation",
                "House No",
                "Age",
                "Gender",
            ]
        )

    # Ensure stable chronological order by source page.
    df["_sort_page"] = pd.to_numeric(df["Page Number"], errors="coerce")
    df = df.sort_values(by=["_sort_page", "Name", "Voter ID"], na_position="last")
    df = df.drop(columns=["_sort_page"])
    df = df.reset_index(drop=True)

    return df


def _extract_field_map(item: Dict[str, Any]) -> Dict[str, str]:
    """
    Return a field/value mapping for the review form.

    Supports:
      1) item["extracted_data"] as a flat dict.
      2) item["extracted_data"]["cards"][0] when cards list exists.
      3) legacy fallback item["data"].
    """
    # Safely support both payload names used by different backend versions.
    record_data = item.get("extracted_data", item.get("data", {}))

    if isinstance(record_data, dict):
        cards = record_data.get("cards")
        if isinstance(cards, list) and cards and isinstance(cards[0], dict):
            first_card = cast(Dict[str, Any], cards[0])
            return {
                k: "" if v is None else str(v)
                for k, v in first_card.items()
                if k not in {"parse_status", "card_index", "region", "raw_ocr_text", "ocr_confidence"}
            }

        # Some payloads wrap the actual values under an inner "data" key.
        nested_data = record_data.get("data")
        if isinstance(nested_data, dict):
            return {k: "" if v is None else str(v) for k, v in nested_data.items()}

        return {
            k: "" if v is None else str(v)
            for k, v in record_data.items()
            if not isinstance(v, (dict, list))
        }

    return {}


def _render_extract_tab() -> None:
    st.subheader("Upload Voter Roll PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key="extract_pdf_uploader",
    )

    if uploaded_file is None:
        return

    if not st.button("Start Extraction", type="primary", key="extract_start_btn"):
        return

    tmp_path: Optional[str] = None
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        with st.status("Running extraction pipeline...", expanded=True) as status:
            st.write("Initializing PageProcessor")
            processor = PageProcessor()

            st.write("Processing PDF pages")
            page_results = processor.process_pdf(tmp_path)

            df = _results_to_dataframe(page_results)
            xlsx_bytes = _xlsx_bytes_from_df(df)
            with EXTRACT_XLSX_PATH.open("wb") as out_fh:
                out_fh.write(xlsx_bytes)

            status.update(label="Extraction complete", state="complete")

        total_rows = len(df)
        pages_extracted = len(page_results)
        pages_skipped = len(processor.skipped_pages)
        review_items = len(processor.human_review_queue)

        st.success("Extraction finished successfully.")
        st.json(
            {
                "pages_extracted": pages_extracted,
                "pages_skipped": pages_skipped,
                "items_sent_to_review": review_items,
                "total_rows": total_rows,
            }
        )

        # Reset review navigation for the newly generated queue.
        _reset_review_state()
        st.session_state["review_index"] = 0

        st.download_button(
            label="Download voter_roll.xlsx",
            data=xlsx_bytes,
            file_name="voter_roll.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="extract_download_xlsx",
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Extraction failed: {type(exc).__name__}: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _render_review_tab() -> None:
    st.subheader("Human Review Queue")

    queue = _read_json_list(QUEUE_PATH)
    if not queue:
        st.success("No items pending review.")
        return

    if "review_index" not in st.session_state:
        st.session_state["review_index"] = 0

    review_index = int(st.session_state["review_index"])
    if review_index >= len(queue):
        st.success("🎉 Review Complete")

        # Read finalized records at export time so we never depend on stale state.
        raw_records = _read_json_list(FINALIZED_PATH)
        if not raw_records and isinstance(st.session_state.get("finalized_records"), list):
            raw_records = cast(List[Dict[str, Any]], st.session_state["finalized_records"])

        final_data_list: List[Dict[str, Any]] = [
            rec for rec in raw_records if isinstance(rec, dict)
        ]
        df = _build_finalized_export_df(final_data_list)

        # Debug visibility before download.
        st.write("Data check:", final_data_list)
        st.dataframe(df, use_container_width=True)

        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)

        st.download_button(
            label="📥 Download Finalized Excel",
            data=buffer.getvalue(),
            file_name="final_corrected_voter_roll.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="review_download_finalized_excel",
        )
        return

    item = queue[review_index]
    item_id = str(item.get("id", f"item_{review_index}"))
    page_no = item.get("page_no", "")
    fields = _extract_field_map(item)

    if not fields:
        st.warning("Current review item has no extracted fields to validate.")

    st.progress((review_index + 1) / len(queue), text=f"Review item {review_index + 1} of {len(queue)}")
    if page_no != "":
        st.caption(f"Page: {page_no}")

    col_left, col_right = st.columns([1, 1], gap="large")
    with col_left:
        crop_path = item.get("crop_path")
        if isinstance(crop_path, str) and crop_path and Path(crop_path).exists():
            st.image(crop_path, caption="Original Crop", use_container_width=True)
        else:
            st.info("No image available for this queue item.")

    with col_right:
        st.markdown("### Validate Extracted Fields")
        with st.form(key=f"review_form_{item_id}_{review_index}"):
            finalized_fields: Dict[str, str] = {}

            for field_name, extracted_value in fields.items():
                st.markdown(f"**{field_name}**")
                radio_key = f"radio_{item_id}_{review_index}_{field_name}"
                input_key = f"input_{item_id}_{review_index}_{field_name}"

                choice = st.radio(
                    label=f"Review action for {field_name}",
                    options=["✅ Keep Extracted", "✏️ Correct Manually"],
                    key=radio_key,
                    horizontal=True,
                    label_visibility="collapsed",
                )

                if choice == "✏️ Correct Manually":
                    corrected = st.text_input(
                        label=f"Correct value for {field_name}",
                        value=extracted_value,
                        key=input_key,
                    )
                    finalized_fields[field_name] = corrected
                else:
                    finalized_fields[field_name] = extracted_value

                st.markdown("---")

            submitted = st.form_submit_button("💾 Submit Review & Next", type="primary")

        if submitted:
            record: Dict[str, Any] = {
                "id": item_id,
                "page_no": page_no,
                "source": "human_review",
                "finalized_data": finalized_fields,
            }
            _append_json_record(FINALIZED_PATH, record)
            finalized_records = st.session_state.get("finalized_records", [])
            if isinstance(finalized_records, list):
                finalized_records.append(record)
                st.session_state["finalized_records"] = finalized_records
            st.session_state["review_index"] = review_index + 1
            st.rerun()


def main() -> None:
    st.title("Electra-Core Unified Dashboard")
    extract_tab, review_tab = st.tabs(["🚀 Extract Data", "🔍 Human Review"])

    with extract_tab:
        _render_extract_tab()

    with review_tab:
        _render_review_tab()


if __name__ == "__main__":
    main()