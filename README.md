# Electra-Core — Forensic-Adaptive Voter Roll PDF Extractor

Electra-Core is a robust data extraction tool designed to process noisy and complex voter roll PDFs. It extracts tabular voter data using a combination of Computer Vision techniques, OCR, and heuristic rules, and provides an intuitive Streamlit dashboard for Human-In-The-Loop (HITL) review and correction.

## Features
- **PDF Page Processing:** Renders pages at configurable DPI and identifies complex grid structures.
- **Adaptive Strategies:** Uses a priority chain of strategies (CV Grid Chop, Grid Projection, Blob Clustering) to handle various levels of scan degradation.
- **OCR Engine:** Pluggable OCR system to recognize voter details (EPIC ID, Name, Relation, Age, Gender, etc.).
- **Human Review Dashboard:** Streamlit UI to validate, correct, and finalize extracted entries that didn't pass strict layout validations.
- **Excel/CSV Exports:** Compiles cleanly structured final results into `.xlsx` and QA metrics into CSV.

## Architecture

Below is the dependency graph of the project codebase:

![Project Architecture](graphify-out/graph.svg)

## Installation

```bash
# Clone the repository
git clone <repository_url>
cd New_pdf2Excel

# Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

*(Note: Ensure that Poppler is installed on your system as it is required by `pdf2image`.)*

## Usage

### 1. Unified Dashboard (Recommended)

Run the Streamlit application for an end-to-end flow with visual extraction and review tabs:

```bash
streamlit run app.py
```

### 2. Command Line Interface

Process a PDF directly from the terminal. The output, QA reports, and the human review queue will be saved to the `output` directory.

```bash
python main.py path/to/voter_roll.pdf --output output_dir
```

- `--dpi`: Rendering resolution (default: 300)
- `--log-level`: Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`)

## Project Structure

- `app.py` & `review_app.py`: Streamlit frontends.
- `main.py`: CLI entry point.
- `pipeline/`: Orchestration and layout validation logic.
- `infrastructure/`: OCR engine implementations and CV extraction strategies.
- `domain/`: Core data models.
- `config/`: Project settings and logging configuration.
- `output/`: Generated CSVs, Excel files, and JSON review queues.

## Exit Codes (CLI)

- `0`: All pages processed successfully.
- `1`: Startup failure (e.g., PDF not found or rendering error).
- `2`: Partial success (some pages have been routed to the human-review queue).
