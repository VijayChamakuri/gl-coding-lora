from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REFERENCE = DATA / "reference" / "stockport_labelled_lines.csv.gz"
PURPOSE_MAP = DATA / "stockport_purpose_map.csv"
RAW = DATA / "raw" / "stockport"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"

REFERENCE_SHA256 = "d202870296c57f95014b9f6ad09bc5bc2b3ca68180eaafb37aad3bbc51d3e37b"
