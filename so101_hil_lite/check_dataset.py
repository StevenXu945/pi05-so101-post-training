import argparse
import json
from pathlib import Path
from dataset_tools import report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate HIL parquet and outcomes; report actual sampling timing")
    parser.add_argument("source", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    text = json.dumps(report(args.source), indent=2, ensure_ascii=False)
    print(text)
    if args.report:
        args.report.write_text(text + "\n")
