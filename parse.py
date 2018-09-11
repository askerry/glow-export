#!/usr/bin/env python

import os

import pandas as pd

import parse_history
import parse_summary

# Directory containing PDF data exports
PDF_DIR = "pdfs"

# Directory where output CSVs will be written
OUTPUT_DIR = "csvs"


def _write_summary(rows):
    """Create single concatenated file for all weekly summaries."""
    df = pd.concat(rows).sort_values(by="week_start_date")
    output_file = os.path.join(OUTPUT_DIR, "summary.csv")
    df.to_csv(output_file, index=False)


def _write_events(events):
    """Create single concatenated file for full event history."""
    df = pd.concat(events).sort_values(by=["timestamp"])
    output_file = os.path.join(OUTPUT_DIR, "events.csv")
    df.to_csv(output_file, index=False)


def process_export_pdfs():

    # Attempt to parse all PDFS in the specified directory
    files = os.listdir(PDF_DIR)
    summary_rows, events = [], []
    for f in files:
        if not f.endswith(".pdf"):
            continue
        summary_rows.append(parse_summary.parse(PDF_DIR, f))
        events.append(parse_history.extract(PDF_DIR, f))

    # Write results to CSV
    if not os.path.isdir(OUTPUT_DIR):
        os.path.mkdir(OUTPUT_DIR)
    _write_summary(summary_rows)
    _write_events(events)


if __name__ == "__main__":
    process_export_pdfs()

