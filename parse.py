import os
import re

import pandas as pd
import numpy as np
import tabula

PDF_DIR = "pdfs"
OUTPUT_DIR = "csvs"


def parse_summary(f):
    path = os.path.join(PDF_DIR, f)
    print("Parsing summary tables from %s" % path)
    (raw_summary, raw_feedings, _, raw_diapers) = tabula.read_pdf(
        path, multiple_tables=True)

    # Parse each of the tables into a flattened, simplified format
    summary = _parse_executive_summary(raw_summary)
    diapers = _parse_diapers(raw_diapers)
    feedings = _parse_feedings(raw_feedings)
    df = pd.concat([summary, diapers, feedings], axis=1)

    # Normalize column names
    df = df.rename(mapper=lambda c: "_".join(c.lower().split(" ")), axis=1)

    # Add date field
    filebase = os.path.splitext(f)[0]
    date = filebase.split("_")[-1]
    date_str = "%s-%s-%s" % (date[4:], date[0:2], date[2:4])
    df.insert(0, "week_start_date", date_str)

    return df


def _parse_executive_summary(df):
    """Parse and flatten top-level "executive summary"."""

    # Exec summary is formatted oddly as two columns of key-value pairs. Flatten
    # into one standard column/row format.
    col1 = list(df[0].values)
    data1 = list(df[1].values)
    col2 = list(df[2].values)
    data2 = list(df[3].values)
    columns = col1 + col2
    data = data1 + data2
    flattened_df = pd.DataFrame(columns=columns, data=[data])

    # Extract some compound columns into simple integer fields
    for col in ["Latest Weight", "Latest Height", "Latest Head Circumference"]:
        results = flattened_df[col].apply(_extract_metric_and_percentile).values
        unit = results[0][1]
        flattened_df["%s %s" % (col, unit)] = [r[0] for r in results]
        flattened_df["%s Percentile" % col] = [r[2] for r in results]
        del flattened_df[col]
    del flattened_df["Due Date"]
    return flattened_df


def _extract_metric_and_percentile(string):
    """Extract integer values from a compound data field."""

    # Expects a format like the following: "3.2 kg / 7.17 lbs (28.54%)"
    match = re.match("([\d.]*) (.*) \/ ([\d.]*) (.*) \(([\d.]*).*\)", string)
    if match:
        (_, _, imperial, unit, percentile) = match.groups()
        return imperial, unit, percentile
    else:
        raise ValueError(
            "Invalid data: %s does not match expected format" % string)


def _parse_diapers(df):
    """Parse and flatten diaper summary."""
    columns = []
    data = []
    # Extract the first two rows, for the two diaper types
    diaper_cols = df.iloc[0].values
    for row_index in [1, 2]:
        row = df.iloc[row_index]
        diaper_type = row.values[0]
        for i, name in enumerate(diaper_cols):
            if i == 0: continue
            column = "%s %s" % (name, diaper_type)
            columns.append(column.lower())
            data.append((row.values[i]))
    return pd.DataFrame(columns=columns, data=[data])


def _parse_feedings(df):
    """Parse and flatten feeding summary."""
    columns = []
    data = []

    # Extract the first two rows, for the two bf sides
    breastfeeding_cols = df.iloc[0].values
    for row_index in [1, 2]:
        row = df.iloc[row_index]
        side = row.values[0]
        for i, name in enumerate(breastfeeding_cols):
            if i == 0 or pd.isnull(name):
                continue
            column = "%s %s" % (name, side)
            value = row.values[i].replace(" mins", "")
            columns.append(column)
            data.append(value)

    # Extract the next two rows, for bottle feeding
    bottle_feeding_cols = df.iloc[3].values
    for row_index in [4, 5]:
        row = df.iloc[row_index]
        bottle_type = row.values[0]
        if "Breastmilk" in bottle_type:
            bottle_type = "breastmilk"
        elif "Formula" in bottle_type:
            bottle_type = "formula"
        for i, name in enumerate(bottle_feeding_cols):
            if i == 0 or pd.isnull(name):
                continue
            column = "%s:%s" % (bottle_type, name)
            value = row.values[i].replace(" mins", "")
            if value == "N/A":
                value = np.nan
            columns.append(column)
            data.append(value)
    return pd.DataFrame(columns=columns, data=[data])


if __name__ == "__main__":

    # Attempt to parse all PDFS in the specified directory
    files = os.listdir(PDF_DIR)
    rows = []
    for f in files:
        if not f.endswith(".pdf"):
            continue
        rows.append(parse_summary(f))

    # Create single concatenated file for all weekly summaries
    df = pd.concat(rows).sort_values(by="week_start_date")
    output_file = os.path.join(OUTPUT_DIR, "summary.csv")
    df.to_csv(output_file, index=False)

