"""Extract individual events from calendar graphic in Glow PDF export.

Individual events (e.g. feedings, diaper changes) are contained in Glow
data exports only in the form of a calendar-like graphic representation.
This script parses the image (using hard-coded assumptions about the pixel
layout) to extract a raw event history.
"""

import datetime
import os
import pathlib
import shutil
import subprocess

import numpy as np
import pandas as pd
from PIL import Image
from pdf2jpg import pdf2jpg


TEMP_DIR = "tmp"

START_COORDINATE = (392, 412)

# Constants for extracting diaper and feeding pixels
SEGMENT_WIDTH = 723
DAY_HEIGHT = 346
FEEDING_HEIGHT = 200
DIAPER_HEIGHT = 60
PADDING_HEIGHT = 20

# There are 7 days in a week and 6 4-hour segments per day
NUM_DAYS = 7
NUM_BINS = 6

# Each segment is 4 hrs long
PIXELS_PER_MINUTE = SEGMENT_WIDTH / 4.0 / 60.0

BLUE = "BLUE"
YELLOW = "YELLOW"
WHITE = "WHITE"
DARK = "DARK"


def extract(PDF_DIR, filename):
    """Extract event history from PDF.

    Individual events (feedings, diaper changes, etc.) are provided in the
    pdf exports as a calendar graphic. Events are extracted by converting
    the pdg to an image and processing the pixel values."""

    pdf_path = os.path.join(PDF_DIR, filename)
    tmp_subdir = _write_tmp_jpg(pdf_path, pdf_page="1")
    results = _read_and_process_tmp_jpg(filename)
    shutil.rmtree(tmp_subdir)
    return results


def _write_tmp_jpg(pdf_path, pdf_page="ALL"):
    """Write event history page of PDF to a temporary jpg file."""
    if not os.path.isdir(TEMP_DIR):
        os.path.mkdir(TEMP_DIR)

    # HACK: due to bug in pdf2jpg, we need to construct command to execute
    # jar file directly, rather than relying on convert_pdf2jpg
    pdf_convert_dir = pathlib.Path(pdf2jpg.__file__).resolve().parent
    jar_file = os.path.join(pdf_convert_dir, "pdf2jpg.jar")
    cmd = 'java -jar %s -i "%s" -o "%s" -p 1' % (jar_file, pdf_path, TEMP_DIR)
    output = subprocess.check_output(cmd, shell=True)
    tmp_subdir = output.split(b'\n')[0]
    return tmp_subdir


def _read_and_process_tmp_jpg(filename):
    """Process tmp image and extract event history."""
    date_str = os.path.splitext(filename)[0].split("_")[-1]
    jpg_file = os.path.join(TEMP_DIR, filename, "1_" + filename + ".jpg")
    start_date = datetime.date(
        int(date_str[4:]), int(date_str[0:2]), int(date_str[2:4]))
    print("Extracting history from %s" % jpg_file)
    img = Image.open(jpg_file)
    pixels = img.load()
    return _extract_windows(pixels, start_date)


def _extract_windows(pixels, start_date):
    """Extract events from image and return as dataframe."""
    columns = ["timestamp", "event", "duration"]
    data = []
    for day in range(NUM_DAYS):
        date = start_date + datetime.timedelta(days=6 - day)
        for bin_num in range(NUM_BINS):
            window_feedings, window_diapers = _extract_window(
                pixels, day, bin_num, date)
            for (dt, duration) in window_feedings:
                data.append([dt, "breastfeeding", duration])
            for (dt, duration) in window_diapers:
                data.append([dt, "diaper", np.nan])
    return pd.DataFrame(columns=columns, data=data)


def _extract_window(pixels, day, bin_num, date):
    """Extract events from a single event window."""
    start_i = START_COORDINATE[0] + SEGMENT_WIDTH * bin_num
    start_j = START_COORDINATE[1] + DAY_HEIGHT * day
    end_i = start_i + SEGMENT_WIDTH
    end_j = start_j + DAY_HEIGHT
    raw_feedings = extract_timeseries(
        "breastfeeding", pixels, start_i, end_i, start_j, end_j)
    raw_diapers = extract_timeseries(
        "diapers", pixels, start_i, end_i, start_j, end_j)
    feedings = process_timeseries(raw_feedings, bin_num, date)
    diapers = process_timeseries(raw_diapers, bin_num, date)
    return feedings, diapers


def _get_color_range(r, g, b):
    """Returns the appropriate color range for rgb value."""
    THRESHOLD = 235
    if r == g == b:
        return WHITE if r >= THRESHOLD else DARK
    elif b >= THRESHOLD and r < THRESHOLD and g < THRESHOLD:
        return BLUE
    elif r > THRESHOLD and b < THRESHOLD and g < THRESHOLD:
        return YELLOW
    else:
        return None


def process_timeseries(values, bin_num, date):
    """Convert array of pixel values into event timestamps and durations.

    Args:
        values: an array in which each value indicates the presence or absence
        of an event at that pixel.
        date: datetime for target day
        bin_num: 0-5 integer specifying a 4 hr bin of the day
    Returns:
        Array of (datetime, duration) tuples representing the events in question
    """
    results = []
    prev_val, current_duration = 0, 0
    for i, val in enumerate(values):
        if val == 1:
            current_duration += 1
        else:
            if prev_val == 1 and current_duration > PIXELS_PER_MINUTE:
                # Log event datetime and duration
                start_i = i - current_duration
                total_pixels = (bin_num * SEGMENT_WIDTH + start_i)
                total_minutes = min(
                    int(total_pixels / PIXELS_PER_MINUTE), 60 * 24 - 1)
                hour = total_minutes // 60
                minutes = total_minutes % 60
                dt = datetime.datetime(
                    date.year, date.month, date.day, hour, minutes)
                results.append((dt, current_duration // PIXELS_PER_MINUTE))
                current_duration = 0
        prev_val = val
    return results


def extract_timeseries(timeseries_type, pixels, start_i, end_i, start_j, end_j):
    """Extract a timeseries from a set of pixel values.

    Returns an array corresponding to the time (x) dimension with a 1 if the
    specified event is occuring at that pixel, a 0 if the event is absent."""

    if timeseries_type == "breastfeeding":
        target_color = YELLOW
        vertical_start = start_j + PADDING_HEIGHT
        vertical_end = vertical_start + FEEDING_HEIGHT
    elif timeseries_type == "diapers":
        target_color = BLUE
        vertical_start = start_j + PADDING_HEIGHT * 2 + FEEDING_HEIGHT
        vertical_end = vertical_start + DIAPER_HEIGHT
    else:
        raise ValueError("Invalid type %s" % timeseries_type)

    results = []
    for x in range(start_i, end_i):
        # For each pixel along the X dimension, get the average RGB value over
        # the relevant range of the Y dimension
        rgb_color = _get_average_rgb(
            pixels, x, x + 1, vertical_start, vertical_end)
        color_range = _get_color_range(*rgb_color)
        if color_range == WHITE:
            results.append(0)
        elif color_range == target_color:
            results.append(1)
        else:
            results.append(np.nan)
    return results


def _get_average_rgb(pixels, start_i, end_i, start_j, end_j):
    """Compute the average rgb values for a matrix.

    Given a matrix defined by start and end coordinates, calculate the
    average rgb value within the matrix."""
    reds, greens, blues = [], [], []
    for i in range(start_i, end_i):
        for j in range(start_j, end_j):
            (r, g, b) = pixels[i, j]
            reds.append(r)
            greens.append(g)
            blues.append(b)
    return np.mean(reds), np.mean(greens), np.mean(blues)
