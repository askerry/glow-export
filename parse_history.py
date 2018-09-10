import datetime
import os

import numpy as np
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


def get_color_range(r, g, b):
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


def _write_tmp_jpg(pdf_path, pdf_page="ALL"):
    result = pdf2jpg.convert_pdf2jpg(pdf_path, TEMP_DIR, pages=pdf_page)
    print(result)


def _read_and_process_tmp_jpg(filename):
    date_str = os.path.splitext(filename)[0].split("_")[-1]
    jpg_file = os.path.join(TEMP_DIR, filename, "1_" + filename + ".jpg")
    start_date = datetime.date(
        int(date_str[4:]), int(date_str[0:2]), int(date_str[2:4]))
    print("Extracting history from %s" % jpg_file)
    img = Image.open(jpg_file)
    pixels = img.load()
    extract_windows(pixels, start_date)


def extract_history_as_image(PDF_DIR, filename):
    pdf_path = os.path.join(PDF_DIR, filename)
    # _write_tmp_jpg(pdf_path, pdf_page="1")
    _read_and_process_tmp_jpg(filename)


def extract_windows(pixels, start_date):
    feedings, diapers = [], []
    for day in range(NUM_DAYS):
        date = start_date + datetime.timedelta(days=6 - day)
        for bin_num in range(NUM_BINS):
            window_feedings, window_diapers = extract_window(
                pixels, day, bin_num, date)
            feedings.extend(window_feedings)
            diapers.extend(window_diapers)
    print(feedings)
    print(diapers)


def extract_window(pixels, day, bin_num, date):
    start_i = START_COORDINATE[0] + SEGMENT_WIDTH * bin_num
    start_j = START_COORDINATE[1] + DAY_HEIGHT * day
    end_i = start_i + SEGMENT_WIDTH
    end_j = start_j + DAY_HEIGHT
    raw_feedings, raw_diapers = get_timeseries(
        pixels, start_i, end_i, start_j, end_j)
    feedings = parse_timeseries(raw_feedings, day, bin_num, date)
    diapers = parse_timeseries(raw_diapers, day, bin_num, date)
    return feedings, diapers


def parse_timeseries(values, day, bin_num, date):
    results = []
    prev_val = 0
    current_duration = 0
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


def get_timeseries(pixels, start_i, end_i, start_j, end_j):
    raw_feedings = []
    raw_diapers = []
    for x in range(start_i, end_i):
        # Handle feeding range of the segment
        feeding_start = start_j + PADDING_HEIGHT
        feeding_end = feeding_start + FEEDING_HEIGHT
        feeding_color = _get_average_rgb(
            pixels, x, x + 1, feeding_start, feeding_end)
        if get_color_range(*feeding_color) == WHITE:
            raw_feedings.append(0)
        elif get_color_range(*feeding_color) == YELLOW:
            raw_feedings.append(1)
        else:
            raw_feedings.append(np.nan)

        # Handle diaper range of the segment
        diaper_start = feeding_end + PADDING_HEIGHT
        diaper_end = diaper_start + DIAPER_HEIGHT
        diaper_color = _get_average_rgb(
            pixels, x, x + 1, diaper_start, diaper_end)
        if get_color_range(*diaper_color) == WHITE:
            raw_diapers.append(0)
        elif get_color_range(*diaper_color) == BLUE:
            raw_diapers.append(1)
        else:
            raw_diapers.append(np.nan)
    return raw_feedings, raw_diapers


def _get_average_rgb(pixels, start_i, end_i, start_j, end_j):
    reds, greens, blues = [], [], []
    for i in range(start_i, end_i):
        for j in range(start_j, end_j):
            (r, g, b) = pixels[i, j]
            reds.append(r)
            greens.append(g)
            blues.append(b)
    return np.mean(reds), np.mean(greens), np.mean(blues)
