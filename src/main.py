import csv
from collections import defaultdict
from typing import TypedDict

import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo


load_dotenv()

# ---------------------------------
# CONSTANTS
# ---------------------------------

# Directory paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIDEO_SOURCE = os.path.join(BASE_DIR, "data", "video.mp4")
CSV_OUTPUT = os.path.join(BASE_DIR, "output", "occupancy_log.csv")

# Model and thresholds
YOLO_MODEL = "yolov8n.pt"
CONF_THRESHOLD = 0.35  # Detections > 35% confidence to be logged
LOGGED_SECONDS = 30.0  # 1/fps is the lowest recommend

# Christchurch
SITE_LAT = -43.5321
SITE_LON = 172.6362

# Year/Month/Day/Hour/Min
RECORDED_START_TIME = datetime(2026, 4, 23, 10, 2, 0, tzinfo=ZoneInfo("Pacific/Auckland"))

# Use "point_finder.py" to set zones
TABLE_ZONES = {
    "table_1": np.array([(1261, 434), (1620, 457), (1598, 635), (1200, 551)], dtype=np.int32),
    "table_2": np.array([(877, 419), (1073, 427), (1047, 529), (851, 493)], dtype=np.int32),
    "table_3": np.array([(695, 415), (822, 414), (805, 487), (684, 474)], dtype=np.int32),
}

# Manual input required
TABLE_INFO = {
    "table_1": {"dist_from_road": 10.0, "in_shadow": 0},
    "table_2": {"dist_from_road": 10.0, "in_shadow": 1},
    "table_3": {"dist_from_road": 10.0, "in_shadow": 1},
}

PERSON_CLASS_ID = 0


# ---------------------------------
# HELPERS
# ---------------------------------


class WeatherCache(TypedDict):
    hour: str | None
    temp_c: float | None


weather_cache: WeatherCache = {
    "hour": None,
    "temp_c": None
}


def datetime_to_site_time(dt: datetime) -> list[int]:
    return [dt.year, dt.month, dt.day, dt.hour, dt.minute]


def get_bottom_center(box) -> tuple[int, int]:
    """
    Return the bottom center point of a boundary box
    This is passed to csv file to indicate when person is truly in target zone
    :param box:
    :return:
    """
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int(y2)


def point_in_zone(point, polygon) -> bool:
    """
    Check if a point lies inside a polygon
    This allows us to know if someone is in target zone
    :param point:
    :param polygon:
    :return:
    """
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def assign_table(box, zones) -> str | None:
    """
    Returns the table name that contains the person's bottom center point
    :param box:
    :param zones:
    :return:
    """
    pt = get_bottom_center(box)
    for table_name, polygon in zones.items():
        if point_in_zone(pt, polygon):
            return table_name
    return None


def build_weather_time(year, month, day, hour, minute=0) -> str:
    """
    Gives a starting time for weather request to start polling from
    :param year:
    :param month:
    :param day:
    :param hour:
    :param minute:
    :return:
    """
    chosen_local = datetime(year, month, day, hour, minute, tzinfo=ZoneInfo("Pacific/Auckland"))
    chosen_utc = chosen_local.astimezone(timezone.utc)
    return chosen_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def get_temperature(lat, lon, site_time) -> float:
    """
    API call each hour to get temp
    :param lat:
    :param lon:
    :param site_time:
    :return:
    """
    API_KEY = os.getenv("METOCEAN_API_KEY")

    url = "https://forecast-v2.metoceanapi.com/point/time"
    headers = {
        "x-api-key": API_KEY,
        "accept": "application/json"
    }

    from_time = build_weather_time(*site_time)

    params = {
        "lat": lat,
        "lon": lon,
        "variables": "air.temperature.at-2m",
        "from": from_time,
        "interval": "1h",
        "repeat": 1
    }

    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()

    data = r.json()
    temp_k = data["variables"]["air.temperature.at-2m"]["data"][0]
    return round(temp_k - 273.15, 2)


def get_cached_temperature(lat, lon, site_time) -> float | None:
    """
    Caches temperature data for easy on pipeline
    :param lat:
    :param lon:
    :param site_time:
    :return:
    """
    requested_hour = build_weather_time(*site_time)

    if weather_cache["hour"] != requested_hour:
        weather_cache["temp_c"] = get_temperature(lat, lon, site_time)
        weather_cache["hour"] = requested_hour

    return weather_cache["temp_c"]


def write_csv_row(csv_writer, timestamp, occupancy,
                  table_info, table_names, temp) -> None:
    for table_name in table_names:
        row = {
            "timestamp": timestamp,
            "table": table_name,
            "occupancy": occupancy.get(table_name, 0),
            "dist_from_road": table_info[table_name]["dist_from_road"],
            "in_shadow": table_info[table_name]["in_shadow"],
            "temp": temp,
        }
        csv_writer.writerow(row)


# ---------------------------------
# MAIN
# ---------------------------------


def main():
    model = YOLO(YOLO_MODEL)
    tracker = Sort()

    # Opens video source and retrieves video fps
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {VIDEO_SOURCE}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("Could not determine video FPS")

    # Sets csv to be written
    table_names = list(TABLE_ZONES.keys())
    os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)
    csv_file = open(CSV_OUTPUT, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "timestamp",
            "table",
            "occupancy",
            "dist_from_road",
            "in_shadow",
            "temp"
        ]
    )
    csv_writer.writeheader()

    last_logged_video_time = -LOGGED_SECONDS

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video. Exiting")
                break

            frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
            video_seconds = frame_index / fps
            current_dt = RECORDED_START_TIME + timedelta(seconds=video_seconds)
            results = model(frame, verbose=False)

            # Process YOLO detections for each frame
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    cls = int(box.cls[0].item())
                    conf = float(box.conf[0].item())

                    # Keep only person detections above confidence threshold
                    if cls != PERSON_CLASS_ID or conf < CONF_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append([x1, y1, x2, y2, conf])

            if len(detections) == 0:
                dets_np = np.empty((0, 5))
            else:
                dets_np = np.array(detections)

            # Update tracker to assign consistent IDs across frames
            tracks = tracker.update(dets_np)

            # Count number of people per table zone
            occupancy = defaultdict(int)

            for track in tracks:
                x1, y1, x2, y2, _ = track
                box = [x1, y1, x2, y2]

                table_name = assign_table(box, TABLE_ZONES)
                if table_name is not None:
                    occupancy[table_name] += 1

            full_occupancy = {name: occupancy.get(name, 0) for name in table_names}

            # Writes a CSV row when enough video time has elapsed since the last log
            if video_seconds - last_logged_video_time >= LOGGED_SECONDS:
                timestamp = current_dt.strftime("%d-%m-%Y %H:%M:%S.%f")[:-3]
                current_site_time = datetime_to_site_time(current_dt)
                temp = get_cached_temperature(SITE_LAT, SITE_LON, current_site_time)
                write_csv_row(csv_writer, timestamp, full_occupancy, TABLE_INFO, table_names, temp)
                csv_file.flush()
                last_logged_video_time = video_seconds

    finally:
        cap.release()
        csv_file.close()


if __name__ == "__main__":
    main()
