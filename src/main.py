import csv
import time
from collections import defaultdict
from typing import TypedDict

import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIDEO_SOURCE = os.path.join(BASE_DIR, "data", "video.mp4")

YOLO_MODEL = "yolov8n.pt"
CONF_THRESHOLD = 0.35
LOG_INTERVAL = 20.0

SITE_LAT = -43.5321
SITE_LON = 172.6362

# Define polygon coordinates for each table zone
# These can be obtained manually with "point_finder.py"
TABLE_ZONES = {
    "table_1": np.array([(126, 304), (416, 303), (411, 530), (104, 535)], dtype=np.int32),
}

TABLE_INFO = {
    "table_1": {"dist_from_road": 12.4, "in_shadow": 0},
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


def get_bottom_center(box: list[float]) -> tuple[int, int]:
    """
    Return the bottom center point of a boundary box
    This is passed to csv file to indicate when person is truly in target zone
    :param box:
    :return:
    """
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int(y2)


def point_in_zone(point: tuple[int, int], polygon: np.ndarray) -> bool:
    """
    Check if a point lies inside a polygon
    This allows us to know if someone is in target zone
    :param point:
    :param polygon:
    :return:
    """
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def assign_table(box: list[float], zones: dict[str, np.ndarray]) -> str | None:
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


def draw_zones(frame: np.ndarray, zones: dict[str, np.ndarray]) -> None:
    """
    Draws zones
    :param frame:
    :param zones:
    :return:
    """
    for table_name, polygon in zones.items():
        cv2.polylines(frame, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)
        x, y = polygon[0]
        cv2.putText(
            frame,
            table_name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )


def get_temperature(lat: float, lon: float) -> float:
    API_KEY = os.getenv("METOCEAN_API_KEY")

    url = "https://forecast-v2.metoceanapi.com/point/time"
    headers = {
        "x-api-key": API_KEY,
        "accept": "application/json"
    }

    now_utc = datetime.now(timezone.utc)
    from_time = now_utc.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")

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


def get_cached_temperature(lat: float, lon: float) -> float | None:
    current_hour = datetime.now().strftime("%Y-%m-%d %H")

    if weather_cache["hour"] != current_hour:
        weather_cache["temp_c"] = get_temperature(lat, lon)
        weather_cache["hour"] = current_hour

    return weather_cache["temp_c"]


def write_csv_row(csv_writer: csv.DictWriter,
                  timestamp: str, occupancy: dict[str, int],
                  table_info: dict[str, dict[str, float]],
                  table_names: list[str], temp: float) -> None:

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

    # Link video file
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {VIDEO_SOURCE}")

    # Potential edit location for bringing more data to be written
    table_names = list(TABLE_ZONES.keys())
    csv_file = open("occupancy_log.csv", "w", newline="", encoding="utf-8")
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

    last_log_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video. Exiting")
                break

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
                x1, y1, x2, y2, track_id = track
                box = [x1, y1, x2, y2]

                table_name = assign_table(box, TABLE_ZONES)
                if table_name is not None:
                    occupancy[table_name] += 1

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )

                label = f"ID {int(track_id)}"
                if table_name:
                    label += f" -> {table_name}"

                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                bx, by = get_bottom_center(box)
                cv2.circle(frame, (bx, by), 4, (0, 0, 255), -1)
            full_occupancy = {name: occupancy.get(name, 0) for name in table_names}

            now = time.time()
            if now - last_log_time >= LOG_INTERVAL:
                timestamp = time.strftime("%d-%m-%Y %H:%M:%S")
                temp = get_cached_temperature(SITE_LAT, SITE_LON)
                write_csv_row(csv_writer, timestamp, full_occupancy, TABLE_INFO, table_names, temp)
                csv_file.flush()
                last_log_time = now

            draw_zones(frame, TABLE_ZONES)

            y_offset = 30
            for table_name in table_names:
                text = f"{table_name}: {full_occupancy[table_name]}"
                cv2.putText(
                    frame,
                    text,
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                y_offset += 30
            cv2.imshow("Table Occupancy Tracker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()


if __name__ == "__main__":
    main()
