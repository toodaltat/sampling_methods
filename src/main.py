import csv
import time
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

CAMERA_SOURCE = 0
YOLO_MODEL = "yolov8n.pt"
CONFIG_THRESHOLD = 0.35
LOG_INTERVAL = 1.0

TABLE_ZONES = {
    "table_1": np.array([(50, 80), (50, 80), (50, 80), (50, 80)], dtype=np.int32),
    "table_2": np.array([(50, 80), (50, 80), (50, 80), (50, 80)], dtype=np.int32),
    "table_3": np.array([(50, 80), (50, 80), (50, 80), (50, 80)], dtype=np.int32),
    "table_4": np.array([(50, 80), (50, 80), (50, 80), (50, 80)], dtype=np.int32),
    "table_5": np.array([(50, 80), (50, 80), (50, 80), (50, 80)], dtype=np.int32),
    "table_6": np.array([(50, 80), (50, 80), (50, 80), (50, 80)], dtype=np.int32),
}

PERSON_CLASS_ID = 0

# ---------------------------------
# HELPERS
# ---------------------------------

def get_bottom_center(box):
    """
    Return the bottom center point of a boundary box
    This is passed to csv file to indicate when person is truly in target zone
    :param box:
    :return:
    """
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int(y2)


def point_in_zone(point, polygon):
    """
    Check if a point lies inside a polygon
    This allows us to know if someone is in target zone
    :param point:
    :param polygon:
    :return:
    """
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def assign_table(box, zones):
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


def draw_zones(frames, zones):
    """
    Draws zones
    :param frames:
    :param zones:
    :return:
    """
    for table_name, polygon in zones.items():
        cv2.polylines(frames, [polygon], isClosed=True, color=(255, 0, 0), thickness =2)
        x,y = polygon[0]
        cv2.putText(
            frames,
            table_name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

def write_csv_row(csv_writer, timestamp, occupancy):
    row = {"timestamp":timestamp}
    row.update(occupancy)
    csv_writer.writerow(row)


# ---------------------------------
# MAIN
# ---------------------------------


def main():
    model = YOLO(YOLO_MODEL)
    tracker = Sort()

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    table_names = list(TABLE_ZONES.keys())
    csv_file = open("occupancy_log.csv", "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=["timestamp"] + table_names)
    csv_writer.writeheader()

    last_log_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame. Exiting")
                break

            results = model(frame, verbose=False)

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    cls = int(box.cls[0].item())
                    config = float(box.conf[0].item())

                    if cls != PERSON_CLASS_ID or config < CONFIG_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append([x1, y1, x2, y2, config])

            if len(detections) == 0:
                dets_np = np.empty((0,5))
            else:
                dets_np = np.array(detections)

            tracks = tracker.update(dets_np)

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
                write_csv_row(csv_writer, timestamp, full_occupancy)
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






















