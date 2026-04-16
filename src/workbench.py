import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIDEO_SOURCE = os.path.join(BASE_DIR, "data")

file = os.listdir(VIDEO_SOURCE)


def capture_time_from_jpeg(directory: str):
    try:
        files = os.listdir(directory)
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
        return []

    results = []
    for entry in files:
        index = entry.find("T")
        results.append(entry[index + 1:] if index != -1 else "")
    return results


parts = capture_time_from_jpeg(VIDEO_SOURCE)
print(parts)
