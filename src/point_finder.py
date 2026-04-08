import cv2

points = []

def mouse_callback(event, x, y, flags, param):
    global points, frame_display

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Clicked point: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            removed = points.pop()
            print(f"Removed point: {removed}")

def draw_points(frame, points):
    display = frame.copy()

    for i, (x, y) in enumerate(points):
        cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(display, str(i+1), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if len(points) > 1:
        for i in range(len(points) - 1):
            cv2.line(display, points[i], points[i+1], (255, 0, 0), 2)

    if len(points) > 2:
        cv2.line(display, points[-1], points[0], (255, 0, 0), 2)

    return display

video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Could not read first frame from video")

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_callback)

while True:
    display = draw_points(frame, points)
    cv2.imshow("Frame", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("c"):
        print("Polygon points:")
        print(points)
    elif key == ord("r"):
        points = []
        print("Points reset")

cv2.destroyAllWindows()