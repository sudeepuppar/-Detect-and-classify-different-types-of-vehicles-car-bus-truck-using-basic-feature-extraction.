import cv2
import numpy as np

# ===== STEP 1: Click Points =====
points = []

def get_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: {x}, {y}")

# Load video
video_path = r"sample video.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Read first frame
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read frame.")
    exit()

first_frame = cv2.resize(first_frame, (1280, 720))
temp_frame = first_frame.copy()

cv2.namedWindow("Select 4 Points")
cv2.setMouseCallback("Select 4 Points", get_points)

print("Click 4 points in order: top-left, top-right, bottom-right, bottom-left")
while True:
    display = temp_frame.copy()
    for p in points:
        cv2.circle(display, p, 5, (0, 0, 255), -1)
    cv2.imshow("Select 4 Points", display)

    if len(points) == 4:
        break
    if cv2.waitKey(1) & 0xFF == 27:
        exit()

cv2.destroyWindow("Select 4 Points")

src_points = np.float32(points)
dst_points = np.float32([
    [0, 0],
    [500, 0],
    [500, 600],
    [0, 600]
])

M = cv2.getPerspectiveTransform(src_points, dst_points)

# ===== STEP 2: Vehicle Detection Setup =====
# 🚫 Disable shadow detection here
fgbg = cv2.createBackgroundSubtractorKNN(history=200, dist2Threshold=400.0, detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

def classify_vehicle(w, h):
    area = w * h
    aspect_ratio = w / float(h)
    if area > 50000 :
        return "truck"
    elif area > 7000:
        return "Car"
    else  :
        return "bike"
    
    

# ===== STEP 3: Main Loop =====
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))

    # Perspective transform
    bird_eye = cv2.warpPerspective(frame, M, (500, 600))

    # Background subtraction (no shadows now)
    fgmask = fgbg.apply(bird_eye)

    # Morphology
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Contour detection
    cnts, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 800:  # Ignore very small objects
            vehicle_type = classify_vehicle(w, h)
            cv2.rectangle(bird_eye, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(bird_eye, vehicle_type, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show original with trapezoid
    temp = frame.copy()
    cv2.polylines(temp, [np.int32(src_points)], True, (0, 0, 255), 2)
    cv2.imshow("Original with ROI", temp)
    cv2.imshow("Bird's Eye View Detection", bird_eye)
    cv2.imshow("Bird's Eye Mask", fgmask)

    if cv2.waitKey(75) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
