
import cv2
import mediapipe as mp
import time
import sys
from pathlib import Path


# ---------------------------------------
# 1. Project Configuration
# ---------------------------------------

print("Python Interpreter:", sys.executable)
print("Starting AI Face Mesh Detection...")


# Get model file location
MODEL_PATH = Path(__file__).parent / "face_landmarker.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Face Landmarker model not found! "
        "Place face_landmarker.task inside your project folder."
    )


# ---------------------------------------
# 2. Initialize MediaPipe
# ---------------------------------------

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Facial mesh connections
connections = (
    mp.tasks.vision.FaceLandmarksConnections
    .FACE_LANDMARKS_TESSELATION
)


# ---------------------------------------
# 3. Open Webcam
# ---------------------------------------

def open_camera():

    for camera_index in range(3):

        camera = cv2.VideoCapture(camera_index)

        if camera.isOpened():
            success, frame = camera.read()

            if success:
                print(f"Camera {camera_index} opened successfully!")
                return camera

        camera.release()

    raise RuntimeError("Unable to open webcam!")


# ---------------------------------------
# 4. Create Face Landmarker
# ---------------------------------------

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# ---------------------------------------
# 5. Draw Facial Mesh
# ---------------------------------------

def draw_face_mesh(image, landmarks):

    height, width, _ = image.shape

    # Convert normalized points into pixel coordinates
    points = []

    for landmark in landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    # Draw connecting lines
    for connection in connections:

        start = connection.start
        end = connection.end

        if start < len(points) and end < len(points):

            cv2.line(
                image,
                points[start],
                points[end],
                (230, 216, 173),
                1,
                cv2.LINE_AA
            )

    # Draw landmark points
    for point in points:

        cv2.circle(
            image,
            point,
            1,
            (0, 255, 0),
            -1
        )


# ---------------------------------------
# 6. Main Program
# ---------------------------------------

cap = None
previous_time = time.monotonic()
start_time = time.monotonic()
last_timestamp = -1

try:

    cap = open_camera()

    with FaceLandmarker.create_from_options(options) as landmarker:

        while cap.isOpened():

            # Capture webcam frame
            success, frame = cap.read()

            if not success:
                print("Unable to read camera frame.")
                break

            # Mirror the webcam
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Make frame read-only during processing
            rgb_frame.flags.writeable = False

            # Create MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Generate increasing video timestamps
            timestamp = int(
                (time.monotonic() - start_time) * 1000
            )

            timestamp = max(timestamp, last_timestamp + 1)
            last_timestamp = timestamp

            # Detect face landmarks
            results = landmarker.detect_for_video(
                mp_image,
                timestamp
            )

            # --------------------------------
            # 7. Face Detection
            # --------------------------------

            face_detected = False

            if results.face_landmarks:

                face_detected = True

                for face_landmarks in results.face_landmarks:

                    draw_face_mesh(
                        frame,
                        face_landmarks
                    )

            # --------------------------------
            # 8. Calculate FPS
            # --------------------------------

            current_time = time.monotonic()

            fps = 1 / max(
                current_time - previous_time,
                0.000001
            )

            previous_time = current_time

            # --------------------------------
            # 9. Display Status
            # --------------------------------

            if face_detected:

                status = "FACE DETECTED"
                color = (0, 255, 0)

            else:

                status = "NO FACE DETECTED"
                color = (0, 0, 255)

            cv2.putText(
                frame,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Press ESC to Exit",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )

            # --------------------------------
            # 10. Display Webcam
            # --------------------------------

            cv2.imshow(
                "AI Face Mesh Detection",
                frame
            )

            # Exit using ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break


except Exception as error:

    print("ERROR:", error)


finally:

    if cap is not None:
        cap.release()

    cv2.destroyAllWindows()

    print("Face Mesh Detection Closed!")
