import cv2
print(cv2.__version__)
from ultralytics import YOLO
import csv
import datetime

# Load the YOLO model
model = YOLO("/home/uk/test11/yolo11n.engine")

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def show_camera():
    output_csv = "detections.csv"
    # CSV 파일 헤더 작성
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])

    print(gstreamer_pipeline(flip_method=0))
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)

    if video_capture.isOpened():
        try:
            while True:
                ret_val, frame = video_capture.read()
                if not ret_val:
                    continue

                results = model(frame)
                annotated_frame = results[0].plot()

                # CSV 저장
                boxes = results[0].boxes
                timestamp = datetime.datetime.now().isoformat()
                with open(output_csv, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        writer.writerow([timestamp, class_id, class_name, confidence, x1, y1, x2, y2])

                cv2.imshow("YOLO Inference", annotated_frame)
                keyCode = cv2.waitKey(10) & 0xFF
                if keyCode == 27 or keyCode == ord('q'):
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")


if __name__ == "__main__":
    show_camera()
