import cv2
print(cv2.__version__)
from ultralytics import YOLO
import csv
import datetime
import os
import time

# YOLO 모델 로드
model = YOLO("/home/uk/test11/yolo11n.engine")

# 저장 디렉토리 만들기
cropped_dir = "cropped_objects"
os.makedirs(cropped_dir, exist_ok=True)


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
    log_csv = "inference_log.csv"
    with open(log_csv, mode="w", newline="") as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["timestamp", "yolo_time", "total_time", "yolo_fps", "total_fps"])

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

                total_start = time.time()

                # ─ YOLO 추론 시간 측정 ─
                yolo_start = time.time()
                results = model(frame)
                yolo_time = time.time() - yolo_start

                annotated_frame = results[0].plot()

                # ─ 결과 저장 ─
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

                        # 객체 자르기 및 저장
                        cropped = frame[y1:y2, x1:x2]
                        filename = f"{timestamp.replace(':', '').replace('.', '')}_{class_name}.jpg"
                        filepath = os.path.join(cropped_dir, filename)
                        cv2.imwrite(filepath, cropped)

                # ─ 전체 처리 시간 계산 ─
                total_time = time.time() - total_start
                yolo_fps = 1.0 / yolo_time if yolo_time > 0 else 0
                total_fps = 1.0 / total_time if total_time > 0 else 0

                # ─ 로그 출력 및 저장 ─
                print(f"[YOLO] Time: {yolo_time:.3f}s | FPS: {yolo_fps:.2f} | "
                      f"[Total] Time: {total_time:.3f}s | FPS: {total_fps:.2f}")

                with open(log_csv, mode="a", newline="") as log_file:
                    log_writer = csv.writer(log_file)
                    log_writer.writerow([timestamp, round(yolo_time, 4), round(total_time, 4),
                                         round(yolo_fps, 2), round(total_fps, 2)])

                # ─ 시각화 ─
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
