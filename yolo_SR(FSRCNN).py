import cv2
import csv
import datetime
import os
import time
import torch
import numpy as np

from ultralytics import YOLO
from models import FSRCNN
from utils import convert_ycbcr_to_rgb

# ─ 모델 로드 ─
model = YOLO("/home/uk/test11/yolo11n.engine")

fsrcnn_model_path = "/home/uk/test11/test/models/fsrcnn_x3.pth"
scale_factor = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fsrcnn = FSRCNN(scale_factor=scale_factor).to(device)
fsrcnn.load_state_dict(torch.load(fsrcnn_model_path, map_location=device))
fsrcnn.eval()
print("✅ FSRCNN 모델 로드 완료")

# ─ 디렉토리 및 CSV ─
result_dir = "/home/uk/test11/test/results"
images_dir = "/home/uk/test11/test/images"
cropped_dir = "/home/uk/test11/test/results/images/cropped_objects"
sr_dir = "/home/uk/test11/test/results/images/sr_objects"
csv_dir = "/home/uk/test11/test/results/csv"

# 디렉토리 생성
os.makedirs(result_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(cropped_dir, exist_ok=True)
os.makedirs(sr_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

output_csv = os.path.join(csv_dir, "detections.csv")
log_csv = os.path.join(csv_dir, "inference_log.csv")

with open(output_csv, mode="w", newline="") as f:
    csv.writer(f).writerow(["timestamp", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])

with open(log_csv, mode="w", newline="") as f:
    csv.writer(f).writerow(["timestamp", "yolo_time", "total_time", "yolo_fps", "total_fps"])

# ─ GStreamer 파이프라인 ─
def gstreamer_pipeline(sensor_id=0, capture_width=1920, capture_height=1080,
                       display_width=960, display_height=540, framerate=30, flip_method=2):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )

# ─ 실시간 처리 루프 ─
def show_camera():
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Error: Unable to open camera")
        return

    try:
        while True:
            total_start = time.time()
            ret, frame = cap.read()
            if not ret:
                continue

            # YOLO 탐지
            yolo_start = time.time()
            results = model(frame)
            yolo_time = time.time() - yolo_start

            boxes = results[0].boxes
            timestamp = datetime.datetime.now().isoformat()

            with open(output_csv, mode="a", newline="") as f:
                writer = csv.writer(f)

                # 최고 confidence 객체 SR 결과 저장용 변수 초기화
                top_conf = -1
                top_sr_bgr = None
                top_class_name = ""

                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    writer.writerow([timestamp, class_id, class_name, confidence, x1, y1, x2, y2])

                    cropped = frame[y1:y2, x1:x2]
                    if cropped.size == 0:
                        continue

                    # 저장용 파일명
                    filename_base = f"{timestamp.replace(':', '').replace('.', '')}_{class_name}"
                    filename_crop = os.path.join(cropped_dir, f"{filename_base}_crop.png")
                    filename_sr = os.path.join(sr_dir, f"{filename_base}_sr.png")

                    # 원본 crop 저장
                    cv2.imwrite(filename_crop, cropped)

                    # ─ FSRCNN Super Resolution ─
                    img_ycbcr = cv2.cvtColor(cropped, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
                    y = img_ycbcr[..., 0]
                    y_tensor = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).to(device)

                    with torch.no_grad():
                        sr_y = fsrcnn(y_tensor).clamp(0.0, 1.0)

                    sr_y_np = sr_y.squeeze().cpu().numpy()
                    h_sr, w_sr = sr_y_np.shape

                    cr_up = cv2.resize(img_ycbcr[..., 1], (w_sr, h_sr), interpolation=cv2.INTER_CUBIC)
                    cb_up = cv2.resize(img_ycbcr[..., 2], (w_sr, h_sr), interpolation=cv2.INTER_CUBIC)

                    sr_ycbcr = np.stack([sr_y_np, cr_up, cb_up], axis=2) * 255.0
                    sr_ycbcr = np.clip(sr_ycbcr, 0, 255).astype(np.uint8)
                    sr_bgr = cv2.cvtColor(sr_ycbcr, cv2.COLOR_YCrCb2BGR)

                    # 최고 confidence 객체 저장
                    if confidence > top_conf:
                        top_conf = confidence
                        top_sr_bgr = sr_bgr.copy()
                        top_class_name = class_name

                    # SR 이미지 저장
                    cv2.imwrite(filename_sr, sr_bgr)

                    # 프레임에 삽입
                    sr_resized = cv2.resize(sr_bgr, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
                    frame[y1:y2, x1:x2] = sr_resized

                    # 테두리 표시
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # FPS 측정
            total_time = time.time() - total_start
            yolo_fps = 1.0 / yolo_time if yolo_time > 0 else 0
            total_fps = 1.0 / total_time if total_time > 0 else 0

            print(f"[YOLO] Time: {yolo_time:.3f}s | FPS: {yolo_fps:.2f} | [Total] Time: {total_time:.3f}s | FPS: {total_fps:.2f}")
            with open(log_csv, mode="a", newline="") as f:
                csv.writer(f).writerow([timestamp, round(yolo_time, 4), round(total_time, 4),
                                        round(yolo_fps, 2), round(total_fps, 2)])

            # 출력
            cv2.putText(frame, f"FPS: {total_fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # confidence 가장 높은 객체 SR 이미지를 우하단 썸네일로 표시
            if top_sr_bgr is not None:
                thumb = cv2.resize(top_sr_bgr, (150, 150))
                h, w, _ = frame.shape
                x_offset = w - 160
                y_offset = h - 160
                frame[y_offset:y_offset + 150, x_offset:x_offset + 150] = thumb
                cv2.putText(frame, f"{top_class_name} ({top_conf:.2f})", (x_offset, y_offset - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("YOLO + FSRCNN", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    show_camera()
