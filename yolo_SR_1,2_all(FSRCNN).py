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
sr_frame_dir = os.path.join(result_dir, "sr_frames")
csv_dir = os.path.join(result_dir, "csv")

os.makedirs(sr_frame_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

output_csv = os.path.join(csv_dir, "detections_sr.csv")
log_csv = os.path.join(csv_dir, "inference_log_sr.csv")

with open(output_csv, mode="w", newline="") as f:
    csv.writer(f).writerow(["timestamp", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])

with open(log_csv, mode="w", newline="") as f:
    csv.writer(f).writerow(["timestamp", "sr_time", "yolo_time", "total_time", "fps"])

# ─ GStreamer 파이프라인 ─
def gstreamer_pipeline(sensor_id=0, capture_width=640, capture_height=360,
                       display_width=640, display_height=360, framerate=30, flip_method=2):
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

            timestamp = datetime.datetime.now().isoformat()

            # ─ FSRCNN 전체 프레임 SR 처리 ─
            sr_start = time.time()
            img_ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
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
            sr_frame = cv2.cvtColor(sr_ycbcr, cv2.COLOR_YCrCb2BGR)
            sr_time = time.time() - sr_start

            # ─ YOLO 추론 ─
            yolo_start = time.time()
            results = model(sr_frame)
            yolo_time = time.time() - yolo_start

            boxes = results[0].boxes

            with open(output_csv, mode="a", newline="") as f:
                writer = csv.writer(f)
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    writer.writerow([timestamp, class_id, class_name, confidence, x1, y1, x2, y2])

                    cv2.rectangle(sr_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(sr_frame, class_name, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ─ FPS 및 시간 기록 ─
            total_time = time.time() - total_start
            fps = 1.0 / total_time if total_time > 0 else 0

            print(f"[SR] {sr_time:.3f}s | [YOLO] {yolo_time:.3f}s | [Total] {total_time:.3f}s | FPS: {fps:.2f}")
            with open(log_csv, mode="a", newline="") as f:
                csv.writer(f).writerow([timestamp, round(sr_time, 4), round(yolo_time, 4), round(total_time, 4), round(fps, 2)])

            cv2.putText(sr_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            cv2.imshow("SR + YOLO", sr_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    show_camera()
