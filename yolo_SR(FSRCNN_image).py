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

# ─ 디렉토리 설정 ─
result_dir = "/home/uk/test11/test/results_image"
images_dir = "/home/uk/test11/test/images"
cropped_dir = os.path.join(result_dir, "images/cropped_objects")
sr_dir = os.path.join(result_dir, "images/sr_objects")
comparison_dir = os.path.join(result_dir, "images/comparison")
combined_dir = os.path.join(result_dir, "images/combined")
csv_dir = os.path.join(result_dir, "csv")

# ─ 디렉토리 생성 ─
os.makedirs(images_dir, exist_ok=True)
os.makedirs(cropped_dir, exist_ok=True)
os.makedirs(sr_dir, exist_ok=True)
os.makedirs(comparison_dir, exist_ok=True)
os.makedirs(combined_dir, exist_ok=True)
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

# ─ 한 장 캡처 후 처리 ─
def capture_and_process_once():
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("❌ 프레임을 캡처하지 못했습니다.")
        return

    timestamp = datetime.datetime.now().isoformat()
    original_filename = os.path.join(images_dir, f"{timestamp.replace(':', '').replace('.', '')}_frame.png")
    cv2.imwrite(original_filename, frame)

    total_start = time.time()
    yolo_start = time.time()
    results = model(frame)
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

            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            filename_base = f"{timestamp.replace(':', '').replace('.', '')}_{class_name}"
            filename_crop = os.path.join(cropped_dir, f"{filename_base}_crop.png")
            filename_sr = os.path.join(sr_dir, f"{filename_base}_sr.png")

            cv2.imwrite(filename_crop, cropped)

            # ─ SR 처리 ─
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
            cv2.imwrite(filename_sr, sr_bgr)

            # 프레임에 덮기
            sr_resized = cv2.resize(sr_bgr, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
            frame[y1:y2, x1:x2] = sr_resized

            # 시각화용 박스
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ─ Crop vs SR 비교 이미지 저장 ─
            comparison = np.hstack([
                cv2.resize(cropped, (sr_bgr.shape[1], sr_bgr.shape[0])),
                sr_bgr
            ])
            comparison_filename = os.path.join(comparison_dir, f"{filename_base}_comparison.png")
            cv2.imwrite(comparison_filename, comparison)

            # ─ 전체 프레임(SR이 덮인 최종 프레임) 저장 ─
            filename_combined = os.path.join(combined_dir, f"{filename_base}_combined.png")
            cv2.imwrite(filename_combined, frame)

            print(f"✅ 저장 완료: {filename_crop}, {filename_sr}, {comparison_filename}, {filename_combined}")

    total_time = time.time() - total_start
    yolo_fps = 1.0 / yolo_time if yolo_time > 0 else 0
    total_fps = 1.0 / total_time if total_time > 0 else 0

    with open(log_csv, mode="a", newline="") as f:
        csv.writer(f).writerow([timestamp, round(yolo_time, 4), round(total_time, 4),
                                round(yolo_fps, 2), round(total_fps, 2)])

    # ─ 시각화 출력 (선택 사항) ─
    cv2.putText(frame, f"FPS: {total_fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.imshow("Final Frame with SR", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ─ 실행 ─
if __name__ == "__main__":
    capture_and_process_once()
