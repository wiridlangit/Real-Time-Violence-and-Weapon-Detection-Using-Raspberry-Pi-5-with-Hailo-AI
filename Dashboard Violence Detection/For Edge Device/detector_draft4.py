import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import csv
import cv2
import hailo
import threading
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, db, storage
from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

LOG_PATH = "latency_log.csv"

if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "detection_type", "confidence", "start_to_upload", "start_to_complete"])

# -----------------------------------------------------------------------------------------------
# Firebase Setup
# -----------------------------------------------------------------------------------------------
cred = credentials.Certificate("(file_name).json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://violence-detection-ta-default-rtdb.asia-southeast1.firebasedatabase.app/',
    'storageBucket': 'violence-detection-ta.firebasestorage.app'
})
bucket = storage.bucket()
LOCATION = "Ruangan"
TEMP_DIR = "temp_frames"
os.makedirs(TEMP_DIR, exist_ok=True)

# -----------------------------------------------------------------------------------------------
# Async Upload Thread
# -----------------------------------------------------------------------------------------------
def upload_to_firebase(frame, detection_type, confidence, inference_start_time=None):
    try:
        upload_start = datetime.now()
        print(f"[UPLOAD] Start: {upload_start.strftime('%H:%M:%S')}")

        if inference_start_time:
            latency_start_to_upload = (upload_start - inference_start_time).total_seconds()
            print(f"[LATENCY] Frame → upload start: {latency_start_to_upload:.6f} s")
        else:
            latency_start_to_upload = None

        timestamp = datetime.now()
        formatted_timestamp = timestamp.strftime("%Y%m%d_%H%M%S")
        readable_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")

        filename = f"{detection_type}_{formatted_timestamp}_{int(confidence * 100)}.jpg"
        local_path = os.path.join(TEMP_DIR, filename)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(local_path, rgb_frame)

        blob = bucket.blob(f"detection_images/{filename}")
        blob.upload_from_filename(local_path)
        blob.make_public()
        image_url = blob.public_url

        db.reference("detections").push().set({
            "confidence": float(confidence),
            "detection_type": detection_type.lower(),
            "firebase_timestamp": {".sv": "timestamp"},
            "image_filename": filename,
            "image_url": image_url,
            "location": LOCATION,
            "timestamp": readable_time
        })

        upload_complete = datetime.now()
        latency_complete = (upload_complete - inference_start_time).total_seconds() if inference_start_time else None
        print(f"[UPLOAD COMPLETE] Took {(upload_complete - upload_start).total_seconds():.6f} s")

        with open(LOG_PATH, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                readable_time,
                detection_type,
                f"{confidence:.6f}",
                f"{latency_start_to_upload:.6f}" if latency_start_to_upload else "",
                f"{latency_complete:.6f}" if latency_complete else ""
            ])

        os.remove(local_path)

    except Exception as e:
        print(f"[UPLOAD ERROR] {e}")

def upload_async(frame, detection_type, confidence, inference_start_time=None):
    threading.Thread(
        target=upload_to_firebase,
        args=(frame.copy(), detection_type, confidence, inference_start_time),
        daemon=True
    ).start()

# -----------------------------------------------------------------------------------------------
# Test Upload
# -----------------------------------------------------------------------------------------------
def test_upload_to_firebase():
    try:
        print("Running test upload to Firebase...")
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        upload_to_firebase(dummy_frame, "FirebaseTest", 0.99)
    except Exception as e:
        print(f"Test upload failed: {e}")

# -----------------------------------------------------------------------------------------------
# Detection State Tracking
# -----------------------------------------------------------------------------------------------
class user_app_callback_class:
    def __init__(self):
        self.last_uploaded_time = {
            "Violence": None,
            "Sharp-Object": None
        }
        self.cooldown = timedelta(seconds=10)

# -----------------------------------------------------------------------------------------------
# Callback Function
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    inference_start_time = datetime.now()

    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    format, width, height = get_caps_from_pad(pad)
    frame = get_numpy_from_buffer(buffer, format, width, height) if format and width and height else None

    if frame is None:
        print("No frame extracted.")
        return Gst.PadProbeReturn.OK

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    for detection in detections:
        label = detection.get_label()
        confidence = detection.get_confidence()

        if label == "NonViolence" or confidence < 0.7:
            continue

        if label not in user_data.last_uploaded_time:
            continue

        print(f"Detected: {label} {confidence:.2f}")
        bbox = detection.get_bbox()
        x_min = int(bbox.xmin() * width)
        y_min = int(bbox.ymin() * height)
        x_max = int(bbox.xmax() * width)
        y_max = int(bbox.ymax() * height)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        now = datetime.now()
        last_time = user_data.last_uploaded_time[label]

        if last_time is None or (now - last_time) > user_data.cooldown:
            upload_async(frame, label, confidence, inference_start_time)
            user_data.last_uploaded_time[label] = now

    return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    user_data = user_app_callback_class()
    test_upload_to_firebase() 
    app = GStreamerDetectionApp(app_callback, user_data)

    try:
        print("Starting detection system...")
        app.run()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Cleaning up...")
