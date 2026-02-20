import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import csv
import os
from datetime import datetime
import psutil

# ---- CONFIGURATION ----
MODEL_PATH = "bestf-int8.tflite"
VIDEO_SOURCE = "test_video1.mp4"
OUTPUT_VIDEO = "processed_potholes.mp4"
DETECTION_LOG = "detections.csv"
SYSTEM_LOG = "system_metrics.csv"

WIDTH, HEIGHT = 320, 320
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.45
SHOW_DISPLAY = False  # Set to False for Headless/SSH

# ---- Setup Logging ----
for log in [DETECTION_LOG, SYSTEM_LOG]:
    if not os.path.exists(log):
        with open(log, 'w', newline='') as f:
            writer = csv.writer(f)
            if log == DETECTION_LOG:
                writer.writerow(["timestamp", "confidence", "x", "y", "w", "h", "fps"])
            else:
                writer.writerow(["timestamp", "fps", "cpu_percent", "ram_percent", "temperature"])

# ---- Initialize TFLite ----
interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]['index']
output_index = output_details[0]['index']
output_scale, output_zero = output_details[0]['quantization']

# ---- Initialize Video ----
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Could not open video {VIDEO_SOURCE}")
    exit()

# Setup Video Writer to save results
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 20.0, (WIDTH, HEIGHT))

print(f"Starting inference on {VIDEO_SOURCE}...")
last_system_log = time.time()

try:
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for model
        frame_resized = cv2.resize(frame, (WIDTH, HEIGHT))
        input_data = np.expand_dims(frame_resized, axis=0).astype(np.uint8)

        # Inference
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)[0].astype(np.int32)

        # Process Detections
        conf = (output[:, 4] - output_zero) * output_scale
        mask = conf > CONF_THRESHOLD
        filtered = output[mask]
        scores = conf[mask]

        boxes = []
        for det in filtered:
            x, y, w, h = det[:4]
            x = (x - output_zero) * output_scale
            y = (y - output_zero) * output_scale
            w = (w - output_zero) * output_scale
            h = (h - output_zero) * output_scale

            x1 = int((x - w / 2) * WIDTH)
            y1 = int((y - h / 2) * HEIGHT)
            boxes.append([x1, y1, int(w * WIDTH), int(h * HEIGHT)])

        fps = 1.0 / (time.time() - start_time)

        # NMS and Drawing
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, scores.tolist(), CONF_THRESHOLD, IOU_THRESHOLD)
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    if scores[i] > 0.6:
                        with open(DETECTION_LOG, 'a', newline='') as f:
                            csv.writer(f).writerow([datetime.now().strftime("%H:%M:%S"), round(float(scores[i]), 2), x, y, w, h, round(fps, 2)])

        # Write frame to output video
        out.write(frame_resized)

        # System Telemetry
        if time.time() - last_system_log >= 2:
            temp = os.popen("vcgencmd measure_temp").readline().replace("temp=","").replace("'C\n","")
            with open(SYSTEM_LOG, 'a', newline='') as f:
                csv.writer(f).writerow([datetime.now().strftime("%H:%M:%S"), round(fps, 2), psutil.cpu_percent(), psutil.virtual_memory().percent, temp])
            last_system_log = time.time()
            print(f"Processing... FPS: {fps:.2f} | Temp: {temp}°C")

finally:
    cap.release()
    out.release()
    print(f"Finished! Result saved as {OUTPUT_VIDEO}")
