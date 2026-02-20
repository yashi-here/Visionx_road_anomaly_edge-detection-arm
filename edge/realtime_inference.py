import cv2
import numpy as np
import subprocess
import tflite_runtime.interpreter as tflite
import time
import csv
import os
from datetime import datetime
import psutil

MODEL_PATH = "bestf-int8.tflite"  # Make sure this is your 320 model

# ---- MATCH MODEL INPUT ----
WIDTH = 320
HEIGHT = 320

CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.45

SHOW_DISPLAY = True

# ---- Logging Files ----
DETECTION_LOG = "detections.csv"
SYSTEM_LOG = "system_metrics.csv"

# Create detection CSV if not exists
if not os.path.exists(DETECTION_LOG):
    with open(DETECTION_LOG, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "confidence",
            "x", "y", "w", "h", "fps"
        ])

# Create system CSV if not exists
if not os.path.exists(SYSTEM_LOG):
    with open(SYSTEM_LOG, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "fps",
            "cpu_percent", "ram_percent",
            "temperature"
        ])

last_system_log = time.time()

# ---- Load Model ----
interpreter = tflite.Interpreter(
    model_path=MODEL_PATH,
    num_threads=4
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model expects input shape:", input_details[0]['shape'])

input_index = input_details[0]['index']
output_index = output_details[0]['index']

output_scale, output_zero = output_details[0]['quantization']

# ---- Camera Command ----
cmd = [
    "rpicam-vid",
    "--width", "320",
    "--height", "320",
    "--framerate", "30",
    "--codec", "yuv420",
    "-t", "0",
    "-o", "-"
]

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
frame_size = WIDTH * HEIGHT * 3 // 2

print("Starting optimized inference with telemetry logging...")

while True:
    start_time = time.time()

    raw = proc.stdout.read(frame_size)
    if len(raw) != frame_size:
        continue

    # ---- YUV → BGR ----
    yuv = np.frombuffer(raw, dtype=np.uint8)
    yuv = yuv.reshape((HEIGHT * 3 // 2, WIDTH))
    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    # ---- Prepare Input (No Resize if 320 model) ----
    input_data = np.expand_dims(frame, axis=0).astype(np.uint8)

    # ---- Inference ----
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_index)[0]

    output = output.astype(np.int32)

    conf = (output[:, 4] - output_zero) * output_scale
    mask = conf > CONF_THRESHOLD

    filtered = output[mask]
    scores = conf[mask]

    boxes = []

    if len(filtered) > 0:
        for i, det in enumerate(filtered):
            x, y, w, h = det[:4]

            x = (x - output_zero) * output_scale
            y = (y - output_zero) * output_scale
            w = (w - output_zero) * output_scale
            h = (h - output_zero) * output_scale

            x1 = int((x - w / 2) * WIDTH)
            y1 = int((y - h / 2) * HEIGHT)
            x2 = int((x + w / 2) * WIDTH)
            y2 = int((y + h / 2) * HEIGHT)

            x1 = max(0, min(WIDTH - 1, x1))
            y1 = max(0, min(HEIGHT - 1, y1))
            x2 = max(0, min(WIDTH - 1, x2))
            y2 = max(0, min(HEIGHT - 1, y2))

            boxes.append([x1, y1, x2 - x1, y2 - y1])

    # ---- FPS Calculation (Before Logging) ----
    fps = 1.0 / (time.time() - start_time)

    # ---- NMS ----
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores.tolist(),
            CONF_THRESHOLD,
            IOU_THRESHOLD
        )

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]

                cv2.rectangle(frame,
                              (x, y),
                              (x + w, y + h),
                              (0, 255, 0), 1)

                # ---- Detection Logging (Confidence > 0.6) ----
                if scores[i] > 0.6:
                    with open(DETECTION_LOG, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            float(scores[i]),
                            x, y, w, h,
                            round(fps, 2)
                        ])

    # ---- System Telemetry Logging (Every 2 Seconds) ----
    current_time = time.time()
    if current_time - last_system_log >= 2:

        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent

        try:
            temp = os.popen("vcgencmd measure_temp").readline()
            temp = temp.replace("temp=", "").replace("'C\n", "")
        except:
            temp = "N/A"

        with open(SYSTEM_LOG, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                round(fps, 2),
                cpu,
                ram,
                temp
            ])

        last_system_log = current_time

    # ---- Display ----
    if SHOW_DISPLAY:
        cv2.putText(frame, f"FPS: {fps:.2f}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 1)

        cv2.imshow("Pothole Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print(f"FPS: {fps:.2f}")

proc.terminate()
cv2.destroyAllWindows()
