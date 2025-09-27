"""
SmartGuard-Laptop v1.0
All-in-one 3D-printer failure-detection proof-of-concept

————————————————————————————————————————
• Runs entirely on a Windows / macOS / Linux laptop
• Uses the built-in webcam (or any USB camera) – NO ESP32 needed
• Live video + AI inference + alert overlay + CSV/JSON logging
• Two modes:
  ─ "Heuristic" (default) – lightweight, no extra model download
  ─ "YOLOv8" – higher accuracy if ultralytics + spaghetti-failure weights present
• Press D to force a demo failure, S to force a demo success
• Tested with Python 3.9, OpenCV 4.9, Ultralytics 8.2

————————————————————————————————————————
Copyright © 2025 Raahul S G
Licence: MIT
"""

import os, time, csv, json, argparse, datetime, random, itertools, collections
import cv2 as cv
import numpy as np

# ─────────────────── CONFIG ─────────────────── #
CFG = {
    "video_source" : 0, # 0 = laptop webcam
    "frame_resize" : (640, 480),
    "log_dir" : "smartguard_logs",
    "heuristic_edge_thres": 0.022, # tuned for default 640×480
    "overlay_font" : cv.FONT_HERSHEY_SIMPLEX,
    "overlay_scale" : 0.6,
    "overlay_color_good" : ( 72,219, 72),
    "overlay_color_bad" : ( 10, 10,255),
    "yolo_weights" : "spaghetti-detect.pt", # optional
}

FAILURE_LABELS = [
    "Layer-Adhesion", "Warping", "Stringing",
    "Under-extrusion", "Support-Failure", "Foreign-Object"
]

# ─────────── optional YOLOv8 engine ─────────── #
class Yolov8Engine:
    def __init__(self, weights):
        from ultralytics import YOLO
        self.model = YOLO(weights)
    
    def detect_failure(self, frame):
        """Return (is_fail, label, conf)"""
        res = self.model(frame, imgsz=640, conf=0.35)[0]
        if not res.boxes: # no defects
            return False, None, 0.0
        best = res.boxes[res.boxes.conf.argmax()]
        label = self.model.names[int(best.cls)]
        return True, label, float(best.conf)

# ───────────── heuristic fallback ───────────── #
def heuristic_detect(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,50,150)
    density = edges.mean() / 255 # 0-1 range
    # simple blob-noise score
    blur = cv.GaussianBlur(gray,(15,15),0)
    diff = cv.absdiff(gray, blur).mean()/255
    score = 0.6*density + 0.4*diff # crude composite  
    is_fail = score > CFG["heuristic_edge_thres"]
    return is_fail, random.choice(FAILURE_LABELS) if is_fail else None, score

# ──────────────── utilities ─────────────────── #
def ensure_log_dir():
    os.makedirs(CFG["log_dir"], exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return ( open(os.path.join(CFG["log_dir"], f"{stamp}.csv"), "w", newline=""),
             open(os.path.join(CFG["log_dir"], f"{stamp}.jsonl"), "w") )

def log_event(csv_fh, json_fh, entry):
    writer = csv.writer(csv_fh)
    writer.writerow(entry.values())
    json_fh.write(json.dumps(entry)+"\n")
    csv_fh.flush(); json_fh.flush()

def overlay_text(img, txt, y, color):
    cv.putText(img, txt, (10,y), CFG["overlay_font"], CFG["overlay_scale"], color, 2, cv.LINE_AA)

# ──────────────────── main ──────────────────── #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo", action="store_true", help="enable YOLOv8 mode (needs ultralytics + weights)")
    args = parser.parse_args()
    
    # pick engine
    if args.yolo and os.path.exists(CFG["yolo_weights"]):
        engine = Yolov8Engine(CFG["yolo_weights"])
        mode = "YOLOv8"
    else:
        engine = None
        mode = "Heuristic"
    
    cam = cv.VideoCapture(CFG["video_source"])
    if not cam.isOpened():
        raise SystemExit("❌ Could not open camera.")
    
    csv_fh, json_fh = ensure_log_dir()
    csv.writer(csv_fh).writerow(["timestamp","result","label","confidence","edgeScore"])
    
    print(f"▶ SmartGuard-Laptop started ({mode} mode) – press Q to quit")
    
    for idx in itertools.count():
        ok, frame = cam.read()
        if not ok: break
        frame = cv.resize(frame, CFG["frame_resize"])
        
        # user-controlled demo injections
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'): break
        demo_force = "none"
        if key == ord('d'): demo_force = "fail" # D key
        if key == ord('s'): demo_force = "succ" # S key
        
        # detection
        if engine:
            is_fail, label, conf = engine.detect_failure(frame)
            edge_score = conf
        else:
            is_fail, label, edge_score = heuristic_detect(frame)
            conf = 1-edge_score if not is_fail else edge_score
        
        # override for demo keys
        if demo_force == "fail": is_fail, label, conf = True, random.choice(FAILURE_LABELS), 0.99
        if demo_force == "succ": is_fail, label, conf = False, None, 0.99
        
        # overlays
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        color = CFG["overlay_color_bad"] if is_fail else CFG["overlay_color_good"]
        overlay_text(frame, f"{stamp} | {('FAIL:'+label) if is_fail else 'PRINT OK'} ({conf*100:0.1f}%)", 25, color)
        overlay_text(frame, f"Detections: {idx}", 50, (255,255,255))
        cv.imshow("SmartGuard-Laptop", frame)
        
        # logging every second frame to limit file size
        if idx % 2 == 0:
            log_entry = collections.OrderedDict(
                timestamp=datetime.datetime.now().isoformat(),
                result="FAIL" if is_fail else "SUCCESS",
                label=label or "",
                confidence=round(conf,3),
                edgeScore=round(edge_score,4)
            )
            log_event(csv_fh, json_fh, log_entry)
    
    cam.release(), cv.destroyAllWindows()
    csv_fh.close(), json_fh.close()
    print("⏹ Session ended – logs saved in", CFG["log_dir"])

if __name__ == "__main__":
    main()