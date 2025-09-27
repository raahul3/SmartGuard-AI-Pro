"""
SmartGuard AI - Combined Version
================================

Two Options:
1. Normal Mode: Exact DEMO.py code (your working version)
2. GUI Mode: Professional interface with same detection algorithm

Usage:
  python smartguard_combined.py           # Normal CLI mode (like DEMO.py)
  python smartguard_combined.py --gui     # GUI mode
  python smartguard_combined.py --yolo    # YOLOv8 mode

Author: Raahul S G (23BAI044)
"""

import os, time, csv, json, argparse, datetime, random, itertools, collections
import cv2 as cv
import numpy as np

# Import tkinter only if GUI mode is requested
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    from PIL import Image, ImageTk
    import threading
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

# ═══════════════════ EXACT DEMO.PY CODE ═══════════════════ #
# This is your exact working code - UNCHANGED
CFG = {
    "video_source" : 0, # 0 = laptop webcam
    "frame_resize" : (640, 480),
    "log_dir" : "smartguard_logs",
    "heuristic_edge_thres": 0.022, # tuned for default 640×480
    "overlay_font" : cv.FONT_HERSHEY_SIMPLEX,
    "overlay_scale" : 0.6,
    "overlay_color_good" : ( 72,219, 72), # Green BGR
    "overlay_color_bad" : ( 10, 10,255), # Red BGR
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
        res = self.model(frame, imgsz=640, conf=0.35, verbose=False)[0]
        if not res.boxes: # no defects
            return False, None, 0.0
        best = res.boxes[res.boxes.conf.argmax()]
        label = self.model.names[int(best.cls)]
        return True, label, float(best.conf)

# ───────────── heuristic fallback - EXACT FROM DEMO.PY ───────────── #
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

# ──────────────── utilities - EXACT FROM DEMO.PY ─────────────────── #
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

# ══════════════════ NORMAL MODE (EXACT DEMO.PY) ══════════════════ #
def run_normal_mode(args):
    """Run in normal CLI mode - EXACT same as your DEMO.py"""
    print("🚀 SmartGuard AI - Normal Mode (Exact DEMO.py)")
    print("=" * 50)
    
    # pick engine - EXACT same logic
    if args.yolo and os.path.exists(CFG["yolo_weights"]):
        try:
            engine = Yolov8Engine(CFG["yolo_weights"])
            mode = "YOLOv8"
            print(f"✅ YOLOv8 engine loaded")
        except:
            engine = None
            mode = "Heuristic"
            print("⚠️ YOLOv8 failed, using Heuristic")
    else:
        engine = None
        mode = "Heuristic"
    
    cam = cv.VideoCapture(CFG["video_source"])
    if not cam.isOpened():
        raise SystemExit("❌ Could not open camera.")
    
    csv_fh, json_fh = ensure_log_dir()
    csv.writer(csv_fh).writerow(["timestamp","result","label","confidence","edgeScore"])
    
    print(f"▶ SmartGuard started ({mode} mode) – press Q to quit")
    print("Press D for demo failure, S for demo success")
    
    # EXACT same main loop as DEMO.py
    for idx in itertools.count():
        ok, frame = cam.read()
        if not ok: break
        frame = cv.resize(frame, CFG["frame_resize"])
        
        # user-controlled demo injections - EXACT same
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'): break
        demo_force = "none"
        if key == ord('d'): demo_force = "fail" # D key
        if key == ord('s'): demo_force = "succ" # S key
        
        # detection - EXACT same logic
        if engine:
            is_fail, label, conf = engine.detect_failure(frame)
            edge_score = conf
        else:
            is_fail, label, edge_score = heuristic_detect(frame)
            conf = 1-edge_score if not is_fail else edge_score
        
        # override for demo keys - EXACT same
        if demo_force == "fail": is_fail, label, conf = True, random.choice(FAILURE_LABELS), 0.99
        if demo_force == "succ": is_fail, label, conf = False, None, 0.99
        
        # overlays - EXACT same
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        color = CFG["overlay_color_bad"] if is_fail else CFG["overlay_color_good"]
        overlay_text(frame, f"{stamp} | {('FAIL:'+label) if is_fail else 'PRINT OK'} ({conf*100:0.1f}%)", 25, color)
        overlay_text(frame, f"Detections: {idx}", 50, (255,255,255))
        cv.imshow("SmartGuard AI - Normal Mode", frame)
        
        # logging every second frame to limit file size - EXACT same
        if idx % 2 == 0:
            log_entry = collections.OrderedDict(
                timestamp=datetime.datetime.now().isoformat(),
                result="FAIL" if is_fail else "SUCCESS",
                label=label or "",
                confidence=round(conf,3),
                edgeScore=round(edge_score,4)
            )
            log_event(csv_fh, json_fh, log_entry)
    
    # cleanup - EXACT same
    cam.release(), cv.destroyAllWindows()
    csv_fh.close(), json_fh.close()
    print("⏹ Session ended – logs saved in", CFG["log_dir"])

# ══════════════════ GUI MODE ══════════════════ #
class SmartGuardGUI:
    """Professional GUI using exact DEMO.py detection algorithm"""
    
    def __init__(self, args):
        self.args = args
        self.root = tk.Tk()
        self.root.title("🚀 SmartGuard AI - Professional GUI (Using Exact DEMO.py Code)")
        self.root.geometry("1300x850")
        self.root.configure(bg='#f5f5f5')
        
        # System state
        self.cam = None
        self.detection_active = False
        self.engine = None
        self.mode = "Heuristic"
        self.detection_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.running = True
        self.detection_results = []
        self.demo_force = None
        
        # Setup engine same as normal mode
        if args.yolo and os.path.exists(CFG["yolo_weights"]):
            try:
                self.engine = Yolov8Engine(CFG["yolo_weights"])
                self.mode = "YOLOv8"
                print("✅ YOLOv8 engine loaded for GUI")
            except:
                print("⚠️ YOLOv8 failed, using Heuristic in GUI")
        
        # Setup logging
        self.csv_fh, self.json_fh = ensure_log_dir()
        csv.writer(self.csv_fh).writerow(["timestamp","result","label","confidence","edgeScore"])
        
        self.create_gui()
    
    def create_gui(self):
        """Create professional GUI interface"""
        # Header
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=70)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame,
                              text="🚀 SmartGuard AI - Professional GUI",
                              font=('Arial', 16, 'bold'),
                              bg='#2c3e50', fg='white')
        title_label.pack(side='left', padx=20, pady=20)
        
        mode_label = tk.Label(header_frame,
                             text=f"Mode: {self.mode} | Using Exact DEMO.py Algorithm",
                             font=('Arial', 10),
                             bg='#2c3e50', fg='#27ae60')
        mode_label.pack(side='right', padx=20, pady=25)
        
        # Control Panel
        control_frame = tk.Frame(self.root, bg='#ecf0f1', relief='raised', bd=2)
        control_frame.pack(fill='x', padx=10, pady=10)
        
        # Camera Controls
        cam_group = tk.LabelFrame(control_frame, text="📹 Camera Controls",
                                 font=('Arial', 10, 'bold'), bg='#ecf0f1')
        cam_group.pack(side='left', padx=10, pady=10, fill='y')
        
        self.start_btn = tk.Button(cam_group, text="▶️ Start Camera",
                                  command=self.toggle_camera,
                                  font=('Arial', 10, 'bold'),
                                  bg='#27ae60', fg='white',
                                  width=15, height=2)
        self.start_btn.pack(padx=10, pady=5)
        
        self.detect_btn = tk.Button(cam_group, text="🔍 Start Detection",
                                   command=self.toggle_detection,
                                   font=('Arial', 10, 'bold'),
                                   bg='#3498db', fg='white',
                                   width=15, height=2,
                                   state='disabled')
        self.detect_btn.pack(padx=10, pady=5)
        
        # Status indicators
        self.cam_status = tk.Label(cam_group, text="📹 Camera: OFF",
                                  font=('Arial', 9), bg='#ecf0f1', fg='#e74c3c')
        self.cam_status.pack(pady=2)
        
        self.detect_status = tk.Label(cam_group, text="🔍 Detection: OFF",
                                     font=('Arial', 9), bg='#ecf0f1', fg='#e74c3c')
        self.detect_status.pack(pady=2)
        
        # Demo Controls
        demo_group = tk.LabelFrame(control_frame, text="🎮 Demo Controls",
                                  font=('Arial', 10, 'bold'), bg='#ecf0f1')
        demo_group.pack(side='left', padx=10, pady=10, fill='y')
        
        tk.Button(demo_group, text="✅ Demo Success",
                 command=lambda: self.inject_demo('succ'),
                 font=('Arial', 9, 'bold'),
                 bg='#27ae60', fg='white',
                 width=15).pack(padx=10, pady=3)
        
        tk.Button(demo_group, text="❌ Demo Failure",
                 command=lambda: self.inject_demo('fail'),
                 font=('Arial', 9, 'bold'),
                 bg='#e74c3c', fg='white',
                 width=15).pack(padx=10, pady=3)
        
        # Statistics Panel
        stats_group = tk.LabelFrame(control_frame, text="📊 Statistics",
                                   font=('Arial', 10, 'bold'), bg='#ecf0f1')
        stats_group.pack(side='left', padx=10, pady=10, fill='y')
        
        self.stats_display = tk.Text(stats_group, height=5, width=20,
                                    font=('Consolas', 9), bg='#34495e', fg='white',
                                    state='disabled')
        self.stats_display.pack(padx=10, pady=10)
        
        # Export Controls
        export_group = tk.LabelFrame(control_frame, text="💾 Export",
                                    font=('Arial', 10, 'bold'), bg='#ecf0f1')
        export_group.pack(side='right', padx=10, pady=10, fill='y')
        
        tk.Button(export_group, text="📊 Export CSV",
                 command=self.export_csv,
                 font=('Arial', 9, 'bold'),
                 bg='#f39c12', fg='white',
                 width=12).pack(padx=10, pady=3)
        
        tk.Button(export_group, text="📋 Export JSON",
                 command=self.export_json,
                 font=('Arial', 9, 'bold'),
                 bg='#9b59b6', fg='white',
                 width=12).pack(padx=10, pady=3)
        
        # Main Content Area
        content_frame = tk.Frame(self.root, bg='#f5f5f5')
        content_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Video Display
        video_frame = tk.LabelFrame(content_frame, text="🎥 Live Video Feed",
                                   font=('Arial', 12, 'bold'), bg='#f5f5f5')
        video_frame.pack(side='left', fill='both', expand=True, padx=(0,5))
        
        self.video_label = tk.Label(video_frame, bg='#2c3e50',
                                   text="SmartGuard AI Professional GUI\n\nCamera feed will appear here\n\nClick 'Start Camera' to begin",
                                   fg='#ecf0f1', font=('Arial', 14))
        self.video_label.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Results Panel
        results_frame = tk.LabelFrame(content_frame, text="📋 Detection Results",
                                     font=('Arial', 12, 'bold'), bg='#f5f5f5')
        results_frame.pack(side='right', fill='both', padx=(5,0))
        results_frame.configure(width=400)
        
        # Status Display
        self.status_display = tk.Text(results_frame, height=15, width=45,
                                     font=('Consolas', 9), bg='#2c3e50', fg='#ecf0f1',
                                     wrap='word', state='disabled')
        self.status_display.pack(fill='x', padx=10, pady=10)
        
        # Detection Log
        log_frame = tk.LabelFrame(results_frame, text="📝 Detection Log",
                                 font=('Arial', 10, 'bold'), bg='#f5f5f5')
        log_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        columns = ('Time', 'Result', 'Type', 'Confidence', 'Score')
        self.log_tree = ttk.Treeview(log_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.log_tree.heading(col, text=col)
            self.log_tree.column(col, width=70, anchor='center')
        
        log_scroll = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_tree.yview)
        self.log_tree.configure(yscrollcommand=log_scroll.set)
        
        self.log_tree.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        log_scroll.pack(side='right', fill='y', pady=5)
        
        self.update_stats()
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if self.cam is None:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start camera feed"""
        try:
            self.cam = cv.VideoCapture(CFG["video_source"])
            if not self.cam.isOpened():
                raise Exception("Cannot access camera")
            
            self.start_btn.config(text="⏹️ Stop Camera", bg='#e74c3c')
            self.detect_btn.config(state='normal')
            self.cam_status.config(text="📹 Camera: ON", fg='#27ae60')
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to start camera: {str(e)}")
    
    def stop_camera(self):
        """Stop camera feed"""
        if self.cam:
            self.cam.release()
            self.cam = None
        
        self.start_btn.config(text="▶️ Start Camera", bg='#27ae60')
        self.detect_btn.config(state='disabled')
        self.cam_status.config(text="📹 Camera: OFF", fg='#e74c3c')
        
        # Reset video display
        self.video_label.config(image='', text="Camera stopped\n\nClick 'Start Camera' to begin")
    
    def camera_loop(self):
        """Main camera processing loop"""
        idx = 0
        while self.cam and self.cam.isOpened() and self.running:
            ret, frame = self.cam.read()
            if not ret:
                break
            
            # Resize frame
            frame = cv.resize(frame, CFG["frame_resize"])
            
            # Process detection if active
            if self.detection_active:
                # Get demo force
                demo_force = self.demo_force
                self.demo_force = None  # Reset
                
                # EXACT same detection logic as DEMO.py
                if self.engine:
                    is_fail, label, conf = self.engine.detect_failure(frame)
                    edge_score = conf
                else:
                    is_fail, label, edge_score = heuristic_detect(frame)
                    conf = 1-edge_score if not is_fail else edge_score
                
                # EXACT same demo override logic
                if demo_force == "fail": is_fail, label, conf = True, random.choice(FAILURE_LABELS), 0.99
                if demo_force == "succ": is_fail, label, conf = False, None, 0.99
                
                # Update counters and log (same as DEMO.py)
                self.detection_count += 1
                if is_fail:
                    self.failure_count += 1
                else:
                    self.success_count += 1
                
                # EXACT same logging logic
                if idx % 2 == 0:  # Every second frame
                    log_entry = collections.OrderedDict(
                        timestamp=datetime.datetime.now().isoformat(),
                        result="FAIL" if is_fail else "SUCCESS",
                        label=label or "",
                        confidence=round(conf,3),
                        edgeScore=round(edge_score,4)
                    )
                    log_event(self.csv_fh, self.json_fh, log_entry)
                
                # Update GUI
                self.root.after_idle(self.update_gui_display, is_fail, label, conf, edge_score)
                
                # EXACT same overlays as DEMO.py
                stamp = datetime.datetime.now().strftime("%H:%M:%S")
                color = CFG["overlay_color_bad"] if is_fail else CFG["overlay_color_good"]
                overlay_text(frame, f"{stamp} | {('FAIL:'+label) if is_fail else 'PRINT OK'} ({conf*100:0.1f}%)", 25, color)
                overlay_text(frame, f"Detections: {self.detection_count}", 50, (255,255,255))
            else:
                # Not detecting
                stamp = datetime.datetime.now().strftime("%H:%M:%S")
                overlay_text(frame, f"{stamp} | READY - Click 'Start Detection'", 25, (255,255,255))
                overlay_text(frame, f"SmartGuard AI GUI | Mode: {self.mode}", 50, (255,255,255))
            
            # Convert and display frame
            try:
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_tk = ImageTk.PhotoImage(frame_pil)
                self.root.after_idle(self.update_video_display, frame_tk)
            except:
                pass
            
            idx += 1
            time.sleep(0.033)  # 30 FPS
    
    def update_video_display(self, frame_tk):
        """Update video display"""
        try:
            if hasattr(self, 'video_label') and self.video_label.winfo_exists():
                self.video_label.config(image=frame_tk, text='')
                self.video_label.image = frame_tk
        except:
            pass
    
    def toggle_detection(self):
        """Toggle detection on/off"""
        if not self.detection_active:
            self.detection_active = True
            self.detect_btn.config(text="⏹️ Stop Detection", bg='#e74c3c')
            self.detect_status.config(text="🔍 Detection: ON", fg='#27ae60')
        else:
            self.detection_active = False
            self.detect_btn.config(text="🔍 Start Detection", bg='#3498db')
            self.detect_status.config(text="🔍 Detection: OFF", fg='#e74c3c')
    
    def inject_demo(self, demo_type):
        """Inject demo detection (like D/S keys in DEMO.py)"""
        if not self.detection_active:
            messagebox.showwarning("Detection Not Active", "Please start detection first!")
            return
        self.demo_force = demo_type
        print(f"Demo injected: {demo_type}")
    
    def update_gui_display(self, is_fail, label, conf, edge_score):
        """Update GUI displays"""
        timestamp_str = datetime.datetime.now().strftime('%H:%M:%S')
        
        # Update status display
        if is_fail:
            status_msg = f"""🚨 FAILURE DETECTED!
Time: {timestamp_str}
Type: {label}
Confidence: {conf*100:.1f}%
Edge Score: {edge_score:.4f}

Algorithm: EXACT DEMO.py heuristic_detect()
Threshold: {CFG['heuristic_edge_thres']}
Formula: 0.6*density + 0.4*diff
Detection #{self.detection_count}"""
            self.status_display.config(fg='#ff6b6b')
        else:
            status_msg = f"""✅ PRINT OK
Time: {timestamp_str}
Confidence: {conf*100:.1f}%
Edge Score: {edge_score:.4f}

Algorithm: EXACT DEMO.py heuristic_detect()
Threshold: {CFG['heuristic_edge_thres']}
Formula: 0.6*density + 0.4*diff
Detection #{self.detection_count}"""
            self.status_display.config(fg='#6bcf7f')
        
        self.status_display.config(state='normal')
        self.status_display.delete('1.0', tk.END)
        self.status_display.insert('1.0', status_msg)
        self.status_display.config(state='disabled')
        
        # Update log tree
        result_text = "❌ FAIL" if is_fail else "✅ OK"
        self.log_tree.insert('', 0, values=(
            timestamp_str,
            result_text,
            label[:10] if label else 'N/A',
            f"{conf*100:.1f}",
            f"{edge_score:.3f}"
        ))
        
        # Keep only last 30 entries
        children = self.log_tree.get_children()
        if len(children) > 30:
            self.log_tree.delete(children[-1])
        
        self.update_stats()
    
    def update_stats(self):
        """Update statistics display"""
        self.stats_display.config(state='normal')
        self.stats_display.delete('1.0', tk.END)
        
        if self.detection_count > 0:
            success_rate = (self.success_count / self.detection_count) * 100
        else:
            success_rate = 0
        
        stats_text = f"""📊 STATISTICS
Total: {self.detection_count}
✅ Success: {self.success_count}
❌ Failures: {self.failure_count}
Success Rate: {success_rate:.1f}%

Algorithm: EXACT
DEMO.py code"""
        
        self.stats_display.insert('1.0', stats_text)
        self.stats_display.config(state='disabled')
    
    def export_csv(self):
        """Export detection data to CSV"""
        if self.detection_count == 0:
            messagebox.showwarning("No Data", "No detection data to export!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Export Detection Data as CSV"
        )
        
        if filename:
            try:
                import shutil
                shutil.copy(self.csv_fh.name, filename)
                messagebox.showinfo("Export Complete", f"✅ CSV exported successfully!\n\nFile: {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Could not export CSV: {str(e)}")
    
    def export_json(self):
        """Export detection data to JSON"""
        if self.detection_count == 0:
            messagebox.showwarning("No Data", "No detection data to export!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Export Detection Data as JSON"
        )
        
        if filename:
            try:
                import shutil
                shutil.copy(self.json_fh.name, filename)
                messagebox.showinfo("Export Complete", f"✅ JSON exported successfully!\n\nFile: {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Could not export JSON: {str(e)}")
    
    def run(self):
        """Run the GUI"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        print("🚀 SmartGuard AI GUI Started!")
        print(f"Mode: {self.mode}")
        print("Using EXACT DEMO.py detection algorithm")
        print("=" * 50)
        self.root.mainloop()
    
    def on_closing(self):
        """Clean shutdown"""
        self.running = False
        if self.cam:
            self.cam.release()
        cv.destroyAllWindows()
        try:
            self.csv_fh.close()
            self.json_fh.close()
        except:
            pass
        self.root.destroy()

# ══════════════════ MAIN FUNCTION ══════════════════ #
def main():
    """Main function with both options"""
    parser = argparse.ArgumentParser(description="SmartGuard AI - Combined Version")
    parser.add_argument("--yolo", action="store_true",
                       help="Enable YOLOv8 mode (needs ultralytics + weights)")
    parser.add_argument("--gui", action="store_true",
                       help="Run in GUI mode (default: normal CLI mode)")
    
    args = parser.parse_args()
    
    if args.gui:
        # GUI Mode
        if not GUI_AVAILABLE:
            print("❌ GUI mode requires tkinter and PIL libraries")
            print("Install with: pip install pillow")
            return
        
        print("🚀 Starting SmartGuard AI in GUI Mode...")
        app = SmartGuardGUI(args)
        app.run()
    else:
        # Normal CLI Mode (exact DEMO.py)
        run_normal_mode(args)

if __name__ == "__main__":
    main()