import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
from pathlib import Path

class SafetyGearDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Safety Gear Detection System")
        self.root.geometry("600x650")
        self.root.resizable(False, False)
        
        # Variables
        self.model = None
        self.tracker = None
        self.cap = None
        self.is_running = False
        self.detection_thread = None
        
        self.selected_model = tk.StringVar()
        self.selected_input = tk.StringVar(value="camera")
        self.video_path = tk.StringVar()
        self.camera_index = tk.StringVar(value="0")
        
        # Setup UI
        self.setup_ui()
        
        # Scan for available models
        self.scan_models()
    
    def setup_ui(self):
        # Title
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="🏍️ Safety Gear Detection System",
            font=("Arial", 20, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(pady=20)
        
        # Main container
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Model Selection Section
        model_frame = tk.LabelFrame(main_frame, text="Model Selection", font=("Arial", 12, "bold"), padx=10, pady=10)
        model_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.model_combo = ttk.Combobox(
            model_frame, 
            textvariable=self.selected_model,
            state="readonly",
            font=("Arial", 10),
            width=50
        )
        self.model_combo.pack(pady=5)
        
        refresh_btn = tk.Button(
            model_frame,
            text="🔄 Refresh Models",
            command=self.scan_models,
            font=("Arial", 9),
            bg="#3498db",
            fg="white",
            cursor="hand2"
        )
        refresh_btn.pack(pady=5)
        
        # Input Source Section
        input_frame = tk.LabelFrame(main_frame, text="Input Source", font=("Arial", 12, "bold"), padx=10, pady=10)
        input_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Camera option
        camera_radio = tk.Radiobutton(
            input_frame,
            text="📹 Live Camera",
            variable=self.selected_input,
            value="camera",
            font=("Arial", 10),
            command=self.toggle_input_options
        )
        camera_radio.pack(anchor=tk.W, pady=5)
        
        camera_frame = tk.Frame(input_frame)
        camera_frame.pack(fill=tk.X, padx=20)
        
        tk.Label(camera_frame, text="Camera Index:", font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        camera_entry = tk.Entry(camera_frame, textvariable=self.camera_index, width=5, font=("Arial", 10))
        camera_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(camera_frame, text="(0 = default camera)", font=("Arial", 8), fg="gray").pack(side=tk.LEFT)
        
        # Video file option
        video_radio = tk.Radiobutton(
            input_frame,
            text="🎬 Video File",
            variable=self.selected_input,
            value="video",
            font=("Arial", 10),
            command=self.toggle_input_options
        )
        video_radio.pack(anchor=tk.W, pady=(10, 5))
        
        video_frame = tk.Frame(input_frame)
        video_frame.pack(fill=tk.X, padx=20)
        
        self.video_entry = tk.Entry(video_frame, textvariable=self.video_path, font=("Arial", 9), state="disabled")
        self.video_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.browse_btn = tk.Button(
            video_frame,
            text="Browse...",
            command=self.browse_video,
            font=("Arial", 9),
            bg="#95a5a6",
            fg="white",
            cursor="hand2",
            state="disabled"
        )
        self.browse_btn.pack(side=tk.LEFT)
        
        # Quick access to any footage directory
        ctms_frame = tk.Frame(input_frame)
        ctms_frame.pack(fill=tk.X, padx=20, pady=(5, 0))
        
        self.ctms_btn = tk.Button(
            ctms_frame,
            text="📁 Browse Video Directory",
            command=self.load_ctms_footage,
            font=("Arial", 9),
            bg="#9b59b6",
            fg="white",
            cursor="hand2",
            state="disabled"
        )
        self.ctms_btn.pack(anchor=tk.W)
        
        # Status Section
        status_frame = tk.LabelFrame(main_frame, text="Status", font=("Arial", 12, "bold"), padx=10, pady=10)
        status_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.status_label = tk.Label(
            status_frame,
            text="Ready to start detection",
            font=("Arial", 10),
            fg="#27ae60"
        )
        self.status_label.pack(pady=5)
        
        # Control Buttons
        button_frame = tk.Frame(main_frame, pady=10)
        button_frame.pack(fill=tk.X)
        
        self.start_btn = tk.Button(
            button_frame,
            text="▶ START DETECTION",
            command=self.start_detection,
            font=("Arial", 14, "bold"),
            bg="#27ae60",
            fg="white",
            cursor="hand2",
            height=3,
            relief=tk.RAISED,
            borderwidth=3
        )
        self.start_btn.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.stop_btn = tk.Button(
            button_frame,
            text="⏹ STOP",
            command=self.stop_detection,
            font=("Arial", 14, "bold"),
            bg="#e74c3c",
            fg="white",
            cursor="hand2",
            height=3,
            state="disabled",
            relief=tk.RAISED,
            borderwidth=3
        )
        self.stop_btn.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Footer
        footer = tk.Label(
            self.root,
            text="Press 'Q' in detection window to stop | Thesis Project 2025",
            font=("Arial", 8),
            fg="gray",
            bg="#ecf0f1"
        )
        footer.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
    
    def toggle_input_options(self):
        if self.selected_input.get() == "video":
            self.video_entry.config(state="normal")
            self.browse_btn.config(state="normal", bg="#3498db")
            self.ctms_btn.config(state="normal", bg="#9b59b6")
        else:
            self.video_entry.config(state="disabled")
            self.browse_btn.config(state="disabled", bg="#95a5a6")
            self.ctms_btn.config(state="disabled", bg="#95a5a6")
    
    def scan_models(self):
        models = []
        
        # Check for best.pt
        if os.path.exists("best.pt"):
            models.append("best.pt")
        
        # Check for models in models/ directory
        if os.path.exists("models"):
            for file in os.listdir("models"):
                if file.endswith(".pt"):
                    models.append(f"models/{file}")
        
        # Check for yolov8 models in root
        for file in ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]:
            if os.path.exists(file) and file not in models:
                models.append(file)
        
        # Add default option
        if "yolov8n.pt" not in models:
            models.append("yolov8n.pt (will download)")
        
        self.model_combo['values'] = models
        if models:
            self.model_combo.current(0)
    
    def browse_video(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("All Files", "*.*")
            ]
        )
        if filename:
            self.video_path.set(filename)
    
    def load_ctms_footage(self):
        # Ask user for CTMS footage directory
        ctms_path = filedialog.askdirectory(
            title="Select CTMS Footage Directory",
            initialdir="C:\\" if os.path.exists("C:\\") else "/"
        )
        
        if not ctms_path:
            return
        
        if not os.path.exists(ctms_path):
            messagebox.showerror("Error", "Selected directory not found!")
            return
        
        video_files = [f for f in os.listdir(ctms_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'))]
        
        if not video_files:
            messagebox.showerror("Error", "No video files found in selected directory!")
            return
        
        # Create selection dialog
        selection_window = tk.Toplevel(self.root)
        selection_window.title("Select Video File")
        selection_window.geometry("600x300")
        selection_window.transient(self.root)
        selection_window.grab_set()
        
        tk.Label(selection_window, text="Select a video file:", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Add scrollbar
        frame = tk.Frame(selection_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(frame, font=("Arial", 10), yscrollcommand=scrollbar.set)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        for video in video_files:
            listbox.insert(tk.END, video)
        
        def select_video():
            selection = listbox.curselection()
            if selection:
                selected_file = video_files[selection[0]]
                full_path = os.path.join(ctms_path, selected_file)
                self.video_path.set(full_path)
                selection_window.destroy()
            else:
                messagebox.showwarning("No Selection", "Please select a video file!")
        
        button_frame = tk.Frame(selection_window)
        button_frame.pack(pady=10)
        
        tk.Button(
            button_frame,
            text="Select",
            command=select_video,
            font=("Arial", 10),
            bg="#27ae60",
            fg="white",
            width=10
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            button_frame,
            text="Cancel",
            command=selection_window.destroy,
            font=("Arial", 10),
            bg="#95a5a6",
            fg="white",
            width=10
        ).pack(side=tk.LEFT, padx=5)
    
    def start_detection(self):
        # Validate inputs
        if not self.selected_model.get():
            messagebox.showerror("Error", "Please select a model!")
            return
        
        if self.selected_input.get() == "video" and not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file!")
            return
        
        # Disable start button
        self.start_btn.config(state="disabled", bg="#95a5a6")
        self.stop_btn.config(state="normal", bg="#e74c3c")
        
        # Start detection in separate thread
        self.is_running = True
        self.detection_thread = threading.Thread(target=self.run_detection, daemon=True)
        self.detection_thread.start()
    
    def stop_detection(self):
        self.is_running = False
        self.status_label.config(text="Stopping detection...", fg="#e74c3c")
        
        # Wait a bit for thread to finish
        if self.detection_thread:
            self.detection_thread.join(timeout=2)
        
        # Clean up
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Re-enable buttons
        self.start_btn.config(state="normal", bg="#27ae60")
        self.stop_btn.config(state="disabled", bg="#95a5a6")
        self.status_label.config(text="Detection stopped", fg="#e74c3c")
    
    def _update_status(self, text, fg="#27ae60"):
        """Thread-safe status update via main thread."""
        self.root.after(0, lambda: self.status_label.config(text=text, fg=fg))

    def _show_error(self, title, message):
        """Thread-safe error dialog via main thread."""
        self.root.after(0, lambda: messagebox.showerror(title, message))

    def _cleanup_after_detection(self):
        """Schedule UI cleanup on the main thread after detection ends."""
        def _do_cleanup():
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            self.start_btn.config(state="normal", bg="#27ae60")
            self.stop_btn.config(state="disabled", bg="#95a5a6")
        self.root.after(0, _do_cleanup)

    def run_detection(self):
        try:
            # Update status
            self._update_status("Loading model...", "#f39c12")
            
            # Load model
            model_path = self.selected_model.get().replace(" (will download)", "")
            self.model = YOLO(model_path)
            
            # Initialize tracker
            self.tracker = DeepSort(
                max_age=30,
                n_init=3,
                max_cosine_distance=0.2,
                nn_budget=None,
                override_track_class=None,
                embedder="mobilenet",
                half=True,
                bgr=True,
                embedder_gpu=True
            )
            
            # Get class names
            class_names = self.model.names
            
            # Open video source
            self._update_status("Opening video source...", "#f39c12")
            
            if self.selected_input.get() == "camera":
                camera_idx = int(self.camera_index.get())
                self.cap = cv2.VideoCapture(camera_idx)
            else:
                self.cap = cv2.VideoCapture(self.video_path.get())
            
            if not self.cap.isOpened():
                raise Exception("Could not open video source")
            
            # Get screen resolution from the existing root (thread-safe read)
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            
            max_display_width = int(screen_width * 0.8)
            max_display_height = int(screen_height * 0.8)
            
            # Create window
            window_name = "Safety Gear Detection – Real-Time"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            
            self._update_status("Running detection... (Press Q to stop)", "#27ae60")
            
            # Main detection loop
            while self.is_running:
                start_time = time.time()
                ret, frame = self.cap.read()
                
                if not ret:
                    self._update_status("End of video reached", "#e74c3c")
                    break
                
                # YOLO Inference
                results = self.model(frame, stream=False, verbose=False, conf=0.25)[0]
                
                detections = []
                helmet_boxes = []
                
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = class_names[cls_id]
                    
                    if conf < 0.25:
                        continue
                    
                    w = x2 - x1
                    h = y2 - y1
                    detections.append([[x1, y1, w, h], conf, cls_name])
                    
                    if 'helmet' in str(cls_name).lower():
                        helmet_boxes.append([x1, y1, x2, y2])
                
                # Update tracker
                tracks = self.tracker.update_tracks(detections, frame=frame)
                
                # Create annotated frame
                annotated_frame = frame.copy()
                
                rider_count = 0
                compliant_count = 0
                object_counts = {}
                
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    
                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    class_name = track.get_det_class()
                    
                    x1, y1, x2, y2 = map(int, ltrb)
                    
                    object_counts[class_name] = object_counts.get(class_name, 0) + 1
                    
                    color = (0, 191, 255)
                    
                    is_rider = 'rider' in str(class_name).lower() or 'person' in str(class_name).lower()
                    
                    compliance_status = ""
                    if is_rider:
                        rider_count += 1
                        is_compliant = False
                        
                        rider_box = [x1, y1, x2, y2]
                        for h_box in helmet_boxes:
                            if self.calculate_intersection(rider_box, h_box) > 0:
                                is_compliant = True
                                break
                        
                        if is_compliant:
                            compliant_count += 1
                            compliance_status = "COMPLIANT"
                            color = (0, 255, 0)
                        else:
                            compliance_status = "NON-COMPLIANT"
                            color = (0, 0, 255)
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"ID {track_id} - {class_name}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    if compliance_status:
                        cv2.putText(annotated_frame, compliance_status, (x1, y2 + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Add metrics
                fps = 1.0 / (time.time() - start_time)
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                y_offset = 80
                cv2.putText(annotated_frame, f"Detections: {len(tracks)} tracked", (20, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                y_offset += 25
                if object_counts:
                    for cls, count in object_counts.items():
                        y_offset += 25
                        cv2.putText(annotated_frame, f"  {cls}: {count}", (30, y_offset), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    y_offset += 25
                    cv2.putText(annotated_frame, "  No objects detected", (30, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
                
                if rider_count > 0:
                    compliance_rate = (compliant_count / rider_count) * 100
                else:
                    compliance_rate = 0.0
                
                cv2.rectangle(annotated_frame, (15, y_offset + 20), (350, y_offset + 60), (0, 0, 0), -1)
                cv2.putText(annotated_frame, f"Safety Compliance: {compliance_rate:.1f}%", 
                            (20, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Side-by-side display
                if frame.shape != annotated_frame.shape:
                    annotated_frame = cv2.resize(annotated_frame, (frame.shape[1], frame.shape[0]))
                
                combined = np.hstack((frame, annotated_frame))
                
                # Scale to fit screen
                h, w = combined.shape[:2]
                scale_w = max_display_width / w
                scale_h = max_display_height / h
                scale = min(scale_w, scale_h, 1.0)
                
                if scale < 1.0:
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    combined = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                cv2.imshow(window_name, combined)
                
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.resizeWindow(window_name, combined.shape[1], combined.shape[0])
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
            
            # Clean up (schedule on main thread to avoid self-join)
            self.is_running = False
            self._cleanup_after_detection()
            
        except Exception as e:
            self.is_running = False
            self._show_error("Error", f"Detection error: {str(e)}")
            self._cleanup_after_detection()
    
    def calculate_intersection(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        return width * height

def main():
    root = tk.Tk()
    app = SafetyGearDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
