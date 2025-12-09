import cv2
import numpy as np
import json
import pickle
from collections import deque

import tensorflow as tf
from ultralytics import YOLO

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk



MODEL_PATH = "exercise_lstm_yolo_v2.keras"
ENCODER_PATH = "label_encoder.pkl"
CONFIG_PATH = "config.json"

print("Učitavam Keras model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Učitavam LabelEncoder...")
with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

print("Učitavam config...")
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

SEQ_LEN = cfg["seq_len"]
NUM_FEATURES = cfg["num_features"]

print(f"SEQ_LEN = {SEQ_LEN}, NUM_FEATURES = {NUM_FEATURES}")

print("Učitavam YOLOv8 pose model...")
yolo_model = YOLO("yolov8n-pose.pt")

yolo_joints = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
joint_index = {name: i for i, name in enumerate(yolo_joints)}



def angle_3d(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return 0.0

    cos_ang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


def angle_2d(a_xy, b_xy, c_xy):
    a = np.array([a_xy[0], a_xy[1], 0.0], dtype=np.float32)
    b = np.array([b_xy[0], b_xy[1], 0.0], dtype=np.float32)
    c = np.array([c_xy[0], c_xy[1], 0.0], dtype=np.float32)
    return angle_3d(a, b, c)


def dist_3d(p1, p2):
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    return float(np.linalg.norm(p1 - p2))


def extract_features_from_yolo_keypoints(kpts):
    

    lm = {}
    for name, idx in joint_index.items():
        x = float(kpts[idx, 0])
        y = float(kpts[idx, 1])
        lm[name] = (x, y, 0.0)


    left_hip = lm["left_hip"]
    right_hip = lm["right_hip"]
    mid_hip = (
        (left_hip[0] + right_hip[0]) / 2.0,
        (left_hip[1] + right_hip[1]) / 2.0,
        (left_hip[2] + right_hip[2]) / 2.0,
    )

    for name in lm.keys():
        x, y, z = lm[name]
        lm[name] = (x - mid_hip[0], y - mid_hip[1], z - mid_hip[2])

    coords = []
    for name in yolo_joints:
        x, y, z = lm[name]
        coords.extend([x, y, z])

    angles = []
    angles.append(angle_3d(lm["right_elbow"], lm["right_shoulder"], lm["right_hip"]))
    angles.append(angle_3d(lm["left_elbow"],  lm["left_shoulder"],  lm["left_hip"]))
    angles.append(angle_3d(lm["right_hip"],   lm["right_knee"],     lm["right_ankle"]))
    angles.append(angle_3d(lm["left_hip"],    lm["left_knee"],      lm["left_ankle"]))
    angles.append(angle_3d(lm["right_wrist"], lm["right_elbow"],    lm["right_shoulder"]))
    angles.append(angle_3d(lm["left_wrist"],  lm["left_elbow"],     lm["left_shoulder"]))

    dists = []
    dists.append(dist_3d(lm["left_shoulder"],  lm["left_wrist"]))
    dists.append(dist_3d(lm["right_shoulder"], lm["right_wrist"]))
    dists.append(dist_3d(lm["left_hip"],       lm["left_ankle"]))
    dists.append(dist_3d(lm["right_hip"],      lm["right_ankle"]))
    dists.append(dist_3d(lm["left_hip"],       lm["left_wrist"]))
    dists.append(dist_3d(lm["right_hip"],      lm["right_wrist"]))
    dists.append(dist_3d(lm["left_shoulder"],  lm["left_ankle"]))
    dists.append(dist_3d(lm["right_shoulder"], lm["right_ankle"]))

    feat = np.array(coords + angles + dists, dtype=np.float32)

    if feat.shape[0] < NUM_FEATURES:
        pad = np.zeros(NUM_FEATURES - feat.shape[0], dtype=np.float32)
        feat = np.concatenate([feat, pad])
    elif feat.shape[0] > NUM_FEATURES:
        feat = feat[:NUM_FEATURES]

    return feat


def predict_from_buffer(buffer):
    if len(buffer) < SEQ_LEN:
        return None, 0.0

    seq = np.array(buffer, dtype=np.float32)   
    seq = np.expand_dims(seq, axis=0)          

    probs = model.predict(seq, verbose=0)[0]
    idx = int(np.argmax(probs))
    class_name = le.inverse_transform([idx])[0]
    conf = float(probs[idx])
    return class_name, conf


def get_joint_2d(kpts, name):
    idx = joint_index[name]
    return kpts[idx, 0:2]


def compute_pushup_angle(kpts_2d):
    shoulder = get_joint_2d(kpts_2d, "right_shoulder")
    elbow    = get_joint_2d(kpts_2d, "right_elbow")
    wrist    = get_joint_2d(kpts_2d, "right_wrist")
    return angle_2d(shoulder, elbow, wrist)


def compute_squat_angle(kpts_2d):
    hip   = get_joint_2d(kpts_2d, "right_hip")
    knee  = get_joint_2d(kpts_2d, "right_knee")
    ankle = get_joint_2d(kpts_2d, "right_ankle")
    return angle_2d(hip, knee, ankle)


def compute_situp_angle(kpts_2d):
    shoulder = get_joint_2d(kpts_2d, "right_shoulder")
    hip      = get_joint_2d(kpts_2d, "right_hip")
    knee     = get_joint_2d(kpts_2d, "right_knee")
    return angle_2d(shoulder, hip, knee)


def compute_jj_metric(kpts_2d):
    la = get_joint_2d(kpts_2d, "left_ankle")
    ra = get_joint_2d(kpts_2d, "right_ankle")
    lh = get_joint_2d(kpts_2d, "left_hip")
    rh = get_joint_2d(kpts_2d, "right_hip")

    ankle_dist = np.linalg.norm(la - ra)
    hip_dist   = np.linalg.norm(lh - rh)
    if hip_dist < 1e-3:
        return 0.0
    return float(ankle_dist / hip_dist)


def compute_pullup_metric(kpts_2d):
    rs = get_joint_2d(kpts_2d, "right_shoulder")
    rw = get_joint_2d(kpts_2d, "right_wrist")
    rh = get_joint_2d(kpts_2d, "right_hip")

    vert = abs(rs[1] - rw[1])
    scale = abs(rs[1] - rh[1])
    if scale < 1e-3:
        return 0.0
    return float(vert / scale)


class ExerciseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Exercise Recognition")

        self.cap = None
        self.source_type = None  
        self.running = False

       
        self.feat_buffer = deque(maxlen=SEQ_LEN)

        self.last_label = None
        self.last_conf = 0.0

        self.pushup_reps = 0
        self.squat_reps = 0
        self.situp_reps = 0
        self.jj_reps = 0
        self.pullup_reps = 0

        self.pushup_state = "up"
        self.squat_state = "up"
        self.situp_state = "down"
        self.jj_state = "closed"
        self.pullup_state = "down"

        
        self.MIN_CONF_FOR_REP = 0.6

        self.PUSHUP_UP_THR = 160.0
        self.PUSHUP_DOWN_THR = 100.0

        self.SQUAT_UP_THR = 165.0
        self.SQUAT_DOWN_THR = 130.0

        self.SITUP_DOWN_THR = 150.0
        self.SITUP_UP_THR = 110.0

        
        self.JJ_CLOSED_THR = 1.0
        self.JJ_OPEN_THR = 1.7

        
        self.PU_DOWN_THR = 1.0   
        self.PU_UP_THR = 0.6     

        self._build_ui()

 

    def _build_ui(self):
        
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

     
        video_frame = ttk.Frame(main_frame)
        video_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))

        self.video_label = ttk.Label(video_frame)
        self.video_label.pack()

       
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=0, column=1, sticky="nsew")

       
        box1 = ttk.LabelFrame(info_frame, text="Vežba")
        box1.pack(fill="x", pady=5)

        self.exercise_name_var = tk.StringVar(value="(još nema)")
        self.exercise_conf_var = tk.StringVar(value="-")

        row1 = ttk.Frame(box1)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="Tip vežbe:").pack(side="left")
        ttk.Label(row1, textvariable=self.exercise_name_var, foreground="blue").pack(side="left", padx=5)

        row2 = ttk.Frame(box1)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text="Verovatnoća:").pack(side="left")
        ttk.Label(row2, textvariable=self.exercise_conf_var, foreground="green").pack(side="left", padx=5)

        box2 = ttk.LabelFrame(info_frame, text="Ponavljanja (gornji deo)")
        box2.pack(fill="x", pady=5)

        self.pushup_reps_var = tk.StringVar(value="0")
        self.pullup_reps_var = tk.StringVar(value="0")
        self.situp_reps_var = tk.StringVar(value="0")

        rowp = ttk.Frame(box2)
        rowp.pack(fill="x", pady=2)
        ttk.Label(rowp, text="push_up:").pack(side="left")
        ttk.Label(rowp, textvariable=self.pushup_reps_var, foreground="purple").pack(side="left", padx=5)

        rowpu = ttk.Frame(box2)
        rowpu.pack(fill="x", pady=2)
        ttk.Label(rowpu, text="pull_up:").pack(side="left")
        ttk.Label(rowpu, textvariable=self.pullup_reps_var, foreground="purple").pack(side="left", padx=5)

        rowsit = ttk.Frame(box2)
        rowsit.pack(fill="x", pady=2)
        ttk.Label(rowsit, text="situp:").pack(side="left")
        ttk.Label(rowsit, textvariable=self.situp_reps_var, foreground="purple").pack(side="left", padx=5)

        box3 = ttk.LabelFrame(info_frame, text="Ponavljanja (donji deo)")
        box3.pack(fill="x", pady=5)

        self.squat_reps_var = tk.StringVar(value="0")
        self.jj_reps_var = tk.StringVar(value="0")

        rowsq = ttk.Frame(box3)
        rowsq.pack(fill="x", pady=2)
        ttk.Label(rowsq, text="squat:").pack(side="left")
        ttk.Label(rowsq, textvariable=self.squat_reps_var, foreground="purple").pack(side="left", padx=5)

        rowjj = ttk.Frame(box3)
        rowjj.pack(fill="x", pady=2)
        ttk.Label(rowjj, text="jumping_jack:").pack(side="left")
        ttk.Label(rowjj, textvariable=self.jj_reps_var, foreground="purple").pack(side="left", padx=5)

   
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=1, column=1, sticky="ew", pady=(10, 0))

        btn_cam = ttk.Button(btn_frame, text="Kamera", command=self.start_camera)
        btn_cam.pack(side="left", padx=5)

        btn_video = ttk.Button(btn_frame, text="Video fajl", command=self.start_video_file)
        btn_video.pack(side="left", padx=5)

        btn_stop = ttk.Button(btn_frame, text="Zaustavi", command=self.stop)
        btn_stop.pack(side="left", padx=5)

        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)


    def reset_state(self):
        self.feat_buffer.clear()
        self.last_label = None
        self.last_conf = 0.0

        self.pushup_reps = 0
        self.squat_reps = 0
        self.situp_reps = 0
        self.jj_reps = 0
        self.pullup_reps = 0

        self.pushup_state = "up"
        self.squat_state = "up"
        self.situp_state = "down"
        self.jj_state = "closed"
        self.pullup_state = "down"

        self.exercise_name_var.set("(još nema)")
        self.exercise_conf_var.set("-")
        self.pushup_reps_var.set("0")
        self.squat_reps_var.set("0")
        self.situp_reps_var.set("0")
        self.jj_reps_var.set("0")
        self.pullup_reps_var.set("0")

    def start_camera(self):
        print("[GUI] Kliknuto dugme Kamera")
        self.stop()

        
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("Greška", "Ne mogu da otvorim kameru (0).")
            print("[GUI] Ne mogu da otvorim kameru (0)")
            self.cap = None
            return

        print("[GUI] Kamera uspešno otvorena")
        self.source_type = "camera"
        self.reset_state()
        self.running = True
        self.update_frame()

    def start_video_file(self):
        self.stop()
        path = filedialog.askopenfilename(
            title="Izaberi video fajl",
            filetypes=[("Video fajlovi", "*.mp4 *.avi *.mov *.mkv"), ("Svi fajlovi", "*.*")]
        )
        if not path:
            return
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Greška", f"Ne mogu da otvorim video:\n{path}")
            self.cap = None
            return
        self.source_type = "video"
        self.reset_state()
        self.running = True
        self.update_frame()

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    
    def update_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            print("[GUI] Nema frejma sa izvora")
            if self.source_type == "video":
                self.stop()
            else:
                self.root.after(200, self.update_frame)
            return

        
        try:
            results = yolo_model(frame, verbose=False)
            if len(results) > 0 and len(results[0].keypoints) > 0:
                kpts_xy = results[0].keypoints.xy.cpu().numpy()  
                person_kpts = kpts_xy[0]

                kpts_full = np.concatenate(
                    [person_kpts, np.ones((person_kpts.shape[0], 1), dtype=np.float32)],
                    axis=1
                )

                feat = extract_features_from_yolo_keypoints(kpts_full)
                self.feat_buffer.append(feat)

                
                label, conf = predict_from_buffer(self.feat_buffer)
                if label is not None:
                    self.last_label = label
                    self.last_conf = conf

                
                if self.last_conf >= self.MIN_CONF_FOR_REP:
                    
                    if self.last_label == "push_up":
                        ang = compute_pushup_angle(person_kpts)
                        if self.pushup_state == "up" and ang < self.PUSHUP_DOWN_THR:
                            self.pushup_state = "down"
                        elif self.pushup_state == "down" and ang > self.PUSHUP_UP_THR:
                            self.pushup_state = "up"
                            self.pushup_reps += 1

                   
                    if self.last_label == "squat":
                        ang = compute_squat_angle(person_kpts)
                        if self.squat_state == "up" and ang < self.SQUAT_DOWN_THR:
                            self.squat_state = "down"
                        elif self.squat_state == "down" and ang > self.SQUAT_UP_THR:
                            self.squat_state = "up"
                            self.squat_reps += 1

                    if self.last_label == "situp":
                        ang = compute_situp_angle(person_kpts)
                        if self.situp_state == "down" and ang < self.SITUP_UP_THR:
                            self.situp_state = "up"
                        elif self.situp_state == "up" and ang > self.SITUP_DOWN_THR:
                            self.situp_state = "down"
                            self.situp_reps += 1

                    
                    if self.last_label == "jumping_jack":
                        m = compute_jj_metric(person_kpts)
                        if self.jj_state == "closed" and m > self.JJ_OPEN_THR:
                            self.jj_state = "open"
                        elif self.jj_state == "open" and m < self.JJ_CLOSED_THR:
                            self.jj_state = "closed"
                            self.jj_reps += 1

                    
                    if self.last_label == "pull_up":
                        m = compute_pullup_metric(person_kpts)
                        if self.pullup_state == "down" and m < self.PU_UP_THR:
                            self.pullup_state = "up"
                        elif self.pullup_state == "up" and m > self.PU_DOWN_THR:
                            self.pullup_state = "down"
                            self.pullup_reps += 1

                annotated = results[0].plot()
            else:
                annotated = frame
        except Exception as e:
            print("[GUI] Greška u YOLO/feature delu:", e)
            annotated = frame

        
        if self.last_label is not None and self.last_conf >= 0.4:
            self.exercise_name_var.set(self.last_label)
            self.exercise_conf_var.set(f"{self.last_conf*100:.1f}%")
        else:
            self.exercise_name_var.set("(prepoznajem...)")
            self.exercise_conf_var.set("-")

        self.pushup_reps_var.set(str(self.pushup_reps))
        self.squat_reps_var.set(str(self.squat_reps))
        self.situp_reps_var.set(str(self.situp_reps))
        self.jj_reps_var.set(str(self.jj_reps))
        self.pullup_reps_var.set(str(self.pullup_reps))

        
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = img.resize((640, 480))
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)




if __name__ == "__main__":
    root = tk.Tk()
    app = ExerciseApp(root)
    root.mainloop()
