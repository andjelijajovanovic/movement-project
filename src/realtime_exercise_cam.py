import cv2
import numpy as np
import tensorflow as tf
import pickle
import json
from collections import deque
from ultralytics import YOLO


model = tf.keras.models.load_model("exercise_lstm_angles.keras")

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open("feature_cols.json", "r") as f:
    feature_cols = json.load(f)

with open("config.json", "r") as f:
    config = json.load(f)

max_len = config["max_len"]



yolo_pose = YOLO("yolov8n-pose.pt")   


yolo_kpts = [
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



def compute_angle(a, b, c):
    """Ugao u tački b (a-b-c) u stepenima."""
    a = np.array(a); b = np.array(b); c = np.array(c)
    v1 = a - b
    v2 = c - b
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cosang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def extract_features_from_yolo_keypoints(kpts):
    """
    kpts: np.array oblika (17, 3) -> [x, y, conf]
    vraća: feature vektor u istom redosledu kao feature_cols
    """

   
    lm = {name: (kpts[i][0], kpts[i][1], 0.0) for i, name in enumerate(yolo_kpts)}

   
    left_hip = lm["left_hip"]
    right_hip = lm["right_hip"]
    mid = (
        (left_hip[0] + right_hip[0]) / 2.0,
        (left_hip[1] + right_hip[1]) / 2.0,
        0.0,
    )

    lm_norm = {}
    for name, (x, y, z) in lm.items():
        lm_norm[f"x_{name}"] = x - mid[0]
        lm_norm[f"y_{name}"] = y - mid[1]
        lm_norm[f"z_{name}"] = 0.0  

    
    def p(j):
        return lm_norm[f"x_{j}"], lm_norm[f"y_{j}"], lm_norm[f"z_{j}"]

    angles = {
        "right_elbow_right_shoulder_right_hip": compute_angle(
            p("right_elbow"), p("right_shoulder"), p("right_hip")
        ),
        "left_elbow_left_shoulder_left_hip": compute_angle(
            p("left_elbow"), p("left_shoulder"), p("left_hip")
        ),
        "right_hip_right_knee_right_ankle": compute_angle(
            p("right_hip"), p("right_knee"), p("right_ankle")
        ),
        "left_hip_left_knee_left_ankle": compute_angle(
            p("left_hip"), p("left_knee"), p("left_ankle")
        ),
        "right_wrist_right_elbow_right_shoulder": compute_angle(
            p("right_wrist"), p("right_elbow"), p("right_shoulder")
        ),
        "left_wrist_left_elbow_left_shoulder": compute_angle(
            p("left_wrist"), p("left_elbow"), p("left_shoulder")
        ),
    }

    
    feat = []
    for col in feature_cols:
        if col in lm_norm:
            feat.append(lm_norm[col])
        elif col in angles:
            feat.append(angles[col])
        else:
            
            feat.append(0.0)

    return np.array(feat, dtype=np.float32)


def predict_from_buffer(buffer):
    """buffer: lista feature vektora [n_frames, num_features]"""
    seq = np.array(buffer, dtype=np.float32)

    seq_padded = tf.keras.preprocessing.sequence.pad_sequences(
        [seq],
        maxlen=max_len,
        padding="post",
        truncating="post",
        dtype="float32",
        value=0.0,
    )

    probs = model.predict(seq_padded, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = le.inverse_transform([idx])[0]
    conf = float(probs[idx])
    return label, conf



cap = cv2.VideoCapture(0)
buffer = deque(maxlen=max_len)
min_frames_for_prediction = 20

if not cap.isOpened():
    print("Ne mogu da otvorim kameru (0).")
    raise SystemExit

print("Pritisni 'q' za izlaz.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Nema frejma sa kamere.")
        break

    results = yolo_pose(frame)[0]

  
    if results.keypoints is not None and len(results.keypoints) > 0:
       
        kpts = results.keypoints.data[0].cpu().numpy()  # (17, 3)

        
        for x, y, conf in kpts:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

        feat_vec = extract_features_from_yolo_keypoints(kpts)
        buffer.append(feat_vec)

        if len(buffer) >= min_frames_for_prediction:
            label, conf = predict_from_buffer(buffer)
            cv2.putText(
                frame,
                f"{label} ({conf*100:.1f}%)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
    else:
        cv2.putText(
            frame,
            "Nema osobe",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    cv2.imshow("Exercise recognition (YOLO Pose + LSTM)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
