import cv2
import numpy as np
import json
import pickle
from collections import deque

from ultralytics import YOLO
import tensorflow as tf


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


def angle_3pts(a, b, c):
    """
    a, b, c: (x, y, z)
    ugao u tački b u stepenima
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b


    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return 0.0

    cos_ang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos_ang))
    return float(ang)


def dist_3d(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return float(np.linalg.norm(p1 - p2))


def extract_features_from_yolo_keypoints(kpts):
    """
    kpts: np.array shape (17, 3) ili (17, 2) iz YOLO,
          [x, y, conf] po jointu.
    vraća: feature vektor dužine NUM_FEATURES
           (mora da bude isto što i u treningu)
    """

    
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

    angles.append(angle_3pts(
        lm["right_elbow"], lm["right_shoulder"], lm["right_hip"]
    ))

    angles.append(angle_3pts(
        lm["left_elbow"], lm["left_shoulder"], lm["left_hip"]
    ))

    angles.append(angle_3pts(
        lm["right_hip"], lm["right_knee"], lm["right_ankle"]
    ))

    angles.append(angle_3pts(
        lm["left_hip"], lm["left_knee"], lm["left_ankle"]
    ))

    angles.append(angle_3pts(
        lm["right_wrist"], lm["right_elbow"], lm["right_shoulder"]
    ))

    angles.append(angle_3pts(
        lm["left_wrist"], lm["left_elbow"], lm["left_shoulder"]
    ))

    dists = []
    dists.append(dist_3d(lm["left_shoulder"], lm["left_wrist"]))
    dists.append(dist_3d(lm["right_shoulder"], lm["right_wrist"]))
    dists.append(dist_3d(lm["left_hip"], lm["left_ankle"]))
    dists.append(dist_3d(lm["right_hip"], lm["right_ankle"]))
    dists.append(dist_3d(lm["left_hip"], lm["left_wrist"]))
    dists.append(dist_3d(lm["right_hip"], lm["right_wrist"]))
    dists.append(dist_3d(lm["left_shoulder"], lm["left_ankle"]))
    dists.append(dist_3d(lm["right_shoulder"], lm["right_ankle"]))

    feat = np.array(coords + angles + dists, dtype=np.float32)


    if feat.shape[0] < NUM_FEATURES:
        pad = np.zeros(NUM_FEATURES - feat.shape[0], dtype=np.float32)
        feat = np.concatenate([feat, pad])
    elif feat.shape[0] > NUM_FEATURES:
        feat = feat[:NUM_FEATURES]

    return feat


def predict_from_buffer(buffer):
    """
    buffer: deque sa poslednjih SEQ_LEN feature vektora
    """
    if len(buffer) < SEQ_LEN:
        return None, 0.0

    seq = np.array(buffer, dtype=np.float32)  
    seq = np.expand_dims(seq, axis=0)         

    probs = model.predict(seq, verbose=0)[0]
    idx = int(np.argmax(probs))
    class_name = le.inverse_transform([idx])[0]
    conf = float(probs[idx])
    return class_name, conf



cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ne mogu da otvorim kameru (0). Proveri uređaj.")
    exit(1)

print("Pritisni 'q' za izlaz.")

feat_buffer = deque(maxlen=SEQ_LEN)
last_label = None
last_conf = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Nema frejma sa kamere.")
        break


    results = yolo_model(frame, verbose=False)
    if len(results) > 0 and len(results[0].keypoints) > 0:
        kpts_xyc = results[0].keypoints.xy.cpu().numpy()
      
        person_kpts = kpts_xyc[0] 
        kpts_full = np.concatenate(
            [person_kpts, np.ones((person_kpts.shape[0], 1), dtype=np.float32)],
            axis=1
        )  

        feat = extract_features_from_yolo_keypoints(kpts_full)
        feat_buffer.append(feat)

        
        label, conf = predict_from_buffer(feat_buffer)
        if label is not None:
            last_label = label
            last_conf = conf

        
        annotated = results[0].plot()
    else:
        annotated = frame

    if last_label is not None:
        txt = f"{last_label} ({last_conf*100:.1f}%)"
    else:
        txt = "Prepoznajem..."

    cv2.putText(
        annotated,
        txt,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.imshow("Exercise recognition", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


