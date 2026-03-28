import argparse
import cv2
import mediapipe as mp
import numpy as np
import os
import sys
from pathlib import Path

try:
    from utils.paths import project_paths
except ModuleNotFoundError:
    # Allow running this file directly: py src/preprocessing/extract_features_enhanced.py
    src_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(src_root))
    from utils.paths import project_paths

# --- KONFIGURACE ---
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,        
    model_complexity=2,           
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def calculate_angle(a, b, c):
    """
    Vypočítá úhel v bodě b mezi body a-b-c
    Vrací úhel ve stupních (0-180)
    """
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_distance(a, b):
    """Euklidovská vzdálenost mezi dvěma body"""
    return np.linalg.norm(a - b)


def _interpolate_nan_1d(values):
    """Lineárně doplní chybějící hodnoty (NaN) v 1D vektoru."""
    values = values.astype(np.float32, copy=True)
    idx = np.arange(values.shape[0])
    valid = ~np.isnan(values)

    if not np.any(valid):
        return np.zeros_like(values)
    if np.any(~valid):
        values[~valid] = np.interp(idx[~valid], idx[valid], values[valid])
    return values


def _interpolate_nan_nd(data):
    """Aplikuje interpolaci NaN po časové ose pro každý kanál zvlášť."""
    filled = data.astype(np.float32, copy=True)
    t = filled.shape[0]
    reshaped = filled.reshape(t, -1)
    for c in range(reshaped.shape[1]):
        reshaped[:, c] = _interpolate_nan_1d(reshaped[:, c])
    return reshaped.reshape(filled.shape)


def _smooth_over_time(data, max_window=11, polyorder=2):
    """Volitelné vyhlazení po čase; zachová tvar dat (bez SciPy)."""
    t = data.shape[0]
    if t < 5:
        return data

    window = min(max_window, t)
    if window % 2 == 0:
        window -= 1
    if window < 5:
        return data

    flat = data.reshape(t, -1)
    smoothed = np.empty_like(flat)
    pad = window // 2

    for c in range(flat.shape[1]):
        channel = flat[:, c]
        padded = np.pad(channel, (pad, pad), mode="edge")
        smoothed[:, c] = np.convolve(padded, np.ones(window, dtype=np.float32) / window, mode="valid")

    return smoothed.reshape(data.shape)


def _extract_frame_landmarks(results):
    """Vrátí landmarky jednoho snímku + příznaky detekce."""
    pose = np.zeros((23, 4), dtype=np.float32)
    left_hand = np.full((21, 3), np.nan, dtype=np.float32)
    right_hand = np.full((21, 3), np.nan, dtype=np.float32)

    face_detected = bool(results.face_landmarks)
    left_detected = bool(results.left_hand_landmarks)
    right_detected = bool(results.right_hand_landmarks)

    if results.pose_landmarks:
        for i, l in enumerate(results.pose_landmarks.landmark[:23]):
            pose[i, 0] = l.x
            pose[i, 1] = l.y
            pose[i, 2] = l.z
            pose[i, 3] = l.visibility
    else:
        pose[:, :3] = np.nan
        pose[:, 3] = 0.0

    if left_detected:
        for i, l in enumerate(results.left_hand_landmarks.landmark):
            left_hand[i, 0] = l.x
            left_hand[i, 1] = l.y
            left_hand[i, 2] = l.z

    if right_detected:
        for i, l in enumerate(results.right_hand_landmarks.landmark):
            right_hand[i, 0] = l.x
            right_hand[i, 1] = l.y
            right_hand[i, 2] = l.z

    mouth_distance = np.nan
    if face_detected:
        upper_lip = np.array(
            [
                results.face_landmarks.landmark[13].x,
                results.face_landmarks.landmark[13].y,
                results.face_landmarks.landmark[13].z,
            ],
            dtype=np.float32,
        )
        lower_lip = np.array(
            [
                results.face_landmarks.landmark[14].x,
                results.face_landmarks.landmark[14].y,
                results.face_landmarks.landmark[14].z,
            ],
            dtype=np.float32,
        )
        mouth_distance = calculate_distance(upper_lip, lower_lip)

    return pose, left_hand, right_hand, mouth_distance, left_detected, right_detected, face_detected


def _build_features_from_clean_landmarks(
    pose_xyz,
    pose_vis,
    left_hand_xyz,
    right_hand_xyz,
    mouth_distance,
    left_detected,
    right_detected,
):
    """Sestaví 243D feature vektor ze stabilizovaných landmarků."""
    distances = []
    distances.append(calculate_distance(pose_xyz[15], pose_xyz[10]))
    distances.append(calculate_distance(pose_xyz[16], pose_xyz[10]))
    distances.append(calculate_distance(pose_xyz[15], pose_xyz[0]))
    distances.append(calculate_distance(pose_xyz[16], pose_xyz[0]))
    distances.append(calculate_distance(pose_xyz[11], pose_xyz[12]))
    distances.append(pose_xyz[0, 1] - pose_xyz[11, 1])
    distances.append(pose_xyz[0, 1] - pose_xyz[12, 1])
    distances.append(calculate_distance(pose_xyz[11], pose_xyz[13]))
    distances.append(calculate_distance(pose_xyz[12], pose_xyz[14]))
    distances.append(calculate_distance(pose_xyz[13], pose_xyz[15]))
    distances.append(mouth_distance)

    angles = []
    angles.append(calculate_angle(pose_xyz[11], pose_xyz[13], pose_xyz[15]))
    angles.append(calculate_angle(pose_xyz[12], pose_xyz[14], pose_xyz[16]))

    torso_center = (pose_xyz[11] + pose_xyz[12]) / 2
    angles.append(calculate_angle(torso_center, pose_xyz[11], pose_xyz[13]))
    angles.append(calculate_angle(torso_center, pose_xyz[12], pose_xyz[14]))

    if pose_vis[10] > 0.5 and pose_vis[0] > 0.5:
        neck_point = (pose_xyz[11] + pose_xyz[12]) / 2
        angles.append(calculate_angle(neck_point, pose_xyz[0], pose_xyz[10]))
    else:
        angles.append(90.0)

    wrist_to_shoulder_l = pose_xyz[15] - pose_xyz[11]
    wrist_angle_l = np.degrees(np.arctan2(wrist_to_shoulder_l[1], wrist_to_shoulder_l[0]))
    angles.append(wrist_angle_l)

    wrist_to_shoulder_r = pose_xyz[16] - pose_xyz[12]
    wrist_angle_r = np.degrees(np.arctan2(wrist_to_shoulder_r[1], wrist_to_shoulder_r[0]))
    angles.append(wrist_angle_r)

    angles.append(calculate_angle(pose_xyz[11], pose_xyz[12], pose_xyz[14]))

    hand_config = []
    if left_detected:
        finger_tips_l = [4, 8, 12, 16, 20]
        avg_finger_dist_l = np.mean([calculate_distance(left_hand_xyz[0], left_hand_xyz[i]) for i in finger_tips_l])
        hand_config.append(avg_finger_dist_l)
        hand_config.append(calculate_distance(left_hand_xyz[4], left_hand_xyz[8]))
        hand_config.append(calculate_distance(left_hand_xyz[4], left_hand_xyz[12]))
    else:
        hand_config.extend([0.0, 0.0, 0.0])

    if right_detected:
        finger_tips_r = [4, 8, 12, 16, 20]
        avg_finger_dist_r = np.mean([calculate_distance(right_hand_xyz[0], right_hand_xyz[i]) for i in finger_tips_r])
        hand_config.append(avg_finger_dist_r)
        hand_config.append(calculate_distance(right_hand_xyz[4], right_hand_xyz[8]))
        hand_config.append(calculate_distance(right_hand_xyz[4], right_hand_xyz[12]))
    else:
        hand_config.extend([0.0, 0.0, 0.0])

    pose_4d = np.concatenate([pose_xyz, pose_vis[:, None]], axis=1)
    combined = np.concatenate(
        [
            pose_4d.flatten(),
            left_hand_xyz.flatten(),
            right_hand_xyz.flatten(),
            np.array(distances, dtype=np.float32),
            np.array(angles, dtype=np.float32),
            np.array(hand_config, dtype=np.float32),
        ]
    )
    return combined.astype(np.float32)

def extract_enhanced_features(results):
    """
    Zachovaná logika 243D features pro jeden snímek.
    Tato funkce se hodí pro rychlý jednotkový test/preview.
    """
    pose, left_hand, right_hand, mouth, left_ok, right_ok, _ = _extract_frame_landmarks(results)

    pose_xyz = _interpolate_nan_nd(pose[:, :3][None, ...])[0]
    left_xyz = _interpolate_nan_nd(left_hand[None, ...])[0]
    right_xyz = _interpolate_nan_nd(right_hand[None, ...])[0]
    mouth_val = _interpolate_nan_1d(np.array([mouth], dtype=np.float32))[0]

    return _build_features_from_clean_landmarks(
        pose_xyz=pose_xyz,
        pose_vis=pose[:, 3],
        left_hand_xyz=left_xyz,
        right_hand_xyz=right_xyz,
        mouth_distance=mouth_val,
        left_detected=left_ok,
        right_detected=right_ok,
    )

def extract(input_root, output_root, overwrite=False):
    """
    Extrahuje ROZŠÍŘENÉ features pro temporal action segmentation
    """
    if not os.path.exists(input_root):
        print(f"CHYBA: Slozka '{input_root}' nebyla nalezena!")
        return

    print(f"Startuji ROZSIŘENOU extrakci features z: {input_root}")
    print(f"Features: 243 hodnot (raw 218 + distances 11 + angles 8 + hand config 6)")

    for root, dirs, files in os.walk(input_root):
        for video_name in files:
            if video_name.lower().endswith((".mp4", ".avi", ".mov")):
                video_path = os.path.join(root, video_name)
                rel_path = os.path.relpath(root, input_root)
                target_folder = os.path.join(output_root, rel_path)
                os.makedirs(target_folder, exist_ok=True)
                
                output_path = os.path.join(target_folder, os.path.splitext(video_name)[0] + ".npy")

                if os.path.exists(output_path) and not overwrite:
                    print(f"Preskakuji (hotovo): {video_name}")
                    continue

                print(f"Zpracovavam: {rel_path}/{video_name}")
                cap = cv2.VideoCapture(video_path)
                pose_seq = []
                left_hand_seq = []
                right_hand_seq = []
                mouth_seq = []
                left_ok_seq = []
                right_ok_seq = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(rgb_frame)

                    pose, left_hand, right_hand, mouth, left_ok, right_ok, _ = _extract_frame_landmarks(results)
                    pose_seq.append(pose)
                    left_hand_seq.append(left_hand)
                    right_hand_seq.append(right_hand)
                    mouth_seq.append(mouth)
                    left_ok_seq.append(left_ok)
                    right_ok_seq.append(right_ok)
                
                cap.release()
                
                # --- ČIŠTĚNÍ DAT ---
                if len(pose_seq) == 0:
                    print(f"  ⚠️ Prazdne video, preskakuji")
                    continue

                pose_seq = np.asarray(pose_seq, dtype=np.float32)
                left_hand_seq = np.asarray(left_hand_seq, dtype=np.float32)
                right_hand_seq = np.asarray(right_hand_seq, dtype=np.float32)
                mouth_seq = np.asarray(mouth_seq, dtype=np.float32)
                left_ok_seq = np.asarray(left_ok_seq, dtype=bool)
                right_ok_seq = np.asarray(right_ok_seq, dtype=bool)

                pose_xyz = _interpolate_nan_nd(pose_seq[:, :, :3])
                left_hand_xyz = _interpolate_nan_nd(left_hand_seq)
                right_hand_xyz = _interpolate_nan_nd(right_hand_seq)
                mouth_seq = _interpolate_nan_1d(mouth_seq)

                pose_xyz = _smooth_over_time(pose_xyz)
                left_hand_xyz = _smooth_over_time(left_hand_xyz)
                right_hand_xyz = _smooth_over_time(right_hand_xyz)
                mouth_seq = _smooth_over_time(mouth_seq[:, None])[:, 0]

                final_data = []
                for t in range(pose_seq.shape[0]):
                    final_data.append(
                        _build_features_from_clean_landmarks(
                            pose_xyz=pose_xyz[t],
                            pose_vis=pose_seq[t, :, 3],
                            left_hand_xyz=left_hand_xyz[t],
                            right_hand_xyz=right_hand_xyz[t],
                            mouth_distance=mouth_seq[t],
                            left_detected=left_ok_seq[t],
                            right_detected=right_ok_seq[t],
                        )
                    )

                final_data = np.asarray(final_data, dtype=np.float32)
                
                np.save(output_path, final_data)
                print(f"  ✓ Ulozeno: {final_data.shape[0]} snimku × {final_data.shape[1]} features")

    print("\n=== HOTOVO ===")
    print("Extrahované features obsahují:")
    print("  • Raw coordinates (218): základní pozice bodů")
    print("  • Distances (11): klíčové vzdálenosti (ruka-ústa, mouth distance, etc.)")
    print("  • Angles (8): úhly kloubů (loket, rameno, krk)")
    print("  • Hand config (6): konfigurace prstů (sevření, rozevření)")
    print("  CELKEM: 243 features na snímek")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrakce 243D features z videi pro inhalacni segmentaci")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Prepise jiz existujici .npy soubory (bez nutnosti mazat slozky).",
    )
    args = parser.parse_args()

    paths = project_paths(__file__)
    input_dir = str(paths["raw_videos"])
    output_dir = str(paths["features_enhanced"])
    
    extract(input_dir, output_dir, overwrite=args.overwrite)
