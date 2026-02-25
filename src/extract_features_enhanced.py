import cv2
import mediapipe as mp
import numpy as np
import os
from scipy.signal import savgol_filter

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

def extract_enhanced_features(results):
    """
    Extrahuje ROZŠÍŘENÉ features optimalizované pro detekci jemných fází
    
    Features:
    1. Raw coordinates (218 hodnot) - jako předtím
    2. Key distances (11 hodnot) - důležité vzdálenosti (včetně mouth distance)
    3. Joint angles (8 hodnot) - úhly kloubů
    4. Hand configuration (6 hodnot) - konfigurace prstů
    
    CELKEM: 243 features
    """
    
    # === 1. RAW COORDINATES (218 hodnot) ===
    if results.pose_landmarks:
        pose = [[l.x, l.y, l.z, l.visibility] for i, l in enumerate(results.pose_landmarks.landmark) if i < 23]
    else:
        pose = [[0.0] * 4] * 23
    
    lh = [[l.x, l.y, l.z] for l in results.left_hand_landmarks.landmark] if results.left_hand_landmarks else [[0.0]*3]*21
    rh = [[l.x, l.y, l.z] for l in results.right_hand_landmarks.landmark] if results.right_hand_landmarks else [[0.0]*3]*21
    
    pose_arr = np.array(pose)
    lh_arr = np.array(lh)
    rh_arr = np.array(rh)
    
    # === 2. KEY DISTANCES (11 hodnot) ===
    # Kritické vzdálenosti pro detekci fází
    distances = []
    
    # Ruce k obličeji
    distances.append(calculate_distance(pose_arr[15, :3], pose_arr[10, :3]))  # Levé zápěstí - ústa
    distances.append(calculate_distance(pose_arr[16, :3], pose_arr[10, :3]))  # Pravé zápěstí - ústa
    distances.append(calculate_distance(pose_arr[15, :3], pose_arr[0, :3]))   # Levé zápěstí - nos
    distances.append(calculate_distance(pose_arr[16, :3], pose_arr[0, :3]))   # Pravé zápěstí - nos
    
    # Šířka ramen (pro detekci dýchání)
    distances.append(calculate_distance(pose_arr[11, :3], pose_arr[12, :3]))  # Šířka ramen
    
    # Výška těla (vertikální změny)
    distances.append(pose_arr[0, 1] - pose_arr[11, 1])   # Nos - levé rameno (Y)
    distances.append(pose_arr[0, 1] - pose_arr[12, 1])   # Nos - pravé rameno (Y)
    
    # Paže - aktivita
    distances.append(calculate_distance(pose_arr[11, :3], pose_arr[13, :3]))  # Levé rameno - loket
    distances.append(calculate_distance(pose_arr[12, :3], pose_arr[14, :3]))  # Pravé rameno - loket
    distances.append(calculate_distance(pose_arr[13, :3], pose_arr[15, :3]))  # Levý loket - zápěstí
    
    # Mouth distance (vzdálenost mezi rty) - indikátor nádechu
    if results.face_landmarks:
        # MediaPipe Face Mesh: bod 13 = horní ret (střed), bod 14 = dolní ret (střed)
        upper_lip = np.array([results.face_landmarks.landmark[13].x,
                             results.face_landmarks.landmark[13].y,
                             results.face_landmarks.landmark[13].z])
        lower_lip = np.array([results.face_landmarks.landmark[14].x,
                             results.face_landmarks.landmark[14].y,
                             results.face_landmarks.landmark[14].z])
        mouth_distance = calculate_distance(upper_lip, lower_lip)
        distances.append(mouth_distance)
    else:
        distances.append(0.0)  # Pokud není detekován obličej
    
    # === 3. JOINT ANGLES (8 hodnot) ===
    # Úhly kloubů - důležité pro držení a manipulaci
    angles = []
    
    # Levý loket (rameno-loket-zápěstí)
    angles.append(calculate_angle(pose_arr[11, :3], pose_arr[13, :3], pose_arr[15, :3]))
    
    # Pravý loket
    angles.append(calculate_angle(pose_arr[12, :3], pose_arr[14, :3], pose_arr[16, :3]))
    
    # Levé rameno (trup-rameno-loket)
    torso_center = (pose_arr[11, :3] + pose_arr[12, :3]) / 2
    angles.append(calculate_angle(torso_center, pose_arr[11, :3], pose_arr[13, :3]))
    
    # Pravé rameno
    angles.append(calculate_angle(torso_center, pose_arr[12, :3], pose_arr[14, :3]))
    
    # Úhel hlavy (krk - nos - ústa) - pro detekci náklonu hlavy
    if pose_arr[10, 3] > 0.5 and pose_arr[0, 3] > 0.5:  # visibility check
        neck_point = (pose_arr[11, :3] + pose_arr[12, :3]) / 2
        angles.append(calculate_angle(neck_point, pose_arr[0, :3], pose_arr[10, :3]))
    else:
        angles.append(90.0)  # default
    
    # Úhel zápěstí vzhledem k horizontále (pro detekci držení)
    wrist_to_shoulder_l = pose_arr[15, :3] - pose_arr[11, :3]
    wrist_angle_l = np.degrees(np.arctan2(wrist_to_shoulder_l[1], wrist_to_shoulder_l[0]))
    angles.append(wrist_angle_l)
    
    wrist_to_shoulder_r = pose_arr[16, :3] - pose_arr[12, :3]
    wrist_angle_r = np.degrees(np.arctan2(wrist_to_shoulder_r[1], wrist_to_shoulder_r[0]))
    angles.append(wrist_angle_r)
    
    # Úhel paží vzhledem k trupu (detekce zvednutí ruky)
    angles.append(calculate_angle(pose_arr[11, :3], pose_arr[12, :3], pose_arr[14, :3]))
    
    # === 4. HAND CONFIGURATION (6 hodnot) ===
    # Konfigurace ruky - pro detekce uchopení/otevírání
    hand_config = []
    
    # Levá ruka: Rozevření prstů (průměrná vzdálenost špiček od zápěstí)
    if results.left_hand_landmarks:
        finger_tips_l = [4, 8, 12, 16, 20]  # špičky 5 prstů
        avg_finger_dist_l = np.mean([calculate_distance(lh_arr[0], lh_arr[i]) for i in finger_tips_l])
        hand_config.append(avg_finger_dist_l)
        
        # Sevření (vzdálenost palec-ukazovák)
        hand_config.append(calculate_distance(lh_arr[4], lh_arr[8]))
        
        # Otevření dlaně (palec vs prostředníček)
        hand_config.append(calculate_distance(lh_arr[4], lh_arr[12]))
    else:
        hand_config.extend([0.0, 0.0, 0.0])
    
    # Pravá ruka: stejné metriky
    if results.right_hand_landmarks:
        finger_tips_r = [4, 8, 12, 16, 20]
        avg_finger_dist_r = np.mean([calculate_distance(rh_arr[0], rh_arr[i]) for i in finger_tips_r])
        hand_config.append(avg_finger_dist_r)
        
        hand_config.append(calculate_distance(rh_arr[4], rh_arr[8]))
        hand_config.append(calculate_distance(rh_arr[4], rh_arr[12]))
    else:
        hand_config.extend([0.0, 0.0, 0.0])
    
    # === SPOJENÍ VŠECH FEATURES ===
    combined = np.concatenate([
        np.array(pose).flatten(),   # 92 hodnot
        np.array(lh).flatten(),     # 63 hodnot
        np.array(rh).flatten(),     # 63 hodnot
        np.array(distances),        # 11 hodnot (včetně mouth distance)
        np.array(angles),           # 8 hodnot
        np.array(hand_config)       # 6 hodnot
    ])
    
    return combined  # 243 features celkem

def extract(input_root, output_root):
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

                if os.path.exists(output_path):
                    print(f"Preskakuji (hotovo): {video_name}")
                    continue

                print(f"Zpracovavam: {rel_path}/{video_name}")
                cap = cv2.VideoCapture(video_path)
                video_data = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(rgb_frame)

                    # Extrakce rozšířených features
                    combined = extract_enhanced_features(results)
                    video_data.append(combined)
                
                cap.release()
                
                # --- ČIŠTĚNÍ DAT ---
                final_data = np.array(video_data)
                
                if len(final_data) == 0:
                    print(f"  ⚠️ Prazdne video, preskakuji")
                    continue

                # Interpolace nul
                for j in range(final_data.shape[1]):
                    col = final_data[:, j]
                    mask = col == 0
                    if np.any(mask) and not np.all(mask):
                        idx = np.arange(len(col))
                        col[mask] = np.interp(idx[mask], idx[~mask], col[~mask])
                
                # Vyhlazení (Savitzky-Golay)
                if len(final_data) > 11:
                    final_data = savgol_filter(final_data, 11, 2, axis=0)

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_dir = os.path.join(project_root, "data", "raw_videos")
    output_dir = os.path.join(project_root, "data", "features_enhanced")  # Nová složka!
    
    extract(input_dir, output_dir)
