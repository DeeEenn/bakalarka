import cv2
import mediapipe as mp
import numpy as np
import os

# Použijeme cestu, která nám v testu zafungovala
from mediapipe.python.solutions import pose as mp_pose

# Inicializace MediaPipe Pose
pose = mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=1, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_landmarks(input_root, output_root):
    # Kontrola existence vstupní složky (vzhledem k src/)
    if not os.path.exists(input_root):
        print(f"CHYBA: Slozka '{input_root}' nebyla nalezena!")
        return

    print(f"Startuji extrakci z: {os.path.abspath(input_root)}")

    for root, dirs, files in os.walk(input_root):
        for video_name in files:
            if video_name.lower().endswith((".mp4", ".avi", ".mov")):
                video_path = os.path.join(root, video_name)
                
                # Zachování struktury složek (01spravne, 08spatne atd.)
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
                    results = pose.process(rgb_frame)

                    if results.pose_landmarks:
                        frame_landmarks = []
                        for landmark in results.pose_landmarks.landmark:
                            # x, y, z a visibility (33 bodů * 4 = 132 hodnot)
                            frame_landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                        video_data.append(frame_landmarks)
                    else:
                        # Pokud postavu nenajde, dáme pole nul (důležité pro časovou osu)Ne 
                np.save(output_path, np.array(video_data))

if __name__ == "__main__":
    # Získáme absolutní cestu ke skriptu a od ní odvozujeme cesty k datům
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_dir = os.path.join(project_root, "data", "raw_videos")
    output_dir = os.path.join(project_root, "data", "features")
    
    extract_landmarks(input_dir, output_dir)
    print("HOTOVO! Vsechna videa byla transformovana na souradnice.")