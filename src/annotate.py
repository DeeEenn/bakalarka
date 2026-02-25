import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.python.solutions import pose as mp_pose

# Inicializace MediaPipe Pose
pose = mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=1, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_landmarks(input_root, output_root):
    if not os.path.exists(input_root):
        print(f"CHYBA: Slozka '{input_root}' nebyla nalezena!")
        return

    print(f"Startuji extrakci z: {os.path.abspath(input_root)}")

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
                    results = pose.process(rgb_frame)

                    if results.pose_landmarks:
                        frame_landmarks = []
                        for landmark in results.pose_landmarks.landmark:
                            # x, y, z a visibility (33 bodů * 4 = 132 hodnot)
                            frame_landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                        video_data.append(frame_landmarks)
                    else:
                        # Pokud postavu nenajde, vložíme nuly, aby seděla délka dat s anotacemi (labels)
                        video_data.append([0.0] * 132)

                cap.release()
                
                # Uložení proběhne až po zpracování CELÉHO videa
                if len(video_data) > 0:
                    np.save(output_path, np.array(video_data))
                    print(f"--- OK: Ulozeno {len(video_data)} snimku.")

    # Zavření MediaPipe instance
    pose.close()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_dir = os.path.join(project_root, "data", "raw_videos")
    output_dir = os.path.join(project_root, "data", "features")
    
    extract_landmarks(input_dir, output_dir)
    print("\nHOTOVO! Vsechna videa byla transformovana na souradnice.")