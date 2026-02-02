import cv2
import mediapipe as mp
import numpy as mp
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

def extract_landmarks(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    
    # prochaze souboru v raw
    for video_name in os.listdir(input_folder):
        if video_name.endswith((".mp4", ".avi")):
            video_path = os.path.join(input_folder, video_name)
            output_path = os.path.join(output_folder, video_name.replace(".mp4", ".npy").replace(".avi", ".npy"))

            print(f"zpracovavam video: {video_name}")

            cap = cv2.VideoCapture(video_path)
            video_data = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # opencv cte BGR, mediapie potrebuje RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                # pokud mediapipe uspesne detekuje postavu, ukladame body
                if results.pose_landmarks:
                    frame_landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        # ukladame x, y, z a visibity
                        frame_landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                    video_data.append(frame_landmarks)
                else:
                    video_data.append([0,0] * 132)

            cap.release()

            np.save(output_path, np.array(video_data))
            print(f"Ulozeno: {output_path}")

