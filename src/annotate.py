import os
import cv2  # Tohle je dulezite pro video!

# CONFIG
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

VIDEO_DIR = os.path.join(project_root, "data", "raw_videos")
OUTPUT_DIR = os.path.join(project_root, "data", "labels")

# FAZE NAZVY
PHASES_NAME = {
    0: "KLID (0)",
    1: "PRIPRAVA (1)",
    2: "ROZDEJCHANI (2)",
    3: "INHALACE (3)",
    4: "ZADRZENI (4)"
}

def annotate_videos():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for root, dirs, files in os.walk(VIDEO_DIR):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov")):
                video_path = os.path.join(root, file)

                # vytvoreni cesty pro label
                rel_path = os.path.relpath(root, VIDEO_DIR)
                target_dir = os.path.join(OUTPUT_DIR, rel_path)
                os.makedirs(target_dir, exist_ok=True)

                label_file = os.path.join(target_dir, os.path.splitext(file)[0] + ".txt")

                if os.path.exists(label_file):
                    print(f"preskakuji: {file} (hotovo)")
                    continue

                print(f"\n>>> START ANOTACE: {file}")
                cap = cv2.VideoCapture(video_path)
                labels = []
                current_label = 0  # inicializace pred smyckou
                
                # vytvoreni okna s moznosti zmeny velikosti
                window_name = "ANOTACE NEXTHALER (Drz cislo 1-4, ESC = konec)"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 1280, 720)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break

                    # snimani klavesy (ceka 33ms = plynule video)
                    key = cv2.waitKey(33) & 0xFF

                    if key == ord('1'):
                        current_label = 1
                    elif key == ord('2'):
                        current_label = 2
                    elif key == ord('3'):
                        current_label = 3
                    elif key == ord('4'):
                        current_label = 4
                    elif key == 27: # ESC pro zrušení
                        print("Anotace prerusena uzivatelem.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    else:
                        current_label = 0

                    # vizualni napoveda
                    text = f"Nahravam: {PHASES_NAME[current_label]}"
                    color = (0, 255, 0) if current_label > 0 else (200, 200, 200)
                    cv2.putText(frame, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    cv2.imshow(window_name, frame)
                    labels.append(current_label)

                cap.release()
                cv2.destroyAllWindows()

                # ulozeni do .txt
                with open(label_file, "w") as f:
                    for label in labels:
                        f.write(f"{label}\n") # Opraveno z 'l' na 'label'
                print(f"--- OK: Soubor ulozen do {label_file}")

if __name__ == "__main__":
    annotate_videos()