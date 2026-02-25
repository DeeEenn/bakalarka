import os
import cv2

# CONFIG
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

VIDEO_DIR = os.path.join(project_root, "data", "raw_videos")
OUTPUT_DIR = os.path.join(project_root, "data", "labels")

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
                rel_path = os.path.relpath(root, VIDEO_DIR)
                target_dir = os.path.join(OUTPUT_DIR, rel_path)
                os.makedirs(target_dir, exist_ok=True)

                label_file = os.path.join(target_dir, os.path.splitext(file)[0] + ".txt")

                if os.path.exists(label_file):
                    print(f"Preskakuji: {file} (hotovo)")
                    continue

                print(f"\n>>> START ANOTACE: {file}")
                cap = cv2.VideoCapture(video_path)
                labels = []
                current_label = 0 
                paused = False
                
                window_name = "ANOTACE - Mezernik = PAUZA, 0-4 = FAZE, ESC = KONEC"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 1280, 720)

                while cap.isOpened():
                    if not paused:
                        ret, frame = cap.read()
                        if not ret: break
                    
                    # Kopie framu pro vykreslení textu (aby se text neukládal do videa, kdybychom ho chtěli nahrávat)
                    display_frame = frame.copy()

                    # Změna barev podle fáze pro lepší vizuální kontrolu
                    color = (0, 255, 0) if current_label > 0 else (200, 200, 200)
                    status_text = f"FAZE: {PHASES_NAME[current_label]} {'[PAUZA]' if paused else ''}"
                    
                    cv2.putText(display_frame, status_text, (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.imshow(window_name, display_frame)

                    # Čekání na klávesu
                    key = cv2.waitKey(30) & 0xFF

                    if key == ord(' '): # Mezerník přepíná pauzu
                        paused = not paused
                    elif key == ord('0'): current_label = 0
                    elif key == ord('1'): current_label = 1
                    elif key == ord('2'): current_label = 2
                    elif key == ord('3'): current_label = 3
                    elif key == ord('4'): current_label = 4
                    elif key == 27: # ESC
                        print("Anotace prerusena.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                    # Ukládáme label POUZE pokud video běží
                    if not paused:
                        labels.append(current_label)

                cap.release()
                cv2.destroyAllWindows()

                # Zápis do souboru
                with open(label_file, "w") as f:
                    for l in labels:
                        f.write(f"{l}\n")
                print(f"--- OK: Ulozeno {len(labels)} snimku.")

if __name__ == "__main__":
    annotate_videos()