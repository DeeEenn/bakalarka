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
    4: "ZADRZENI (4)",
    5: "VYDECH (5)"
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
                
                # Získání celkového počtu snímků
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"    Video: {total_frames} snimku, {fps:.1f} FPS")
                
                labels = []
                current_frame_idx = 0
                current_label = 0
                
                window_name = "ANOTACE - SIPKY: <-/-> SNIMEK, 0-5: FAZE, ENTER: POTVRDIT, ESC: KONEC"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 1280, 720)
                
                # Načteme všechny snímky předem (rychlejší navigace)
                print(f"    Nacitam snimky...")
                all_frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    all_frames.append(frame)
                cap.release()
                
                if len(all_frames) != total_frames:
                    print(f"    VAROVANI: Ocekavano {total_frames}, nacteno {len(all_frames)} snimku")
                    total_frames = len(all_frames)
                
                print(f"    Nacteno {total_frames} snimku. Zacina anotace...")
                
                # Inicializace labelů (všechny začínají jako 0 = KLID)
                labels = [0] * total_frames
                
                # Frame-by-frame anotace
                while True:
                    frame = all_frames[current_frame_idx].copy()
                    
                    # Info overlay
                    color = (0, 255, 0) if current_label > 0 else (200, 200, 200)
                    status_text = f"Snimek: {current_frame_idx+1}/{total_frames} | FAZE: {PHASES_NAME[current_label]}"
                    
                    cv2.putText(frame, status_text, (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Ukazatel průběhu
                    bar_width = frame.shape[1] - 60
                    bar_x = 30
                    bar_y = frame.shape[0] - 30
                    progress = int((current_frame_idx / total_frames) * bar_width)
                    cv2.rectangle(frame, (bar_x, bar_y - 10), (bar_x + bar_width, bar_y + 10), (100, 100, 100), -1)
                    cv2.rectangle(frame, (bar_x, bar_y - 10), (bar_x + progress, bar_y + 10), (0, 255, 0), -1)
                    
                    cv2.imshow(window_name, frame)
                    
                    # Čekání na klávesu
                    key = cv2.waitKey(0) & 0xFF  # Čeká na stisk
                    
                    if key == 27:  # ESC - konec
                        print("Anotace prerusena.")
                        cv2.destroyAllWindows()
                        return
                    elif key == 13:  # ENTER - uložení a pokračování na další video
                        break
                    elif key == 83 or key == ord('d'):  # Šipka vpravo nebo D - další snímek
                        labels[current_frame_idx] = current_label
                        current_frame_idx = min(current_frame_idx + 1, total_frames - 1)
                    elif key == 81 or key == ord('a'):  # Šipka vlevo nebo A - předchozí  
                        current_frame_idx = max(current_frame_idx - 1, 0)
                    elif key == ord('0'): 
                        current_label = 0
                        labels[current_frame_idx] = current_label
                    elif key == ord('1'): 
                        current_label = 1
                        labels[current_frame_idx] = current_label
                    elif key == ord('2'): 
                        current_label = 2
                        labels[current_frame_idx] = current_label
                    elif key == ord('3'): 
                        current_label = 3
                        labels[current_frame_idx] = current_label
                    elif key == ord('4'): 
                        current_label = 4
                        labels[current_frame_idx] = current_label
                    elif key == ord('5'): 
                        current_label = 5
                        labels[current_frame_idx] = current_label
                    elif key == ord('s'):  # S - Skip (nastav všechny zbývající na aktuální label)
                        for i in range(current_frame_idx, total_frames):
                            labels[i] = current_label
                        break

                cv2.destroyAllWindows()

                # Zápis do souboru - každý label na řádek
                with open(label_file, "w") as f:
                    for l in labels:
                        f.write(f"{l}\n")
                
                print(f"--- OK: Ulozeno {len(labels)} anotaci (= pocet snimku ve videu).")

if __name__ == "__main__":
    annotate_videos()