import csv
import os
import cv2
import sys
from pathlib import Path

try:
    from utils.paths import project_paths
except ModuleNotFoundError:
    # Allow direct execution from src/annotation_tools.
    src_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(src_root))
    from utils.paths import project_paths

# CONFIG
paths = project_paths(__file__)
project_root = str(paths["root"])

VIDEO_DIR = str(paths["raw_videos"])
OUTPUT_DIR = str(paths["labels"])
METADATA_FILE = str(paths["metadata_csv"])

PHASES_NAME = {
    0: "KLID (0)",
    1: "PRIPRAVA (1)",
    2: "ROZDEJCHANI (2)",
    3: "INHALACE (3)",
    4: "ZADRZENI (4)",
    5: "VYDECH (5)",
}

CSV_COLUMNS = [
    "video_id",
    "is_correct",
    "error_type",
    "notes",
    "label_file",
    "num_frames",
    "fps",
]


def load_metadata_rows(csv_path):
    if not os.path.exists(csv_path):
        return []

    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def save_metadata_rows(csv_path, rows):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def upsert_metadata_row(csv_path, new_row):
    rows = load_metadata_rows(csv_path)
    updated = False

    for i, row in enumerate(rows):
        if row.get("video_id") == new_row["video_id"]:
            rows[i] = new_row
            updated = True
            break

    if not updated:
        rows.append(new_row)

    save_metadata_rows(csv_path, rows)


def infer_default_correctness(video_id):
    video_id_lower = video_id.lower()
    if "spravne" in video_id_lower:
        return "1"
    return "0"


def prompt_video_metadata(video_id, label_file, num_frames, fps):
    print("\n--- VIDEO METADATA ---")
    print(f"Video: {video_id}")
    default_correct = infer_default_correctness(video_id)

    is_correct = input(
        f"Je postup spravny? [1=spravne, 0=spatne] (default {default_correct}): "
    ).strip()
    if is_correct not in {"0", "1"}:
        is_correct = default_correct

    error_type = ""
    if is_correct == "0":
        error_type = input(
            "Typ chyby (napr. malo_vydech;kratke_zadrzeni;spatne_poradi): "
        ).strip()

    notes = input("Poznamka (volitelne): ").strip()

    return {
        "video_id": video_id,
        "is_correct": is_correct,
        "error_type": error_type,
        "notes": notes,
        "label_file": label_file,
        "num_frames": str(num_frames),
        "fps": f"{fps:.3f}",
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

                video_id = os.path.normpath(os.path.join(rel_path, file)).replace("\\", "/")
                label_file = os.path.join(target_dir, os.path.splitext(file)[0] + ".txt")

                if os.path.exists(label_file):
                    print(f"Preskakuji: {file} (hotovo)")
                    continue

                print(f"\n>>> START ANOTACE: {file}")
                cap = cv2.VideoCapture(video_path)

                # Ziskani celkoveho poctu snimku
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                print(f"    Video: {total_frames} snimku, {fps:.1f} FPS")

                current_frame_idx = 0
                current_label = 0

                window_name = "ANOTACE - SIPKY: <-/-> SNIMEK, 0-5: FAZE, ENTER: POTVRDIT, ESC: KONEC"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 1280, 720)

                # Nacteme vsechny snimky predem (rychlejsi navigace)
                print("    Nacitam snimky...")
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

                # Inicializace labelu (vsechny zacinaji jako 0 = KLID)
                labels = [0] * total_frames

                # Frame-by-frame anotace
                while True:
                    frame = all_frames[current_frame_idx].copy()

                    # Info overlay
                    color = (0, 255, 0) if current_label > 0 else (200, 200, 200)
                    status_text = f"Snimek: {current_frame_idx + 1}/{total_frames} | FAZE: {PHASES_NAME[current_label]}"

                    cv2.putText(
                        frame,
                        status_text,
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                    )

                    # Ukazatel prubehu
                    bar_width = frame.shape[1] - 60
                    bar_x = 30
                    bar_y = frame.shape[0] - 30
                    progress = int((current_frame_idx / total_frames) * bar_width)
                    cv2.rectangle(frame, (bar_x, bar_y - 10), (bar_x + bar_width, bar_y + 10), (100, 100, 100), -1)
                    cv2.rectangle(frame, (bar_x, bar_y - 10), (bar_x + progress, bar_y + 10), (0, 255, 0), -1)

                    cv2.imshow(window_name, frame)

                    # Cekani na klavesu
                    key = cv2.waitKey(0) & 0xFF

                    if key == 27:  # ESC - konec
                        print("Anotace prerusena.")
                        cv2.destroyAllWindows()
                        return
                    elif key == 13:  # ENTER - ulozeni a pokracovani na dalsi video
                        break
                    elif key == 83 or key == ord("d"):  # Sipka vpravo nebo D - dalsi snimek
                        labels[current_frame_idx] = current_label
                        current_frame_idx = min(current_frame_idx + 1, total_frames - 1)
                    elif key == 81 or key == ord("a"):  # Sipka vlevo nebo A - predchozi
                        current_frame_idx = max(current_frame_idx - 1, 0)
                    elif key == ord("0"):
                        current_label = 0
                        labels[current_frame_idx] = current_label
                    elif key == ord("1"):
                        current_label = 1
                        labels[current_frame_idx] = current_label
                    elif key == ord("2"):
                        current_label = 2
                        labels[current_frame_idx] = current_label
                    elif key == ord("3"):
                        current_label = 3
                        labels[current_frame_idx] = current_label
                    elif key == ord("4"):
                        current_label = 4
                        labels[current_frame_idx] = current_label
                    elif key == ord("5"):
                        current_label = 5
                        labels[current_frame_idx] = current_label
                    elif key == ord("s"):  # S - Skip (nastav vsechny zbyvajici na aktualni label)
                        for i in range(current_frame_idx, total_frames):
                            labels[i] = current_label
                        break

                cv2.destroyAllWindows()

                # Zapis do souboru - kazdy label na radek
                with open(label_file, "w", encoding="utf-8") as f:
                    for l in labels:
                        f.write(f"{l}\n")

                print(f"--- OK: Ulozeno {len(labels)} anotaci (= pocet snimku ve videu).")

                # Ulozeni video-level metadata pro detekci kvality postupu.
                metadata_row = prompt_video_metadata(
                    video_id=video_id,
                    label_file=os.path.relpath(label_file, project_root).replace("\\", "/"),
                    num_frames=len(labels),
                    fps=fps,
                )
                upsert_metadata_row(METADATA_FILE, metadata_row)
                print(f"--- OK: Metadata ulozena do {METADATA_FILE}")


if __name__ == "__main__":
    annotate_videos()
