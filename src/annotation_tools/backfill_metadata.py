import os
import cv2
import sys
from pathlib import Path

try:
    from annotation_tools.annotate import (
        METADATA_FILE,
        OUTPUT_DIR,
        VIDEO_DIR,
        load_metadata_rows,
        prompt_video_metadata,
        upsert_metadata_row,
    )
except ModuleNotFoundError:
    # Allow running directly from src/annotation_tools
    src_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(src_root))
    from annotation_tools.annotate import (
        METADATA_FILE,
        OUTPUT_DIR,
        VIDEO_DIR,
        load_metadata_rows,
        prompt_video_metadata,
        upsert_metadata_row,
    )

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov"]


def count_label_frames(label_path):
    with open(label_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip() != "")


def find_video_path(rel_no_ext):
    for ext in VIDEO_EXTENSIONS:
        candidate = os.path.join(VIDEO_DIR, rel_no_ext + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def read_video_fps(video_path):
    if not video_path:
        return 0.0

    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps


def collect_label_files():
    label_files = []
    for root, _, files in os.walk(OUTPUT_DIR):
        for name in files:
            if name.lower().endswith(".txt"):
                label_files.append(os.path.join(root, name))
    label_files.sort()
    return label_files


def main():
    if not os.path.exists(OUTPUT_DIR):
        print(f"Labels folder neexistuje: {OUTPUT_DIR}")
        return

    rows = load_metadata_rows(METADATA_FILE)
    existing_ids = {r.get("video_id", "") for r in rows}

    mode = input(
        "Rezim [m=jen chybejici metadata, a=vsechny/update]: "
    ).strip().lower()
    only_missing = mode != "a"

    labels = collect_label_files()
    if not labels:
        print("Nebyly nalezeny zadne .txt anotace.")
        return

    print(f"Nalezeno anotaci: {len(labels)}")
    print(f"Metadata soubor: {METADATA_FILE}")

    processed = 0
    skipped = 0

    for label_path in labels:
        rel_label = os.path.relpath(label_path, OUTPUT_DIR)
        rel_no_ext = os.path.splitext(rel_label)[0]

        video_path = find_video_path(rel_no_ext)
        if video_path is None:
            # fallback, aby metadata sla ulozit i bez nalezeneho videa
            video_id = rel_no_ext.replace("\\", "/") + ".unknown"
        else:
            rel_video = os.path.relpath(video_path, VIDEO_DIR)
            video_id = rel_video.replace("\\", "/")

        if only_missing and video_id in existing_ids:
            skipped += 1
            continue

        num_frames = count_label_frames(label_path)
        fps = read_video_fps(video_path)

        metadata_row = prompt_video_metadata(
            video_id=video_id,
            label_file=os.path.relpath(label_path, os.path.dirname(OUTPUT_DIR)).replace("\\", "/"),
            num_frames=num_frames,
            fps=fps,
        )
        upsert_metadata_row(METADATA_FILE, metadata_row)
        existing_ids.add(video_id)
        processed += 1

        print(f"--- OK: metadata ulozena ({processed}/{len(labels)})")

    print("\nHotovo.")
    print(f"Zpracovano: {processed}")
    print(f"Preskoceno: {skipped}")


if __name__ == "__main__":
    main()
