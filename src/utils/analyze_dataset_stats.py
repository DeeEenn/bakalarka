import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    from utils.paths import project_paths
except ModuleNotFoundError:
    # Allow direct execution from src/utils.
    src_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(src_root))
    from utils.paths import project_paths

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def _iter_video_files(video_root):
    for root, _, files in os.walk(video_root):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                yield os.path.join(root, name)


def _video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0, 0

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, frame_count


def _plot_histogram(durations_sec, output_png, bins):
    plt.figure(figsize=(10, 5))
    plt.hist(durations_sec, bins=bins, color="#2f6ea5", edgecolor="black", alpha=0.9)
    plt.xlabel("Delka videa (s)")
    plt.ylabel("Pocet videi")
    plt.title("Distribuce delky videi")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def _resolve_against_root(path_value, project_root):
    p = Path(path_value)
    if p.is_absolute():
        return str(p)
    return str((Path(project_root) / p).resolve())


def _category_from_relpath(rel_path):
    parts = rel_path.split("/")
    if len(parts) <= 1:
        return "_root"
    return parts[0]


def analyze_video_lengths(video_root, output_png=None, output_csv=None, bins=20):
    abs_videos = os.path.abspath(video_root)
    print(f"Analyzuji videa v: {abs_videos}")

    if not os.path.exists(abs_videos):
        print(f"CHYBA: Slozka {abs_videos} neexistuje")
        return

    rows = []
    durations_sec = []
    invalid = 0
    category_counts_all = {}
    category_counts_valid = {}

    for video_path in sorted(_iter_video_files(abs_videos)):
        fps, frame_count = _video_info(video_path)
        rel = os.path.relpath(video_path, abs_videos).replace("\\", "/")
        category = _category_from_relpath(rel)

        category_counts_all[category] = category_counts_all.get(category, 0) + 1

        if fps <= 0 or frame_count <= 0:
            invalid += 1
            rows.append(
                {
                    "category": category,
                    "video": rel,
                    "fps": f"{fps:.5f}",
                    "frame_count": frame_count,
                    "duration_sec": "",
                    "valid": 0,
                }
            )
            continue

        duration_sec = frame_count / fps
        durations_sec.append(duration_sec)
        category_counts_valid[category] = category_counts_valid.get(category, 0) + 1
        rows.append(
            {
                "category": category,
                "video": rel,
                "fps": f"{fps:.5f}",
                "frame_count": frame_count,
                "duration_sec": f"{duration_sec:.4f}",
                "valid": 1,
            }
        )

    if not durations_sec:
        print("Nebyla nalezena zadna validni videa s fps + frame_count")
        return

    print("\n--- STATISTIKA DELKY VIDEI ---")
    print(f"Pocet videi (validnich): {len(durations_sec)}")
    print(f"Pocet videi (nevalidnich): {invalid}")
    print(f"Prumerna delka: {np.mean(durations_sec):.2f} s")
    print(f"Median delky: {np.median(durations_sec):.2f} s")
    print(f"Min delka: {np.min(durations_sec):.2f} s")
    print(f"Max delka: {np.max(durations_sec):.2f} s")

    print("\n--- POCET VIDEI PODLE KATEGORIE ---")
    for category in sorted(category_counts_all.keys()):
        all_count = category_counts_all.get(category, 0)
        valid_count = category_counts_valid.get(category, 0)
        print(f"{category:<20} celkem: {all_count:>4} | validni: {valid_count:>4}")

    if output_png:
        os.makedirs(os.path.dirname(output_png), exist_ok=True)
        _plot_histogram(durations_sec, output_png, bins=bins)
        print(f"Histogram ulozen: {output_png}")

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["category", "video", "fps", "frame_count", "duration_sec", "valid"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Detailni CSV ulozeno: {output_csv}")

    print("\nHotovo.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyza delky videi (fps + frame_count) a histogram pro praktickou cast"
    )
    paths = project_paths(__file__)
    parser.add_argument("--video_root", default=str(paths["raw_videos"]))
    parser.add_argument(
        "--output_png",
        default=str(paths["results"] / "thesis_report" / "video_length_histogram.png"),
    )
    parser.add_argument(
        "--output_csv",
        default=str(paths["results"] / "thesis_report" / "video_length_stats.csv"),
    )
    parser.add_argument("--bins", type=int, default=20)
    args = parser.parse_args()

    video_root = _resolve_against_root(args.video_root, paths["root"])
    output_png = _resolve_against_root(args.output_png, paths["root"])
    output_csv = _resolve_against_root(args.output_csv, paths["root"])

    analyze_video_lengths(
        video_root=video_root,
        output_png=output_png,
        output_csv=output_csv,
        bins=args.bins,
    )