import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from models.registry import get_device, load_model
    from utils.paths import project_paths
except ModuleNotFoundError:
    # Allow direct execution from src/evaluation and project root invocations.
    src_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(src_root))
    from models.registry import get_device, load_model
    from utils.paths import project_paths


METRIC_COLUMNS = [
    "model",
    "samples",
    "frame_acc",
    "macro_f1",
    "edit",
    "f1_10",
    "f1_25",
    "f1_50",
]


def find_feature_label_pairs(features_dir, labels_dir):
    pairs = []
    for root, _, files in os.walk(features_dir):
        for file_name in files:
            if not file_name.endswith(".npy"):
                continue
            feat_path = os.path.join(root, file_name)
            rel = os.path.relpath(feat_path, features_dir)
            label_path = os.path.join(labels_dir, rel.replace(".npy", ".txt"))
            if os.path.exists(label_path):
                pairs.append((feat_path, label_path))
    pairs.sort()
    return pairs


def collapse_segments(seq):
    labels = []
    starts = []
    ends = []
    if len(seq) == 0:
        return labels, starts, ends

    last = int(seq[0])
    start = 0
    for i in range(1, len(seq)):
        cur = int(seq[i])
        if cur != last:
            labels.append(last)
            starts.append(start)
            ends.append(i)
            last = cur
            start = i
    labels.append(last)
    starts.append(start)
    ends.append(len(seq))
    return labels, starts, ends


def levenshtein(a, b):
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )
    return int(dp[n, m])


def edit_score(pred, gt):
    p_lab, _, _ = collapse_segments(pred)
    g_lab, _, _ = collapse_segments(gt)
    if len(g_lab) == 0:
        return 100.0 if len(p_lab) == 0 else 0.0
    dist = levenshtein(p_lab, g_lab)
    return (1.0 - dist / max(len(g_lab), len(p_lab), 1)) * 100.0


def f1_at_overlap(pred, gt, overlap):
    p_lab, p_start, p_end = collapse_segments(pred)
    g_lab, g_start, g_end = collapse_segments(gt)

    tp = 0
    fp = 0
    hits = np.zeros(len(g_lab), dtype=bool)

    for j in range(len(p_lab)):
        iou = np.zeros(len(g_lab), dtype=np.float32)
        for k in range(len(g_lab)):
            if p_lab[j] != g_lab[k]:
                continue
            inter = min(p_end[j], g_end[k]) - max(p_start[j], g_start[k])
            union = max(p_end[j], g_end[k]) - min(p_start[j], g_start[k])
            if union > 0:
                iou[k] = max(0.0, inter / union)

        idx = int(np.argmax(iou)) if len(iou) > 0 else -1
        if idx >= 0 and iou[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = True
        else:
            fp += 1

    fn = int((~hits).sum())
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2.0 * tp / denom) * 100.0


def infer_logits(model_name, model, x, device):
    with torch.no_grad():
        if model_name == "asformer":
            t_steps = x.shape[-1]
            mask = torch.ones((1, t_steps), dtype=torch.bool, device=device)
            logits = model(x, mask=mask)
        elif model_name == "mstcn":
            stage_logits = model(x)
            logits = stage_logits[-1]
        else:
            raise ValueError(model_name)
    return logits


def evaluate_model(model_name, ckpt, pairs, device):
    model, ckpt_path = load_model(model_name, checkpoint_path=ckpt, device=device)

    total_correct = 0
    total_frames = 0
    cls_tp = defaultdict(int)
    cls_fp = defaultdict(int)
    cls_fn = defaultdict(int)

    edit_scores = []
    f1_10_scores = []
    f1_25_scores = []
    f1_50_scores = []

    per_video = []

    for feat_path, label_path in pairs:
        feat = np.load(feat_path).T
        gt = np.loadtxt(label_path, dtype=np.int64)

        t_steps = min(feat.shape[1], len(gt))
        if t_steps <= 1:
            continue

        feat = feat[:, :t_steps]
        gt = gt[:t_steps]

        x = torch.from_numpy(feat).float().unsqueeze(0).to(device)
        logits = infer_logits(model_name, model, x, device)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()[:t_steps]

        correct = int((pred == gt).sum())
        total_correct += correct
        total_frames += int(t_steps)

        # Macro F1 over classes
        for c in range(6):
            p_c = pred == c
            g_c = gt == c
            tp = int(np.logical_and(p_c, g_c).sum())
            fp = int(np.logical_and(p_c, ~g_c).sum())
            fn = int(np.logical_and(~p_c, g_c).sum())
            cls_tp[c] += tp
            cls_fp[c] += fp
            cls_fn[c] += fn

        vid_edit = edit_score(pred, gt)
        vid_f1_10 = f1_at_overlap(pred, gt, 0.10)
        vid_f1_25 = f1_at_overlap(pred, gt, 0.25)
        vid_f1_50 = f1_at_overlap(pred, gt, 0.50)
        vid_acc = (correct / t_steps) * 100.0

        edit_scores.append(vid_edit)
        f1_10_scores.append(vid_f1_10)
        f1_25_scores.append(vid_f1_25)
        f1_50_scores.append(vid_f1_50)

        per_video.append(
            {
                "model": model_name,
                "video": os.path.basename(feat_path),
                "feature_path": feat_path,
                "label_path": label_path,
                "frames": int(t_steps),
                "frame_acc": vid_acc,
                "edit": vid_edit,
                "f1_10": vid_f1_10,
                "f1_25": vid_f1_25,
                "f1_50": vid_f1_50,
            }
        )

    frame_acc = 0.0 if total_frames == 0 else total_correct / total_frames

    class_f1 = []
    for c in range(6):
        tp = cls_tp[c]
        fp = cls_fp[c]
        fn = cls_fn[c]
        denom = 2 * tp + fp + fn
        f1 = 0.0 if denom == 0 else (2.0 * tp / denom)
        class_f1.append(f1)

    macro_f1 = float(np.mean(class_f1)) if class_f1 else 0.0

    summary = {
        "model": model_name,
        "checkpoint": ckpt_path,
        "samples": len(per_video),
        "frame_acc": frame_acc * 100.0,
        "macro_f1": macro_f1 * 100.0,
        "edit": float(np.mean(edit_scores)) if edit_scores else 0.0,
        "f1_10": float(np.mean(f1_10_scores)) if f1_10_scores else 0.0,
        "f1_25": float(np.mean(f1_25_scores)) if f1_25_scores else 0.0,
        "f1_50": float(np.mean(f1_50_scores)) if f1_50_scores else 0.0,
    }

    return summary, per_video


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_summary_metrics(summary_rows, output_png):
    metrics = ["frame_acc", "macro_f1", "edit", "f1_10", "f1_25", "f1_50"]
    metric_labels = ["FrameAcc", "MacroF1", "Edit", "F1@10", "F1@25", "F1@50"]

    x = np.arange(len(metrics))
    width = 0.36

    asf = next((r for r in summary_rows if r["model"] == "asformer"), None)
    mst = next((r for r in summary_rows if r["model"] == "mstcn"), None)

    if asf is None or mst is None:
        return

    y_asf = [asf[m] for m in metrics]
    y_mst = [mst[m] for m in metrics]

    plt.figure(figsize=(11, 5))
    plt.bar(x - width / 2, y_asf, width=width, label="ASFormer")
    plt.bar(x + width / 2, y_mst, width=width, label="MS-TCN")
    plt.xticks(x, metric_labels)
    plt.ylabel("Score (%)")
    plt.title("Model Comparison Metrics")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=170)
    plt.close()


def print_table(summary_rows):
    print("")
    print("=" * 110)
    print("THESIS MODEL COMPARISON")
    print("=" * 110)
    print(
        f"{'Model':<12} {'Samples':>7} {'FrameAcc':>10} {'MacroF1':>10} "
        f"{'Edit':>10} {'F1@10':>10} {'F1@25':>10} {'F1@50':>10}"
    )
    print("-" * 110)
    for r in summary_rows:
        print(
            f"{r['model']:<12} {r['samples']:>7d} {r['frame_acc']:>9.2f}% {r['macro_f1']:>9.2f}% "
            f"{r['edit']:>9.2f} {r['f1_10']:>9.2f} {r['f1_25']:>9.2f} {r['f1_50']:>9.2f}"
        )
    print("=" * 110)
    print("")


def main():
    paths = project_paths(__file__)
    parser = argparse.ArgumentParser(description="Generate thesis-ready tables and plots for ASFormer vs MS-TCN")
    parser.add_argument("--features_dir", default=str(paths["features_enhanced"]))
    parser.add_argument("--labels_dir", default=str(paths["labels"]))
    parser.add_argument("--asformer_ckpt", default="asformer_attention_v1.pth")
    parser.add_argument("--mstcn_ckpt", default="mstcn_v1.pth")
    parser.add_argument("--out_dir", default=str(paths["results"] / "thesis_report"))
    parser.add_argument(
        "--include_substring",
        default=None,
        help="Optional: evaluate only pairs whose relative feature path contains this text.",
    )
    args = parser.parse_args()

    pairs = find_feature_label_pairs(args.features_dir, args.labels_dir)
    if args.include_substring:
        needle = args.include_substring.lower()
        filtered = []
        for feat_path, label_path in pairs:
            rel = os.path.relpath(feat_path, args.features_dir).replace("\\", "/").lower()
            if needle in rel:
                filtered.append((feat_path, label_path))
        pairs = filtered

    if not pairs:
        print("No feature-label pairs found for evaluation.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()

    print(f"Pairs: {len(pairs)} | device: {device}")

    summary_asf, per_video_asf = evaluate_model("asformer", args.asformer_ckpt, pairs, device)
    summary_mst, per_video_mst = evaluate_model("mstcn", args.mstcn_ckpt, pairs, device)

    summary_rows = [summary_asf, summary_mst]
    per_video_rows = per_video_asf + per_video_mst

    print_table(summary_rows)

    summary_csv = os.path.join(args.out_dir, "summary_metrics.csv")
    per_video_csv = os.path.join(args.out_dir, "per_video_metrics.csv")
    summary_plot = os.path.join(args.out_dir, "summary_metrics_bar.png")

    write_csv(summary_csv, summary_rows, fieldnames=METRIC_COLUMNS + ["checkpoint"])
    write_csv(
        per_video_csv,
        per_video_rows,
        fieldnames=[
            "model",
            "video",
            "feature_path",
            "label_path",
            "frames",
            "frame_acc",
            "edit",
            "f1_10",
            "f1_25",
            "f1_50",
        ],
    )
    plot_summary_metrics(summary_rows, summary_plot)

    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved per-video CSV: {per_video_csv}")
    print(f"Saved summary plot: {summary_plot}")


if __name__ == "__main__":
    main()
