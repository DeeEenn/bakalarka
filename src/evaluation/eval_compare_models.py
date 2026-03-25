import argparse
import os
from collections import defaultdict

import numpy as np
import torch

from models.registry import get_device, load_model
from utils.paths import project_paths


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
    return pairs


def collapse_segments(seq):
    labels, starts, ends = [], [], []
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
    cls_tp, cls_fp, cls_fn = defaultdict(int), defaultdict(int), defaultdict(int)

    edit_scores, f1_10_scores, f1_25_scores, f1_50_scores = [], [], [], []

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

        total_correct += int((pred == gt).sum())
        total_frames += int(t_steps)

        for c in range(6):
            p_c = pred == c
            g_c = gt == c
            cls_tp[c] += int(np.logical_and(p_c, g_c).sum())
            cls_fp[c] += int(np.logical_and(p_c, ~g_c).sum())
            cls_fn[c] += int(np.logical_and(~p_c, g_c).sum())

        edit_scores.append(edit_score(pred, gt))
        f1_10_scores.append(f1_at_overlap(pred, gt, 0.10))
        f1_25_scores.append(f1_at_overlap(pred, gt, 0.25))
        f1_50_scores.append(f1_at_overlap(pred, gt, 0.50))

    frame_acc = 0.0 if total_frames == 0 else total_correct / total_frames

    class_f1 = []
    for c in range(6):
        tp, fp, fn = cls_tp[c], cls_fp[c], cls_fn[c]
        denom = 2 * tp + fp + fn
        class_f1.append(0.0 if denom == 0 else (2.0 * tp / denom))

    macro_f1 = float(np.mean(class_f1)) if class_f1 else 0.0

    return {
        "model": model_name,
        "checkpoint": ckpt_path,
        "frame_acc": frame_acc * 100.0,
        "macro_f1": macro_f1 * 100.0,
        "edit": float(np.mean(edit_scores)) if edit_scores else 0.0,
        "f1_10": float(np.mean(f1_10_scores)) if f1_10_scores else 0.0,
        "f1_25": float(np.mean(f1_25_scores)) if f1_25_scores else 0.0,
        "f1_50": float(np.mean(f1_50_scores)) if f1_50_scores else 0.0,
        "samples": len(edit_scores),
    }


def print_table(results):
    print("")
    print("=" * 110)
    print("POROVNANI MODELU")
    print("=" * 110)
    print(
        f"{'Model':<12} {'Samples':>7} {'FrameAcc':>10} {'MacroF1':>10} "
        f"{'Edit':>10} {'F1@10':>10} {'F1@25':>10} {'F1@50':>10}"
    )
    print("-" * 110)
    for r in results:
        print(
            f"{r['model']:<12} {r['samples']:>7d} {r['frame_acc']:>9.2f}% {r['macro_f1']:>9.2f}% "
            f"{r['edit']:>9.2f} {r['f1_10']:>9.2f} {r['f1_25']:>9.2f} {r['f1_50']:>9.2f}"
        )
    print("=" * 110)
    print("")


def main():
    paths = project_paths(__file__)
    parser = argparse.ArgumentParser(description="Eval + porovnani ASFormer vs MS-TCN")
    parser.add_argument("--features_dir", default=str(paths["features_enhanced"]))
    parser.add_argument("--labels_dir", default=str(paths["labels"]))
    parser.add_argument("--asformer_ckpt", default="asformer_attention_v1.pth")
    parser.add_argument("--mstcn_ckpt", default="mstcn_v1.pth")
    args = parser.parse_args()

    pairs = find_feature_label_pairs(args.features_dir, args.labels_dir)
    if not pairs:
        print("Nebyly nalezeny zadne feature-label pary.")
        return

    device = get_device()
    print(f"Nalezeno paru: {len(pairs)} | device: {device}")

    res_asformer = evaluate_model("asformer", args.asformer_ckpt, pairs, device)
    res_mstcn = evaluate_model("mstcn", args.mstcn_ckpt, pairs, device)
    print_table([res_asformer, res_mstcn])


if __name__ == "__main__":
    main()
