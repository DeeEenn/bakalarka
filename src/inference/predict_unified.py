import argparse
import csv
import os
import sys
from tkinter import Tk, filedialog
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from models.registry import get_device, load_model
except ModuleNotFoundError:
    # Allow direct execution from src/inference and project root invocations.
    src_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(src_root))
    from models.registry import get_device, load_model


PHASES_INFO = {
    0: {"name": "KLID", "color": "lightgray"},
    1: {"name": "PRIPRAVA", "color": "royalblue"},
    2: {"name": "ROZDEJCHANI", "color": "orange"},
    3: {"name": "INHALACE", "color": "green"},
    4: {"name": "ZADRZENI", "color": "red"},
    5: {"name": "VYDECH", "color": "purple"},
}


def pick_npy_file():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Vyber .npy soubor pro predikci",
        filetypes=[("NumPy files", "*.npy")],
    )
    root.destroy()
    return path


def infer_one(model_name, model, feat_path, device):
    features = np.load(feat_path).T
    x = torch.from_numpy(features).float().unsqueeze(0).to(device)

    with torch.no_grad():
        if model_name == "asformer":
            t_steps = x.shape[-1]
            mask = torch.ones((1, t_steps), dtype=torch.bool, device=device)
            logits = model(x, mask=mask)
        elif model_name == "mstcn":
            stage_logits = model(x)
            logits = stage_logits[-1]
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    return pred


def load_ground_truth(feat_path):
    gt_path = feat_path.replace("features_enhanced", "labels").replace(".npy", ".txt")
    if not os.path.exists(gt_path):
        return None, gt_path
    gt = np.loadtxt(gt_path, dtype=int)
    return gt, gt_path


def _compress_labels(labels):
    if len(labels) == 0:
        return []
    compact = [int(labels[0])]
    for val in labels[1:]:
        val = int(val)
        if val != compact[-1]:
            compact.append(val)
    return compact


def _extract_segments(labels):
    segments = []
    if len(labels) == 0:
        return segments

    start = 0
    current = int(labels[0])
    for i in range(1, len(labels)):
        val = int(labels[i])
        if val != current:
            segments.append(
                {
                    "label": current,
                    "start": start,
                    "end": i - 1,
                    "length": i - start,
                }
            )
            current = val
            start = i

    segments.append(
        {
            "label": current,
            "start": start,
            "end": len(labels) - 1,
            "length": len(labels) - start,
        }
    )
    return segments


def _infer_fps_from_metadata(feat_path, fallback_fps=30.0):
    """
    Tries to recover FPS from data/video_metadata.csv.
    Falls back to provided value if metadata is missing or unmatched.
    """
    try:
        path_obj = Path(feat_path).resolve()
        project_root = path_obj.parents[2]
        metadata_path = project_root / "data" / "video_metadata.csv"
        if not metadata_path.exists():
            return float(fallback_fps)

        rel_parts = path_obj.parts
        if "features_enhanced" not in rel_parts:
            return float(fallback_fps)
        idx = rel_parts.index("features_enhanced")
        rel_after = rel_parts[idx + 1 :]
        if not rel_after:
            return float(fallback_fps)

        stem = Path(*rel_after).with_suffix("").as_posix()
        expected_label_suffix = stem + ".txt"

        with open(metadata_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_file = (row.get("label_file") or "").replace("\\", "/")
                if label_file.endswith(expected_label_suffix):
                    fps_raw = row.get("fps")
                    if fps_raw is None or str(fps_raw).strip() == "":
                        return float(fallback_fps)
                    return float(fps_raw)

    except Exception:
        pass

    return float(fallback_fps)


def evaluate_logic_checker(prediction, fps, min_breath_hold_sec=4.5):
    """
    Rule set:
    1) Order check where PRIPRAVA (1) and ROZDEJCHANI (2) are interchangeable.
    2) Average ZADRZENI (4) duration must be >= min_breath_hold_sec.
    """
    pred = np.asarray(prediction, dtype=int)
    compact = [p for p in _compress_labels(pred) if p in {1, 2, 3, 4, 5}]
    segments = _extract_segments(pred)

    # Stage map: PRIPRAVA and ROZDEJCHANI are same stage (mix allowed)
    stage_map = {1: 0, 2: 0, 3: 1, 4: 2, 5: 3}

    order_ok = True
    order_issues = []
    mapped_stages = []

    for label in compact:
        mapped = stage_map[label]
        if mapped_stages and mapped < mapped_stages[-1]:
            order_ok = False
            order_issues.append(
                f"Porusene poradi: faze {label} nasleduje po vyssi fazi ({mapped_stages[-1]})."
            )
        else:
            mapped_stages.append(mapped)

    # Optional basic completeness checks for core clinical phases
    for required in (3, 4, 5):
        if required not in compact:
            order_ok = False
            order_issues.append(f"Chybi faze {required} ({PHASES_INFO[required]['name']}).")

    zadrzeni_segments = [s for s in segments if s["label"] == 4]
    zadrzeni_secs = [(s["length"] / float(fps)) for s in zadrzeni_segments]
    avg_zadrzeni_sec = float(np.mean(zadrzeni_secs)) if zadrzeni_secs else 0.0
    breath_hold_ok = avg_zadrzeni_sec >= float(min_breath_hold_sec) and len(zadrzeni_secs) > 0

    breath_hold_issue = None
    if len(zadrzeni_secs) == 0:
        breath_hold_issue = "Faze ZADRZENI nebyla detekovana."
    elif not breath_hold_ok:
        breath_hold_issue = (
            f"Prumerna delka ZADRZENI je {avg_zadrzeni_sec:.2f}s, "
            f"minimum je {min_breath_hold_sec:.2f}s."
        )

    overall_ok = order_ok and breath_hold_ok
    return {
        "overall_ok": overall_ok,
        "order_ok": order_ok,
        "order_issues": order_issues,
        "breath_hold_ok": breath_hold_ok,
        "breath_hold_issue": breath_hold_issue,
        "avg_zadrzeni_sec": avg_zadrzeni_sec,
        "fps": float(fps),
        "compact_sequence": compact,
        "zadrzeni_count": len(zadrzeni_secs),
    }


def print_logic_report(report):
    print("\n=== LOGIC CHECKER ===")
    print(f"Pouzite FPS: {report['fps']:.3f}")
    print(f"Kompaktni sekvence fazi: {report['compact_sequence']}")

    if report["order_ok"]:
        print("[OK] Poradi kroku je validni (PRIPRAVA/ROZDEJCHANI mix je povolen).")
    else:
        print("[CHYBA] Poradi kroku je nevalidni:")
        for issue in report["order_issues"]:
            print(f"  - {issue}")

    if report["breath_hold_ok"]:
        print(
            f"[OK] Prumerne ZADRZENI = {report['avg_zadrzeni_sec']:.2f}s "
            f"(pocet segmentu: {report['zadrzeni_count']})."
        )
    else:
        print(f"[CHYBA] {report['breath_hold_issue']}")

    verdict = "PROSEL" if report["overall_ok"] else "NEPROSEL"
    print(f"Finalni verdikt: {verdict}")


def plot_prediction(feat_path, model_name, checkpoint_path, prediction, gt=None):
    if gt is not None:
        t_steps = min(len(gt), len(prediction))
        gt = gt[:t_steps]
        prediction = prediction[:t_steps]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    else:
        t_steps = len(prediction)
        fig, ax2 = plt.subplots(1, 1, figsize=(15, 3))
        ax1 = None

    time_axis = np.arange(t_steps)

    if ax1 is not None:
        gt_colors = [PHASES_INFO[int(g)]["color"] for g in gt]
        ax1.bar(time_axis, [1] * t_steps, color=gt_colors, width=1.0)
        ax1.set_title("Ground Truth", fontsize=12, fontweight="bold")
        ax1.set_yticks([])
        ax1.grid(False)

    pred_colors = [PHASES_INFO[int(p)]["color"] for p in prediction[:t_steps]]
    ax2.bar(time_axis, [1] * t_steps, color=pred_colors, width=1.0)
    ax2.set_title(f"Predikce modelu: {model_name}", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Snimek (cas)")
    ax2.set_yticks([])
    ax2.grid(False)

    legend_patches = [
        mpatches.Patch(color=info["color"], label=f"{idx}: {info['name']}")
        for idx, info in PHASES_INFO.items()
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=6, fontsize=10, bbox_to_anchor=(0.5, 0.02))

    plt.suptitle(
        f"Soubor: {os.path.basename(feat_path)} | Model: {model_name} | Ckpt: {checkpoint_path}",
        fontsize=12,
        fontweight="bold",
        y=0.97,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    plt.show()


def plot_comparison(feat_path, asformer_pred, mstcn_pred, asformer_ckpt, mstcn_ckpt, gt=None):
    if gt is not None:
        t_steps = min(len(gt), len(asformer_pred), len(mstcn_pred))
        gt = gt[:t_steps]
        asformer_pred = asformer_pred[:t_steps]
        mstcn_pred = mstcn_pred[:t_steps]
        fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
        ax_gt, ax_a, ax_m = axes
    else:
        t_steps = min(len(asformer_pred), len(mstcn_pred))
        asformer_pred = asformer_pred[:t_steps]
        mstcn_pred = mstcn_pred[:t_steps]
        fig, axes = plt.subplots(2, 1, figsize=(16, 5), sharex=True)
        ax_gt = None
        ax_a, ax_m = axes

    time_axis = np.arange(t_steps)

    if ax_gt is not None:
        gt_colors = [PHASES_INFO[int(g)]["color"] for g in gt]
        ax_gt.bar(time_axis, [1] * t_steps, color=gt_colors, width=1.0)
        ax_gt.set_title("Ground Truth", fontsize=11, fontweight="bold")
        ax_gt.set_yticks([])
        ax_gt.grid(False)

    a_colors = [PHASES_INFO[int(p)]["color"] for p in asformer_pred]
    ax_a.bar(time_axis, [1] * t_steps, color=a_colors, width=1.0)
    ax_a.set_title("Predikce: ASFormer", fontsize=11, fontweight="bold")
    ax_a.set_yticks([])
    ax_a.grid(False)

    m_colors = [PHASES_INFO[int(p)]["color"] for p in mstcn_pred]
    ax_m.bar(time_axis, [1] * t_steps, color=m_colors, width=1.0)
    ax_m.set_title("Predikce: MS-TCN", fontsize=11, fontweight="bold")
    ax_m.set_xlabel("Snimek (cas)")
    ax_m.set_yticks([])
    ax_m.grid(False)

    legend_patches = [
        mpatches.Patch(color=info["color"], label=f"{idx}: {info['name']}")
        for idx, info in PHASES_INFO.items()
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=6, fontsize=10, bbox_to_anchor=(0.5, 0.02))

    plt.suptitle(
        (
            f"Soubor: {os.path.basename(feat_path)} | "
            f"ASFormer ckpt: {asformer_ckpt} | MS-TCN ckpt: {mstcn_ckpt}"
        ),
        fontsize=11,
        fontweight="bold",
        y=0.97,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    plt.show()


def print_single_result(model_name, ckpt_path, feat_path, prediction, gt, gt_path):
    print(f"\n=== {model_name.upper()} ===")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Input: {feat_path}")
    if gt is None:
        print(f"Ground truth nenalezena: {gt_path}")
    else:
        print(f"Ground truth: {gt_path}")
        t_steps = min(len(gt), len(prediction))
        frame_acc = (prediction[:t_steps] == gt[:t_steps]).mean()
        print(f"Frame accuracy na tomto souboru: {frame_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Unified predict pro ASFormer i MS-TCN")
    parser.add_argument(
        "--model",
        choices=["asformer", "mstcn", "both"],
        required=True,
        help="Ktery model pouzit (both = spolecne porovnani)",
    )
    parser.add_argument("--ckpt", default=None, help="Cesta k checkpointu")
    parser.add_argument("--asformer-ckpt", default=None, help="Cesta k ASFormer checkpointu")
    parser.add_argument("--mstcn-ckpt", default=None, help="Cesta k MS-TCN checkpointu")
    parser.add_argument("--input", default=None, help="Cesta k .npy (kdyz neni, otevre se dialog)")
    parser.add_argument("--no-plot", action="store_true", help="Nevykresluj graf")
    parser.add_argument("--no-logic-check", action="store_true", help="Preskocit logic checker")
    parser.add_argument(
        "--min-breath-hold-sec",
        type=float,
        default=4.5,
        help="Minimalni prumerna delka faze ZADRZENI v sekundach",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Vynutene FPS pro logic checker (jinak se zkusi metadata, fallback 30)",
    )
    args = parser.parse_args()

    device = get_device()

    feat_path = args.input if args.input else pick_npy_file()
    if not feat_path:
        print("Nebyl vybran zadny .npy soubor.")
        return

    gt, gt_path = load_ground_truth(feat_path)
    fps = (
        float(args.fps)
        if args.fps is not None
        else _infer_fps_from_metadata(feat_path, fallback_fps=30.0)
    )

    if args.model == "both":
        asformer_ckpt = args.asformer_ckpt or args.ckpt
        mstcn_ckpt = args.mstcn_ckpt or args.ckpt

        asformer_model, asformer_ckpt_path = load_model("asformer", checkpoint_path=asformer_ckpt, device=device)
        mstcn_model, mstcn_ckpt_path = load_model("mstcn", checkpoint_path=mstcn_ckpt, device=device)

        asformer_pred = infer_one("asformer", asformer_model, feat_path, device)
        mstcn_pred = infer_one("mstcn", mstcn_model, feat_path, device)

        print_single_result("asformer", asformer_ckpt_path, feat_path, asformer_pred, gt, gt_path)
        if not args.no_logic_check:
            asformer_logic = evaluate_logic_checker(
                asformer_pred,
                fps=fps,
                min_breath_hold_sec=args.min_breath_hold_sec,
            )
            print_logic_report(asformer_logic)

        print_single_result("mstcn", mstcn_ckpt_path, feat_path, mstcn_pred, gt, gt_path)
        if not args.no_logic_check:
            mstcn_logic = evaluate_logic_checker(
                mstcn_pred,
                fps=fps,
                min_breath_hold_sec=args.min_breath_hold_sec,
            )
            print_logic_report(mstcn_logic)

        if not args.no_plot:
            plot_comparison(
                feat_path,
                asformer_pred,
                mstcn_pred,
                asformer_ckpt=asformer_ckpt_path,
                mstcn_ckpt=mstcn_ckpt_path,
                gt=gt,
            )
        return

    model, ckpt_path = load_model(args.model, checkpoint_path=args.ckpt, device=device)
    prediction = infer_one(args.model, model, feat_path, device)

    print_single_result(args.model, ckpt_path, feat_path, prediction, gt, gt_path)

    if not args.no_logic_check:
        logic_report = evaluate_logic_checker(
            prediction,
            fps=fps,
            min_breath_hold_sec=args.min_breath_hold_sec,
        )
        print_logic_report(logic_report)

    if not args.no_plot:
        plot_prediction(feat_path, args.model, ckpt_path, prediction, gt=gt)


if __name__ == "__main__":
    main()
