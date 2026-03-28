import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Unified predict pro ASFormer i MS-TCN")
    parser.add_argument("--model", choices=["asformer", "mstcn"], required=True, help="Ktery model pouzit")
    parser.add_argument("--ckpt", default=None, help="Cesta k checkpointu")
    parser.add_argument("--input", default=None, help="Cesta k .npy (kdyz neni, otevre se dialog)")
    parser.add_argument("--no-plot", action="store_true", help="Nevykresluj graf")
    args = parser.parse_args()

    device = get_device()
    model, ckpt_path = load_model(args.model, checkpoint_path=args.ckpt, device=device)

    feat_path = args.input if args.input else pick_npy_file()
    if not feat_path:
        print("Nebyl vybran zadny .npy soubor.")
        return

    prediction = infer_one(args.model, model, feat_path, device)
    gt, gt_path = load_ground_truth(feat_path)

    print(f"Model: {args.model}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Input: {feat_path}")
    if gt is None:
        print(f"Ground truth nenalezena: {gt_path}")
    else:
        print(f"Ground truth: {gt_path}")
        t_steps = min(len(gt), len(prediction))
        frame_acc = (prediction[:t_steps] == gt[:t_steps]).mean()
        print(f"Frame accuracy na tomto souboru: {frame_acc:.4f}")

    if not args.no_plot:
        plot_prediction(feat_path, args.model, ckpt_path, prediction, gt=gt)


if __name__ == "__main__":
    main()
