"""
Inference script for multi-task models.

Uses model predictions for:
1. Frame-level phase segmentation
2. Video-level error type classification (direct from model)
3. Video-level error step classification (direct from model)
4. Video-level correctness classification (direct from model)

This replaces the rule-based logic in predict_unified.py with learned predictions.
"""
import argparse
import csv
import os
import sys
from pathlib import Path
from tkinter import Tk, filedialog

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from models.asformer_multitask import ASFormerMultitask
    from models.mstcn_multitask import MSTCNMultitask
    from models.registry import get_device, _resolve_checkpoint_path
    from data_io.dataset_multitask import idx_to_error_type, idx_to_error_step
except ModuleNotFoundError:
    src_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(src_root))
    from models.asformer_multitask import ASFormerMultitask
    from models.mstcn_multitask import MSTCNMultitask
    from models.registry import get_device, _resolve_checkpoint_path
    from data_io.dataset_multitask import idx_to_error_type, idx_to_error_step


PHASES_INFO = {
    0: {"name": "KLID", "color": "lightgray"},
    1: {"name": "PRIPRAVA", "color": "royalblue"},
    2: {"name": "ROZDEJCHANI", "color": "orange"},
    3: {"name": "INHALACE", "color": "green"},
    4: {"name": "ZADRZENI", "color": "red"},
    5: {"name": "VYDECH", "color": "purple"},
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def load_multitask_model(model_name, checkpoint_path, device):
    """Load a multi-task model with checkpoint."""
    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    
    if model_name == "asformer_multitask":
        model = ASFormerMultitask(
            num_layers=8,
            d_model=128,
            input_dim=243,
            num_phases=6,
            num_error_types=9,
            num_error_steps=5,
            num_heads=8,
            dropout=0.1,
            max_dilation=16,
        ).to(device)
    elif model_name == "mstcn_multitask":
        model = MSTCNMultitask(
            num_stages=4,
            num_layers=8,
            num_f_maps=64,
            dim_in=243,
            num_phases=6,
            num_error_types=9,
            num_error_steps=5,
            dropout=0.3,
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"⚠ Checkpoint not found: {checkpoint_path} (using untrained model)")
    
    model.eval()
    return model


def infer_multitask(model_name, model, feat_path, device):
    """
    Run inference with multi-task model.
    
    Returns:
        pred: Frame-level phase predictions (numpy array)
        error_type: Predicted error type (string)
        error_step: Predicted error step (string)
        is_correct: Predicted correctness (int: 0 or 1)
        confidences: Dictionary with prediction confidences
    """
    features = np.load(feat_path).T  # (C, T)
    x = torch.from_numpy(features).float().unsqueeze(0).to(device)  # (1, C, T)

    with torch.no_grad():
        t_steps = x.shape[-1]
        mask = torch.ones((1, t_steps), dtype=torch.bool, device=device)
        
        # Get all multi-task outputs
        phase_logits, error_type_logits, error_step_logits, correctness_logits = model(
            x, mask=mask
        )
        
        # Phase predictions (frame-level)
        phase_pred = torch.argmax(phase_logits, dim=1).squeeze(0).cpu().numpy()
        phase_probs = torch.softmax(phase_logits, dim=1)
        phase_max_prob = torch.max(phase_probs, dim=1).values.squeeze(0).cpu().numpy()
        
        # Error type prediction (video-level)
        error_type_pred = torch.argmax(error_type_logits, dim=1).item()
        error_type_probs = torch.softmax(error_type_logits, dim=1).squeeze(0).cpu().numpy()
        error_type = idx_to_error_type(error_type_pred)
        
        # Error step prediction (video-level)
        error_step_pred = torch.argmax(error_step_logits, dim=1).item()
        error_step_probs = torch.softmax(error_step_logits, dim=1).squeeze(0).cpu().numpy()
        error_step = idx_to_error_step(error_step_pred)
        
        # Correctness prediction (video-level)
        correctness_pred = torch.argmax(correctness_logits, dim=1).item()
        correctness_probs = torch.softmax(correctness_logits, dim=1).squeeze(0).cpu().numpy()
        is_correct = correctness_pred
        
        confidences = {
            "phase_mean_prob": float(np.mean(phase_max_prob)),
            "phase_min_prob": float(np.min(phase_max_prob)),
            "error_type_confidence": float(error_type_probs[error_type_pred]),
            "error_step_confidence": float(error_step_probs[error_step_pred]),
            "correctness_confidence": float(correctness_probs[correctness_pred]),
        }

    return phase_pred, error_type, error_step, is_correct, confidences


def load_ground_truth(feat_path):
    """Load ground truth labels if available."""
    gt_path = feat_path.replace("features_enhanced", "labels").replace(".npy", ".txt")
    if not os.path.exists(gt_path):
        return None, gt_path
    gt = np.loadtxt(gt_path, dtype=int)
    return gt, gt_path


def _extract_segments(labels):
    """Extract continuous segments from label sequence."""
    segments = []
    if len(labels) == 0:
        return segments

    start = 0
    current = int(labels[0])
    for i in range(1, len(labels)):
        val = int(labels[i])
        if val != current:
            segments.append({
                "label": current,
                "start": start,
                "end": i - 1,
                "length": i - start,
            })
            current = val
            start = i

    segments.append({
        "label": current,
        "start": start,
        "end": len(labels) - 1,
        "length": len(labels) - start,
    })
    return segments


def visualize_results(
    pred,
    gt,
    feat_path,
    error_type,
    error_step,
    is_correct,
    confidences,
    output_path=None,
):
    """Visualize prediction with multi-task outputs."""
    fig, axes = plt.subplots(2 if gt is not None else 1, 1, figsize=(14, 6))
    if gt is None:
        axes = [axes]

    # Plot prediction
    ax_pred = axes[0]
    for seg in _extract_segments(pred):
        label = seg["label"]
        info = PHASES_INFO.get(label, {"name": f"Class {label}", "color": "gray"})
        ax_pred.barh(
            0,
            seg["length"],
            left=seg["start"],
            color=info["color"],
            edgecolor="black",
            linewidth=0.5,
        )
    ax_pred.set_xlim(0, len(pred))
    ax_pred.set_ylim(-0.5, 0.5)
    ax_pred.set_yticks([])
    ax_pred.set_xlabel("Frame")
    
    # Add multi-task predictions to title
    correctness_str = "SPRAVNE" if is_correct == 1 else "SPATNE"
    title = f"Predikce | {correctness_str}"
    if error_type != "none":
        title += f" | Chyba: {error_type}"
    if error_step != "none":
        title += f" | Krok: {error_step}"
    ax_pred.set_title(title, fontsize=10, fontweight="bold")

    # Plot ground truth if available
    if gt is not None:
        ax_gt = axes[1]
        for seg in _extract_segments(gt):
            label = seg["label"]
            info = PHASES_INFO.get(label, {"name": f"Class {label}", "color": "gray"})
            ax_gt.barh(
                0,
                seg["length"],
                left=seg["start"],
                color=info["color"],
                edgecolor="black",
                linewidth=0.5,
            )
        ax_gt.set_xlim(0, len(gt))
        ax_gt.set_ylim(-0.5, 0.5)
        ax_gt.set_yticks([])
        ax_gt.set_xlabel("Frame")
        ax_gt.set_title("Ground Truth", fontsize=10)

    # Legend
    patches = [
        mpatches.Patch(color=info["color"], label=info["name"])
        for info in PHASES_INFO.values()
    ]
    fig.legend(handles=patches, loc="upper right", bbox_to_anchor=(0.99, 0.98), fontsize=9)

    # Add confidences as text
    conf_text = (
        f"Confidences:\n"
        f"Phase: {confidences['phase_mean_prob']:.3f}\n"
        f"Error type: {confidences['error_type_confidence']:.3f}\n"
        f"Error step: {confidences['error_step_confidence']:.3f}\n"
        f"Correctness: {confidences['correctness_confidence']:.3f}"
    )
    fig.text(0.02, 0.98, conf_text, fontsize=8, verticalalignment="top", family="monospace")

    fig.suptitle(f"Multi-task Prediction: {Path(feat_path).name}", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✓ Visualization saved: {output_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Multi-task inference for inhaler technique")
    parser.add_argument(
        "--model",
        type=str,
        choices=["asformer_multitask", "mstcn_multitask"],
        default="asformer_multitask",
        help="Multi-task model to use",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, help="Input .npy file (or use GUI)")
    parser.add_argument("--output-viz", type=str, help="Save visualization to file")
    parser.add_argument("--output-csv", type=str, help="Save predictions to CSV")

    args = parser.parse_args()
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Get input file
    if args.input:
        feat_path = args.input
    else:
        feat_path = pick_npy_file()
        if not feat_path:
            print("No file selected. Exiting.")
            return
    
    if not os.path.exists(feat_path):
        print(f"Error: File not found: {feat_path}")
        return
    
    print(f"Input file: {feat_path}")
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Use default from training folder
        checkpoint_path = str(PROJECT_ROOT / "src" / "training" / f"{args.model}_best.pth")
        # Fallback to final if best doesn't exist
        if not os.path.exists(checkpoint_path):
            checkpoint_path = str(PROJECT_ROOT / "src" / "training" / f"{args.model}_final.pth")
    
    # Load model
    print(f"Loading model: {args.model}")
    model = load_multitask_model(args.model, checkpoint_path, device)
    
    # Run inference
    print("Running inference...")
    pred, error_type, error_step, is_correct, confidences = infer_multitask(
        args.model, model, feat_path, device
    )
    
    # Load ground truth if available
    gt, gt_path = load_ground_truth(feat_path)
    if gt is not None:
        print(f"Ground truth found: {gt_path}")
    
    # Print results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"Correctness: {'SPRAVNE' if is_correct == 1 else 'SPATNE'} (confidence: {confidences['correctness_confidence']:.3f})")
    print(f"Error type: {error_type} (confidence: {confidences['error_type_confidence']:.3f})")
    print(f"Error step: {error_step} (confidence: {confidences['error_step_confidence']:.3f})")
    print(f"Phase mean confidence: {confidences['phase_mean_prob']:.3f}")
    print(f"Total frames: {len(pred)}")
    print("="*70 + "\n")
    
    # Save predictions to CSV if requested
    if args.output_csv:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "video_path",
                "is_correct",
                "error_type",
                "error_step",
                "correctness_confidence",
                "error_type_confidence",
                "error_step_confidence",
                "phase_mean_confidence",
                "num_frames",
            ])
            writer.writerow([
                feat_path,
                is_correct,
                error_type,
                error_step,
                f"{confidences['correctness_confidence']:.4f}",
                f"{confidences['error_type_confidence']:.4f}",
                f"{confidences['error_step_confidence']:.4f}",
                f"{confidences['phase_mean_prob']:.4f}",
                len(pred),
            ])
        print(f"✓ Results saved: {args.output_csv}")
    
    # Visualize
    visualize_results(
        pred, gt, feat_path, error_type, error_step, is_correct, confidences,
        output_path=args.output_viz
    )
    
    print("✓ Inference complete!")


if __name__ == "__main__":
    main()
