"""
Evaluation script for multi-task models.

Evaluates:
1. Frame-level phase segmentation accuracy (MoF, F1, Edit Score)
2. Video-level error type classification accuracy
3. Video-level error step classification accuracy
4. Video-level correctness classification accuracy
"""
import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

try:
    from data_io.dataset_multitask import InhalerDatasetMultitask, idx_to_error_type, idx_to_error_step
    from models.asformer_multitask import ASFormerMultitask
    from models.mstcn_multitask import MSTCNMultitask
    from models.registry import get_device, _resolve_checkpoint_path
    from utils.paths import project_paths
except ModuleNotFoundError:
    src_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(src_root))
    from data_io.dataset_multitask import InhalerDatasetMultitask, idx_to_error_type, idx_to_error_step
    from models.asformer_multitask import ASFormerMultitask
    from models.mstcn_multitask import MSTCNMultitask
    from models.registry import get_device, _resolve_checkpoint_path
    from utils.paths import project_paths


def load_multitask_model(model_name, checkpoint_path, device):
    """Load a multi-task model."""
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
    
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def calculate_frame_accuracy(pred_phases, gt_phases, valid_mask):
    """Calculate frame-level phase accuracy (MoF)."""
    correct = ((pred_phases == gt_phases) & valid_mask).sum().item()
    total = valid_mask.sum().item()
    return correct, total


def evaluate_multitask(model, dataloader, device):
    """
    Evaluate multi-task model on all tasks.
    
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    model.eval()
    
    # Phase-level metrics (frame-wise)
    phase_correct = 0
    phase_total = 0
    
    # Video-level metrics
    all_error_type_pred = []
    all_error_type_gt = []
    all_error_step_pred = []
    all_error_step_gt = []
    all_correctness_pred = []
    all_correctness_gt = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            phase_labels = batch["phase_labels"].to(device)
            error_type_labels = batch["error_type"].to(device)
            error_step_labels = batch["error_step"].to(device)
            correctness_labels = batch["is_correct"].to(device)
            
            valid_mask = phase_labels != -100
            
            # Forward pass
            phase_logits, error_type_logits, error_step_logits, correctness_logits = model(
                features, mask=valid_mask
            )
            
            # Phase predictions (frame-level)
            phase_pred = phase_logits.argmax(dim=1)
            correct, total = calculate_frame_accuracy(phase_pred, phase_labels, valid_mask)
            phase_correct += correct
            phase_total += total
            
            # Video-level predictions
            error_type_pred = error_type_logits.argmax(dim=1)
            error_step_pred = error_step_logits.argmax(dim=1)
            correctness_pred = correctness_logits.argmax(dim=1)
            
            all_error_type_pred.extend(error_type_pred.cpu().numpy())
            all_error_type_gt.extend(error_type_labels.cpu().numpy())
            all_error_step_pred.extend(error_step_pred.cpu().numpy())
            all_error_step_gt.extend(error_step_labels.cpu().numpy())
            all_correctness_pred.extend(correctness_pred.cpu().numpy())
            all_correctness_gt.extend(correctness_labels.cpu().numpy())
    
    # Calculate metrics
    metrics = {}
    
    # Phase accuracy (MoF)
    metrics["phase_accuracy"] = phase_correct / phase_total if phase_total > 0 else 0.0
    
    # Error type classification
    metrics["error_type_accuracy"] = accuracy_score(all_error_type_gt, all_error_type_pred)
    error_type_prec, error_type_rec, error_type_f1, _ = precision_recall_fscore_support(
        all_error_type_gt, all_error_type_pred, average="weighted", zero_division=0
    )
    metrics["error_type_precision"] = error_type_prec
    metrics["error_type_recall"] = error_type_rec
    metrics["error_type_f1"] = error_type_f1
    
    # Per-class metrics for error types
    error_type_prec_per_class, error_type_rec_per_class, error_type_f1_per_class, error_type_support = precision_recall_fscore_support(
        all_error_type_gt, all_error_type_pred, average=None, zero_division=0
    )
    metrics["error_type_per_class"] = {
        "precision": error_type_prec_per_class,
        "recall": error_type_rec_per_class,
        "f1": error_type_f1_per_class,
        "support": error_type_support,
    }
    
    # Error step classification
    metrics["error_step_accuracy"] = accuracy_score(all_error_step_gt, all_error_step_pred)
    error_step_prec, error_step_rec, error_step_f1, _ = precision_recall_fscore_support(
        all_error_step_gt, all_error_step_pred, average="weighted", zero_division=0
    )
    metrics["error_step_precision"] = error_step_prec
    metrics["error_step_recall"] = error_step_rec
    metrics["error_step_f1"] = error_step_f1
    
    # Per-class metrics for error steps
    error_step_prec_per_class, error_step_rec_per_class, error_step_f1_per_class, error_step_support = precision_recall_fscore_support(
        all_error_step_gt, all_error_step_pred, average=None, zero_division=0
    )
    metrics["error_step_per_class"] = {
        "precision": error_step_prec_per_class,
        "recall": error_step_rec_per_class,
        "f1": error_step_f1_per_class,
        "support": error_step_support,
    }
    
    # Correctness classification
    metrics["correctness_accuracy"] = accuracy_score(all_correctness_gt, all_correctness_pred)
    correctness_prec, correctness_rec, correctness_f1, _ = precision_recall_fscore_support(
        all_correctness_gt, all_correctness_pred, average="binary", zero_division=0
    )
    metrics["correctness_precision"] = correctness_prec
    metrics["correctness_recall"] = correctness_rec
    metrics["correctness_f1"] = correctness_f1
    
    # Confusion matrices
    metrics["error_type_confusion"] = confusion_matrix(all_error_type_gt, all_error_type_pred)
    metrics["error_step_confusion"] = confusion_matrix(all_error_step_gt, all_error_step_pred)
    metrics["correctness_confusion"] = confusion_matrix(all_correctness_gt, all_correctness_pred)
    
    return metrics


def print_metrics(metrics, model_name):
    """Print evaluation metrics in a readable format."""
    from data_io.dataset_multitask import ERROR_TYPE_TO_IDX, ERROR_STEP_TO_IDX
    
    # Reverse mappings
    idx_to_error_type_map = {v: k for k, v in ERROR_TYPE_TO_IDX.items()}
    idx_to_error_step_map = {v: k for k, v in ERROR_STEP_TO_IDX.items()}
    
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS: {model_name}")
    print(f"{'='*70}\n")
    
    print("--- PHASE SEGMENTATION (Frame-level) ---")
    print(f"  Accuracy (MoF): {metrics['phase_accuracy']*100:.2f}%")
    
    print("\n--- ERROR TYPE CLASSIFICATION (Video-level) ---")
    print(f"  Accuracy:  {metrics['error_type_accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['error_type_precision']*100:.2f}%")
    print(f"  Recall:    {metrics['error_type_recall']*100:.2f}%")
    print(f"  F1-Score:  {metrics['error_type_f1']*100:.2f}%")
    
    # Per-class error type metrics
    print("\n  Per-class metrics:")
    print(f"    {'Class':<30s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>8s}")
    print(f"    {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for idx in range(len(metrics['error_type_per_class']['support'])):
        class_name = idx_to_error_type_map.get(idx, f"class_{idx}")
        prec = metrics['error_type_per_class']['precision'][idx]
        rec = metrics['error_type_per_class']['recall'][idx]
        f1 = metrics['error_type_per_class']['f1'][idx]
        support = int(metrics['error_type_per_class']['support'][idx])
        if support > 0:  # Only show classes that exist in the dataset
            print(f"    {class_name:<30s} {prec*100:>9.2f}% {rec*100:>9.2f}% {f1*100:>9.2f}% {support:>8d}")
    
    print("\n--- ERROR STEP CLASSIFICATION (Video-level) ---")
    print(f"  Accuracy:  {metrics['error_step_accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['error_step_precision']*100:.2f}%")
    print(f"  Recall:    {metrics['error_step_recall']*100:.2f}%")
    print(f"  F1-Score:  {metrics['error_step_f1']*100:.2f}%")
    
    # Per-class error step metrics
    print("\n  Per-class metrics:")
    print(f"    {'Class':<30s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>8s}")
    print(f"    {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for idx in range(len(metrics['error_step_per_class']['support'])):
        class_name = idx_to_error_step_map.get(idx, f"class_{idx}")
        prec = metrics['error_step_per_class']['precision'][idx]
        rec = metrics['error_step_per_class']['recall'][idx]
        f1 = metrics['error_step_per_class']['f1'][idx]
        support = int(metrics['error_step_per_class']['support'][idx])
        if support > 0:  # Only show classes that exist in the dataset
            print(f"    {class_name:<30s} {prec*100:>9.2f}% {rec*100:>9.2f}% {f1*100:>9.2f}% {support:>8d}")
    
    print("\n--- CORRECTNESS CLASSIFICATION (Video-level) ---")
    print(f"  Accuracy:  {metrics['correctness_accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['correctness_precision']*100:.2f}%")
    print(f"  Recall:    {metrics['correctness_recall']*100:.2f}%")
    print(f"  F1-Score:  {metrics['correctness_f1']*100:.2f}%")
    
    print("\n--- CONFUSION MATRICES ---")
    
    print("\nCorrectness Confusion Matrix:")
    print("              Predicted")
    print("             Incorrect  Correct")
    print(f"Actual Incorrect {metrics['correctness_confusion'][0,0]:>6}    {metrics['correctness_confusion'][0,1]:>6}")
    print(f"       Correct   {metrics['correctness_confusion'][1,0]:>6}    {metrics['correctness_confusion'][1,1]:>6}")
    
    print("\nError Type Confusion Matrix:")
    print(f"(Rows: Actual, Columns: Predicted)")
    cm = metrics['error_type_confusion']
    # Only show non-zero classes
    active_classes = sorted(set(range(cm.shape[0])))
    class_names = [idx_to_error_type_map.get(i, f"cls_{i}") for i in active_classes]
    
    # Print header
    header = "       " + " ".join(f"{name[:6]:>6s}" for name in class_names)
    print(header)
    
    # Print rows
    for i, actual_idx in enumerate(active_classes):
        row_name = class_names[i][:6]
        row_values = " ".join(f"{cm[actual_idx, pred_idx]:>6d}" for pred_idx in active_classes)
        print(f"{row_name:>6s} {row_values}")
    
    print("\nError Step Confusion Matrix:")
    print(f"(Rows: Actual, Columns: Predicted)")
    cm = metrics['error_step_confusion']
    active_classes = sorted(set(range(cm.shape[0])))
    class_names = [idx_to_error_step_map.get(i, f"cls_{i}") for i in active_classes]
    
    # Print header
    header = "       " + " ".join(f"{name[:6]:>6s}" for name in class_names)
    print(header)
    
    # Print rows
    for i, actual_idx in enumerate(active_classes):
        row_name = class_names[i][:6]
        row_values = " ".join(f"{cm[actual_idx, pred_idx]:>6d}" for pred_idx in active_classes)
        print(f"{row_name:>6s} {row_values}")
    
    print(f"\n{'='*70}\n")


def save_metrics_csv(metrics, output_path, model_name):
    """Save metrics to CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Metric", "Value"])
        writer.writerow([model_name, "phase_accuracy", f"{metrics['phase_accuracy']:.4f}"])
        writer.writerow([model_name, "error_type_accuracy", f"{metrics['error_type_accuracy']:.4f}"])
        writer.writerow([model_name, "error_type_precision", f"{metrics['error_type_precision']:.4f}"])
        writer.writerow([model_name, "error_type_recall", f"{metrics['error_type_recall']:.4f}"])
        writer.writerow([model_name, "error_type_f1", f"{metrics['error_type_f1']:.4f}"])
        writer.writerow([model_name, "error_step_accuracy", f"{metrics['error_step_accuracy']:.4f}"])
        writer.writerow([model_name, "error_step_precision", f"{metrics['error_step_precision']:.4f}"])
        writer.writerow([model_name, "error_step_recall", f"{metrics['error_step_recall']:.4f}"])
        writer.writerow([model_name, "error_step_f1", f"{metrics['error_step_f1']:.4f}"])
        writer.writerow([model_name, "correctness_accuracy", f"{metrics['correctness_accuracy']:.4f}"])
        writer.writerow([model_name, "correctness_precision", f"{metrics['correctness_precision']:.4f}"])
        writer.writerow([model_name, "correctness_recall", f"{metrics['correctness_recall']:.4f}"])
        writer.writerow([model_name, "correctness_f1", f"{metrics['correctness_f1']:.4f}"])
    print(f"✓ Metrics saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-task models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["asformer_multitask", "mstcn_multitask"],
        required=True,
        help="Multi-task model to evaluate",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--output-csv", type=str, help="Save metrics to CSV")
    
    args = parser.parse_args()
    
    paths = project_paths(__file__)
    device = get_device()
    
    print(f"Loading model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    
    # Load model
    model = load_multitask_model(args.model, args.checkpoint, device)
    
    # Load dataset
    metadata_csv = paths["data"] / "video_metadata.csv"
    dataset = InhalerDatasetMultitask(
        str(paths["features_enhanced"]),
        str(paths["labels"]),
        str(metadata_csv),
        max_len=1000,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Dataset size: {len(dataset)} videos")
    print("\nEvaluating...")
    
    # Evaluate
    metrics = evaluate_multitask(model, dataloader, device)
    
    # Print results
    print_metrics(metrics, args.model)
    
    # Save to CSV if requested
    if args.output_csv:
        save_metrics_csv(metrics, args.output_csv, args.model)
    
    print("✓ Evaluation complete!")


if __name__ == "__main__":
    main()
