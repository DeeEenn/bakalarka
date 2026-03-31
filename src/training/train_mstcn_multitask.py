"""
Training script for multi-task MS-TCN.

Trains on:
1. Frame-level phase segmentation (main task)
2. Video-level error type classification (auxiliary)
3. Video-level error step classification (auxiliary)
4. Video-level correctness classification (auxiliary)
"""
import torch
import torch.nn.functional as F
import time
import csv
import sys
from pathlib import Path
from torch.utils.data import DataLoader

try:
    from data_io.dataset_multitask import InhalerDatasetMultitask
    from models.mstcn_multitask import MSTCNMultitask
    from utils.paths import project_paths
except ModuleNotFoundError:
    src_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(src_root))
    from data_io.dataset_multitask import InhalerDatasetMultitask
    from models.mstcn_multitask import MSTCNMultitask
    from utils.paths import project_paths


EPOCHS = 50
BATCH_SIZE = 4
LR = 0.0005
MAX_LEN = 1000
TRAIN_RATIO = 0.8  # 80% train, 20% validation
EARLY_STOP_PATIENCE = 10  # Stop if no improvement for N epochs

NUM_STAGES = 4
NUM_LAYERS = 8
NUM_F_MAPS = 64
INPUT_DIM = 243
NUM_PHASES = 6
NUM_ERROR_TYPES = 9
NUM_ERROR_STEPS = 5
DROPOUT = 0.3

# Loss weights
WEIGHT_PHASE = 1.0
WEIGHT_ERROR_TYPE = 0.5
WEIGHT_ERROR_STEP = 0.5
WEIGHT_CORRECTNESS = 0.5

# TMSE parameters
LAMBDA_TMSE = 0.15
TAU = 4.0


def lengths_to_mask(lengths, max_len, device):
    lengths = lengths.to(device)
    rng = torch.arange(max_len, device=device).unsqueeze(0)
    return rng < lengths.unsqueeze(1)


def temporal_mse_loss(logits, valid_mask, tau=4.0):
    """Smoothing loss for temporal consistency."""
    log_probs = F.log_softmax(logits, dim=1)
    diff = (log_probs[:, :, 1:] - log_probs[:, :, :-1]) ** 2
    diff = torch.clamp(diff, max=tau ** 2)
    diff = diff.mean(dim=1)

    pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
    pair_mask_f = pair_mask.float()

    denom = pair_mask_f.sum()
    if denom.item() == 0:
        return logits.new_tensor(0.0)

    return (diff * pair_mask_f).sum() / denom


def validate_model(model, val_loader, device, criterion_phase, criterion_error_type, 
                   criterion_error_step, criterion_correctness):
    """Validate model on validation set."""
    model.eval()
    
    val_total_loss = 0.0
    val_phase_loss = 0.0
    val_error_type_loss = 0.0
    val_error_step_loss = 0.0
    val_correctness_loss = 0.0
    
    val_phase_correct = 0
    val_phase_total = 0
    val_error_type_correct = 0
    val_error_step_correct = 0
    val_correctness_correct = 0
    val_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
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

            # Calculate losses
            loss_phase = criterion_phase(phase_logits, phase_labels)
            loss_error_type = criterion_error_type(error_type_logits, error_type_labels)
            loss_error_step = criterion_error_step(error_step_logits, error_step_labels)
            loss_correctness = criterion_correctness(correctness_logits, correctness_labels)

            # Add TMSE smoothing loss for phase prediction
            loss_tmse = temporal_mse_loss(phase_logits, valid_mask, tau=TAU)

            # Combined loss
            total_loss = (
                WEIGHT_PHASE * (loss_phase + LAMBDA_TMSE * loss_tmse)
                + WEIGHT_ERROR_TYPE * loss_error_type
                + WEIGHT_ERROR_STEP * loss_error_step
                + WEIGHT_CORRECTNESS * loss_correctness
            )

            # Accumulate losses
            val_total_loss += total_loss.item()
            val_phase_loss += loss_phase.item()
            val_error_type_loss += loss_error_type.item()
            val_error_step_loss += loss_error_step.item()
            val_correctness_loss += loss_correctness.item()

            # Calculate accuracies
            phase_pred = phase_logits.argmax(dim=1)
            phase_correct = (phase_pred == phase_labels) & valid_mask
            val_phase_correct += phase_correct.sum().item()
            val_phase_total += valid_mask.sum().item()

            error_type_pred = error_type_logits.argmax(dim=1)
            val_error_type_correct += (error_type_pred == error_type_labels).sum().item()

            error_step_pred = error_step_logits.argmax(dim=1)
            val_error_step_correct += (error_step_pred == error_step_labels).sum().item()

            correctness_pred = correctness_logits.argmax(dim=1)
            val_correctness_correct += (correctness_pred == correctness_labels).sum().item()

            val_samples += features.size(0)
    
    # Average metrics
    avg_total_loss = val_total_loss / len(val_loader)
    avg_phase_loss = val_phase_loss / len(val_loader)
    avg_error_type_loss = val_error_type_loss / len(val_loader)
    avg_error_step_loss = val_error_step_loss / len(val_loader)
    avg_correctness_loss = val_correctness_loss / len(val_loader)
    
    phase_acc = 100.0 * val_phase_correct / val_phase_total if val_phase_total > 0 else 0
    error_type_acc = 100.0 * val_error_type_correct / val_samples
    error_step_acc = 100.0 * val_error_step_correct / val_samples
    correctness_acc = 100.0 * val_correctness_correct / val_samples
    
    return {
        "total_loss": avg_total_loss,
        "phase_loss": avg_phase_loss,
        "error_type_loss": avg_error_type_loss,
        "error_step_loss": avg_error_step_loss,
        "correctness_loss": avg_correctness_loss,
        "phase_acc": phase_acc,
        "error_type_acc": error_type_acc,
        "error_step_acc": error_step_acc,
        "correctness_acc": correctness_acc,
    }


def main():
    paths = project_paths(__file__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logs_dir = paths["results"] / "thesis_report" / "training_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    txt_log_path = logs_dir / "mstcn_multitask_train.log"
    csv_log_path = logs_dir / "mstcn_multitask_train_metrics.csv"

    # Load multi-task dataset with train/val split
    metadata_csv = paths["data"] / "video_metadata.csv"
    
    train_dataset = InhalerDatasetMultitask(
        str(paths["features_enhanced"]),
        str(paths["labels"]),
        str(metadata_csv),
        max_len=MAX_LEN,
        mode="train",
        train_ratio=TRAIN_RATIO,
    )
    
    val_dataset = InhalerDatasetMultitask(
        str(paths["features_enhanced"]),
        str(paths["labels"]),
        str(metadata_csv),
        max_len=MAX_LEN,
        mode="val",
        train_ratio=TRAIN_RATIO,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Initialize multi-task model
    model = MSTCNMultitask(
        num_stages=NUM_STAGES,
        num_layers=NUM_LAYERS,
        num_f_maps=NUM_F_MAPS,
        dim_in=INPUT_DIM,
        num_phases=NUM_PHASES,
        num_error_types=NUM_ERROR_TYPES,
        num_error_steps=NUM_ERROR_STEPS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Loss functions
    criterion_phase = torch.nn.CrossEntropyLoss(ignore_index=-100)
    criterion_error_type = torch.nn.CrossEntropyLoss()
    criterion_error_step = torch.nn.CrossEntropyLoss()
    criterion_correctness = torch.nn.CrossEntropyLoss()

    print(f"Start MS-TCN multi-task trenovani...")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    txt_log = open(txt_log_path, "w", encoding="utf-8")
    csv_f = open(csv_log_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(
        csv_f,
        fieldnames=[
            "epoch",
            "train_total_loss",
            "train_phase_loss",
            "train_error_type_loss",
            "train_error_step_loss",
            "train_correctness_loss",
            "train_phase_acc",
            "train_error_type_acc",
            "train_error_step_acc",
            "train_correctness_acc",
            "val_total_loss",
            "val_phase_loss",
            "val_error_type_loss",
            "val_error_step_loss",
            "val_correctness_loss",
            "val_phase_acc",
            "val_error_type_acc",
            "val_error_step_acc",
            "val_correctness_acc",
            "epoch_time_sec",
        ],
    )
    csv_writer.writeheader()
    
    # Early stopping tracking
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0

    try:
        for epoch in range(EPOCHS):
            epoch_start = time.time()
            model.train()
            
            epoch_total_loss = 0.0
            epoch_phase_loss = 0.0
            epoch_error_type_loss = 0.0
            epoch_error_step_loss = 0.0
            epoch_correctness_loss = 0.0
            
            epoch_phase_correct = 0
            epoch_phase_total = 0
            epoch_error_type_correct = 0
            epoch_error_step_correct = 0
            epoch_correctness_correct = 0
            epoch_samples = 0

            for batch in train_loader:
                features = batch["features"].to(device)
                phase_labels = batch["phase_labels"].to(device)
                error_type_labels = batch["error_type"].to(device)
                error_step_labels = batch["error_step"].to(device)
                correctness_labels = batch["is_correct"].to(device)

                valid_mask = phase_labels != -100

                optimizer.zero_grad()

                # Forward pass
                phase_logits, error_type_logits, error_step_logits, correctness_logits = model(
                    features, mask=valid_mask
                )

                # Calculate losses
                loss_phase = criterion_phase(phase_logits, phase_labels)
                loss_error_type = criterion_error_type(error_type_logits, error_type_labels)
                loss_error_step = criterion_error_step(error_step_logits, error_step_labels)
                loss_correctness = criterion_correctness(correctness_logits, correctness_labels)

                # Add TMSE smoothing loss for phase prediction
                loss_tmse = temporal_mse_loss(phase_logits, valid_mask, tau=TAU)

                # Combined loss
                total_loss = (
                    WEIGHT_PHASE * (loss_phase + LAMBDA_TMSE * loss_tmse)
                    + WEIGHT_ERROR_TYPE * loss_error_type
                    + WEIGHT_ERROR_STEP * loss_error_step
                    + WEIGHT_CORRECTNESS * loss_correctness
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Accumulate losses
                epoch_total_loss += total_loss.item()
                epoch_phase_loss += loss_phase.item()
                epoch_error_type_loss += loss_error_type.item()
                epoch_error_step_loss += loss_error_step.item()
                epoch_correctness_loss += loss_correctness.item()

                # Calculate accuracies
                # Phase accuracy (frame-level)
                phase_pred = phase_logits.argmax(dim=1)
                phase_correct = (phase_pred == phase_labels) & valid_mask
                epoch_phase_correct += phase_correct.sum().item()
                epoch_phase_total += valid_mask.sum().item()

                # Error type accuracy (video-level)
                error_type_pred = error_type_logits.argmax(dim=1)
                epoch_error_type_correct += (error_type_pred == error_type_labels).sum().item()

                # Error step accuracy (video-level)
                error_step_pred = error_step_logits.argmax(dim=1)
                epoch_error_step_correct += (error_step_pred == error_step_labels).sum().item()

                # Correctness accuracy (video-level)
                correctness_pred = correctness_logits.argmax(dim=1)
                epoch_correctness_correct += (correctness_pred == correctness_labels).sum().item()

                epoch_samples += features.size(0)

            # Average metrics
            avg_total_loss = epoch_total_loss / len(train_loader)
            avg_phase_loss = epoch_phase_loss / len(train_loader)
            avg_error_type_loss = epoch_error_type_loss / len(train_loader)
            avg_error_step_loss = epoch_error_step_loss / len(train_loader)
            avg_correctness_loss = epoch_correctness_loss / len(train_loader)
            
            phase_acc = 100.0 * epoch_phase_correct / epoch_phase_total if epoch_phase_total > 0 else 0
            error_type_acc = 100.0 * epoch_error_type_correct / epoch_samples
            error_step_acc = 100.0 * epoch_error_step_correct / epoch_samples
            correctness_acc = 100.0 * epoch_correctness_correct / epoch_samples

            epoch_time = time.time() - epoch_start
            
            # Run validation
            val_metrics = validate_model(
                model, val_loader, device,
                criterion_phase, criterion_error_type, criterion_error_step, criterion_correctness
            )

            line_train = (
                f"Epoch {epoch + 1:03d}/{EPOCHS} | TRAIN - "
                f"Total: {avg_total_loss:.4f} | "
                f"Phase: {avg_phase_loss:.4f} ({phase_acc:.1f}%) | "
                f"ErrorType: {avg_error_type_loss:.4f} ({error_type_acc:.1f}%) | "
                f"ErrorStep: {avg_error_step_loss:.4f} ({error_step_acc:.1f}%) | "
                f"Correct: {avg_correctness_loss:.4f} ({correctness_acc:.1f}%)")
            
            line_val = (
                f"Epoch {epoch + 1:03d}/{EPOCHS} | VAL   - "
                f"Total: {val_metrics['total_loss']:.4f} | "
                f"Phase: {val_metrics['phase_loss']:.4f} ({val_metrics['phase_acc']:.1f}%) | "
                f"ErrorType: {val_metrics['error_type_loss']:.4f} ({val_metrics['error_type_acc']:.1f}%) | "
                f"ErrorStep: {val_metrics['error_step_loss']:.4f} ({val_metrics['error_step_acc']:.1f}%) | "
                f"Correct: {val_metrics['correctness_loss']:.4f} ({val_metrics['correctness_acc']:.1f}%) | "
                f"Time: {epoch_time:.1f}s")
            
            print(line_train)
            print(line_val)
            txt_log.write(line_train + "\n")
            txt_log.write(line_val + "\n")
            txt_log.flush()

            csv_writer.writerow(
                {
                    "epoch": epoch + 1,
                    "train_total_loss": f"{avg_total_loss:.6f}",
                    "train_phase_loss": f"{avg_phase_loss:.6f}",
                    "train_error_type_loss": f"{avg_error_type_loss:.6f}",
                    "train_error_step_loss": f"{avg_error_step_loss:.6f}",
                    "train_correctness_loss": f"{avg_correctness_loss:.6f}",
                    "train_phase_acc": f"{phase_acc:.2f}",
                    "train_error_type_acc": f"{error_type_acc:.2f}",
                    "train_error_step_acc": f"{error_step_acc:.2f}",
                    "train_correctness_acc": f"{correctness_acc:.2f}",
                    "val_total_loss": f"{val_metrics['total_loss']:.6f}",
                    "val_phase_loss": f"{val_metrics['phase_loss']:.6f}",
                    "val_error_type_loss": f"{val_metrics['error_type_loss']:.6f}",
                    "val_error_step_loss": f"{val_metrics['error_step_loss']:.6f}",
                    "val_correctness_loss": f"{val_metrics['correctness_loss']:.6f}",
                    "val_phase_acc": f"{val_metrics['phase_acc']:.2f}",
                    "val_error_type_acc": f"{val_metrics['error_type_acc']:.2f}",
                    "val_error_step_acc": f"{val_metrics['error_step_acc']:.2f}",
                    "val_correctness_acc": f"{val_metrics['correctness_acc']:.2f}",
                    "epoch_time_sec": f"{epoch_time:.3f}",
                }
            )
            csv_f.flush()
            
            # Early stopping and checkpoint saving
            if val_metrics["total_loss"] < best_val_loss:
                best_val_loss = val_metrics["total_loss"]
                best_epoch = epoch + 1
                epochs_no_improve = 0
                
                # Save best model
                best_model_path = paths["training"] / "mstcn_multitask_best.pth"
                torch.save(model.state_dict(), str(best_model_path))
                print(f"✓ New best model saved (val_loss: {best_val_loss:.4f})")
                txt_log.write(f"✓ New best model saved (val_loss: {best_val_loss:.4f})\n")
            else:
                epochs_no_improve += 1
                msg = f"No improvement for {epochs_no_improve} epochs (best: {best_val_loss:.4f} at epoch {best_epoch})"
                print(msg)
                txt_log.write(msg + "\n")
                
                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    stop_msg = f"Early stopping triggered after {EARLY_STOP_PATIENCE} epochs without improvement"
                    print(stop_msg)
                    txt_log.write(stop_msg + "\n")
                    break

    finally:
        txt_log.close()
        csv_f.close()

    # Save final model
    save_path = paths["training"] / "mstcn_multitask_final.pth"
    torch.save(model.state_dict(), str(save_path))
    print(f"Final model saved: {save_path}")
    print(f"Best model saved: {paths['training'] / 'mstcn_multitask_best.pth'} (epoch {best_epoch}, val_loss: {best_val_loss:.4f})")
    print(f"TXT log: {txt_log_path}")
    print(f"CSV log: {csv_log_path}")


if __name__ == "__main__":
    main()
