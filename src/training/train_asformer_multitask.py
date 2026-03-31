"""
Training script for multi-task ASFormer.

Trains on:
1. Frame-level phase segmentation (main task)
2. Video-level error type classification (auxiliary)
3. Video-level error step classification (auxiliary)
4. Video-level correctness classification (auxiliary)
"""
import logging
import time
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from data_io.dataset_multitask import InhalerDatasetMultitask
    from models.asformer_multitask import ASFormerMultitask
    from utils.paths import project_paths
except ModuleNotFoundError:
    src_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(src_root))
    from data_io.dataset_multitask import InhalerDatasetMultitask
    from models.asformer_multitask import ASFormerMultitask
    from utils.paths import project_paths


# Training hyperparameters
EPOCHS = 50
BATCH_SIZE = 4
LR = 0.0005
MAX_LEN = 1000
TRAIN_RATIO = 0.8  # 80% train, 20% validation
EARLY_STOP_PATIENCE = 10  # Stop if no improvement for N epochs

# Multi-task loss weights
WEIGHT_PHASE = 1.0  # Main task
WEIGHT_ERROR_TYPE = 0.5  # Auxiliary
WEIGHT_ERROR_STEP = 0.5  # Auxiliary
WEIGHT_CORRECTNESS = 0.5  # Auxiliary


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

            # Combined loss
            total_loss = (
                WEIGHT_PHASE * loss_phase
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
    txt_log_path = logs_dir / "asformer_multitask_train.log"
    csv_log_path = logs_dir / "asformer_multitask_train_metrics.csv"

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
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize multi-task model
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

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Loss functions
    criterion_phase = torch.nn.CrossEntropyLoss(ignore_index=-100)
    criterion_error_type = torch.nn.CrossEntropyLoss()
    criterion_error_step = torch.nn.CrossEntropyLoss()
    criterion_correctness = torch.nn.CrossEntropyLoss()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.handlers = []

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    )
    file_handler = logging.FileHandler(str(txt_log_path), mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    with open(csv_log_path, "w", newline="", encoding="utf-8") as csv_f:
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

        logger.info("Start multi-task trenovani ASFormer...")
        logger.info("Train samples: %d | Val samples: %d", len(train_dataset), len(val_dataset))
        logger.info(
            "Config: epochs=%d, batch_size=%d, lr=%s, max_len=%d, device=%s",
            EPOCHS,
            BATCH_SIZE,
            LR,
            MAX_LEN,
            device,
        )
        logger.info(
            "Loss weights: phase=%.2f, error_type=%.2f, error_step=%.2f, correctness=%.2f",
            WEIGHT_PHASE,
            WEIGHT_ERROR_TYPE,
            WEIGHT_ERROR_STEP,
            WEIGHT_CORRECTNESS,
        )
        
        # Early stopping tracking
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_epoch = 0

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

            for step, batch in enumerate(train_loader, start=1):
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

                # Combined loss
                total_loss = (
                    WEIGHT_PHASE * loss_phase
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

                if step % 5 == 0 or step == len(train_loader):
                    logger.info(
                        "Epoch %d/%d | Step %d/%d | Total loss: %.4f | Phase: %.4f | ErrorType: %.4f | ErrorStep: %.4f | Correct: %.4f",
                        epoch + 1,
                        EPOCHS,
                        step,
                        len(train_loader),
                        total_loss.item(),
                        loss_phase.item(),
                        loss_error_type.item(),
                        loss_error_step.item(),
                        loss_correctness.item(),
                    )

            epoch_time = time.time() - epoch_start
            
            # Average training metrics
            avg_total_loss = epoch_total_loss / len(train_loader)
            avg_phase_loss = epoch_phase_loss / len(train_loader)
            avg_error_type_loss = epoch_error_type_loss / len(train_loader)
            avg_error_step_loss = epoch_error_step_loss / len(train_loader)
            avg_correctness_loss = epoch_correctness_loss / len(train_loader)
            
            phase_acc = 100.0 * epoch_phase_correct / epoch_phase_total if epoch_phase_total > 0 else 0
            error_type_acc = 100.0 * epoch_error_type_correct / epoch_samples
            error_step_acc = 100.0 * epoch_error_step_correct / epoch_samples
            correctness_acc = 100.0 * epoch_correctness_correct / epoch_samples
            
            # Run validation
            val_metrics = validate_model(
                model, val_loader, device,
                criterion_phase, criterion_error_type, criterion_error_step, criterion_correctness
            )

            logger.info(
                "Epoch %d/%d | TRAIN - Total: %.4f Phase: %.4f (%.2f%%) ErrorType: %.4f (%.2f%%) ErrorStep: %.4f (%.2f%%) Correct: %.4f (%.2f%%)",
                epoch + 1,
                EPOCHS,
                avg_total_loss,
                avg_phase_loss,
                phase_acc,
                avg_error_type_loss,
                error_type_acc,
                avg_error_step_loss,
                error_step_acc,
                avg_correctness_loss,
                correctness_acc,
            )
            
            logger.info(
                "Epoch %d/%d | VAL   - Total: %.4f Phase: %.4f (%.2f%%) ErrorType: %.4f (%.2f%%) ErrorStep: %.4f (%.2f%%) Correct: %.4f (%.2f%%) | Time: %.1fs",
                epoch + 1,
                EPOCHS,
                val_metrics["total_loss"],
                val_metrics["phase_loss"],
                val_metrics["phase_acc"],
                val_metrics["error_type_loss"],
                val_metrics["error_type_acc"],
                val_metrics["error_step_loss"],
                val_metrics["error_step_acc"],
                val_metrics["correctness_loss"],
                val_metrics["correctness_acc"],
                epoch_time,
            )
            
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
                best_model_path = paths["training"] / "asformer_multitask_best.pth"
                torch.save(model.state_dict(), str(best_model_path))
                logger.info("✓ New best model saved (val_loss: %.4f)", best_val_loss)
            else:
                epochs_no_improve += 1
                logger.info("No improvement for %d epochs (best: %.4f at epoch %d)", 
                           epochs_no_improve, best_val_loss, best_epoch)
                
                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    logger.info("Early stopping triggered after %d epochs without improvement", EARLY_STOP_PATIENCE)
                    break

    # Save final model
    save_path = paths["training"] / "asformer_multitask_final.pth"
    torch.save(model.state_dict(), str(save_path))
    logger.info("Final model saved: %s", save_path)
    logger.info("Best model saved: %s (epoch %d, val_loss: %.4f)", 
                paths["training"] / "asformer_multitask_best.pth", best_epoch, best_val_loss)
    logger.info("TXT log: %s", txt_log_path)
    logger.info("CSV log: %s", csv_log_path)


if __name__ == "__main__":
    main()
