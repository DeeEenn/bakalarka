"""
Multi-task dataset for inhaler technique recognition.
Returns both frame-level phase labels and video-level error metadata.
"""
import csv
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# Error type to index mapping
ERROR_TYPE_TO_IDX = {
    "none": 0,
    "kratke_zadrzeni": 1,
    "chybi_zadrzeni": 2,
    "chybi_inhalace": 3,
    "chybi_vydech": 4,
    "spatne_poradi": 5,
    "zadrzeni_otevrena_pusa": 6,
    "malo_vydech": 7,
    "other": 8,
}

# Error step to index mapping (which phase has error)
ERROR_STEP_TO_IDX = {
    "none": 0,
    "sequence": 1,  # poradi kroku
    "3": 2,  # INHALACE
    "4": 3,  # ZADRZENI
    "5": 4,  # VYDECH
}


class InhalerDatasetMultitask(Dataset):
    """
    Extended dataset that loads:
    - Features (243D skeleton features per frame)
    - Phase labels (frame-level, 0-5)
    - Error metadata (video-level: is_correct, error_type, error_step)
    
    Args:
        features_dir: Path to features_enhanced directory
        labels_dir: Path to labels directory
        metadata_csv: Path to video_metadata.csv
        max_len: Maximum sequence length (padding/truncation)
        mode: Dataset split mode - "train", "val", or "all" (default: "all")
        train_ratio: Ratio of data for training if mode is train/val (default: 0.8)
        random_seed: Random seed for reproducible splits (default: 42)
    """

    def __init__(self, features_dir, labels_dir, metadata_csv, max_len=2000, 
                 mode="all", train_ratio=0.8, random_seed=42):
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.max_len = max_len
        self.mode = mode
        
        # Load metadata CSV into memory
        self.metadata = self._load_metadata(metadata_csv)
        
        # Build list of samples (only those with labels AND metadata)
        all_samples = self._get_data_list()
        
        # Apply train/val split if needed
        if mode in ["train", "val"]:
            import random
            random.seed(random_seed)
            shuffled = all_samples.copy()
            random.shuffle(shuffled)
            
            split_idx = int(len(shuffled) * train_ratio)
            if mode == "train":
                self.data_list = shuffled[:split_idx]
            else:  # mode == "val"
                self.data_list = shuffled[split_idx:]
        else:
            self.data_list = all_samples
        
        # Print dataset statistics
        self._print_statistics()

    def _load_metadata(self, metadata_csv):
        """Parse video_metadata.csv into dict keyed by label_file path."""
        metadata = {}
        unknown_error_types = []
        unknown_error_steps = []
        
        if not os.path.exists(metadata_csv):
            print(f"⚠ Warning: metadata CSV not found: {metadata_csv}")
            return metadata
        
        with open(metadata_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_file = row.get("label_file", "").replace("\\", "/")
                if not label_file:
                    continue
                
                # Parse metadata fields
                is_correct = int(row.get("is_correct", 1))
                error_type = row.get("error_type", "").strip()
                if error_type == "" or is_correct == 1:
                    error_type = "none"
                
                error_step = row.get("error_step", "").strip()
                if error_step == "" or is_correct == 1:
                    error_step = "none"
                
                # Map to indices with validation
                if error_type not in ERROR_TYPE_TO_IDX:
                    unknown_error_types.append(error_type)
                    error_type_idx = ERROR_TYPE_TO_IDX["other"]  # Fallback
                else:
                    error_type_idx = ERROR_TYPE_TO_IDX[error_type]
                
                if error_step not in ERROR_STEP_TO_IDX:
                    unknown_error_steps.append(error_step)
                    error_step_idx = ERROR_STEP_TO_IDX["none"]  # Fallback
                else:
                    error_step_idx = ERROR_STEP_TO_IDX[error_step]
                
                metadata[label_file] = {
                    "is_correct": is_correct,
                    "error_type": error_type,
                    "error_type_idx": error_type_idx,
                    "error_step": error_step,
                    "error_step_idx": error_step_idx,
                }
        
        # Report unknown values
        if unknown_error_types:
            unique_unknown_types = set(unknown_error_types)
            print(f"⚠ Warning: {len(unknown_error_types)} unknown error_type values found (mapped to 'other'): {unique_unknown_types}")
        
        if unknown_error_steps:
            unique_unknown_steps = set(unknown_error_steps)
            print(f"⚠ Warning: {len(unknown_error_steps)} unknown error_step values found (mapped to 'none'): {unique_unknown_steps}")
        
        return metadata

    def _get_data_list(self):
        """Build list of (feature_path, label_path, metadata) tuples."""
        samples = []
        
        for root, _, files in os.walk(self.features_dir):
            for file in files:
                if not file.endswith(".npy"):
                    continue
                
                # Get corresponding label file
                rel_path = os.path.relpath(os.path.join(root, file), self.features_dir)
                label_file_path = os.path.join(self.labels_dir, rel_path.replace(".npy", ".txt"))
                
                if not os.path.exists(label_file_path):
                    continue
                
                # Construct metadata key (relative path from data/)
                # e.g. "labels/01spravne/video.txt"
                label_key = str(Path("labels") / Path(rel_path).with_suffix(".txt")).replace("\\", "/")
                
                # Get metadata (if missing, assume correct)
                meta = self.metadata.get(label_key, {
                    "is_correct": 1,
                    "error_type": "none",
                    "error_type_idx": 0,
                    "error_step": "none",
                    "error_step_idx": 0,
                })
                
                samples.append((
                    os.path.join(root, file),  # feature path
                    label_file_path,            # label path
                    meta                        # metadata dict
                ))
        
        return samples

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        feature_path, label_path, meta = self.data_list[idx]

        # Load features and phase labels (same as before)
        features = np.load(feature_path).T  # (C, T)
        labels = np.loadtxt(label_path, dtype=np.int64)  # (T,)

        t_steps = features.shape[1]

        # Pad or truncate to max_len
        if t_steps < self.max_len:
            pad_width = self.max_len - t_steps
            features = np.pad(features, ((0, 0), (0, pad_width)), mode="constant")
            labels = np.pad(labels, (0, pad_width), mode="constant", constant_values=-100)
        else:
            features = features[:, : self.max_len]
            labels = labels[: self.max_len]

        t_eff = min(t_steps, self.max_len)

        # Convert to tensors
        features_t = torch.from_numpy(features).float()
        labels_t = torch.from_numpy(labels).long()
        
        # Error metadata (video-level scalars)
        is_correct_t = torch.tensor(meta["is_correct"], dtype=torch.long)
        error_type_t = torch.tensor(meta["error_type_idx"], dtype=torch.long)
        error_step_t = torch.tensor(meta["error_step_idx"], dtype=torch.long)

        return {
            "features": features_t,
            "phase_labels": labels_t,
            "is_correct": is_correct_t,
            "error_type": error_type_t,
            "error_step": error_step_t,
            "t_eff": t_eff,
        }
    
    def _print_statistics(self):
        """Print dataset statistics."""
        if len(self.data_list) == 0:
            print(f"⚠ Dataset ({self.mode}): 0 samples found!")
            return
        
        # Count by correctness
        correct_count = sum(1 for _, _, meta in self.data_list if meta["is_correct"] == 1)
        incorrect_count = len(self.data_list) - correct_count
        
        # Count by error type
        error_type_counts = {}
        for _, _, meta in self.data_list:
            et = meta["error_type"]
            error_type_counts[et] = error_type_counts.get(et, 0) + 1
        
        # Count by error step
        error_step_counts = {}
        for _, _, meta in self.data_list:
            es = meta["error_step"]
            error_step_counts[es] = error_step_counts.get(es, 0) + 1
        
        print(f"\n{'='*60}")
        print(f"Dataset Statistics ({self.mode.upper()})")
        print(f"{'='*60}")
        print(f"Total samples: {len(self.data_list)}")
        print(f"  ✓ Correct:   {correct_count:3d} ({100*correct_count/len(self.data_list):.1f}%)")
        print(f"  ✗ Incorrect: {incorrect_count:3d} ({100*incorrect_count/len(self.data_list):.1f}%)")
        
        if incorrect_count > 0:
            print(f"\nError type distribution:")
            for et, count in sorted(error_type_counts.items(), key=lambda x: -x[1]):
                if et != "none":
                    print(f"  - {et:30s}: {count:3d}")
            
            print(f"\nError step distribution:")
            for es, count in sorted(error_step_counts.items(), key=lambda x: -x[1]):
                if es != "none":
                    print(f"  - {es:30s}: {count:3d}")
        
        print(f"{'='*60}\n")


# Utility functions for reverse mapping
def idx_to_error_type(idx):
    """Convert error_type index back to string."""
    idx_to_name = {v: k for k, v in ERROR_TYPE_TO_IDX.items()}
    return idx_to_name.get(idx, "unknown")


def idx_to_error_step(idx):
    """Convert error_step index back to string."""
    idx_to_name = {v: k for k, v in ERROR_STEP_TO_IDX.items()}
    return idx_to_name.get(idx, "unknown")
