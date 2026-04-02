# Inhaler Technique Recognition using Deep Learning

**Bachelor Thesis Project** - Multi-task temporal action segmentation for automated assessment of inhaler usage.

---

## Project Overview

This project uses **multi-task learning** with temporal convolutional networks (ASFormer, MS-TCN) to automatically recognize and assess inhaler technique from video recordings. The system:

1. **Extracts skeleton features** from videos using MediaPipe (243D per frame)
2. **Segments phases** (frame-level): REST, PREPARATION, BREATHING, INHALATION, BREATH-HOLD, EXHALATION
3. **Detects errors** (video-level): type of error, which step contains the error, overall correctness

### Why Multi-task Learning?

Instead of rule-based error detection, the model **learns to recognize errors directly from annotations**:
- **Shared representations** for all tasks → better generalization
- **End-to-end learning** → no manual rules needed
- **Robust to variations** → learns patterns that rules miss

---

## Project Structure

```
bakalarka/
├── data/
│   ├── video_metadata.csv           # Annotations: phases + error metadata
│   ├── features_enhanced/           # Extracted 243D skeleton features (.npy)
│   │   ├── 01spravne/              # Correct technique videos
│   │   ├── 07spatne/               # Incorrect technique (various errors)
│   │   └── ...
│   ├── labels/                      # Frame-level phase labels (.txt)
│   ├── raw_videos/                  # Original MP4 videos
│   └── raw_videos_unseen/           # Unseen test set
│
├── src/
│   ├── preprocessing/
│   │   ├── extract_features_enhanced.py  # MediaPipe feature extraction
│   │   └── visualize_features.py         # 3D skeleton visualization
│   │
│   ├── annotation_tools/
│   │   ├── annotate.py                   # Interactive annotation tool
│   │   ├── validate_annotations.py       # Check annotation consistency
│   │   └── backfill_metadata.py          # Add FPS/frame counts to CSV
│   │
│   ├── data_io/
│   │   └── dataset_multitask.py          # Multi-task PyTorch Dataset
│   │
│   ├── models/
│   │   ├── asformer_multitask.py         # Multi-task ASFormer
│   │   ├── mstcn_multitask.py            # Multi-task MS-TCN
│   │   └── registry.py                   # Model factory (build/load)
│   │
│   ├── training/
│   │   ├── train_asformer_multitask.py   # Train ASFormer with validation
│   │   ├── train_mstcn_multitask.py      # Train MS-TCN with validation
│   │   ├── asformer_multitask_best.pth   # Best checkpoint (after training)
│   │   └── mstcn_multitask_best.pth
│   │
│   ├── inference/
│   │   └── predict_multitask.py          # Predict on single video
│   │
│   ├── evaluation/
│   │   └── eval_multitask.py             # Evaluate on full dataset
│   │
│   └── utils/
│       ├── paths.py                      # Centralized path management
│       └── analyze_dataset_stats.py      # Dataset statistics
│
├── results/
│   └── thesis_report/                    # Evaluation outputs
│       ├── summary_metrics.csv
│       ├── per_video_metrics.csv
│       ├── confusion_matrices/
│       └── training_logs/
│
       
├── requirements.md                       # Python dependencies
└── README.md                             # This file
```

---

## Complete Pipeline

### **Step 0: Setup Environment**

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.md
```

---

### **Step 1: Preprocessing & Annotation**

#### 1.1 Extract Skeleton Features from Videos

```bash
python src/preprocessing/extract_features_enhanced.py
```

**What it does:**
- Processes all videos in `data/raw_videos/`
- Extracts 243D skeleton features per frame using MediaPipe:
  - 33 pose landmarks (x, y, z, visibility) = 132D
  - 21 left hand landmarks (x, y, z) = 63D
  - 21 right hand landmarks (x, y, z) = 63D
- Applies normalization (shoulder-center based)
- Computes geometric features (angles, distances)
- Interpolates missing detections
- Applies temporal smoothing
- Saves to `data/features_enhanced/<category>/<video_id>.npy`

**Duration:** ~1-2 hours for 13GB dataset on AMD Ryzen 5 7600X (CPU-based)

---

#### 1.2 Annotate Phase Sequences

```bash
python src/annotation_tools/annotate.py
```

**Interactive tool for annotating phase boundaries:**
- Opens video in OpenCV window
- Mark start/end of each phase (KLID, PRIPRAVA, ROZDYCHANI, INHALACE, ZADRZENI, VYDECH)
- Saves frame-level labels to `data/labels/<category>/<video_id>.txt`
- Auto-updates `data/video_metadata.csv` with metadata

---

#### 1.3 Annotate Error Metadata

**Manually edit `data/video_metadata.csv`:**

```csv
video_id,is_correct,error_type,error_step,error_start_frame,error_end_frame,notes,...
07spatne/video.mp4,0,chybi_zadrzeni,4,120,180,User skipped breath-hold,...
08spatne/video.mp4,0,zadrzeni_otevrena_pusa,4,95,145,Mouth open during hold,...
01spravne/video.mp4,1,,,,,Correct technique,...
```

**Columns:**
- `is_correct`: 0 = incorrect, 1 = correct
- `error_type`: Type of error (e.g., `kratke_zadrzeni`, `chybi_zadrzeni`, `zadrzeni_otevrena_pusa`)
- `error_step`: Which phase has error? (3=INHALACE, 4=ZADRZENI, 5=VYDECH, or "sequence")
- `error_start_frame`, `error_end_frame`: Optional frame range of error

**Supported error types** (see `src/data_io/dataset_multitask.py`):
- `kratke_zadrzeni` - Breath-hold too short (< 3s)
- `chybi_zadrzeni` - Missing breath-hold phase
- `zadrzeni_otevrena_pusa` - Mouth open during breath-hold
- `chybi_inhalace` - Missing inhalation
- `chybi_vydech` - Missing exhalation
- `spatne_poradi` - Wrong phase order
- `malo_vydech` - Insufficient exhalation
- `malo_rozdychani` - Insufficient breathing before inhalation
- `vynechane_rozdychani` - Completely skipped breathing phase
- `other` - Other errors

---

#### 1.4 Validate Annotations

```bash
python src/annotation_tools/validate_annotations.py
```

**Checks:**
- All videos in `features_enhanced/` have corresponding labels in `labels/`
- Feature dimensions match (243D)
- Frame counts are synchronized
- Reports missing or mismatched files

---

### **Step 2: Training**

#### 2.1 Train ASFormer Multi-task Model

```bash
python src/training/train_asformer_multitask.py
```

**What it does:**
- Splits dataset → 80% train, 20% validation (reproducible with seed=42)
- Trains with **4 loss components**:
  - Frame-level phase segmentation (CE loss)
  - Video-level correctness (BCE loss)
  - Video-level error type (CE loss)
  - Video-level error step (CE loss)
- **Early stopping** (patience=10 epochs) when validation loss stops improving
- Saves **best model** (`asformer_multitask_best.pth`) based on validation loss
- Saves **final model** (`asformer_multitask_final.pth`) after all epochs

**Outputs:**
- `src/training/asformer_multitask_best.pth` ← Use this for inference!
- `src/training/asformer_multitask_final.pth`
- `results/thesis_report/training_logs/asformer_multitask_train_metrics.csv`
- Console logs show train/val metrics per epoch

**Hyperparameters:**
- Epochs: 50 (with early stopping)
- Batch size: 4
- Learning rate: 5e-4
- Loss weights: phase=1.0, correctness=0.5, error_type=0.3, error_step=0.3

---

#### 2.2 Train MS-TCN Multi-task Model

```bash
python src/training/train_mstcn_multitask.py
```

**Same as ASFormer**, with MS-TCN architecture (10 stages, 64 features per stage).

---

### **Step 3: Evaluation**

#### 3.1 Evaluate on Full Dataset

```bash
python src/evaluation/eval_multitask.py \
    --model asformer_multitask \
    --checkpoint src/training/asformer_multitask_best.pth \
    --batch-size 4
```

**Metrics computed:**
- **Frame-level**: Phase accuracy, edit score, F1@[10,25,50]
- **Video-level**: 
  - Correctness accuracy
  - Error type: per-class precision/recall/F1 + weighted avg
  - Error step: per-class precision/recall/F1 + weighted avg
- **Confusion matrices** for all classification tasks

**Outputs:**
- `results/thesis_report/summary_metrics.csv` - Overall metrics
- `results/thesis_report/per_video_metrics.csv` - Per-video breakdown
- Console: Detailed per-class metrics + confusion matrices

---

### **Step 4: Inference on New Videos**

#### 4.1 Predict with GUI File Picker

```bash
python src/inference/predict_multitask.py --model asformer_multitask
```
→ Opens file dialog to select `.npy` feature file

---

#### 4.2 Predict with CLI

```bash
python src/inference/predict_multitask.py \
    --model asformer_multitask \
    --checkpoint src/training/asformer_multitask_best.pth \
    --input data/features_enhanced/07spatne/20251208_114006.npy \
    --output-viz prediction.png \
    --output-csv prediction.csv
```

**Output example:**
```
--- MULTI-TASK PREDICTIONS ---
Correctness: INCORRECT (confidence: 0.923)
Error Type: chybi_zadrzeni (confidence: 0.876)
Error Step: 4 (ZADRZENI) (confidence: 0.901)
Phase Avg Confidence: 0.784
Total frames: 241

✓ Visualization saved: prediction.png
✓ Results saved: prediction.csv
```

**Visualization includes:**
- Phase sequence timeline
- Confidence scores per frame
- Error metadata predictions

---

## Utility Scripts

### Analyze Dataset Statistics

```bash
python src/utils/analyze_dataset_stats.py
```

**Generates:**
- Video length statistics (mean, median, std, min, max)
- Per-category breakdown
- Histogram visualization
- Export: `results/thesis_report/video_length_stats.csv`

---

### Visualize 3D Skeleton

```bash
python src/preprocessing/visualize_features.py
```

Interactive 3D visualization of skeleton keypoints from `.npy` features.

---

## Dataset Format

### video_metadata.csv

```csv
video_id,is_correct,error_type,error_step,error_start_frame,error_end_frame,notes,label_file,num_frames,fps
01spravne/video.mp4,1,,,,,,labels/01spravne/video.txt,679,30.002
07spatne/video.mp4,0,chybi_zadrzeni,4,120,180,No breath-hold,labels/07spatne/video.txt,512,30.0
```

### Phase Labels (.txt files)

```
0 0 150    # Frames 0-150: REST (KLID)
1 151 200  # Frames 151-200: PREPARATION (PRIPRAVA)
2 201 250  # Frames 201-250: BREATHING (ROZDYCHANI)
3 251 300  # Frames 251-300: INHALATION (INHALACE)
4 301 400  # Frames 301-400: BREATH-HOLD (ZADRZENI)
5 401 500  # Frames 401-500: EXHALATION (VYDECH)
```

---

## How to Add New Error Types

When you annotate new videos with previously unseen errors:

### Step 1: Add to CSV

```csv
video_id,is_correct,error_type,error_step,...
video.mp4,0,new_error_name,4,...
```

### Step 2: Update ERROR_TYPE_MAPPING

Edit `src/data_io/dataset_multitask.py`:

```python
ERROR_TYPE_MAPPING = {
    "none": 0,
    "kratke_zadrzeni": 1,
    "chybi_zadrzeni": 2,
    "chybi_inhalace": 3,
    "chybi_vydech": 4,
    "spatne_poradi": 5,
    "zadrzeni_otevrena_pusa": 6,
    "malo_vydech": 7,
    "malo_rozdychani": 8,         # krátké/nedostatečné rozdýchání
    "vynechane_rozdychani": 9,    # kompletně vynechal rozdýchání
    "other": 10
}
```

### Step 3: Update Model Config

Edit `src/models/registry.py`:

```python
"asformer_multitask": {
    ...
    "num_error_types": 11,  # Was 10, now 11
    ...
}
```

### Step 4: Retrain from Scratch

```bash
python src/training/train_asformer_multitask.py
```

**Important:** You must retrain the model from scratch when changing the number of output classes. You cannot fine-tune an existing model with new classes.

---

## Model Architectures

### ASFormer Multi-task

**Architecture:**
- **Encoder**: 10 transformer layers (feature dimension: 64)
- **Decoders**: 
  - Phase decoder (frame-level): outputs 6 classes per frame
  - Error classifiers (video-level): 3 separate heads
    - Correctness classifier: binary (correct/incorrect)
    - Error type classifier: 11 classes
    - Error step classifier: 6 classes (none, sequence, 3, 4, 5)

**Features:**
- Self-attention for long-range temporal dependencies
- Masked pooling for video-level features (ignores padding frames)
- Multi-scale temporal convolutions

---

### MS-TCN Multi-task

**Architecture:**
- **Encoder**: Single-stage TCN (64 channels, 10 layers)
- **Refinement**: 10 dilated TCN stages with increasing dilation
- **Decoders**: Same as ASFormer (frame + video-level heads)

**Features:**
- Dilated convolutions for multi-scale temporal context
- Temporal MSE loss for smooth predictions
- Efficient training (faster than ASFormer)

---

## Training Details

### Loss Function

Total loss is a weighted sum of 4 components:

```python
total_loss = (
    1.0 * phase_loss              # Frame-level CE loss
    + 0.5 * correctness_loss      # Video-level BCE loss
    + 0.3 * error_type_loss       # Video-level CE loss
    + 0.3 * error_step_loss       # Video-level CE loss
)
```

### Train/Validation Split

- **80% training** (177 videos)
- **20% validation** (44 videos)
- Reproducible split with `random_seed=42`
- Splits from `video_metadata.csv` to ensure both train/val have correct and incorrect examples

### Early Stopping

- Monitors **validation loss**
- Patience: 10 epochs (stops if no improvement for 10 consecutive epochs)
- Saves **best model** (lowest validation loss)
- Saves **final model** (last epoch)

### Checkpointing

Models saved to `src/training/`:
- `<model_name>_best.pth` - Best validation performance ← **Use this!**
- `<model_name>_final.pth` - Final epoch

---

## Evaluation Metrics

### Frame-level (Phase Segmentation)

- **Accuracy**: Percentage of correctly classified frames
- **Edit Score**: Normalized edit distance (penalizes over-segmentation)
- **F1@k**: F1 score with frame-level tolerance [10, 25, 50] frames

### Video-level (Error Detection)

**Per-task metrics:**

1. **Correctness** (binary):
   - Accuracy
   - Precision, Recall, F1
   - Confusion matrix

2. **Error Type** (9 classes):
   - Per-class precision, recall, F1, support
   - Weighted average F1
   - Confusion matrix

3. **Error Step** (6 classes):
   - Per-class precision, recall, F1, support
   - Weighted average F1
   - Confusion matrix

---

## Feature Extraction Details

### MediaPipe Configuration

- **Model complexity**: 2 (highest accuracy)
- **Tracking**: Enabled (smooth between frames)
- **Min detection/tracking confidence**: 0.5

### Feature Vector (243D per frame)

1. **Pose landmarks (132D)**: 33 keypoints × (x, y, z, visibility)
2. **Left hand (63D)**: 21 keypoints × (x, y, z)
3. **Right hand (63D)**: 21 keypoints × (x, y, z)

### Preprocessing Steps

1. **Normalization**: Shoulder-center reference point
2. **Geometric features**: Angles (elbow, shoulder), distances (hand-to-mouth)
3. **Interpolation**: Linear interpolation for missing detections
4. **Temporal smoothing**: Moving average filter (window=5)

---

## Configuration

### Model Hyperparameters

Edit `src/models/registry.py`:

```python
MODEL_CONFIGS = {
    "asformer_multitask": {
        "n_features": 243,        # Input feature dimension
        "n_classes": 6,           # Phase classes
        "num_error_types": 9,     # Error type classes
        "num_error_steps": 6,     # Error step classes
        "n_layers": 10,           # ASFormer layers
        "feature_dim": 64,        # Hidden dimension
        "num_heads": 8,           # Attention heads
    },
    ...
}
```

### Training Hyperparameters

Edit training scripts directly:

```python
# src/training/train_asformer_multitask.py
epochs = 50
batch_size = 4
lr = 5e-4
patience = 10  # Early stopping patience

# Loss weights
w_phase = 1.0
w_correct = 0.5
w_error_type = 0.3
w_error_step = 0.3
```

---

## Troubleshooting

### Model doesn't load

**Error:** `KeyError: 'asformer_multitask'`
**Fix:** Make sure you're using the registry:
```python
from src.models.registry import build_model, load_model
model = load_model("asformer_multitask", checkpoint_path)
```

---

### Dataset split warnings

**Warning:** `Unknown error_type in video X: typo_error → mapped to 'other'`
**Fix:** Check `video_metadata.csv` for typos in error_type column

---

### Training takes too long

**Options:**
- Reduce batch size (default: 4 → try 2)
- Use MS-TCN instead of ASFormer (faster)
- Reduce number of epochs (50 → 30)
- Check if early stopping is working (should stop before 50 epochs if converged)

---

### Validation loss not improving

**Possible causes:**
- **Overfitting**: Try adding dropout or data augmentation
- **Learning rate too high**: Reduce from 5e-4 to 1e-4
- **Imbalanced data**: Check dataset statistics - might need class weights
- **Insufficient data**: Consider augmentation or simplifying the model

---

## Research Background

### Phase Definitions

| Phase | ID | Czech Name | Description |
|-------|----|-----------|------------------------------------|
| REST | 0 | KLID | Initial rest position |
| PREPARATION | 1 | PRIPRAVA | Shake/prepare inhaler |
| BREATHING | 2 | ROZDYCHANI | Exhale before inhalation |
| INHALATION | 3 | INHALACE | Inhale medication |
| BREATH-HOLD | 4 | ZADRZENI | Hold breath (~5-10 sec) |
| EXHALATION | 5 | VYDECH | Exhale slowly |

### Common Errors

| Error Type | Czech Name | Description |
|-----------|-----------|------------------------------------------------|
| Short breath-hold | kratke_zadrzeni | Hold duration < 4.5 seconds |
| Missing breath-hold | chybi_zadrzeni | Phase 4 (ZADRZENI) not performed |
| Mouth open | zadrzeni_otevrena_pusa | Lips not sealed during hold |
| Wrong position | spatna_pozice | Inhaler not at mouth |
| Slow inhalation | inhalace_pomala | Inhalation too slow |
| Fast inhalation | inhalace_rychla | Inhalation too fast |
| Insufficient breathing | malo_rozdychani | Short/shallow breathing before inhalation |
| Skipped breathing | vynechane_rozdychani | Completely omitted breathing phase |

---

## Dependencies

See `requirements.md` for full list. Key libraries:

- PyTorch (CUDA 11.8)
- MediaPipe (pose detection)
- OpenCV (video processing)
- NumPy, Pandas (data handling)
- Matplotlib (visualization)

---

## Notes

### Phase Order Flexibility

**Current behavior:** The system treats **PREPARATION (1)** and **BREATHING (2)** as interchangeable phases. Some users may exhale before shaking the inhaler, which is medically acceptable.

- If swapped order should be **correct**: Mark `is_correct=1` in CSV
- If swapped order should be **an error**: Add `error_type="prohozeni_fazi"` and update mappings

### Partial Errors

When a video has an error in only **one phase** (e.g., phase 4):
- **Frame-level model** still learns correct behavior from phases 0,1,2,3,5
- **Video-level metadata** tells the model "this video has error type X in step 4"
- This is correct behavior! Model learns both correct and incorrect patterns.

---

## Acknowledgments

**Models:**
- **ASFormer**: "ASFormer: Transformer for Action Segmentation" (Fangqiu Yi et al., 2021)
- **MS-TCN**: "MS-TCN: Multi-Stage Temporal Convolutional Network" (Yazan Abu Farha et al., 2019)

**Pose Estimation:**
- **MediaPipe Holistic**: Google's pose/hand/face detection framework

---

## Contact

For questions about this project, contact the author or refer to `bakalarka.md` for full thesis text.

## How to set up project
### Creating venv
- python -m venv venv
### Activating venv
- venv\Scripts\activate
### Install libraries
- **run:** pip install opencv-python pandas mediapipe numpy matplotlib torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
- **or:** install dependencies from requirements.md

## How does the whole pipeline work?
### First steps, aquiring dataset
- First thing you need to do is to aquire your much needed raw videos of the activity you actually want "deep learn".
### Extracting skeleton features using MediaPipe 
- In this project instead of processing RGB videos, we're extracting skeleton of those using mediapipe.
- This can be done by **running:** python src/extract_features.py
- This might take a lot of time depending on the size of your dataset and the efficiency of your hardware.
- This process is done via **cpu** and for example my dataset that has around 13gb of videos took around 1 hour and 30 minutes on AMD Ryzen 5 7600X
## Current Recommended Workflow (BP v3)
Run all commands from project root:

### 0) Preprocessing and annotation
- `py src/preprocessing/extract_features_enhanced.py`
- `py src/preprocessing/normalize_features.py`
- `py src/preprocessing/visualize_features.py`
- `py src/annotation_tools/annotate.py`
- `py src/annotation_tools/backfill_metadata.py`
- `py src/annotation_tools/validate_annotations.py`

### 1) Train models
- `py src/training/train_asformer.py`
- `py src/training/train_mstcn.py`

### 2) Compare models (quick)
- `py src/evaluation/eval_compare_models.py --asformer_ckpt src/asformer_attention_v1.pth --mstcn_ckpt src/mstcn_v1.pth`

### 3) Visual prediction check
- `py src/inference/predict_unified.py --model asformer --ckpt src/asformer_attention_v1.pth`
- `py src/inference/predict_unified.py --model mstcn --ckpt src/mstcn_v1.pth`
#### Compare both models on same video
- `py src/inference/predict_unified.py --model both --asformer-ckpt src/asformer_attention_v1.pth --mstcn-ckpt src/mstcn_v1.pth --input data/features_enhanced/01spravne/NECO.npy`
- Optional no-plot mode: add `--no-plot`

Logic checker in `predict_unified.py`:
- validates step order with tolerance that `PRIPRAVA (1)` and `ROZDEJCHANI (2)` can be mixed
- validates average breath-hold (`ZADRZENI = class 4`) with threshold `>= 4.5s`
- threshold can be changed by `--min-breath-hold-sec`

### 4) Thesis-ready outputs (tables + graphs)
- `py src/evaluation/report_thesis.py --asformer_ckpt src/asformer_attention_v1.pth --mstcn_ckpt src/mstcn_v1.pth`


- `py analyze_dataset_stats.py --video_root data/raw_videos --output_png results/thesis_report/video_length_histogram.png --output_csv results/thesis_report/video_length_stats.csv --bins 20`

Outputs are saved to `results/thesis_report/`:
- `summary_metrics.csv`
- `per_video_metrics.csv`
- `summary_metrics_bar.png`

Optional subset evaluation (e.g. only `01spravne`):
- `py src/evaluation/report_thesis.py --include_substring 01spravne`

### 5) Unseen raw videos -> automatic BP report (no manual console review)
1. Put new videos (not used in training) into a separate folder, e.g. `data/raw_videos_unseen/`.
2. Run:
- `py src/evaluation/report_unseen_raw.py --raw_dir data/raw_videos_unseen --asformer_ckpt src/training/asformer_attention_v1.pth --mstcn_ckpt src/training/mstcn_v1.pth`
3. Outputs are saved into a timestamped folder under:
- `results/thesis_report/unseen_report/`

Generated outputs include:
- `unseen_per_video_predictions.csv` (per-video diagnosis for both models)
- `unseen_model_consensus.csv` (agreement/disagreement ASFormer vs MS-TCN)
- `unseen_summary_by_model.csv` (counts/percentages correct vs wrong)
- `unseen_predicted_correctness.png`
- `unseen_error_type_distribution.png`
- `unseen_model_agreement.png`

## Annotation standard for wrong videos (important)
When annotating wrong procedures, always fill metadata so each error is tied to a concrete step/class.

Metadata fields in `data/video_metadata.csv`:
- `is_correct` (`1` correct, `0` wrong)
- `error_type` (e.g. `kratke_zadrzeni`, `malo_vydech`, `spatne_poradi`)
- `error_step` (phase id `0-5` or `sequence`)
- `error_start_frame` (optional)
- `error_end_frame` (optional)

Recommended mapping:
- `kratke_zadrzeni` -> `error_step=4`
- `malo_vydech` -> `error_step=5`
- `spatne_poradi` -> `error_step=sequence`

For `07spatne` (very short breath-hold):
- set `is_correct=0`
- set `error_type=kratke_zadrzeni`
- set `error_step=4`
- optionally fill `error_start_frame/error_end_frame` for better future analysis

- py src/evaluation/report_unseen_raw.py --raw_dir data/raw_videos_unseen --asformer_ckpt src/training/asformer_attention_v1.pth --mstcn_ckpt src/training/mstcn_v1.pth

