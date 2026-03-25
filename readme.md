# Bachelor Thesis - Reconignizing right inhalation techniques using deep learning.
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

### 4) Thesis-ready outputs (tables + graphs)
- `py src/evaluation/report_thesis.py --asformer_ckpt src/asformer_attention_v1.pth --mstcn_ckpt src/mstcn_v1.pth`

Outputs are saved to `results/thesis_report/`:
- `summary_metrics.csv`
- `per_video_metrics.csv`
- `summary_metrics_bar.png`

Optional subset evaluation (e.g. only `01spravne`):
- `py src/evaluation/report_thesis.py --include_substring 01spravne`

