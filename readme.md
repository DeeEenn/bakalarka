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

