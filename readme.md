# Bachelor Thesis - Reconignizing right inhalation techniques using deep learning.
## How to set up project
### Creating venv
- python -m venv venv
### Activating venv
- venv\Scripts\activate
- 
# Install libraries
- **run:** pip install opencv-python mediapipe numpy matplotlib torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
- **or:** install dependencies from requirements.md

## How does the whole pipeline work?
### First steps, aquiring dataset
- First thing you need to do is to aquire your much needed raw videos of the activity you actually want "deep learn".
### Extracting skeleton features using MediaPipe 
- In this project instead of processing RGB videos, we're extracting skeleton of those using mediapipe.
- This can be done by **running:** python src/extract_features.py
- This might take a lot of time depending on the size of your dataset and the efficiency of your hardware.
- This process is done via **cpu** and for example my dataset that has around 13gb of videos took around 1 hour and 30 minutes on AMD Ryzen 5 7600X
