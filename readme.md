# venv creation
- python -m venv venv

# activation
- venv\Scripts\activate

# install libraries
- pip install opencv-python mediapipe numpy matplotlib torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# run features extraction 
- python src/extract_features.py