
# Face MediaPipe Analysis

A real-time face landmark detection and classification system using MediaPipe and OpenCV.

## Overview

This project captures video from your camera in real-time, detects facial landmarks using MediaPipe, and classifies them with a trained MLP model.

## Requirements

- Python 3.12
- No GPU required

## Installation

1. Clone the repository
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Collection
```bash
python app.py
```
- A window will open showing your camera feed
- Press **S** to enter save mode
- Press **0-9** to save current landmarks with that label
- Landmarks are saved to `face_landmarks.csv`
- Press **Q** to quit

### Model Training
- Open and run all cells in `train_model.ipynb`

### Evaluation
```bash
python app.py
```
- Press **E** to enter evaluate mode
- Labels appear on screen (mapped from `face_labels.csv`)

## Output Files

- `face_landmarks.csv` - Saved landmark data with labels
- `face_labels.csv` - Label mappings (0-9)