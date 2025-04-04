# 🎭 Emotion Detection Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0-green.svg)](https://opencv.org/)
[![DeepFace](https://img.shields.io/badge/DeepFace-0.0.93-orange.svg)](https://github.com/serengil/deepface)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A comprehensive emotion analysis toolkit featuring real-time facial emotion detection, time series analysis, and stock prediction capabilities.

## 📋 Overview

This repository contains multiple implementations of emotion detection and analysis using different approaches. The main focus is on facial emotion recognition using OpenCV and DeepFace, along with additional implementations for stock prediction and time series analysis.

## 🗂️ Project Structure

| Component | Description |
|-----------|-------------|
| `emotion.py` | 🎥 Main implementation of real-time facial emotion detection using OpenCV and DeepFace |
| `TTC.py`, `TTC-NEW.py`, `TTC_tensor.py` | 📊 Various implementations of time series analysis and prediction |
| `ttc2.py`, `ttc_INDI.py` | 🔄 Alternative time series implementations |
| `LTSM_StockPredict.py` | 📈 Stock price prediction using LSTM neural networks |
| `testing.py` | 🔍 Testing and validation scripts |
| `model_log.txt` | 📝 Log file containing model training and evaluation metrics |

## ⚙️ Dependencies

The project requires the following main dependencies:

```bash
opencv-python~=4.10.0.84  # Computer vision operations
deepface~=0.0.93         # Deep learning facial analysis
tf_keras                 # TensorFlow backend
moviepy~=2.1.1          # Video processing
```

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone [your-repository-url]
   cd Emotion_Detection
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### 🎭 Real-time Emotion Detection

To run the real-time emotion detection:

1. Ensure you have a webcam connected
2. Run the emotion detection script:
   ```bash
   python emotion.py
   ```
3. Press 'q' to quit the application

**Features:**
- 📹 Real-time video capture
- 👤 Face detection
- 😊 Emotion analysis
- 📦 Bounding box visualization

### 📊 Time Series Analysis

Multiple implementations available:
- `TTC.py` - Basic time series analysis
- `TTC-NEW.py` - Enhanced version with additional features
- `TTC_tensor.py` - TensorFlow-based implementation
- `ttc2.py` and `ttc_INDI.py` - Alternative implementations

### 📈 Stock Prediction

Run the stock prediction model:
```bash
python LTSM_StockPredict.py
```

## ✨ Key Features

- 🎥 Real-time facial emotion detection
- 😊 Multiple emotion categories (happy, sad, angry, etc.)
- 👤 Face detection using Haar Cascade Classifier
- 🧠 Deep learning-based emotion analysis
- 📊 Time series analysis implementations
- 📈 Stock price prediction using LSTM

## 🤝 Contributing

Contributions are welcome! Feel free to:
- 🐛 Report bugs
- 💡 Suggest enhancements
- 🔧 Submit pull requests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
<div align="center">
  <sub>Built with ❤️ by <a href="https://github.com/yourusername">Your Name</a></sub>
</div>