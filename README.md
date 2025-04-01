# Mental Health Support Chatbot

[English](README.md) | [简体中文](README_zh_CN.md) | [香港繁體中文](README_zh_HK.md)

An AI-based conversational system designed to provide mental health support through emotion detection and responsive dialogue.

## Core Features

- **Dual-mode Emotion Analysis**: Combines facial expression recognition and text sentiment analysis
- **Real-time Facial Emotion Detection**: Uses computer vision to detect emotions from facial expressions
- **Text-based Emotion Analysis**: Analyzes emotional content in user messages
- **Crisis Indicator Detection**: Identifies potential mental health crisis signals
- **Conversation History Management**: Maintains context for more relevant responses
- **Two Operating Modes**:
  - User Mode: Simplified interface for regular users
  - Developer Mode: Advanced interface with real-time metrics and debugging information

## Technical Implementation

### 1. Emotion Recognition Model (EmotionDetector)

- **Deep Learning Architecture**: ResNet-18 based convolutional neural network
  - Modified ResNet-18 model, adapted for grayscale image input (single channel)
  - Output layer adjusted for 7-class emotion classification (happy, neutral, sad, surprise, angry, disgust, fear)
  
- **Dataset**: 
  - Trained on the FER-2013 (Facial Expression Recognition 2013) dataset
  - Contains 35,887 grayscale images of facial expressions categorized into 7 emotions
  - Images are 48x48 pixels in size

- **Model Training and Optimization**:
  - Implemented using PyTorch framework
  - Supports CPU and CUDA acceleration (automatically detects available devices)
  - Model weights stored in `models/best_model.pth`

- **Face Detection**:
  - Uses OpenCV's Haar cascade classifier for face detection
  - Real-time video frame processing and emotion probability visualization

### 2. Language Processing System (LanguageProcessor)

- **Text Emotion Analysis Algorithm**:
  - Based on keyword matching and sentiment intensity analysis
  - Sentiment intensity assessment based on exclamation marks, capitalization, letter repetition, and intensity vocabulary
  
- **Crisis Indicator Extraction**:
  - Identifies potential crisis situations based on predefined keywords
  - Supports multiple crisis category recognition (self-harm, violence, emergencies, etc.)

### 3. Response Generation System (ResponseGenerator)

- **Context-aware Responses**:
  - Generates personalized responses based on current emotional state and conversation history
  - Emotion change detection and corresponding response adjustment
  
- **Crisis Response Mechanism**:
  - Priority-based crisis response strategies
  - Integration of support resources and helpline information

### 4. Multimodal Emotion Fusion

- **Emotion State Verification Algorithm**:
  - Combines visual and textual emotion analysis results
  - Confidence-based weight allocation
  - Non-neutral emotion priority strategy

## System Requirements

- Python 3.9
- CUDA 11.6
- torch 11.2+cu116
- torchaudio 0.12.0+cu116
- torchvision 0.13.0+cu116
- OpenCV 4.7.0+

## Installation Guide

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```
2. Install required packages:
```bash
pip install -r requirements.txt
 ```
```
3. Download dataset (for model training):
   
   - The pre-trained model weights are included in models/best_model.pth
   - If you want to retrain the model, you need to download the FER-2013 dataset
   - Download from Kaggle FER-2013
   - Extract the dataset to the archive folder with the following structure:
     ```plaintext
     archive/
     ├── train/
     │   ├── angry/
     │   ├── disgust/
     │   ├── fear/
     │   ├── happy/
     │   ├── neutral/
     │   ├── sad/
     │   └── surprise/
     └── test/
         ├── angry/
         ├── disgust/
         ├── fear/
         ├── happy/
         ├── neutral/
         ├── sad/
         └── surprise/
      ```
4. Run the application:

```bash
python main.py
 ```