# Distracted Driver Detection Using Deep Learning

This repository provides a comprehensive solution for detecting distracted driving behaviors using deep learning. The project focuses on three primary models: a **Custom CNN**, a **Hybrid CNN+ViT+BiLSTM**, and a **Vision Transformer (ViT)**. Each model is designed to address specific challenges in image classification, and their performance is evaluated using various metrics. Additionally, Grad-CAM heatmaps are employed for explainability.

---

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
  - [Custom CNN Architecture](#custom-cnn-architecture)
  - [Hybrid CNN+ViT+BiLSTM Model](#hybrid-cnnvitbilstm-model)
  - [Vision Transformer (ViT) Model](#vision-transformer-vit-model)
- [Explainability with Grad-CAM](#explainability-with-grad-cam)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [How to Use](#how-to-use)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

---

## Introduction

Distracted driving is a significant cause of road accidents globally. This project aims to classify driver behaviors into ten categories, such as safe driving, texting, talking on the phone, and more. By leveraging state-of-the-art deep learning models, we provide a robust solution for detecting these behaviors from images.

---

## Dataset

The dataset used is the **State Farm Distracted Driver Detection Dataset**, which contains labeled images of drivers performing various activities. The dataset is structured into ten classes:
- **c0**: Safe driving
- **c1**: Texting - right
- **c2**: Talking on the phone - right
- **c3**: Texting - left
- **c4**: Talking on the phone - left
- **c5**: Operating the radio
- **c6**: Drinking
- **c7**: Reaching behind
- **c8**: Hair and makeup
- **c9**: Talking to passenger

---

## Models

### Custom CNN Architecture

#### Overview
The Custom CNN model was designed from scratch to efficiently classify images into ten categories. It is lightweight and optimized for real-time applications.

#### Architecture
1. **Input Layer**: Accepts images resized to `(180, 180, 3)`.
2. **Convolutional Layers**:
   - Three convolutional layers with kernel sizes `(3, 3)`.
   - Activation Function: **ReLU** (Rectified Linear Unit) to introduce non-linearity and prevent vanishing gradients.
3. **Pooling Layers**:
   - MaxPooling layers to reduce spatial dimensions and computational complexity.
4. **Fully Connected Layers**:
   - Dense layer with 128 neurons and **ReLU** activation.
   - Output layer with 10 neurons and **softmax** activation for multi-class classification.

#### Design Rationale
- **ReLU Activation**: Chosen for its simplicity and efficiency in training deep networks.
- **Softmax Output**: Ensures probabilities sum to 1, making it suitable for multi-class classification.
- **MaxPooling**: Reduces overfitting by down-sampling feature maps.

#### Metrics
- **Training Accuracy**: 85.4%
- **Validation Accuracy**: 83.7%
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam with a learning rate of `1e-3`.

---

### Hybrid CNN+ViT+BiLSTM Model

#### Overview
This hybrid model combines the strengths of CNNs, Vision Transformers (ViTs), and Bidirectional Long Short-Term Memory (BiLSTM) networks. It captures both spatial and temporal features for improved classification.

#### Architecture
1. **CNN Block**:
   - Two convolutional layers with **ReLU** activation and MaxPooling.
   - Batch normalization for regularization.
2. **Transformer Block**:
   - Multi-head self-attention with 4 heads.
   - Dense layers for feed-forward processing.
3. **BiLSTM Block**:
   - Bidirectional LSTM with 64 units to capture temporal dependencies.
4. **Fusion**:
   - Concatenation of outputs from the Transformer and BiLSTM blocks.
   - Global Average Pooling followed by dense layers.
5. **Output Layer**:
   - Dense layer with 10 neurons and **softmax** activation.

#### Design Rationale
- **CNN**: Extracts spatial features from images.
- **ViT**: Captures global dependencies using self-attention.
- **BiLSTM**: Models temporal relationships in sequential data.
- **Dropout**: Prevents overfitting by randomly deactivating neurons during training.

#### Metrics
- **Training Accuracy**: 91.2%
- **Validation Accuracy**: 89.8%
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam with a learning rate of `1e-3`.

---

### Vision Transformer (ViT) Model

#### Overview
The ViT model leverages the power of transformers for image classification. It divides images into patches and processes them as sequences, similar to natural language processing tasks.

#### Architecture
1. **Patch Extraction**:
   - Images are divided into patches of size `(8, 8)`.
2. **Patch Embedding**:
   - Dense layer to project patches into a higher-dimensional space.
   - Positional embeddings added to retain spatial information.
3. **Transformer Encoder**:
   - Two transformer blocks with multi-head self-attention and feed-forward layers.
4. **Classification Head**:
   - Global Average Pooling followed by dense layers.
   - Output layer with 10 neurons and **softmax** activation.

#### Design Rationale
- **Attention Mechanism**: Captures global dependencies in the image.
- **Positional Embeddings**: Retain spatial information lost during patch extraction.
- **Softmax Output**: Ensures probabilities sum to 1.

#### Metrics
- **Training Accuracy**: 89.7%
- **Validation Accuracy**: 88.5%
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam with a learning rate of `1e-4`.

---

## Explainability with Grad-CAM

To ensure transparency and interpretability, Grad-CAM (Gradient-weighted Class Activation Mapping) was used to visualize the regions of the image that influenced the model's predictions.

### Steps
1. **Gradient Computation**:
   - Gradients of the target class are computed with respect to the feature maps of the last convolutional layer.
2. **Heatmap Generation**:
   - Feature maps are weighted by the gradients and combined to create a heatmap.
3. **Overlay**:
   - The heatmap is superimposed on the original image for visualization.

### Tools
- **Matplotlib**: For plotting heatmaps.
- **OpenCV**: For resizing and overlaying heatmaps.

### Results
- Provided insights into the model's focus areas.
- Helped identify potential biases or errors in predictions.

---

## Training and Evaluation

### Training
- **Batch Size**: 32
- **Epochs**: 30
- **Data Augmentation**:
  - Horizontal flip, rotation, zoom, and shear transformations.

### Evaluation Metrics
- Accuracy
- Precision, Recall, and F1-Score
- Confusion Matrix

---

## Results

| Model                  | Training Accuracy | Validation Accuracy | Precision | Recall | F1-Score |
|------------------------|-------------------|---------------------|-----------|--------|----------|
| Custom CNN             | 85.4%            | 83.7%              | 84.7%     | 85.2%  | 84.9%    |
| Hybrid CNN+ViT+BiLSTM  | 91.2%            | 89.8%              | 90.8%     | 91.0%  | 90.9%    |
| Vision Transformer (ViT) | 89.7%          | 88.5%              | 89.2%     | 89.5%  | 89.3%    |

---

## Future Work

- Explore lightweight models for deployment on edge devices.
- Incorporate real-time video analysis.
- Extend the dataset to include more diverse driving scenarios.

---

## Acknowledgments

- **State Farm** for providing the dataset.
- **TensorFlow** and **Keras** for the deep learning frameworks.
- **OpenCV** and **Matplotlib** for visualization tools.

---
