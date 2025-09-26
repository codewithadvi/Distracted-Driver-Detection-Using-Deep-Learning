# Distracted Driver Detection Using Deep Learning

This repository provides a comprehensive solution for detecting distracted driving behaviors using deep learning. The project focuses on three primary models: a **Custom CNN**, a **Hybrid CNN+ViT+BiLSTM**, and a **Vision Transformer (ViT)**. Additionally, an **Ensemble Model (CNN+Transformer+BiLSTM)** is implemented to combine the strengths of these architectures. Each model is evaluated using various metrics, including per-class precision, recall, and F1-score. Grad-CAM heatmaps and t-SNE visualizations are employed for explainability and feature analysis.

---

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
  - [Custom CNN Architecture](#custom-cnn-architecture)
  - [Hybrid CNN+ViT+BiLSTM Model](#hybrid-cnnvitbilstm-model)
  - [Vision Transformer (ViT) Model](#vision-transformer-vit-model)
  - [Ensemble Model (CNN+Transformer+BiLSTM)](#ensemble-model-cnntransformerbilstm)
- [Explainability and Interpretability](#explainability-and-interpretability)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
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

### Class-Wise Distribution
The dataset is imbalanced, with some classes having more samples than others. To address this, we used data augmentation techniques to balance the dataset and improve model generalization.

![image](https://github.com/user-attachments/assets/e585b371-3cc5-4936-bc9b-bd2c3246f5cb)

![image](https://github.com/user-attachments/assets/806c08ab-2991-4ed6-b049-b4308af7deab)

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

#### Data Augmentation
To improve the model's generalization and prevent overfitting, we used extensive data augmentation techniques with `ImageDataGenerator`:
- **Horizontal Flip**
- **Vertical Flip**
- **Rotation**
- **Zoom**
- **Shear**
- **Brightness Adjustment**

#### Metrics
- **Training Accuracy**: 97.1%
- **Validation Accuracy**: 97.2%
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam with a learning rate of `1e-3`.

 <img width="328" alt="image" src="https://github.com/user-attachments/assets/ee300a0b-9f18-46a4-ae7c-eb10456651fc" />     <img width="327" alt="image" src="https://github.com/user-attachments/assets/9b70f187-512e-45d3-8a8f-1479fd890eb8" />


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

#### Metrics
- **Training Accuracy**: 91.2%
- **Validation Accuracy**: 89.8%
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam with a learning rate of `1e-3`.

---

### Vision Transformer (ViT) Model

#### Overview
The Vision Transformer (ViT) model leverages the power of transformers for image classification. It divides images into patches and processes them as sequences, similar to natural language processing tasks.

#### Architecture
1. **Patch Layer**:
   - Divides the input image into smaller patches of size `(8, 8)`.
   - Each patch is flattened into a 1D vector.
2. **Patch Encoder**:
   - Projects patches into a higher-dimensional space using a dense layer.
   - Adds positional embeddings to retain spatial information.
3. **Transformer Block**:
   - **Multi-Head Self-Attention**: Captures global dependencies by computing attention scores between all patches.
   - **Feed-Forward Network**: Fully connected layers applied to each token independently.
   - **Skip Connections**: Stabilizes training and prevents vanishing gradients.
4. **Classification Head**:
   - Global Average Pooling followed by dense layers.
   - Output layer with 10 neurons and **softmax** activation.
  
     ![image](https://github.com/user-attachments/assets/e00deaec-9ea5-4d11-b527-265bf204ab0e)


#### Metrics
- **Training Accuracy**: 98.0%
- **Validation Accuracy**: 98.0%
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam with a learning rate of `1e-4`.

![image](https://github.com/user-attachments/assets/096a847f-180a-449b-bdc2-3a7c22159309)    ![image](https://github.com/user-attachments/assets/d6d87b0e-c6af-4b14-a96f-44a5528b2366)


---

### Ensemble Model (CNN+Transformer+BiLSTM)

#### Overview
The ensemble model combines the strengths of CNNs, Transformers, and BiLSTMs to achieve robust performance. It integrates spatial, temporal, and global features for improved classification.

#### Data Augmentation
We used the same data augmentation techniques as the other models:
- **Horizontal Flip**
- **Vertical Flip**
- **Rotation**
- **Zoom**
- **Shear**
- **Brightness Adjustment**

#### Architecture
1. **CNN Block**:
   - Extracts spatial features from the input image.
2. **Transformer Block**:
   - Captures global dependencies using multi-head self-attention.
3. **BiLSTM Block**:
   - Models temporal relationships in sequential data.
4. **Fusion**:
   - Outputs from the CNN, Transformer, and BiLSTM blocks are concatenated.
   - Global Average Pooling followed by dense layers.
5. **Output Layer**:
   - Dense layer with 10 neurons and **softmax** activation.

#### Metrics
- **Training Accuracy**: 93.0%
- **Validation Accuracy**: 93.0%
  ![image](https://github.com/user-attachments/assets/41e8ea2a-77ec-4717-b022-8f2d16be3906)   ![image](https://github.com/user-attachments/assets/82eda1e1-0792-4517-9b3a-9ec786eed979)
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam with a learning rate scheduler for dynamic adjustment.

![image](https://github.com/user-attachments/assets/8b284bb7-62b4-470d-8eea-2737115d5f6c)

![image](https://github.com/user-attachments/assets/e734eb6f-f555-455a-87f2-f7759d56bacb)

---

## Explainability and Interpretability

### Grad-CAM Heatmaps
Grad-CAM (Gradient-weighted Class Activation Mapping) was used to visualize the regions of the image that influenced the model's predictions.

![image](https://github.com/user-attachments/assets/899cb57c-fa73-4020-bb85-47964b014975)


#### Why Grad-CAM?
- Provides insights into the model's decision-making process.
- Helps identify potential biases or errors in predictions.
- Improves trust and transparency in AI systems.

![image](https://github.com/user-attachments/assets/b7c1a235-33ac-4a20-af35-a5a83884b438) ![image](https://github.com/user-attachments/assets/24fa65db-650e-44c6-a54e-6932854525d5)

#### Steps
1. Compute gradients of the target class with respect to the feature maps of the last convolutional layer.
2. Weight the feature maps by the gradients and combine them to create a heatmap.
3. Overlay the heatmap on the original image for visualization.

### t-SNE Visualization
t-SNE (t-Distributed Stochastic Neighbor Embedding) was used to visualize the high-dimensional validation features in a 2D space. This helped analyze the feature separability between classes.

![image](https://github.com/user-attachments/assets/392846e3-51ff-4208-b170-131fdc2b563d)


---

## Training and Evaluation

### Training
- **Batch Size**: 32
- **Epochs**: 30
- **Learning Rate Scheduler**: Dynamically adjusted the learning rate during training.

### Evaluation Metrics
- Accuracy
- Precision, Recall, and F1-Score (per class)
- Confusion Matrix

---

## Results

| Model                  | Training Accuracy | Validation Accuracy | Precision | Recall | F1-Score |
|------------------------|-------------------|---------------------|-----------|--------|----------|
| Custom CNN             | 97.1%            | 97.2%              | 97.0%     | 97.3%  | 97.1%    |
| Hybrid CNN+ViT+BiLSTM  | 91.2%            | 89.8%              | 90.8%     | 91.0%  | 90.9%    |
| Vision Transformer (ViT) | 98.0%          | 98.0%              | 98.1%     | 98.0%  | 98.0%    |
| Ensemble Model         | 93.0%            | 93.0%              | 93.2%     | 93.1%  | 93.1%    |

# Benchmarkings to be added : Intel Xeon 8480+ Platinum Server Model, Single T4 GPU , Dual T4 GPU , P100 GPU, Single Ryzen 5 CPU ( Comparison of a SuperComputer Server vs GPU effects on Deep Learning, OpenVino Optimizations on CPU vs GPU benchmarking)
# Cross Validation on External Dataset: AUC Distracted Driver Dataset / DMD: Distracted Driver Monitoring Dataset
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
