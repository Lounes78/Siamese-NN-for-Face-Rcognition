# Siamese-NN-for-Face-Recognition
My goal throw this project is to go from a research paper (in this case, Siamese Neural Networks for One-shot Image Recognition) all the way to a complete functionning application.

This repository contains code for implementing a Siamese Neural Network for image recognition using TensorFlow and Keras. The Siamese Neural Network is trained to distinguish between similar and dissimilar pairs of images.

## Table of Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Data Collection](#data-collection)
4. [Preprocessing](#preprocessing)
5. [Model Architecture](#model-architecture)
6. [Training](#training)
7. [Evaluation](#evaluation)
   
## Data Collection
The dataset used in this project is organized into three folders: positive, negative, and anchor. Positive and negative folders contain images for training, while the anchor folder contains images used as reference points.

## Preprocessing
The dataset is preprocessed using TensorFlow Datasets. Images are resized and scaled before being fed into the Siamese Neural Network. The data is split into training and testing sets.

## Model Architecture
The Siamese Neural Network architecture consists of a shared embedding network followed by a distance calculation layer. The model is defined using the Keras API.

## Training
The model is trained using a binary cross-entropy loss function and the Adam optimizer. Training checkpoints are saved, allowing for the possibility of resuming training.

## Evaluation
The trained model is evaluated on a separate testing set. Precision and recall metrics are calculated to assess the model's performance.
