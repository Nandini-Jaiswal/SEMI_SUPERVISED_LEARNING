# Semi-Supervised Learning with Clustering on Fashion MNIST and Overhead MNIST Datasets

This repository contains an implementation of a semi-supervised learning experiment using K-Means clustering to seed the classification process on the Fashion MNIST and Overhead MNIST datasets. This approach is inspired by a similar experiment conducted on the MNIST dataset.

## Overview

In this project, we aim to leverage K-Means clustering to identify a small subset of labeled images that can be used to train a classifier. The workflow involves the following steps:

1. **Clustering**: Apply K-Means clustering to partition the datasets into clusters.
2. **Labeling**: Select a representative sample of images from each cluster to create a labeled training set.
3. **Training**: Train Logistic Regression model on the labeled subset.
4. **Evaluation**: Compare accuracy scores for different trained model.

## Datasets

### 1. Fashion MNIST
The Fashion MNIST dataset consists of 70,000 grayscale images in 10 categories. The images are of fashion products such as clothing, shoes, and bags. Each image is 28x28 pixels.

### 2. Overhead MNIST
The Overhead-MNIST dataset is composed of 76,690 grayscale satellite images of shape (28, 28, 1), four comma-separated value (. csv) files containing either flattened picture arrays or label mapping/summary data, and ubyte file duplicates.

## Requirements

To run the experiments, you need the following libraries:
- Python 3.x
- NumPy
- Scikit-learn
- TensorFlow/Keras

## Workflow Explained

1.   First, we will take 100 random samples (for Fashion MNIST dataset)/40 random samples (for Overhead MNIST dataset) from the dataset and use it to train the
 logistic regression model.
2.   Next, we will use k-means clustering to get 100 clusters and take the nearest points as centroids. Then we will use those 100 data-points to train the logistic regression classifier.
3. Then we will propagate the labelling to the whole cluster and do the same.
4. Repeat the same, but only propagate to the 20% of the dataset now.


## References
1. Fashion MNIST dataset can be easily accessed in Python through the Keras library, which provides a simple interface to download and load the dataset.
   To load the Fashion MNIST dataset, use the following code:
```python
from keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
```
2. The Overhead MNIST dataset can be downloaded from [here](https://www.kaggle.com/datasets/datamunge/overheadmnist/code).

3. [Keras Documentation](https://keras.io/)

4. [Scikit-learn Documentation](https://scikit-learn.org/)
