Dog Breed Classification with Stanford Dogs Dataset

Project Overview

This project focuses on building a deep learning model to classify dog breeds using the Stanford Dogs Dataset. By leveraging Convolutional Neural Networks (CNNs) and transfer learning with VGG16, the model aims to accurately distinguish between various dog breeds.

Key Steps

Dataset Preparation: Downloaded and extracted the Stanford Dogs Dataset, organizing images by breed and extracting bounding box annotations.
Preprocessing: Resized images to 224x224 pixels, normalized pixel values to [0,1], and applied label encoding and one-hot encoding for breed labels.
Data Augmentation: Implemented transformations like rotation, flipping, zoom, and shifting using Keras' ImageDataGenerator to enhance model generalization.
Model Architecture: Utilized the VGG16 architecture pre-trained on ImageNet, fine-tuning the top layers for breed classification.
Training: Split the dataset into an 80/20 train-test ratio and trained the model with augmented data.
Evaluation: Assessed model performance using accuracy, precision, recall, and F1 score, with options for fine-tuning hyperparameters to improve results.
Technologies Used

TensorFlow / Keras
VGG16 Transfer Learning
Python
LabelEncoder / One-Hot Encoding
ImageDataGenerator for Augmentation
Key Takeaways

Successfully implemented transfer learning to improve classification accuracy.
Preprocessing and data augmentation significantly boosted model robustness.
Gained hands-on experience with preparing and deploying a real-world dataset for image classification tasks.
