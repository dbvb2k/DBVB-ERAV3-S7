# MNIST Digit Classification

This README.md provides a comprehensive overview of your project, including:
- Assignment objective
- Project structure
- Detailed descriptions of all model architectures
- Training configuration and approach
- Data augmentation strategy
- Learning rate scheduling details
- Usage instructions
- Basic requirements

This repository contains various CNN model implementations for MNIST digit classification using PyTorch. The objective was to achieve the following aspects
1. 99.4% test accuracy for at least 3+ epochs
2. Total epochs to be less than or equal to 15 epochs 
3. Model to have less than 8000 parameters
4. Using modular code for models. Every model that was tried out must be there in the Model.py file 
5. Each File must have a "Target, Result, Analysis" block

## Project Structure 
├── Models.py # Contains all model architectures
├── train.py # Training script with data loading and training loops
├── data/ # Directory for MNIST dataset (auto-downloaded)
├── models/ # Directory for saved model checkpoints
└── images/ # Directory for saved screenshots

## Model Architectures - Steps tried out (1, 2, 3)

# Step 1

### Target
- Get the set-up right
- Set the data transforms (Train and Test)
- Set data loader (after obtaining train / test data)
- Get the basic skeleton right (initial working model architecture)
- Set basic working code for training & test loop
- Batch Normalization and ReLU activation incorporated

### Result
- Parameters: 12,090
- Best Training Accuracy: 99.24% (14th Epoch)
- Best Test Accuracy: 99.01% (10th Epoch)

### Analysis
- Number of model parameters (12K) is > than the required number (of 8K)
- Good model, The gap between training and test accuracy is less (Epochs 3 to 8)
- Model is overfitting slighly after the 9th epoch

=====================================================================================================

# Step 2

### Target
- Make the model lighter (< 8K parameters)
- Dropouts and GAP implemented

### Result
- Parameters: 7002 
- Best Training Accuracy: 99.50% (15th Epoch)
- Best Test Accuracy: 99.25% (10th Epoch)

### Analysis
- Good model, The gap between training and test accuracy is less
- Training accuracy is 99.50% in the last epoch, model is slow in learning ?

=====================================================================================================

# Step 3

### Target
- Implement Image augmentation (Random Rotation of 5 to 7 degrees)
- Implement Step LR policy
- Check Adam and SGD optimizers for better accuracy

### Result
- Parameters: 7,984
- Best Training Accuracy: 99.30% (13th Epoch)
- Best Test Accuracy: 99.44% (15th Epoch)

### Analysis
- Number of model parameters is the required limit of 8K
- The difference betwen training and test accuracy is minimal and model is consistently underfitting (in all epochs)
- Model's test accuracy is consistent in the range of 99.37% to 98.44% (Epochs 7 to 15)
- SGD Optimizer (99.44% best accuracy) gave a slightly better accuracy than Adam optimizer (99.38% best accuracy), remaining all parameters were being the same
- Increased the capacity of the model by adding a layer after GAP!


## Training Details

- **Optimizer**: SGD with momentum (0.9)
- **Learning Rate**: 0.01 with ReduceLROnPlateau scheduling / StepLR 
- **Batch Size**: 128 (64 for CUDA)
- **Epochs**: 15
- **Data Augmentation**: Random rotation (-7° to 7°)
- **Normalization**: Mean = 0.1307, Std = 0.3081

## Data Augmentation

Training transforms include:
- Random rotation (-7° to 7°)
- Normalization
- Optional ColorJitter (commented out)

## Learning Rate Scheduling

Uses ReduceLROnPlateau scheduler with:
- Mode: max (tracking validation accuracy)
- Factor: 0.1
- Patience: 3 epochs
- Minimum LR: 1e-6

## Usage

1. Clone the repository
2. Run train.py to start training:

The MNIST dataset will be automatically downloaded to the `data/` directory on first run.

## Requirements

- PyTorch
- torchvision
- tqdm
- matplotlib
- numpy

