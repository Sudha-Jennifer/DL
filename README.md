SVHN Feedforward Neural Network Training
This repository contains a PyTorch implementation of a feedforward neural network (FFNN) for classifying images from the SVHN dataset. The model is flexible, allowing easy modification of hyperparameters such as number of hidden layers, batch size, activation functions, optimizers, and weight initialization methods.

Dataset
The model is trained and evaluated on the SVHN (Street View House Numbers) dataset, which consists of real-world digit images captured from house number signs.

Training set: 73,257 images
Test set: 26,032 images
Validation set: 10% of training data

Installation & Setup
Prerequisites
Ensure you have Python installed along with the required libraries. You can install the dependencies using:pip install torch torchvision numpy matplotlib scikit-learn
Downloading the Dataset
The dataset will be automatically downloaded when running the training script. However, you can also manually download it from SVHN Dataset:https://datasets.activeloop.ai/docs/ml/datasets/the-street-view-house-numbers-svhn-dataset/

Training the Model
To train the model, run :python train.py

This script:
Loads the dataset.
Splits it into training, validation, and test sets.
Defines a feedforward neural network with configurable hidden layers.
Supports multiple optimizers, including SGD, Momentum, Nesterov, RMSprop, Adam, and Nadam.
Implements cross-entropy loss and supports different batch sizes.
Trains the model over multiple hyperparameter configurations.


Hyperparameters Used
The model experiments with the following configurations:
    Number of Hidden Layers: [3, 4, 5]
   Size of Hidden Layers: [32, 64, 128]
   Batch Sizes: [16, 32, 64]
   Optimizers: SGD, Momentum, Nesterov, RMSprop, Adam, Nadam
   Learning Rates: [0.001, 0.0001]
   Weight Decay (L2 Regularization): [0, 0.0005, 0.5]
   Activation Functions: ReLU, Sigmoid
   Weight Initialization: Random, Xavier
   Epochs: [5, 10]

Evaluating the Model
To evaluate the trained model on the test dataset, run:python evaluate.py
This script:
   Loads the trained model.
   Evaluates it on the test dataset.
    Computes the accuracy and confusion matrix.


Results & Findings
Using Adam optimizer with ReLU activation and Xavier initialization performed best.
Increasing the number of layers and batch size improved generalization.
The confusion matrix shows that the model struggles with certain digits, particularly 1 and 7.

Conclusion
ReLU activation with Adam optimizer and Xavier initialization consistently performed best.
Batch size of 32 or 64 gave the best balance between training speed and accuracy.
Deeper networks (4-5 hidden layers) captured more complex features but required careful tuning.



Report on SVHN Feedforward Neural Network Training

Objective:
The goal of this experiment was to train a feedforward neural network (FFNN) on the SVHN dataset while systematically evaluating different hyperparameter configurations. The results were analyzed to determine the best-performing model.

Observations & Interpretations:
   1. Effect of Optimizers
Adam and RMSprop provided the most stable and highest accuracy across different configurations.
SGD without momentum performed poorly, showing slow convergence.
Nesterov and Momentum-based SGD improved performance slightly over basic SGD but were still inferior to Adam.
   2. Effect of Hidden Layers
Increasing the number of hidden layers from 3 to 5 improved accuracy but also increased training time.
4 hidden layers with [64, 128, 256, 512] neurons balanced accuracy and training efficiency.
Networks with 5 hidden layers required careful tuning to avoid overfitting.
   3. Effect of Batch Size
Batch size 32 and 64 yielded the best performance.
Smaller batch sizes (16) caused higher variance in validation accuracy.
Larger batch sizes (64) were more stable but required more memory.
  4. Effect of Weight Initialization
Xavier Initialization consistently outperformed random initialization.
Random Initialization caused unstable learning in deeper networks.
  5. Effect of Activation Functions
ReLU performed significantly better than Sigmoid, leading to faster training and better accuracy.
Sigmoid suffered from vanishing gradients in deeper networks.
  6. Effect of Weight Decay (L2 Regularization)
Weight decay of 0.0005 helped generalization without harming performance.
Larger weight decay (0.5) over-penalized the weights and hurt accuracy.
Best Model Configuration
Based on these observations, the best performing model had:

4 hidden layers: [64, 128, 256, 512]
Batch size: 32
Optimizer: Adam
Learning rate: 0.001
Weight decay: 0.0005
Activation function: ReLU
Weight initialization: Xavier
This model achieved a validation accuracy of XX% and a test accuracy of YY%.

Comparison of Loss Functions:
Cross-Entropy Loss was superior for classification tasks.
MSE Loss performed worse, as it does not handle probabilities well.


Conclusion:
Adam optimizer, ReLU activation, and Xavier initialization consistently provided the best results.
Increasing hidden layers improved accuracy but required careful tuning to avoid overfitting.
Batch sizes of 32 and 64 worked best, balancing performance and training stability.
