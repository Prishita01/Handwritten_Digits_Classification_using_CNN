# Handwritten_Digits_Classification_using_CNN
CNN Digit Classifier

Handwritten Digits Classification using CNN
This repository contains code for training and evaluating a Convolutional Neural Network (CNN) model to classify handwritten digits using the MNIST dataset.

**Dependencies**

TensorFlow
NumPy
Dataset
The code utilizes the MNIST dataset, which consists of 60,000 training images and 10,000 testing images. The dataset is preprocessed and split into training and testing sets.

**Model Architecture**
The CNN model architecture used for classification is as follows:

Convolutional layer with 32 filters, a kernel size of (3, 3), and ReLU activation.
MaxPooling layer with a pool size of (2, 2).
Convolutional layer with 64 filters, a kernel size of (3, 3), and ReLU activation.
MaxPooling layer with a pool size of (2, 2).
Convolutional layer with 64 filters, a kernel size of (3, 3), and ReLU activation.
Flatten layer to convert the 2D feature maps to 1D.
Dense layer with 64 units and ReLU activation.
Dense layer with 10 units and softmax activation for classification.

**Training**
The model is compiled with the Adam optimizer and categorical cross-entropy loss. It is trained on the training set with a batch size of 128 and for 10 epochs. The validation data is used to evaluate the model's performance during training.

**Evaluation**
After training, the model is evaluated on the testing set to measure its performance. The test loss and accuracy are calculated and printed.

Please refer to the code for detailed implementation.

**Usage**
Install the required dependencies.
Run the code in a Python environment.
Feel free to modify and experiment with the code to explore different settings and improve the classification performance.

Note: Ensure that you have the necessary dependencies installed and the MNIST dataset is correctly loaded before running the code.
