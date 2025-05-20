Human Action Recognition Using CNN
Objective
The goal of this project is to build a Convolutional Neural Network (CNN) that can classify human actions based on images. The model is trained on a dataset containing various human action categories and predicts the class of a new uploaded image.

1. Setting Up Kaggle API to Download Dataset
To begin, we need access to the Human Action Recognition dataset, which is available on Kaggle. To download it programmatically, we use the Kaggle API.

First, the kaggle.json API key file is uploaded.

Then, we create a hidden .kaggle directory and copy the uploaded JSON file there.

We also change the file permissions to ensure it's secure and usable by the Kaggle API.

This setup allows us to authenticate and use Kaggle's command-line tools within Google Colab.

2. Downloading and Extracting the Dataset
After setting up the API, we use it to download the dataset:
"!kaggle datasets download -d meetnagadia/human-action-recognition-har-dataset"
The dataset is downloaded as a zip file.

We then unzip it into a directory named human_action_data.

This directory contains training and validation images categorized by different human actions such as walking, running, standing, etc.

3. Importing Required Libraries
We import all the essential libraries for:

Image preprocessing (ImageDataGenerator)

Model building (tensorflow.keras)

Plotting and visualization (matplotlib, seaborn)

Model evaluation (sklearn.metrics)

File handling and directory operations (os, numpy)

TensorFlow is also checked to confirm it's properly installed.

4. Dataset Directory Structure
We define the path to our dataset. The dataset consists of image folders structured in subdirectories by action class. The training and validation data are split from this structure using the validation_split feature of ImageDataGenerator.

5. Data Preprocessing and Augmentation
To feed the data into our CNN, we perform preprocessing using ImageDataGenerator:

All image pixels are rescaled to the range [0, 1] by dividing by 255.

We use an 80-20 split for training and validation.

Using flow_from_directory, the data is read and converted into batches that are ready to be used in model training.

Image specifications used:

Target size: 150x150 pixels

Batch size: 32

Classes: Automatically inferred from folder names

6. CNN Model Architecture
A Sequential CNN model is built with the following structure:

Input Layer: 150x150x3 RGB image

Conv2D + MaxPooling layers:

Layer 1: 32 filters, 3x3 kernel

Layer 2: 64 filters, 3x3 kernel

Layer 3: 128 filters, 3x3 kernel

Flatten Layer: Converts 3D output to 1D

Dense Layer: 128 neurons with ReLU activation

Dropout Layer: 50% dropout to prevent overfitting

Output Layer: Softmax activation with number of units equal to number of classes (actions)

The model is compiled using:

Optimizer: Adam

Loss Function: Categorical Crossentropy

Metric: Accuracy

7. Model Training
The model is trained for 15 epochs using the prepared train_generator and val_generator. Both training and validation accuracy and loss values are tracked.

A history object is returned, which is then used to plot training vs. validation accuracy and loss graphs. These help us visually inspect the model’s performance and detect overfitting or underfitting.

8. Model Evaluation
After training, the model is evaluated on the validation data.

Predictions are made using the validation set.

The predicted class labels are compared to the true labels.

We use classification_report to print precision, recall, f1-score, and support for each class.

A confusion matrix is plotted using seaborn’s heatmap to show correct vs. incorrect predictions in a grid format.

This step helps to identify which classes are well predicted and which are confused with others.

9. Saving the Model
Once the model is trained and evaluated, it is saved in .h5 format as human_action_cnn_model.h5. This saved model can be loaded later for inference without retraining.

10. Predicting on a Single Image
To demonstrate real-world usage, we include a function called predict_image(img_path):

It loads an image from a given path and resizes it to the input shape expected by the model.

It converts the image to an array, normalizes it, and makes a prediction.

It prints out the predicted class along with the confidence score.

This is helpful for testing custom images without modifying the full pipeline.

Conclusion
This project successfully demonstrates how to:

Load and preprocess a dataset using the Kaggle API.

Build and train a CNN to classify human actions in images.

Evaluate and visualize model performance.

Save the model and use it for real-time predictions on custom input.

It is a full-fledged pipeline from raw data to a usable deep learning model for image classification tasks.
