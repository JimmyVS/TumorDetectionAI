# Breast Cancer Diagnosis with Neural Networks

[![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/)

This repository contains a Python script that uses TensorFlow to build, train, and evaluate a neural network for breast cancer diagnosis using a dataset (`cancer.csv`).

## Description

The script processes the dataset by separating features and target variables, then splits the data into training and testing sets. A sequential neural network model is defined with three dense layers using the sigmoid activation function. The model is compiled with the Adam optimizer and binary cross-entropy loss function. Users can interactively choose to train the model or evaluate its performance on the test set through a simple command-line interface.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- tensorflow

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/JimmyVS/TumorDetectionAI.git
    ```
2. Navigate to the project directory:
    ```bash
    cd TumorDetectionAI
    ```
3. Install the required packages:
    ```bash
    pip install pandas scikit-learn tensorflow
    ```

## Usage

1. Ensure you have the `cancer.csv` dataset in the project directory.
2. Run the script:
    ```bash
    set PYTHONIOENCODING=UTF-8
    python TumorDetection.py
    ```
3. Follow the interactive prompts to train or test the model.

## Dataset

The `cancer.csv` file should contain the dataset with features and a target column named `diagnosis(1=m, 0=b)` where `1` represents malignant and `0` represents benign diagnoses.
This repository already contains a dataset. You can change it whenever you want, but make sure to add the required features.

## Script Overview

- **Loading and Preparing Data**: Loads the dataset and preprocesses it.
- **Model Definition and Compilation**: Defines and compiles the neural network.
- **Training and Testing Functions**: Encapsulates training and evaluation logic.
- **Interactive Menu**: Allows users to train or test the model based on user input.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
