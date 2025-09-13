# Customer Churn Prediction using Artificial Neural Networks

## Description

This project focuses on predicting customer churn using an Artificial Neural Network (ANN). Customer churn is a critical problem for many businesses, as retaining existing customers is often more cost-effective than acquiring new ones. This solution leverages machine learning to identify customers who are likely to churn, enabling businesses to take proactive measures to retain them.

## Features

- Data preprocessing and exploration
- Implementation of an Artificial Neural Network (ANN) for classification
- Model training and evaluation
- Prediction of customer churn likelihood

## Technologies Used

- Python
- scikit-learn
- pandas
- numpy
- Keras/TensorFlow
- matplotlib(for visualization)

## Setup Instructions

To set up the project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/BinCha1/Customer-Churn-Prediction-using-Artificial-Neural-Networks.git
    cd Customer-Churn-Prediction-using-Artificial-Neural-Networks
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    _Note: A `requirements.txt` file will be created if not present._

## Usage

1.  **Run the Jupyter Notebook:**

    ```bash
    jupyter notebook
    ```

    Open the `Customer_Churn_Prediction.ipynb` (or similar) notebook and execute the cells to see the data preprocessing, model training, and prediction in action.

2.  **Alternatively, run the Python script (if available):**
    ```bash
    python main.py
    ```

## Model Architecture

The Artificial Neural Network (ANN) used in this project typically consists of:

- An input layer corresponding to the number of features in the dataset.
- One or more hidden layers with ReLU activation functions.
- An output layer with a sigmoid activation function for binary classification (churn or no churn).
- The model is compiled with an Adam optimizer and binary cross-entropy loss.
