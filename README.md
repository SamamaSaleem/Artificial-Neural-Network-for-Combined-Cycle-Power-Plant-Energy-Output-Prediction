# Artificial Neural Network for Combined Cycle Power Plant Energy Output Prediction

This project demonstrates the implementation of an Artificial Neural Network (ANN) model to predict the net hourly electrical energy output of a Combined Cycle Power Plant (CCPP). The steps below detail the process from importing libraries and data, to training the model and evaluating its performance.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Implementation](#implementation)
  - [Importing the Libraries](#importing-the-libraries)
  - [Importing the Dataset](#importing-the-dataset)
  - [Splitting the Dataset](#splitting-the-dataset)
  - [Building the ANN](#building-the-ann)
  - [Training the Model](#training-the-model)
  - [Predicting Test Results](#predicting-test-results)
  - [Evaluating the Model](#evaluating-the-model)
- [Results](#results)

## Installation

Ensure you have Python and the following libraries installed:

- pandas
- scikit-learn
- tensorflow

You can install the required libraries using pip:

```bash
pip install pandas scikit-learn tensorflow
```

## Dataset

The dataset used is `Folds5x2_pp.xlsx`, sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant). Make sure this file is in the same directory as your script or provide the correct path to the file.

## Implementation

### Importing the Libraries

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
```

### Importing the Dataset

```python
dataset = pd.read_excel('Folds5x2_pp.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

dataset.head()
```

### Splitting the Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

### Building the ANN

```python
# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=12, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=12, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1))
```

### Training the Model

```python
# Compiling the ANN
ann.compile(optimizer='adam', loss='mean_squared_error')

# Training the ANN model on the Training set
ann.fit(X_train, y_train, batch_size=32, epochs=100)
```

### Predicting Test Results

```python
y_pred = ann.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
```

### Evaluating the Model

```python
# R-Squared score
print(r2_score(y_test, y_pred))
```

## Results

This Artificial Neural Network model achieved a high R-Squared score of **0.9164** on the test set, demonstrating strong predictive performance for estimating the energy output of a Combined Cycle Power Plant based on environmental factors. This project highlights the effectiveness of deep learning techniques in handling complex regression tasks in energy management.
