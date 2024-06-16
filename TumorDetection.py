import os
os.environ['PYTHONIOENCODING'] = 'UTF-8'

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load the dataset
dataset = pd.read_csv('cancer.csv')

# Separate features and target variable
x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])
y = dataset["diagnosis(1=m, 0=b)"]

# Split the data into a training set and a testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

# Define the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(x_train.shape[1],)))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def TrainModel():
    # Train the model
    model.fit(x_train, y_train, epochs=700)

def TestModel():
    # Evaluate the model
    model.evaluate(x_test, y_test)

while True:
    print("Select an Option:")
    print("1. Train Model, 2. Test Model, 3. Exit")
    
    option = input("Choose 1/2/3: ")

    if option == "1":
        TrainModel()
    elif option == "2":
        TestModel()
    elif option == "3":
        break
    else:
        print("Please write only 1, 2 or 3.")
        