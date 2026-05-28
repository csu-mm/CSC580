'''
MS - Artificial Intelligence and Machine Learning
Course: CSC580 - Applying Machine Learning and Neural Networks - Capstone
Module 3: Critical Thinking Assignment
Professor: Dr. Brian Holbert
Created by Mukul Mondal
May 27, 2026

Problem statement: Option #2: Predicting Fuel Efficiency Using TensorFlow
In a regression problem, we aim to predict the output of a continuous value, like a price or a probability.

This assignment uses the classic Auto MPG ( https://archive.ics.uci.edu/ml/datasets/auto+mpg ).
Dataset and builds a model to predict the fuel efficiency of late-1970s and early-1980s automobiles. 
The data model includes descriptions of many automobiles from that time period. 
The description for an automobile includes attributes such as cylinders, displacement, horsepower, and weight.

( logical Process / Steps ) to be performed:
Download the dataset ==> Import the dataset using Pandas ==> Inspect the data ==> Take some basic Stats 
==> Split data ==> Normalize the data ==> Build models (mse and mae) ==> Train the models for 1000 epochs
==> Compare the two models ==> Write Analysis Report.

More details can be found in: Course 580 Module 3: Critical Thinking Assignment

'''

import os
from os import system, name

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
#print(tf.__version__) # 2.21.0
import seaborn as sns
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling


# Helper function.
# Clears the terminal
def clearScreen():
    if name == 'nt':  # For windows
        _ = system('cls')
    else:             # For mac and linux(here, os.name is 'posix')
        _ = system('clear')
    return

# Data Preparation and Preprocessing
# downloads data from the provided link
# cleanup for null/empty rows
# Saves locally as .csv file
# display basic tail() data
# prepares training and test dataframes
# shows pairplot(..) # showing corelations between two features
# shows 'MPG' feature's statistics
# Normalize the data
# create and retrun training and test datasets
def prepareData():
    # Download the dataset
    data_file = keras.utils.get_file("auto-mpg.data","http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    column_names =["MPG","Cylinders","Displacement","Horsepower","Weight","Acceleration","Model Year","Origin"]

    # Step 2: Import database using Pandas.
    raw_dataset = pd.read_csv(data_file, names=column_names, na_values="?", comment='\t', sep=" ", skipinitialspace=True)
    if raw_dataset is None:
        print("Data read error")
        exit(1)

    dataset = raw_dataset.copy()
    dataset.dropna(inplace=True) # remove empty/null data

    try:
        # Save data file as .csv
        file1 = "./datafiles/Module3/AutoMPG.csv"    
        if os.path.exists(file1):
            os.remove(file1)
        dataset.to_csv(file1, index=False)
    except Exception as e:
        print("Saving data as .csv file locally failed with exception: ", e)

    #Step 3: Take a screenshot of the tail of the dataset.
    print(dataset.tail())  # ok
    # print(dataset.describe().T)

    # Step 4: Split the data into train and test.
    train_dataset = dataset.sample(frac=0.8, random_state=0)  
    test_dataset = dataset.drop(train_dataset.index)

    # Step 5: Inspect the data.
    sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
    plt.show()

    # Step 6: Take a screenshot of the tail of the plots.
    # Step 7: Review the statistics.
    train_stats = train_dataset.describe()
    #train_stats.pop("MPG")
    train_stats = train_stats.pop("MPG").T
    print(train_stats)

    # Step 8: Take a screenshot of the tail of the statistics.
    # Step 9: Split features from labels.

    # Step 11: Normalize the data.
    def norm(x):    
        return (x - train_stats['mean']) / train_stats['std']  

    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)

    # This normalized data is what you will use to train the model.

    # Separate features and labels
    X_train, Y_train = normed_train_data.drop("MPG", axis=1), train_dataset["MPG"]
    X_test, Y_test = normed_test_data.drop("MPG", axis=1), test_dataset["MPG"]

    return  X_train, Y_train, X_test, Y_test

# Step 12: Build the model.
""" Not Used
def build_model():
    model = keras.Sequential([ keras.layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]), keras.layers.Dense(64, activation='relu'), keras.layers.Dense(1) ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',  optimizer=optimizer,  metrics=['mae', 'mse'])
    return model """


# buld model with 3 debse layers, optimizer and loss function: 'mse'
# returns the model
def build_model_mse(input_len: int = 9):
    model = keras.Sequential(name='mse')
    #model.add(keras.layers.Input(input_len))
    model.add(keras.layers.Input(shape=(input_len,), name="input"))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae','mse'])
    return model


# buld model with 3 Dense layers, optimizer and loss function: 'mae'
# returns the model
def build_model_mae(input_len: int = 9):
    model = keras.Sequential(name='mae')
    #model.add(keras.layers.Input(input_len))
    model.add(keras.layers.Input(shape=(input_len,), name="input"))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.RMSprop(0.001)
    model.compile(loss='mae', optimizer=optimizer, metrics=['mae','mse'])
    return model


# Application execution main entry point.
# calls other functions from above
# shows models summary
# runs both models for 10000 epochs
# prepares model run history in dataframe.
# displays the history in text and graphs
if __name__ == "__main__":
    clearScreen()
    print("Course: CSC580 - Applying Machine Learning and Neural Networks - Capstone")
    print("Module 3: Critical Thinking Assignment")
    print("  Option 2: Predicting Fuel Efficiency Using TensorFlow\n")

    # Get Data
    X_train, Y_train, X_test, Y_test = prepareData()

    # Step 14: Take a screenshot of the model summary.
    # Step 15: Now, try out the model. Take a batch of  10 examples from the training data and call  model.predict on it.
    # Step 16: Provide a screenshot of the model summary.
    model_mse = build_model_mse(len(X_train.keys()))
    model_mae = build_model_mae(len(X_train.keys()))
    
    print(model_mse.summary())
    print(model_mae.summary())

    # Step 17: Train the model.
    # Step 18: Train the model for 1000 epochs, and record the training and validation accuracy in the  history  object.
    print("        \nModel Training: MSE        ")
    EPOCHS = 1000
    history = model_mse.fit( X_train, Y_train, epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[tfdocs.modeling.EpochDots()] ) # giving error here

    #Step 19: Visualize the model's training progress using the stats stored in the  history  object.
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print("\n")

    # Step 20: Provide a screenshot of the tail of the history.
    print(hist.tail())

    #Step 21: Provide a screenshot of the history plot.
    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    plotter.plot({'Basic': history}, metric = "mae")  
    plt.ylim([0, 10])  
    plt.ylabel('MAE [MPG]')
    plt.show()
    plotter.plot({'Basic': history}, metric = "mse")  
    plt.ylim([0, 20])  
    plt.ylabel('MSE [MPG^2]')   
    plt.show()

    # Step 22: Compare the two models, one using Mean Absolute Error and the other using Mean Square Error.
    print("        \nModel Training: MAE        ")
    history = model_mae.fit(X_train, Y_train, epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[tfdocs.modeling.EpochDots()])
    
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print("\n")
    print(hist.tail())

    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    plotter.plot({'Basic': history}, metric = "mae")  
    plt.ylim([0, 10])  
    plt.ylabel('MAE [MPG]')
    plt.show()
    plotter.plot({'Basic': history}, metric = "mse")  
    plt.ylim([0, 20])  
    plt.ylabel('MSE [MPG^2]')   
    plt.show()
