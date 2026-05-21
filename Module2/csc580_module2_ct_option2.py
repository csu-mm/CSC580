
'''
MS - Artificial Intelligence and Machine Learning
Course: CSC580 - Applying Machine Learning and Neural Networks - Capstone
Module 2: Critical Thinking Assignment
Professor: Dr. Brian Holbert
Created by Mukul Mondal
May 20, 2026

Problem statement: 
Option #2: Predicting Future Sales
In this assignment, you will work with a neural network that can be used to predict 
future revenues from the sales of a new video game. A dataset is provided that 
you'll use to train a neural network to predict how much money you can expect 
future video games to earn based on historical data. The data are contained in 
a file named: sales_data_training.csv
.In this spreadsheet, there is one row for each video game that a store has sold in the past.

You’ll use Keras to train the neural network that will try to predict the total earnings of 
a new game based on these characteristics. Along with the sales_data_training.csv file, 
there is also a second data file called: sales_data_test.csv
.Links to an external site. This file is exactly like the first one. The machine learning 
system should only use the training dataset during the training phase. Then, you'll use the 
test data to check how well the neural network is working. To use this data to train a 
neural network, you first have to scale this data so that each value is between zero and one. 
Neural networks train best when data in each column is all scaled to the same range. 
Use the following Python code to scale the earning and unit price columns in both the 
training and test datasets. You will use Pandas to generate scaled training and test data sets.


'''
import os
from os import system, name

import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential  
from keras.layers import *
import keras_tuner as kt
from keras.models import Sequential
from keras import layers 
from keras import activations




# Helper function.
# Clears the terminal
def clearScreen():
    if name == 'nt':  # For windows
        _ = system('cls')
    else:             # For mac and linux(here, os.name is 'posix')
        _ = system('clear')
    return

# Helper function.
# recreate input data files for multiple runs
def restoreOriginalDataFiles():
    df1 = pd.read_csv("./datafiles/Module2/sales_data_testing_org.csv")
    df2 = pd.read_csv("./datafiles/Module2/sales_data_training_org.csv")
    
    file1 = "./datafiles/Module2/sales_data_testing.csv"
    file2 = "./datafiles/Module2/sales_data_training.csv"
    
    if os.path.exists(file1):
        os.remove(file1)
    if os.path.exists(file2):
        os.remove(file2)
    
    df1.to_csv(file1, index=False)
    df2.to_csv(file2, index=False)
    return

"""
 To use this data to train a neural network, you first have to scale this data 
 so that each value is between zero and one. Neural networks train best 
 when data in each column is all scaled to the same range. 
 Use the following Python code to scale the earning and unit price columns 
 in both the training and test datasets.
"""
def prepareData():
    restoreOriginalDataFiles()

    file_testing = "./datafiles/Module2/sales_data_testing.csv"
    file_training = "./datafiles/Module2/sales_data_training.csv"

    training_data_df = pd.read_csv(file_training) # Load training data set from CSV file    
    testing_data_df = pd.read_csv(file_testing)   # Load testing data set from CSV file    

    # Data needs to be scaled to a small range like 0 to 1 for the neural# network to work well.
    scaler = MinMaxScaler(feature_range=(0, 1))# Scale both the training inputs and output
    scaled_training = scaler.fit_transform(training_data_df)
    scaled_testing = scaler.transform(testing_data_df)
    # Print out the adjustment that the scaler applied to the total_earnings column of data
    print("Note: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[8], scaler.min_[8]))
    
    # Create new pandas DataFrame objects from the scaled data
    scaled_training_df = pd.DataFrame(scaled_training, columns=training_data_df.columns.values)
    scaled_testing_df = pd.DataFrame(scaled_testing, columns=testing_data_df.columns.values)
    
    # Save scaled data dataframes to new CSV files
    file_testing_scaled = "./datafiles/Module2/sales_data_testing_scaled.csv"
    file_training_scaled = "./datafiles/Module2/sales_data_training_scaled.csv"
    scaled_training_df.to_csv(file_training_scaled, index=False)
    scaled_testing_df.to_csv(file_testing_scaled, index=False)
    
    return scaler

"""
...to get just the input features, we grab all of the columns of the training data but drop
the total earnings column. Then, on line eight, extract just the total earnings column as shown. 
Now, X contains all the input features for each game, and 
     Y contains only the expected earnings for each game. 
Now, you can build a neural network ...
"""
def load_training_data():
    #file_testing_scaled = "./datafiles/Module2/sales_data_testing_scaled.csv"
    file_training_scaled = "./datafiles/Module2/sales_data_training_scaled.csv"

    # Load the training data
    training_data_df = pd.read_csv(file_training_scaled)
    X = training_data_df.drop('total_earnings', axis=1).values
    Y = training_data_df[['total_earnings']].values
    
    return X,Y


"""
Incorporate the following parameters into your model definition:
   use a sequential model
   use nine inputs and one output
   make the model dense
   use the ReLU activation function for the hidden layers
   use the linear activation function for the output layer.
"""
def build_model():
    model = Sequential()
    model.add(layers.Input((9,)))  # input
    model.add(layers.Dense(64, activation=activations.relu))
    model.add(layers.Dense(32, activation=activations.relu))
    model.add(layers.Dense(1, activation=activations.linear))  # output
    model.compile('adam', loss='mse')
    return model

# builds the model with Tuning
def build_model_tuning(hp):
    model = Sequential()
    model.add(layers.Input((9,))) # Input

    # Tune the number of layers
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(
            layers.Dense(
                # Tune number of units separately
                units=hp.Int("units_{i}", min_value=32, max_value=512, step=32),
                activation=activations.relu
            )
        )
    model.add(layers.Dense(1, activation=activations.linear)) # Output

    # Tune the learning rate
    learning_rate = hp.Float("lr", min_value = 1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss=hp.Choice("loss", ['mse','mae']),
        metrics=['mse',tf.keras.metrics.RootMeanSquaredError(),'mae'])

    return model

"""
Complete the final segment of Python code. Be sure to rescale your final prediction 
using the two  parameters during the scaling of the training and testing data sets.
"""
def tune_hyperparameters(X, Y, val_size: float = 0.2, verbose: int = 2):
    global barRunning
    print("\n* * * * * Hyperparemter Tuning Under Way* * * * *")

    # Tuning the model's hyperparameters
    tuner = kt.BayesianOptimization(
        hypermodel=build_model_tuning,
        objective=kt.Objective("val_root_mean_squared_error", direction="min"),
        max_trials=15,
        executions_per_trial=2,
        overwrite=True,
        directory="hpo",
        project_name="sales_predictions"
    )

    # Summary of the search space
    tuner.search_space_summary()

    X_TRAIN,X_VAL,Y_TRAIN,Y_VAL = train_test_split(X,Y,test_size=val_size)
    # Search for the best hyperparemeter configurations
    tuner.search(X_TRAIN, Y_TRAIN, epochs=10, batch_size=25, validation_data=(X_VAL, Y_VAL), verbose=verbose)

    print("\n* * * * * Best Tuned Model * * * * *")
    # Get the best model and return it
    models = tuner.get_best_models(num_models=1)
    best_model = models[0]

    best_model.build()
    best_model.summary()

    return best_model


"""
Train your model using both X and Y as well as the following:
   50 epochs
   shuffle=True; this action will make Keras shuffle the data randomly during each epoch
    verbose = 2; this tells Keras to print detailed information during the processing. 
Take a screenshot of these messages for your submission.
Save your trained model. You will submit this model as part of your assignment.
"""
def train_model(X,Y,model,name: str, verbose: int = 2, validation_split: float = 0.2):

    print("\n* * * * * Training Model * * * * *")
    model.fit(X,Y,batch_size = 100, epochs = 50, verbose=verbose, shuffle=True, validation_split=validation_split)

    print("\n* * * * * Saving Model: {} * * * * *".format(name))
    model.save("{}.h5".format(name))

    return model


"""
Evaluate your neural network model using model.evaluate(...) method. Print out the MSE for the test dataset.
"""
def test_model(model):
    # Load the testing data
    file_testing_scaled = "./datafiles/Module2/sales_data_testing_scaled.csv"
    file_training_scaled = "./datafiles/Module2/sales_data_training_scaled.csv"

    testing_data_df = pd.read_csv(file_testing_scaled)

    X_TEST = testing_data_df.drop('total_earnings', axis=1).values
    Y_TEST = testing_data_df[['total_earnings']].values

    test_error_rate = model.evaluate(X_TEST, Y_TEST, verbose=2)

    if type(test_error_rate) is float:
        error = test_error_rate
    else:
        error = test_error_rate[1]
        
    print("The mean squared error (MSE) for the test data set is: {}".format(error))


"""
Complete the final segment of Python code. Be sure to rescale your final prediction using 
the two  parameters during the scaling of the training and testing data sets.
"""
def predict(model, scaler):
    file_proposed_new_product = "./datafiles/Module2/proposed_new_product.csv"
    #X_PRED = pd.read_csv("proposed_new_product.csv").values
    X_PRED = pd.read_csv(file_proposed_new_product).values
    pred = model.predict(X_PRED)

    temp = np.concatenate((X_PRED[0,0:8],pred[0], [X_PRED[0][8]]))
    scaled = scaler.inverse_transform([temp])

    print("Earnings Prediction for Proposed Product: ${}".format(scaled[0][8]))



# Application execution main entry point.
if __name__ == "__main__":
    clearScreen()
    print("Course: CSC580 - Applying Machine Learning and Neural Networks - Capstone")
    print("Module 2: Critical Thinking Assignment")
    print("  Option 2: Predicting Future Sales\n")

    scaler = prepareData()

    X_train, Y_train = load_training_data()

    usrPrompt: str = "Do you want to use: 'Hyper Parameter Tuning'? y(yes), n(no) or 'q' to quit: "
    usrInput: str = "x"
    okInputs = ['y','n','q']
    while usrInput not in okInputs:
        print("\n")
        usrInput = input(usrPrompt).strip().lower()
    if usrInput == 'q':
        exit(0)    

    if usrInput == 'y':  # Run with hyperparetmer tuning steps        
        tuned_model = tune_hyperparameters(X_train, Y_train, verbose=1)
        trained_model = train_model(X_train, Y_train, tuned_model, "trained_tuned_model", verbose=2, validation_split=0)
    else: # Run without tuning
        model = build_model()
        trained_model = train_model(X_train, Y_train, model, "trained_untuned_model", verbose=2, validation_split=0)
        #print(trained_model.input_shape)
    
    test_model(trained_model)
    predict(trained_model, scaler)

