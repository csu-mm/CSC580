'''
MS - Artificial Intelligence and Machine Learning
Course: CSC580 - Applying Machine Learning and Neural Networks - Capstone
Module 6: Critical Thinking Assignment
Professor: Dr. Brian Holbert
Created by Mukul Mondal
June 20, 2026

Problem statement: Option #1: Implementation of CIFAR10 with CNNs Using TensorFlow
For this assignment, you will train a network to classify images from the CIFAR
dataset ( https://www.cs.toronto.edu/~kriz/cifar.html ) using a 
Convolutional Neural Network (CNN) built in TensorFlow.
Provide an analysis of the veracity of your model. 

( logical Process / Steps ) to be performed:
Download the dataset ==> Preprocess / Normalize the data ==> Build models ==> Train the models for 20(or more) epochs
==> Evaluate model ==> Prediction ==> Write Analysis Report.

More details can be found in: Course 580 Module 6: Critical Thinking Assignment

'''


import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import confusion_matrix
import seaborn as sns


# class labels of all data in the data set
class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# Helper function.
# Clears the terminal
def clearScreen():
    if os.name == 'nt':  # For windows
        _ = os.system('cls')
    else:             # For mac and linux(here, os.name is 'posix')
        _ = os.system('clear')
    return


# Load CIFAR‑10 dataset
#  Normalize data items
#  input: normalizeFactor. Normally this value is: 255.0 because the max value of any single color = 255
def load_CIFAR_data_and_Prepare(normalizeFactor: float):
    if normalizeFactor < 1:
        normalizeFactor = 1.0
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # 2. Normalize pixel values
    x_train_normalized = x_train.astype("float32") / normalizeFactor
    x_test_normalized = x_test.astype("float32") / normalizeFactor
    
    return x_train_normalized, y_train, x_test_normalized, y_test

# This function creates a Sequential CNN model with 3 layers.
# CIFAR images are 32×32 RGB, and this has to match with the input_shape.
def createModel1():
    return tf.keras.models.Sequential([
        # Block #1
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Block #2
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Final classification
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')  # Fully‑connected classifier at the end
    ])


# Model with improved veracity
def createModel3():
    return tf.keras.models.Sequential([
        # Block 1
        tf.keras.layers.Conv2D(32, 3, padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(32, 3, padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(2),

        # Block 2
        tf.keras.layers.SeparableConv2D(64, 3, padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.SeparableConv2D(64, 3, padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(2),

        # Global pooling instead of Flatten
        tf.keras.layers.GlobalAveragePooling2D(),

        # Modern classifier head
        tf.keras.layers.Dense(128, activation='relu', 
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
                            tf.keras.layers.Dropout(0.3),
                            tf.keras.layers.Dense(10, activation='softmax')
    ])


# This function creates a Sequential CNN model with 3 layers.
# This model has more layers and offers little better accuray, 
#    but I've noticed that it takes almost twice execution time compared to the above model.
def createModel2():
    return tf.keras.models.Sequential([
        # Block 1
        tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                    kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                    input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),

        # Block 2
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),

        # Block 3
        tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.5),

        # Classification head
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    

# This function does the model training.
#   returns history
# inputs:
#     xtrain: normalized input
#     epch: epochs to run
#     btchSize: batch_size  # we chan change this based on our GPU and hence it can be used for hyperparameter tuning.
#     val_split: validation_split # defines split of data items collection for the training activity.
#     xtrain: input data to be trained.
#     ytrain: corresponding actual output
def trainModel(epch: int, btchSize: int, val_split: float, xtrain, ytrain):
    if epch < 1:
        return
    if val_split <= 0:
        return
    if xtrain is None or len(xtrain) < 1:
        return
    if ytrain is None or len(ytrain) < 1:
        return
    return model.fit( xtrain, ytrain, epochs=epch, batch_size=btchSize, validation_split=val_split )


# This function shows one image.
# This is only for visual verification
def showImage(normalizedImage, normalizeFactor):
    if normalizedImage is None:
        return
    if normalizeFactor < 1:
        return
    img_big = tf.image.resize(normalizedImage*normalizeFactor, (256, 256), method='bicubic')
    plt.imshow(tf.clip_by_value(img_big, 0, 255).numpy().astype("uint8"))
    plt.title("True: " + class_names[y_test[idx][0]])
    #plt.ion()
    plt.show()  #plt.show(block=False)



# Application execution main entry point.
# It calls above functions and perforns all needed tasks for this classification
if __name__ == "__main__":
    clearScreen()
    print("Course: CSC580 - Applying Machine Learning and Neural Networks - Capstone")
    print("Module 6: Critical Thinking Assignment")
    print("  Option 1: Implementation of CIFAR10 with CNNs Using TensorFlow\n")


    # 1. Load CIFAR‑10 dataset
    normalizationFactor: float = 255.0
    x_train_normalized, y_train, x_test_normalized, y_test = load_CIFAR_data_and_Prepare(normalizationFactor)

    # Build a CNN model
    #model = createModel1()
    #model = createModel2()
    model = createModel3()
    model.summary()  # show model summary

    # Compile the model
    model.compile( optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'] )

    # Train the CNN model
    history = trainModel(20, 128, 0.1, x_train_normalized, y_train)
    #print(history.history.keys()) # dict_keys(['accuracy', 'loss', 'val_accuracy', 'val_loss'])
    print("max. accuracy=",max(history.history['accuracy']))
    print("max. loss=", max(history.history['loss']))
    #print("val_accuracy=",history.history['val_accuracy']) 
    #print("val_loss=",history.history['val_loss'])

    # Evaluate model on test data
    test_loss, test_acc = model.evaluate(x_test_normalized, y_test)
    print("Test accuracy:", test_acc)
    print("Test loss:", test_loss)

    # confusion matrix
    y_pred = model.predict(x_test_normalized)
    y_pred_classes = y_pred.argmax(axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)
    print(cm)
    plt.figure(figsize=(8, 6))
    #sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    #plt.ion()
    plt.show()

    # Prediction/verfication on single test image
    idx: int = 9  # predict image at this index
    img = x_test_normalized[idx]
    pred = model.predict(img.reshape(1, 32, 32, 3))
    print("Predicted:", class_names[np.argmax(pred)])    
    showImage(img, normalizationFactor) # show actual image


# based on my random experiment (random value initialized in 'idx' variable above).
# idx = 2: Predicted: ship
# idx = 8: Predicted: cat
# idx = 9: Predicted: automobile
# idx = 10: Predicted: airplane
# idx = 12: Predicted: dog



#
# Environment setup and Installs
#
# C:\Projs\Python\CSU>python -m venv csc580
# C:\Projs\Python\CSU\csc580>scripts\activate
# (csc580) C:\Projs\Python\CSU\csc580>python.exe -m pip install --upgrade pip
# (csc580) C:\Projs\Python\CSU\csc580>pip install numpy 
# (csc580) C:\Projs\Python\CSU\csc580>pip install matplotlib
# (csc580) C:\Projs\Python\CSU\csc580>pip install sklearn
# (csc580) C:\Projs\Python\CSU\csc580>pip install seaborn
# (csc580) C:\Projs\Python\CSU\csc580>pip install tensorflow  # not supported for: Python 3.14.5, supported versions: 3.8,3.9,3.10
# (csc580) C:\Projs\Python\CSU\csc580>pip install tensorflow[and-cuda]  # Optional: GPU support (if we've an NVIDIA GPU)
# 