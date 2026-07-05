"""
MS - Artificial Intelligence and Machine Learning
Course: CSC580 - Applying Machine Learning and Neural Networks - Capstone
Module 8: Portfolio Project
Professor: Dr. Brian Holbert
Created by Mukul Mondal
July 3, 2026


Option #2: Encoder-Decoder Model for Sequence-to-Sequence Prediction

Part 1 (Research Write-up):
Research and analyze the use of encoder-decoder models in industry and in applications. 
Your use cases should be uniquely distinct and should cover multiple areas of industry. 
Ensure that your paper meeting the following guidelines:

> Identify at least 4 pertinent use cases in which this model is used and the benefit for using it in each.
> Your paper should be a maximum of 4 pages and include at least 3 scholarly references in APA format. 
     Ensure that your assignment is formatted according to the CSU Global Writing Center. 
     You can easily access the Writing Center by clicking on the tab in the course navigation panel.

Part 2 (Programming Implementation):
For this Portfolio Project assignment, you will develop an encoder-decoder model for 
sequence-to-sequence prediction using Keras. (Keras library is a wrapper for low-level 
TensorFlow commands, and you will import this as a Python library.) 
The encoder-decoder model provides a pattern for using recurrent neural networks to address 
challenging sequence-to-sequence prediction problems such as machine translation.

Encoder-decoder models can be developed in the Keras Python deep learning library. 
For this assignment, you will develop a sophisticated encoder-decoder recurrent 
neural network for a sequence-to-sequence prediction problem with Keras.

There are three Python-based programming tasks for this Portfolio Project. The three parts are:
1. Encoder-Decoder Model in Keras
2. Scalable Sequence-to-Sequence Problem
3. Encoder-Decoder LSTM for Sequence Prediction.
...
Encoder-Decoder Model in Keras
The encoder-decoder model is a way of organizing recurrent neural networks for sequence-to-sequence 
prediction problems. 
The approach involves two recurrent neural networks: one to encode the source sequence, 
called the encoder, and a second to decode the encoded source sequence into the target sequence, 
called the decoder.

Use the following code example as a starting point to define an encoder-decoder recurrent neural network. 
....

"""


# Imports
import os
import numpy as np
from random import randint
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, LSTM, Dense


# Helper function.
# Clears the terminal
def clearScreen():
    if os.name == 'nt':  # For windows
        _ = os.system('cls')
    else:             # For mac and linux(here, os.name is 'posix')
        _ = os.system('clear')
    return


# Encoder–Decoder Model
# The function define_models is the core of a classic encoder–decoder (seq2seq) architecture.
# It actually builds three separate Keras models, each serving a different role in training and inference.
# 
# input arguments:
# Three input arguments in the function correspond to the core dimensional choices in the encoder–decoder architecture.
#   n_input: input feature dimensionality, size of each input vector.
#   n_output: output feature dimensionality, size of each output vector.
#   n_units: hidden size / model capacity, number of hidden units in the encoder/decoder LSTM (model capacity).
#            This is the number of LSTM (or GRU) units in both encoder and decoder.
def define_models(n_input, n_output, n_units):
    # Encoder input: variable-length sequence with n_input features per timestep
    encoder_inputs = Input(shape=(None, n_input))

    # Encoder LSTM: returns final hidden + cell states (not full sequence)
    encoder = LSTM(n_units, return_state=True)

    # Run encoder on input sequence; capture final states
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    # Store encoder states to pass into decoder as initial state
    encoder_states = [state_h, state_c]

    # Decoder input: variable-length sequence with n_output features per timestep
    decoder_inputs = Input(shape=(None, n_output))

    # Decoder LSTM: returns full output sequence + states for inference
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)

    # Run decoder using encoder states as initial state (training mode)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    # Dense layer to map decoder outputs to probability distribution over output tokens
    decoder_dense = Dense(n_output, activation='softmax')

    # Apply dense layer to each timestep of decoder output
    decoder_outputs = decoder_dense(decoder_outputs)

    # Full training model: takes encoder + decoder inputs, outputs predicted sequence
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Inference encoder model: outputs encoder states given new input sequence
    encoder_model = Model(encoder_inputs, encoder_states)

    # Decoder inference: placeholders for previous timestep hidden + cell states
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    # Run decoder for inference using provided states instead of encoder states
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

    # Updated decoder states for next timestep during inference loop
    decoder_states = [state_h, state_c]

    # Apply dense layer to inference decoder output
    decoder_outputs = decoder_dense(decoder_outputs)

    # Final inference decoder model:
    # Inputs: decoder input token + previous states
    # Outputs: predicted token + updated states
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states )

    # Return training model + encoder inference model + decoder inference model
    return model, encoder_model, decoder_model



# Predict a sequence:
# This function is doing the core work of turning an encoded input into a decoded output sequence.
# This is a sequence-to-sequence inference function.
#  Input arguments:
#    infenc: inference encoder model. A trained encoder model used only for inference (prediction), not training.
#    infdec: inference decoder model. A decoder model configured specifically for step-by-step prediction.
#    source: encoded input sequence. The input sequence we want to translate/transform.
#    n_steps: number of timesteps to predict -- length of the output sequence we want the decoder to generate.
#    cardinality: number of possible output classes. The size of the output vocabulary or number of categories per timestep.
#                 softmax output from infdec, will be of length cardinality.
#    vrbs=1: verbosity flag - controlls logging/printing of the function.
def predict_sequence(infenc, infdec, source, n_steps, cardinality, vrbs=1):
    # Run the encoder model on the input sequence.
    # This produces the initial hidden + cell states for the decoder.
    # 'source' must be shaped as (1, sequence_length, cardinality).
    state = infenc.predict(source, verbose=vrbs)  # encoder: initial decoder states

    # Initialize the decoder input with a "start token".
    # Here the start token is represented as a one-hot vector of zeros.
    # Shape: (1 batch, 1 timestep, cardinality)
    target_seq = np.array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)

    # List to store predicted probability vectors for each timestep
    output = list()

    # Generate n_steps output tokens autoregressively
    for t in range(n_steps):

        # Run the decoder for one timestep.
        # Inputs:
        #   - target_seq: the previous predicted token (or start token initially)
        #   - state: the previous hidden + cell states
        # Outputs:
        #   - yhat: predicted probability distribution for next token
        #   - h, c: updated decoder states for next timestep
        yhat, h, c = infdec.predict([target_seq] + state, verbose=vrbs)

        # Store the predicted probability vector for this timestep
        # yhat[0, 0, :] : batch 0, timestep 0, full categorical distribution
        output.append(yhat[0, 0, :])

        # Update decoder state for next iteration
        state = [h, c]

        # The predicted output becomes the next decoder input.
        # Feeding predictions back into the model.
        target_seq = yhat

    # Convert list of predictions into a NumPy array having shape (n_steps, cardinality)
    return np.array(output)


# 
# Generate random integer sequence.
# Creates a sequence of random integers, each representing a categorical class index.
# It generates a list of integers, where each integer is a class label drawn randomly from a fixed range.
# Input arguments:
#     length: The number of timesteps (items) in the generated sequence.
#              If length = 5, the output will contain 5 integers, like [3, 1, 4, 2, 1].
#     n_unique: The number of possible unique class values.
#                If n_unique = 10, the valid class indices are: 1, 2, 3, ..., 8, 9
#                   The function uses randint(1, n_unique - 1), so it never generates 0, and never generates n_unique. 
def generate_sequence(length, n_unique):
    return [randint(1, n_unique - 1) for _ in range(length)]


# Dataset generator.
# It builds a full synthetic dataset of input/output sequence pairs, each made of categorical tokens drawn from a fixed vocabulary.
# Input arguments:
#   n_in: input sequence length. Defines how many timesteps each input sequence contains.
#         If n_in = 5, every input sample looks like: [3, 1, 4, 2, 5]
#   n_out: output sequence length. Defines how many timesteps each target/output sequence contains.
#         If n_out = 3, every output sample looks like: [2, 7, 1]
#   cardinality: number of possible token values. This is the size of the vocabulary.
#         If cardinality = 10, valid token values are: 1, 2, 3, ..., 8, 9, it should not include 0 and 10.
#   n_samples: number of sequences to generate. Defines how many (input, output) pairs the dataset contains.
#         If n_samples = 1000, the function randomly generates and returns: 1000 input sequences and 1000 output sequences.
def get_dataset(n_in, n_out, cardinality, n_samples):
    # Initialize lists to hold encoder inputs (X1), decoder inputs (X2), and decoder outputs (y)
    X1, X2, y = list(), list(), list()

    # Generate n_samples training examples
    for _ in range(n_samples):

        # Generate a random source sequence of length n_in using integers from 0..cardinality-1
        source = generate_sequence(n_in, cardinality)

        # Define the target sequence as the first n_out elements of the source
        target = source[:n_out]

        # Reverse the target sequence (common in seq2seq toy tasks)
        target.reverse()

        # Create decoder input sequence:
        # Prepend a start-of-sequence token (0) and shift the target right by one position
        target_in = [0] + target[:-1]

        # One-hot encode the source sequence (shape: 1 × n_in × cardinality)
        src_encoded = to_categorical([source], num_classes=cardinality)

        # One-hot encode the target output sequence (decoder expected output)
        tar_encoded = to_categorical([target], num_classes=cardinality)

        # One-hot encode the shifted target sequence (decoder input during training)
        tar2_encoded = to_categorical([target_in], num_classes=cardinality)

        # Append encoded sequences to dataset lists
        X1.append(src_encoded)   # Encoder input
        X2.append(tar2_encoded)  # Decoder input
        y.append(tar_encoded)    # Decoder output

    # Convert lists to NumPy arrays for model training
    X1 = np.array(X1)
    X2 = np.array(X2)
    y  = np.array(y)

    # Remove the extra dimension added by to_categorical (axis=1)
    X1 = np.squeeze(X1, axis=1)
    X2 = np.squeeze(X2, axis=1)
    y  = np.squeeze(y, axis=1)

    # Return encoder inputs, decoder inputs, and decoder outputs
    return X1, X2, y


#
# One-hot decode
# It takes a sequence of one‑hot vectors and converts each vector back into its class index—the position where the “1” appears.
# If each vector has cardinality classes.
#   Example one‑hot vector: [0, 1, 0, 0] -- this means class at index 1.
# This loops through each one‑hot vector in the sequence and returns the index of the largest value in the vector.
#   For example: argmax([0, 1, 0]) -> 1; argmax([1, 0, 0]) -> 0 etc.
#
# Input argument:
#   encoded_seq must be a sequence (list or array) of one‑hot vectors.
#      example: [ [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1] ]
#             so, class indices: 1, 0, 3
# Return:
#  a list of class indices. with above example: [1, 0, 3]
def one_hot_decode(encoded_seq):
    return [int(np.argmax(vector)) for vector in encoded_seq]  


# ----------------------------
# MAIN EXECUTION
# ----------------------------
# Application execution main entry point.
# It calls above functions and perforns all needed tasks for this classification
if __name__ == "__main__":
    clearScreen()
    # print("tf.__version__ : ", tf.__version__)
    print("Course: CSC580 - Applying Machine Learning and Neural Networks - Capstone")
    print("Module 8: Portfolio Project")
    print("  Option #2: Encoder-Decoder Model for Sequence-to-Sequence Prediction")
    print("         Part 2: (Programming Implementation)\n")

    # Problem configuration
    # Number of features (i.e., vocabulary size for one-hot vectors).
    # The +1 is usually because index 0 is reserved for a "start" token.
    n_features = 50 + 1

    # Length of input sequence (encoder timesteps)
    n_steps_in = 6

    # Length of output sequence (decoder timesteps)
    n_steps_out = 3

    # Define model
    # define_models() returns:
    # 1. train  -- the full training model (encoder + decoder combined)
    # 2. infenc -- the standalone encoder model used during inference
    # 3. infdec -- the standalone decoder model used during inference
    train, infenc, infdec = define_models(n_features, n_features, 128)

    # Compile the training model:
    # - optimizer: Adam
    # - loss: categorical crossentropy (because outputs are one-hot vectors)
    # - metric: accuracy (percentage of correct one-hot predictions)
    train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    # Train model
    # Generate training data
    # get_dataset() returns:
    # X1 -- encoder input sequences (shape: n_samples × n_steps_in)
    # X2 -- decoder input sequences (teacher forcing inputs)
    # y  -- decoder output sequences (one-hot encoded targets)
    X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 5000)
    
    print("----- Training starting -----")
    # Train the model:
    # - Inputs: [X1, X2] -- encoder inputs + decoder inputs
    # - Target: y -- decoder outputs
    # - epochs: 30 -- number of passes through the dataset
    train.fit([X1, X2], y, epochs=30, verbose=1)

    # Evaluate accuracy
    total: int = 100  # number of random test samples.
    correct: int = 0  # counter for correct predictions, initialilly must be: 0
    for _ in range(total):
        # Generate ONE new random test sample
        X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)

        # Predict output sequence using inference encoder + decoder
        # predict_sequence() returns a sequence of one-hot vectors
        target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
    
        # Compare predicted sequence with true sequence:
        # one_hot_decode() converts one-hot vectors → integer class indices
        if np.array_equal(one_hot_decode(y[0]), one_hot_decode(target)):
            correct += 1
    
    print("----- Training completed -----")
    print("\nAccuracy: %.2f%%" % (correct / total * 100.0)) # prints: Accuracy 

    
    # Show example predictions
    print("\n---  10 example predictions ---")
    for _ in range(10):
        # Generate one random test sample
        X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
        
        # predict output sequence
        # yhat = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)  # ok, verbose always
        yhat = predict_sequence(infenc, infdec, X1, n_steps_out, n_features,vrbs=0)  # (vrbs=0 -- silent mode)

        # Decode one-hot sequences into integer sequences
        src = one_hot_decode(X1[0])       # input sequence
        expected = one_hot_decode(y[0])   # true output sequence
        predicted = one_hot_decode(yhat)  # model prediction

        # Print input, expected output, and predicted output
        print(f"X={src} y={expected}, yhat={predicted}")
