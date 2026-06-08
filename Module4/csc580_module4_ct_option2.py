'''
MS - Artificial Intelligence and Machine Learning
Course: CSC580 - Applying Machine Learning and Neural Networks - Capstone
Module 4: Critical Thinking Assignment
Professor: Dr. Brian Holbert
Created by Mukul Mondal
June 7, 2026

Problem statement: Option #2: Logistic Regression with TensorFlow
In this assignment, you will analyze the quality of a TensorFlow prediction by generating synthetic data.

1) Generate the synthetic data using the following Python code snippet.
    # Generate synthetic data
    N = 100
    # Zeros form a Gaussian centered at (-1, -1)
    x_zeros = np.random.multivariate_normal(
    mean=np.array((-1, -1)), cov=.1*np.eye(2), size=(N//2,))
    y_zeros = np.zeros((N//2,))

    # Ones form a Gaussian centered at (1, 1)
    
    x_ones = np.random.multivariate_normal(
    mean=np.array((1, 1)), cov=.1*np.eye(2), size=(N//2,))
    y_ones = np.ones((N//2,))
    x_np = np.vstack([x_zeros, x_ones])
    y_np = np.concatenate([y_zeros, y_ones])

2) Plot x_zeros and x_ones on the same graph.

3) Generate a TensorFlow graph.
    with tf.name_scope("placeholders"):
        x = tf.compat.v1.placeholder(tf.float32, (N, 2))
        y = tf.compat.v1.placeholder(tf.float32, (N,))
    
    with tf.name_scope("weights"):
        W = tf.Variable(tf.random.normal((2, 1)))
        b = tf.Variable(tf.random.normal((1,)))

    with tf.name_scope("prediction"):
        y_logit = tf.squeeze(tf.matmul(x, W) + b)

        # the sigmoid gives the class probability of 1
        y_one_prob = tf.sigmoid(y_logit)

        # Rounding P(y=1) will give the correct prediction.
        y_pred = tf.round(y_one_prob)

    with tf.name_scope("loss"):
        # Compute the cross-entropy term for each datapoint
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y)

        # Sum all contributions
        l = tf.reduce_sum(entropy)

    with tf.name_scope("optim"):
        train_op = tf.compat.v1.train.AdamOptimizer(.01).minimize(l)

    with tf.name_scope("summaries"):
        tf.summary.scalar("loss", l)
        merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter('logistic-train', tf.get_default_graph())

4) Train the model, get the weights, and make predictions.
5) Plot the predicted outputs on top of the data.

For your deliverable, provide a detailed analysis using your screenshots as supporting content.
Write up your analysis using a Word document. Submit your Python code and Word document in a zip archive file. 
Name your archive file: CSC580_CTA_4_2_last_name_first_name.zip.

Your paper should be a minimum of two pages in length and conform to the CSU Global Writing Center.

'''


import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
# above line should be present here, otherwise it won't run successfully
#    because some part of some functions in the provided code snippet uses
#       code from old version of tf.
# 
# My installation has:  tf.__version__ =  2.21.0
# 

def clearScreen():
    if os.name == 'nt':  # For windows
        _ = os.system('cls')
    else:             # For mac and linux(here, os.name is 'posix')
        _ = os.system('clear')
    return

clearScreen()
print("Course: CSC580 - Applying Machine Learning and Neural Networks - Capstone")
print("Module 4: Critical Thinking Assignment")
print("  Option 2: Logistic Regression with TensorFlow\n")

# check tf version
print("tf.__version__ = ", tf.__version__) # 2.21.0




# 1) Generate the synthetic data using the following Python code snippet.
# Generate synthetic data
#  
N = 100
# Zeros form a Gaussian centered at (-1, -1)
x_zeros = np.random.multivariate_normal(
    mean=np.array((-1, -1)),
    cov=.1*np.eye(2),
    size=(N//2,)
)
y_zeros = np.zeros((N//2,))

# Ones form a Gaussian centered at (1, 1)
x_ones = np.random.multivariate_normal(
    mean=np.array((1, 1)),
    cov=.1*np.eye(2),
    size=(N//2,)
)
y_ones = np.ones((N//2,))

x_np = np.vstack([x_zeros, x_ones]).astype(np.float32)
y_np = np.concatenate([y_zeros, y_ones]).astype(np.float32)


# 2) Plot x_zeros and x_ones on the same graph.
plt.figure(figsize=(6,6))
plt.scatter(x_zeros[:,0], x_zeros[:,1], color='blue', label='Class 0')
plt.scatter(x_ones[:,0], x_ones[:,1], color='red', label='Class 1')
plt.legend()
plt.title("Synthetic Data")
plt.show()


# 3) Generate a TensorFlow graph.
with tf.name_scope("placeholders"):
    x = tf.compat.v1.placeholder(tf.float32, (N, 2))
    y = tf.compat.v1.placeholder(tf.float32, (N,))

with tf.name_scope("weights"):
    W = tf.Variable(tf.random.normal((2, 1)))
    b = tf.Variable(tf.random.normal((1,)))

with tf.name_scope("prediction"):
    y_logit = tf.squeeze(tf.matmul(x, W) + b)
    y_one_prob = tf.sigmoid(y_logit)  # the sigmoid gives the class probability of 1.
    y_pred = tf.round(y_one_prob)     # Rounding P(y=1) will give the correct prediction.

with tf.name_scope("loss"):
    # Compute the cross-entropy term for each datapoint
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y)
    l = tf.reduce_sum(entropy)  # Sum all contributions

with tf.name_scope("optim"):
    train_op = tf.compat.v1.train.AdamOptimizer(0.01).minimize(l)

with tf.name_scope("summaries"):
    #tf.summary.scalar("loss", l)              # it won't work in my installed tf version
    tf.compat.v1.summary.scalar("loss", l)     # lets use it from earlier tf version
    #merged = tf.summary.merge_all()           # it won't work in my installed tf version
    merged = tf.compat.v1.summary.merge_all()  # lets use it from earlier tf version
    #train_writer = tf.summary.FileWriter('logistic-train', tf.compat.v1.get_default_graph())           # it won't work in my installed tf version
    train_writer = tf.compat.v1.summary.FileWriter('logistic-train', tf.compat.v1.get_default_graph())  # lets use it from earlier tf version
    

# 4) Train the model, get the weights, and make predictions.
num_epochs = 2000

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(num_epochs):
        _, loss_val, summary = sess.run(
            [train_op, l, merged],
            feed_dict={x: x_np, y: y_np}
        )
        train_writer.add_summary(summary, step)

        if step % 200 == 0:
            print(f"Step {step}, Loss = {loss_val:.4f}")

    # Get trained parameters
    W_val, b_val = sess.run([W, b])
    y_pred_val = sess.run(y_pred, feed_dict={x: x_np})
    print(f"Prediction: {y_pred_val}")


# 5) Plot the predicted outputs on top of the data.
plt.figure(figsize=(6,6))
plt.scatter(x_np[:,0], x_np[:,1], c=y_pred_val, cmap='bwr', edgecolors='k')
plt.title("Predicted Classes")
plt.show()
