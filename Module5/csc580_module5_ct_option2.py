'''
MS - Artificial Intelligence and Machine Learning
Course: CSC580 - Applying Machine Learning and Neural Networks - Capstone
Module 5: Critical Thinking Assignment
Professor: Dr. Brian Holbert
Created by Mukul Mondal
June 10, 2026

Problem statement: 
Option #2: Building a Random Forest Classifier
In this assignment, you will use the iris dataset ( https://gist.github.com/curran/a08a1080b88344b0c8a7#file-iris-csv )
to classify iris plants based on measurements of their petal widths and sepal lengths. 
The dataset contains four variables measuring various parts of iris flowers of three related species 
and a fourth variable with the species name.

More details can be found in: Course 580 Module 5: Option2 - Critical Thinking Assignment.

'''

# My development environment: Windows 11 Enterprise, Python 3.13.5

# Load libraries
import os

import numpy as np     # Load numpy
import matplotlib.pyplot as plt

import pandas as pd   # Load pandas

# Load the library with the iris dataset
from sklearn.datasets import load_iris

# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier


# Helper function.
# Clears the terminal
def clearScreen():
    if os.name == 'nt':  # For windows
        _ = os.system('cls')
    else:             # For mac and linux(here, os.name is 'posix')
        _ = os.system('clear')
    return


clearScreen()
print("Course: CSC580 - Applying Machine Learning and Neural Networks - Capstone")
print("Module 5: Critical Thinking Assignment")
print("  Option #2: Building a Random Forest Classifier.")
print("  The dataset(iris) contains four variables measuring various parts of iris flowers\n  of three related species and a fourth variable with the species name.\n")

# 1) Load the data

np.random.seed(0)  # Set random seed

# Create an object called iris with the iris data
iris = load_iris()

# Create a dataframe with the four feature variables
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add a new column with the species names; this is what we are going to try to predict
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# View the top 5 rows
print("\nTop 5 data rows:")
print(df.head())
# Make a screenshot of the head of the dataset.

# 2) Create training and test data
# Create a new column that, for each row, generates a random number between 0 and 1, and
# if that value is less than or equal to .75, then sets the value of that cell as True

# df['is_train'] = np.random.uniform(0, 1, len(df)) = 0.75 # SyntaxError: cannot assign to function call
df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.75

# View the top 5 rows
print("\nTop 5 data rows:")
print(df.head())

train = df[df['is_train'] == True]
test  = df[df['is_train'] == False]
# Show the number of observations for the test and training dataframes
print("\nNumber of observations in the training data:", len(train))
print("Number of observations in the test data:", len(test))
# Make a screenshot of the outputs.

# 3) Preprocess the data
# Create a list of the feature column's names
features = df.columns[:4]
print(f"\nfeatures:\n{list(features)}") # View features

# train['species'] contains the actual species names. Before we can use it,
# we need to convert each species name into a digit. So, in this case, there
# are three species, which have been coded as 0, 1, or 2.
y = pd.factorize(train['species'])[0]
# View target
print(f"\ntarget:\n{y}")
# Make a screenshot of the outputs.


# 4) Train the Random Forest Classifier
# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf = clf.fit(train[features], y)
"""
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=2, oob_score=False, random_state=0,
            verbose=0, warm_start=False)
"""
# My comment: Some parameters shown (like min_impurity_split) are deprecated and no longer appear in modern versions of scikit‑learn.
# clf2 = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, n_jobs=2, random_state=0)
#
# feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width']
# for name, importance in zip(feature_names, clf2.feature_importances_):
#     print(name, importance)
#

# Apply the Classifier we trained to the test data (which, remember, it has never seen before)
preds = clf.predict(test[features])

#6) Evaluate the classifier by comparing the predicted and actual species for the first five observations.
clf.predict_proba(test[features])[:10]  #  predicted probabilities for first 10 observations:

# Compare predicted vs actual species (first 5)
comparison = pd.DataFrame({
    'Actual': test['species'].head().values,
    'Predicted': preds[:5]
})
print(f"\nComparison:\n{comparison}")

# 7) Create a confusion matrix and use it to interpret the classification method.
# Supply a screenshot of the confusion matrix.
# Create confusion matrix
cm = pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames=['Predicted Species'])
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')
plt.colorbar()

plt.xticks(np.arange(len(cm.columns)), cm.columns, rotation=45)
plt.yticks(np.arange(len(cm.index)), cm.index)

plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.title("Confusion Matrix")
print(f"\nConfusion matric: {cm}")
# Add numbers inside the squares
for i in range(len(cm.index)):
    for j in range(len(cm.columns)):
        plt.text(j, i, cm.iloc[i, j], ha='center', va='center', color='black')

plt.tight_layout()
plt.show()  # Display confusion matrix

# 8) View the list of features and their importance scores.
print("\nfeature_importances_:")
for name, importance in list(zip(train[features], clf.feature_importances_)):
    print(f"{name}: {importance:.4f}")
