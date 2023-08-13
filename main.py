import numpy as np
from sklearn.datasets import load_digits, load_wine, load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Import necessary libraries
# Import NumPy for numerical operations, scikit-learn for datasets, model selection, and metrics,
# MLPClassifier for building neural networks, and matplotlib for data visualization.

# Import Digits dataset: Handwritten digits (0-9) images
digits = load_digits()
X_digits = digits.data.astype('float32')
y_digits = digits.target.astype('int')

# Import Wine dataset: Features of wines categorized into classes
wine = load_wine()
X_wine = wine.data.astype('float32')
y_wine = wine.target.astype('int')

# Import Iris dataset: Features of iris flowers categorized into species
iris = load_iris()
X_iris = iris.data.astype('float32')
y_iris = iris.target.astype('int')

# Load and preprocess dataset features and labels for Digits, Wine, and Iris datasets.
# Digits dataset: X_digits (feature variables) and y_digits (target variable - labels).
# Wine dataset: X_wine (feature variables) and y_wine (target variable - labels).
# Iris dataset: X_iris (feature variables) and y_iris (target variable - labels).

# Split the datasets into training and testing sets
# For each dataset, split the features (X) and labels (y) into training and testing sets
# Data Processing: Splitting the data into training and testing sets
X_digits_train, X_digits_test, y_digits_train, y_digits_test = train_test_split(X_digits, y_digits, test_size=0.2, random_state=42)  # Train-test split for Digits dataset
X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(X_wine, y_wine, test_size=0.2, random_state=42)  # Train-test split for Wine dataset
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)  # Train-test split for Iris dataset

# Split each dataset into training and testing subsets for feature variables (X) and target labels (y).
# Randomly shuffle and partition the data, reserving a portion for testing (20% of the data).

# Data Visualization

# Display a random sample of Digits dataset images
plt.figure(figsize=(8, 8))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_digits_train[i].reshape(8, 8), cmap='gray')
    plt.title(f'Digit: {y_digits_train[i]}')
    plt.axis('off')
plt.suptitle('Random Samples from Digits Dataset')
plt.show()

# Display a grid of 10 random handwritten digit images from the training set.
# The images are shown with their corresponding digit labels.

# Create and train a Multi-layer Perceptron (MLP) classifier for each dataset

# Modeling Method: Multi-layer Perceptron (MLP) Classifier for Handwritten Digits Classification
# Architecture: Single hidden layer with 100 neurons, SGD optimizer, learning rate of 0.001
clf_digits = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, alpha=1e-4,
                           solver='sgd', verbose=10, random_state=1,
                           learning_rate_init=0.001)
clf_digits.fit(X_digits_train, y_digits_train)

# Modeling Method: Multi-layer Perceptron (MLP) Classifier for Wine Classification
# Architecture: Single hidden layer with 100 neurons, SGD optimizer, learning rate of 0.001
clf_wine = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, alpha=1e-4,
                         solver='sgd', verbose=10, random_state=1,
                         learning_rate_init=0.001)
clf_wine.fit(X_wine_train, y_wine_train)

# Modeling Method: Multi-layer Perceptron (MLP) Classifier for Iris Species Classification
# Architecture: Single hidden layer with 100 neurons, SGD optimizer, learning rate of 0.001
clf_iris = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, alpha=1e-4,
                         solver='sgd', verbose=10, random_state=1,
                         learning_rate_init=0.001)
clf_iris.fit(X_iris_train, y_iris_train)

# Create an MLP classifier for each dataset with specified architecture and optimization parameters.
# Train the classifiers using the training data (feature variables and target labels).

# Predict on the test data for each dataset
# For each trained model, make predictions on the corresponding test data
# Data Processing: Making predictions
y_digits_pred = clf_digits.predict(X_digits_test)
y_wine_pred = clf_wine.predict(X_wine_test)
y_iris_pred = clf_iris.predict(X_iris_test)

# Use the trained models to make predictions on the test feature variables (X) for each dataset.

# Calculate accuracy for each dataset
# For each set of predictions, calculate the accuracy by comparing predicted labels to actual labels
# Data Processing: Calculating accuracy
accuracy_digits = accuracy_score(y_digits_test, y_digits_pred)
accuracy_wine = accuracy_score(y_wine_test, y_wine_pred)
accuracy_iris = accuracy_score(y_iris_test, y_iris_pred)

# Calculate the accuracy of the predictions by comparing them to the actual target labels (y).

# Print the accuracies for each dataset
print(f"Digits Accuracy: {accuracy_digits:.2f}")
print(f"Wine Accuracy: {accuracy_wine:.2f}")
print(f"Iris Accuracy: {accuracy_iris:.2f}")

# Display the calculated accuracy values for each dataset.
