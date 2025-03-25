# Imports:
import tensorflow.keras as ks
from Classification_Utils import build_cnn, build_fnn
from numpy import argmax, reshape
from seaborn import heatmap
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use("TkAgg") # Since i am running this in an GNU terminal.
import matplotlib.pyplot as plt


### Load and Visualize the dataset:

# Loading dataset:
(x_train, y_train), (x_test, y_test) = ks.datasets.mnist.load_data()


# Displaying a few of the sample images
fig, axes = plt.subplots(3, 3, figsize=(9, 9))

for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i], cmap="plasma", interpolation="hanning")
    ax.set_facecolor((0.15, 0.15, 0.156))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(y_train[i], fontsize=12)

plt.tight_layout()
plt.show()


### Preprocess the Images:

# Normalize the pixel values to [0 -> 1]
x_train = x_train / 255
x_test = x_test / 255

# Convert labels to one-hot encoding

# num_classes=4
# [2 3 1 0] -> [[0 0 1 0] [0 0 0 1] [0 1 0 0] [1 0 0 0]]
y_train = ks.utils.to_categorical(y_train, num_classes=10)
y_test = ks.utils.to_categorical(y_test, num_classes=10)

# Rashaping the input data to hold the color channel (grey)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# Building the CNN
CNN = build_cnn()

# Summary of the model architecture
CNN.summary()


# Training the CNN:
# Epochs: Amount of times to backpropogate, more times ≃ better perfomance, but with the cost of overfitting.
history_cnn = CNN.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))

# Finding its accuracy:
cnn_test_loss, cnn_test_acc = CNN.evaluate(x_test, y_test)


# Computing and plotting the confusion matrix:


# Getting the prediction labels from the predicted data:

# argmax takes the index of the higest number, since we use softmax in the classification layer
# we can take the higest number from the predicted value and store its index as the class it choose.
# i.e., [0.3 0.2 0.5] -> 2
y_pred_cnn = argmax(CNN.predict(x_test), axis=1)

# We du the same with the actual values, since we one hot encoded the data we turned
# e.g., 2 -> [0 0 1] we find the index of this again with argmax [0 0 1] -> 2
# then we get the class the CNN predicted and the actual class.
y_true = argmax(y_test, axis=1)


# Generating the confusion matrix:
cm_cnn = confusion_matrix(y_true, y_pred_cnn)

# Plotting:
plt.figure(figsize=(8, 6))
heatmap(cm_cnn, annot=True, fmt="d", cmap="Blues", 
        xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted Class:")
plt.ylabel("True Class:")
plt.show()


# Building the FNN
FNN = build_fnn()

# Training the FNN
history_fnn = FNN.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))

fnn_test_loss, fnn_test_acc = FNN.evaluate(x_test, y_test)

# FNN predicting:
y_pred_fnn = argmax(FNN.predict(x_test), axis=1)

# Generating comfusion matrix for the FNN:
cm_fnn = confusion_matrix(y_true, y_pred_fnn)

# Plotting:
plt.figure(figsize=(8, 6))
heatmap(cm_fnn, annot=True, fmt="d", cmap="Blues", 
        xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted Class:")
plt.ylabel("True Class:")
plt.show()

### Comparing the FNN and CNN
print(f"\nCNN Accuracy: {cnn_test_acc}")
print(f"FNN Accuracy: {fnn_test_acc}\n")

# Differance:
print(f"CNN has: {cnn_test_acc - fnn_test_acc}, better accuracy.")
