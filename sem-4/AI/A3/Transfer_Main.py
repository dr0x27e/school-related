import tensorflow as tf

import matplotlib
matplotlib.use("TkAgg") # Since i am running this in an GNU terminal.
import matplotlib.pyplot as plt

from Transfer_Utils import preprocess_image, augment_image, custom_ResNet50
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from numpy import argmax, array

## As far as i know there is no way unless you want to download the entire mnist dataset
## to Load the dataset WITHOUT splitting it, as this is how it comes out of the 'box'
## from tensorflow.

# Loading dataset:
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

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


## Resizing all the images to match the input size of pre-trained models:

# ResNet50 and mobileNetV2 both takes in an image of (224, 224, 3)
# That is 244 x 244 pixels with 3 color channels (RGB)

# Processing all those images at once requires ALOT of ram.
# Given that the mnist dataset consists of 70000 images, and each image consists
# of 28 * 28 * 8 (uint8 -> 8 bits) we are already looking at a lot of ram just
# to keep them loaded, but scaling the images up to 224 * 224 * 3 * 32 (float32) makes it
# ~38 GB of data which is not something my/most computer will allow you to do.

# So we will use pipelining and lazy lists to process -> augment -> train images on the fly.

# First we create a lazy list (a list that is not loaded fully but loads on demand)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# Apply preprocessing and augmentation:
train_dataset = (train_dataset
                 .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
                 .shuffle(buffer_size=1000)
                 .map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
                 .batch(32)
                 .repeat() # Loop indefinetly
                 .prefetch(tf.data.AUTOTUNE)
                 )

# No need to augment the test data as there will be no back proagation with it.
test_dataset = (test_dataset
                .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(32)
                .repeat()
                .prefetch(tf.data.AUTOTUNE)
                )

# Here we are chaining operations:

# First .map:
# We apply preprocess_image (Function in src/utils.py) on every element.
# This takes the original image and formats it into the size required by ResNet50.
# num_parallel_calls, this tells tensorflow to create X amount of threads on the 
# CPU, in our case this is tf.data.AUTOTUNE which tells tensorflow to dynamically determine
# the number of threads to create.
# This preproccessing happens on the fly so that we can avoid the problem of ~40 GB ram.

# .shuffle:
# Shuffle randomly shuffles the order of elements in the dataset, this ensures that the model
# wont become overfitted with the input data. Buffer_size is the buffer size.
# The buffer holds X images in our example 1000, which would be around ~600MB.
# This buffer is essentially now the biggest element in ram.

# Second .map:
# We apply augment_image (Function in src/utils.py) this augments each image, and again
# dynamically determines how many CPU threads to use.

# .batch:
# This batches the dataset into batches e.g., 32 so [32, 224, 224, 3]. We do this since the GPU
# depending on if you have CUDA & CUDNN installed and the CPU works more efficiently when it does
# not need to create a new thread for each and every image. Instead we batch them up so that each
# thread can work on 32 images at a time.

# .repeat:
# This makes it so that when the model reaches the end of the dataset the dataset will be
# generated from the start again.

# .prefetch
# Prefetches data even before the GPU / CPU is done with training on the current images.
# This speeds up the process as when the model is done training on its data, it instantly gets new
# data to train on.

# Originally starting with ~40 GB ram needed, we now only load, the original datasets with uint8,
# and size of (70000, 28, 28, 1) this is about ~120 MB + 1 image per thread in each map
# e.g., 4 images per map (224, 224, 3) * 32 (float32) * 4 * 2 + 2 batches of 32 images and 1
# prefetch buffer that holds 1 batch of 32 + finally the shuffle buffer of 1000 images bringing
# our total AT WORST CASE: 1104 images of size (224, 224, 3) * 32, this is about 700-750 MB of ram,
# ~55 times less than what we started with.


## Initiali and heatmap?zing the model:

# Frozen with ImageNet weights:
print("\n...Training with Frozen ImageNet weights\n")
model_frozen = custom_ResNet50(freeze_base=True, weights="imagenet")
model_frozen.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Fitting with explicit steps.
# This is needed becuase we are using a .reapet lazy list, so tensorflow does
# not know when to stop.
history_frozen = model_frozen.fit(
    train_dataset, 
    epochs=2, 
    validation_data=test_dataset,
    # Divide by 32 since for each step we do batch(32) (groups of 32)
    steps_per_epoch=len(x_train) // 32,  # 1875 steps
    validation_steps=len(x_test) // 32  # 312 steps
)

# Evaluation with explicit steps
frozen_eval = model_frozen.evaluate(
    test_dataset,
    steps=len(x_test) // 32  # 312 steps
)

# Predicting classes:
y_true = []
y_pred = []

# Since we are using a lazy list, so we have to visit each index, to "activate" them
# We also need to limit how many we take as we have an infinitely repeating lazy list
for images, labels in test_dataset.take(len(x_test) // 32):
    preds = model_frozen.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(argmax(preds, axis=1))

# Turning to numpy arrays:
y_ture, y_pred = array(y_true), array(y_pred)

# Getting the confusion matrix:
cm = confusion_matrix(y_true, y_pred)

# Plotting:
plt.figure(figsize=(8, 6))
heatmap(cm, annot=True, fmt="d", cmap="Blues", 
        xticklabels=range(10), yticklabels=range(10))
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.show()
