import tensorflow as tf
import tensorflow.keras as ks

import numpy as np

# Function for preprocessing a single image at a time:
def preprocess_image(image, label):
    # Normalizing the image
    image = tf.cast(image, tf.float32) / 255.0

    # Resizing the image to 224 x 224
    image = tf.image.resize(image[..., tf.newaxis], [224, 224])

    # Converting the 1 channel (Grey) into 3 channel (RGB) by just repeating
    # The greyscale 3 times.
    image = tf.repeat(image, 3, axis=-1) # Shape: (224, 224, 3)

    return image, label


# Function to augment the picture:
# I avoid flipping/ rotating the images since this might confuse digits
# e.g., 9 turning into 6.
def augment_image(image, label):
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    return image, label


# Function for building / modifying the ResNet50 model:
def custom_ResNet50(freeze_base, weights):
    # Loading ResNet50 without thop FC layer
    # ResNet50 only has 1 FC layer, that has 1000 classes used for classification.
    base_model = tf.keras.applications.ResNet50(
        weights = weights,
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freezing the weights:
    # Not so that we can freeze by writing True in as a parameter (looks better.)
    base_model.trainable = not freeze_base

    # Building the custom model:
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),  # Converts to 2D
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    return model
