import tensorflow as tf
import json

def build_fnn():
    # Building the model:
    model = tf.keras.models.Sequential([
        # A Fully connected neural network, is just a bunch of nodes
        # that all connect to the next layer. here We have 128 "Neurons"
        # conencted to 64, then 32 and finally we have our classification layer
        # That uses softmax to do a propability distribute.
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    
    # Same as in the CNN.
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def build_cnn():
    # Building the model:
    model = tf.keras.models.Sequential([
    
        # Adding 3 layers of convelution and 2 layers of max pooling in between:
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
    
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
    
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    
        # Flattening the the output so that we can send it to the FC layers.
        tf.keras.layers.Flatten(),
    
        # Adding 2 FC layers, 1 64 in size and 1 in the final output 
        # size of 10 (classification layer).
        tf.keras.layers.Dense(64, activation="relu"),
    
        # Final layer uses softmax activation to tun the layer into a propability distrobution.
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    # Compile the model:
    # Optimizer: This takes the outputted gradients from the backpropagation
    # and updates the weights in the model.

    # Adam short for Adaptive Moment Estimation is one of the most popular
    # optimization algorithms to minimize the loss function during training.
    # loss function: categorical_crossentropy
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

