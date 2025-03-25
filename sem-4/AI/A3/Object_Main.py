import xml.etree.ElementTree as ET
import tensorflow as tf
import numpy as np
import json
import cv2
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Directories:
annon_dir = "VOC2012_train_val/Annotations"
images_dir = "VOC2012_train_val/JPEGImages"

# Loading config:
with open("config.json", "r") as f:
    config = json.load(f)

# Subset and dictionary for class to index.
subset = config["classes"]
num_classes = len(subset)
class_index = {c: i for i, c in enumerate(subset)}

print("Loaded: ", len(subset), " classes: ")
for i, c in enumerate(subset):
    print(i, ": Loaded ", c, " class")

# Grid parameters
grid_size = 7
output_filters = 5 + num_classes  # Confidence + 4 (Bounding box) + class prob

# Load and preprocess data:
print("\n... Loading and preprocessing data")
images      = []
y_true      = []
image_paths = [] # For predictions later in predict.py

# Go through all xml files:
for xml_file in os.listdir(annon_dir):
    tree = ET.parse(os.path.join(annon_dir, xml_file))
    root = tree.getroot()
    filename = os.path.join(images_dir, root.find("filename").text)
    
    # We need to find the scale that the picture later gets resized with
    # so that we can resize the bounding boxes
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)
    
    scale_x = 224 / width
    scale_y = 224 / height
    
    # Creating a grid 7x7 with a vector that contains:
    # [P, bnd box, class]
    # P = Confidence that htere is a class in the grid.
    # bnd box = bounding box coordinates.
    # Class = "one hot encoded" which class it in.
    y = np.zeros((grid_size, grid_size, output_filters))

    for obj in root.findall("object"):
        class_name = obj.find("name").text

        # If the class is not in our subset jump over.
        if class_name in subset:
            bndbox = obj.find("bndbox")
            # Original coordinates scaled and normalized to 0 -> 1:
            xmin = float(bndbox.find("xmin").text) * scale_x / 224
            ymin = float(bndbox.find("ymin").text) * scale_y / 224
            xmax = float(bndbox.find("xmax").text) * scale_x / 224
            ymax = float(bndbox.find("ymax").text) * scale_y / 224
            
            # Compute center, width, and height:
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin
            
            # Compute the grid cell:
            cell_x = int(center_x * grid_size)
            cell_y = int(center_y * grid_size)
            
            # Make sure it is inside the image. (Note: Might not be needed anymore remove later.)
            if 0 <= cell_x < grid_size and 0 <= cell_y < grid_size:
                # Adjusting x and y to be relative to the grid cell:
                x_cell = (center_x * grid_size) - cell_x
                y_cell = (center_y * grid_size) - cell_y
                
                box = np.array([x_cell, y_cell, w, h])
                
                class_idx = class_index[class_name]
                y[cell_y, cell_x, 0] = 1            # Confidence (Always 1 since we are not predicting here)
                y[cell_y, cell_x, 1:5] = box         # Bounding box (x, y, w, h)
                y[cell_y, cell_x, 5 + class_idx] = 1  # Class probability (Always 1 again, no prediction)
    
    # Check if there is a least a single class from our subset:
    if y[:, :, 0].any():
        # Reading, resizing and normalizing the pictures:
        img = cv2.imread(filename)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        images.append(img)
        y_true.append(y)
        image_paths.append(filename)  # For predict.py


print("Data preprocessed")

# Convert to numpy arrays:
print("\n... Converting to numpy arrays")
images = np.array(images)
y_true = np.array(y_true)
print("Conversion done\n")

#### FOR PREDICT.PY
# Save image paths for predict.py
with open("image_paths.json", "w") as f:
    json.dump(image_paths, f)
print("Image paths saved to image_paths.json")
####

# Build the model with split outputs (Branches):
print("\n... Building the model")
inputs = tf.keras.Input(shape=(224, 224, 3)) # 224, 224
# Feature extraction "layers":
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # 112, 112
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # 56, 56
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # 28, 28

# FC layer to branch with:
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)          # Dactivate 30% of neurons per pass, so that
                                             # the model does not over rely on a few neurons.

# Confidence Branch: Sigmoid classifictaion layer as we want output from 0 -> 1:
# In the different Branches we reshape the FC layer into a grid of (gridsize, gridsize, x)
# Where x is the part of the vector per grid (In this case only 1 since confidence is 1 long)
conf = tf.keras.layers.Dense(grid_size * grid_size * 1)(x)
conf = tf.keras.layers.Reshape((grid_size, grid_size, 1))(conf)
conf = tf.keras.layers.Activation('sigmoid', name='conf')(conf)

# Box Brach: Many FC layers to get just a tiny bit better accuracy.
# Sigmoid as we want the normalized bounding boxes back 0 -> 1:
# Rehsape (gridsize, gridsize, len(boundingboxes))
boxes = tf.keras.layers.Dense(512, activation='relu')(x)
boxes = tf.keras.layers.Dense(256, activation='relu')(boxes)
boxes = tf.keras.layers.Dropout(0.3)(boxes)
boxes = tf.keras.layers.Dense(128, activation='relu')(boxes)
boxes = tf.keras.layers.Dropout(0.3)(boxes)
boxes = tf.keras.layers.Dense(grid_size * grid_size * 4)(boxes)
boxes = tf.keras.layers.Reshape((grid_size, grid_size, 4))(boxes)
boxes = tf.keras.layers.Activation('sigmoid', name='boxes')(boxes)

# Classes branch: Softmax as we want a probability distrobution back.
# Reshape: Same deal.
classes = tf.keras.layers.Dense(grid_size * grid_size * num_classes)(x)
classes = tf.keras.layers.Reshape((grid_size, grid_size, num_classes))(classes)
classes = tf.keras.layers.Activation('softmax', name='classes')(classes)

outputs = tf.keras.layers.Concatenate(axis=-1)([conf, boxes, classes])
model   = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()

# Custom loss function:
def custom_loss(y_true, y_pred):
    # Spliting the tensors into components
    true_conf    = y_true[..., 0]  # (batch, 7, 7)               (Confidence)
    true_boxes   = y_true[..., 1:5]  # (batch, 7, 7, 4)          (Bounding boxes)
    true_classes = y_true[..., 5:]  # (batch, 7, 7, num_classes) (class list)
    
    pred_conf    = y_pred[..., 0]  # (batch, 7, 7)               (Same deal)         
    pred_boxes   = y_pred[..., 1:5]  # (batch, 7, 7, 4)          ----||----
    pred_classes = y_pred[..., 5:]  # (batch, 7, 7, num_classes) ----||----
    
    # Masks for cells with and without objects:
    # Having a mask for the cells that dont have classes in the allows
    # us to apply different weights to grid cells with objects and without.
    # Then we can create priorities for grids with and without weights on
    # bounding boxes and class predictions:
    obj_mask   = tf.cast(true_conf > 0, tf.float32)
    noobj_mask = 1.0 - obj_mask
    
    # Parameters for weighting
    weight_noobj = 0.3   # Weight for confidence when no object
    weight_coord = 9.0   # Weight for bounding box coordinates (Super high)
                         # Still sucks though.
    
    # 1. Confidence loss (using MSE):
    conf_diff       = tf.square(true_conf - pred_conf)
    conf_loss_obj   = tf.reduce_sum(obj_mask * conf_diff)
    conf_loss_noobj = tf.reduce_sum(noobj_mask * conf_diff)
    conf_loss       = conf_loss_obj + weight_noobj * conf_loss_noobj
    
    # 2. Bounding box loss (using MSE, only for cells with objects):
    box_diff = tf.square(true_boxes - pred_boxes)
    box_loss = tf.reduce_sum(obj_mask[..., tf.newaxis] * box_diff)
    box_loss = weight_coord * box_loss
    
    # 3. Class loss (using cross-entropy, only for cells with objects (Thanks mask!))
    cross_entropy = tf.keras.losses.categorical_crossentropy(
        true_classes, pred_classes, from_logits=False
    )
    class_loss = tf.reduce_sum(obj_mask * cross_entropy)
    
    # Total loss:
    total_loss = conf_loss + box_loss + class_loss
    return total_loss


# Compile and train model:
model.compile(optimizer='adam', loss=custom_loss)
print("\n... Training the model")
history = model.fit(images, y_true, epochs=35, batch_size=16, validation_split=0.2)
model.save("custom_model.keras")
print("Model saved as custom_model.keras")

# Plotting loss curve:
print("\n... Plotting the loss curve")
plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Training Loss", color="blue")
plt.plot(history.history["val_loss"], label="Validation Loss", color="red")
plt.title("Training and Validation Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
