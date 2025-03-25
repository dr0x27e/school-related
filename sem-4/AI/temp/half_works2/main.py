import xml.etree.ElementTree as ET
import tensorflow as tf
import numpy as np
import json
import cv2
import os

# Directories
annon_dir = "VOC2012_train_val/Annotations"
images_dir = "VOC2012_train_val/JPEGImages"

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

subset = config["classes"]
num_classes = len(subset)
class_index = {c: i for i, c in enumerate(subset)}

print("Loaded: ", len(subset), " classes: ")
for i, c in enumerate(subset):
    print(i, ": Loaded ", c, " class")

# Grid parameters
grid_size = 7
output_filters = 1 + num_classes  # Only confidence + class probabilities (no boxes)

# Load and preprocess data
print("\n... Loading and preprocessing data")
images = []
y_true = []

for xml_file in os.listdir(annon_dir):
    tree = ET.parse(os.path.join(annon_dir, xml_file))
    root = tree.getroot()
    filename = os.path.join(images_dir, root.find("filename").text)

    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)
    
    scale_x = 224 / width
    scale_y = 224 / height
    
    y = np.zeros((grid_size, grid_size, output_filters))

    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name in subset:
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            
            # Normalize coordinates to [0, 1] in 224x224 image
            xmin = (xmin * scale_x) / 224
            ymin = (ymin * scale_y) / 224
            xmax = (xmax * scale_x) / 224
            ymax = (ymax * scale_y) / 224

            # Compute center to determine grid cell
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            
            cell_x = int(cx * grid_size)
            cell_y = int(cy * grid_size)
            if 0 <= cell_x < grid_size and 0 <= cell_y < grid_size:
                class_idx = class_index[class_name]
                y[cell_y, cell_x, 0] = 1  # Confidence (object present)
                y[cell_y, cell_x, 1 + class_idx] = 1  # Class probability
   
    if y[:, :, 0].any():
        img = cv2.imread(filename)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        images.append(img)
        y_true.append(y)

print("Data preprocessed")

# Convert to numpy arrays
print("\n... Converting to numpy arrays")
images = np.array(images)
y_true = np.array(y_true)
print("Conversion done\n")
print("Predicate shape: ", y_true.shape)
print("Images shape: ", images.shape)

# Build the model with split outputs (no boxes)
print("\n... Building the model")
inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)

conf = tf.keras.layers.Dense(grid_size * grid_size * 1)(x)
conf = tf.keras.layers.Reshape((grid_size, grid_size, 1))(conf)
conf = tf.keras.layers.Activation('sigmoid', name='conf')(conf)

classes = tf.keras.layers.Dense(grid_size * grid_size * num_classes)(x)
classes = tf.keras.layers.Reshape((grid_size, grid_size, num_classes))(classes)
classes = tf.keras.layers.Activation('softmax', name='classes')(classes)

outputs = tf.keras.layers.Concatenate(axis=-1)([conf, classes])
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()

# Custom loss function (no box loss)
def yolo_loss(y_true, y_pred):
    """
    Custom loss function for confidence and class prediction only.
    
    Args:
        y_true: Ground truth tensor of shape (batch, 7, 7, 1 + num_classes)
        y_pred: Predicted tensor of shape (batch, 7, 7, 1 + num_classes)
    
    Returns:
        Total loss
    """
    # Split the tensors into components
    true_conf = y_true[..., 0]  # (batch, 7, 7)
    true_classes = y_true[..., 1:]  # (batch, 7, 7, num_classes)
    
    pred_conf = y_pred[..., 0]  # (batch, 7, 7)
    pred_classes = y_pred[..., 1:]  # (batch, 7, 7, num_classes)
    
    # Masks for cells with and without objects
    obj_mask = tf.cast(true_conf > 0, tf.float32)  # (batch, 7, 7)
    noobj_mask = 1.0 - obj_mask  # (batch, 7, 7)
    
    # Hyperparameters for weighting
    lambda_noobj = 0.4  # Weight for confidence when no object
    
    # 1. Confidence loss
    conf_diff = tf.square(true_conf - pred_conf)  # (batch, 7, 7)
    conf_loss_obj = tf.reduce_sum(obj_mask * conf_diff)
    conf_loss_noobj = tf.reduce_sum(noobj_mask * conf_diff)
    conf_loss = conf_loss_obj + lambda_noobj * conf_loss_noobj
   
    # 2. Class loss (using cross-entropy, only for cells with objects)
    # Compute cross-entropy loss for each grid cell
    cross_entropy = tf.keras.losses.categorical_crossentropy(
        true_classes, pred_classes, from_logits=False
    )  # (batch, 7, 7)
    
    # Apply obj_mask to only consider cells with objects
    class_loss = tf.reduce_sum(obj_mask * cross_entropy)
    
    # Total loss
    total_loss = conf_loss + class_loss
    return total_loss

# Compile and train
model.compile(optimizer='adam', loss=yolo_loss)
print("\n... Training the model")
model.fit(images, y_true, epochs=25, batch_size=16, validation_split=0.2)
model.save("yolo_model.keras")
print("Model saved as yolo_model.keras")
