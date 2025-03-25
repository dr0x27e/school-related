import xml.etree.ElementTree as ET
import tensorflow as tf
import numpy as np
import json
import cv2
import os
import random

# Directories:
annon_dir = "VOC2012_train_val/Annotations"
images_dir = "VOC2012_train_val/JPEGImages"

# Loading config:
with open("config.json", "r") as f:
    config = json.load(f)

subset = config["classes"]
num_classes = len(subset)
class_index = {c: i for i, c in enumerate(subset)}

# Grid parameters:
grid_size = 7
output_filters = 5 + num_classes  # Confidence + 4 for bounding box + class probabilities

print("\n... Fetching image paths from image_paths.json")
# Load image paths
with open("image_paths.json", "r") as f:
    image_paths = json.load(f)
print(f"Loaded {len(image_paths)} image paths")

# Load and preprocess data (same as main.py):
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
            xmin = float(bndbox.find("xmin").text) * scale_x / 224
            ymin = float(bndbox.find("ymin").text) * scale_y / 224
            xmax = float(bndbox.find("xmax").text) * scale_x / 224
            ymax = float(bndbox.find("ymax").text) * scale_y / 224
            
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin
            
            cell_x = int(center_x * grid_size)
            cell_y = int(center_y * grid_size)
            
            if 0 <= cell_x < grid_size and 0 <= cell_y < grid_size:
                x_cell = (center_x * grid_size) - cell_x
                y_cell = (center_y * grid_size) - cell_y
                
                box = np.array([x_cell, y_cell, w, h])
                
                class_idx = class_index[class_name]
                y[cell_y, cell_x, 0] = 1
                y[cell_y, cell_x, 1:5] = box
                y[cell_y, cell_x, 5 + class_idx] = 1
   
    if y[:, :, 0].any():
        img = cv2.imread(filename)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        images.append(img)
        y_true.append(y)

print("Data preprocessed")

print("\n... Converting to numpy arrays")
images = np.array(images)
y_true = np.array(y_true)
print("Conversion done\n")

##################################

# Load the trained model:
model = tf.keras.models.load_model("custom_model.keras", compile=False)
print("Model loaded successfully")

# Function to extract predictions:
def predictions(y_pred):
    classes = []
    scores = []
    boxes = []
    grid_positions = []
    
    y_pred = y_pred[0]  # Removeing "batch" dimension
    
    # Go through all grids.
    for i in range(grid_size):
        for j in range(grid_size):
            conf = y_pred[i, j, 0]
            # Check if there is atleast 50% confidence
            if conf > 0.5:
                # Extract bounding box (x, y, w, h)
                x_cell, y_cell, w, h = y_pred[i, j, 1:5]
                
                # Convert x, y from cell relative to image relative
                x = (j + x_cell) / grid_size
                y = (i + y_cell) / grid_size
                
                # Convert to back to xmin, ymin, xmax, ymax:
                xmin = (x - w / 2) * 224
                ymin = (y - h / 2) * 224
                xmax = (x + w / 2) * 224
                ymax = (y + h / 2) * 224
                
                # Clip to image boundaries:
                xmin = max(0, min(xmin, 224 - 1))
                ymin = max(0, min(ymin, 224 - 1))
                xmax = max(0, min(xmax, 224 - 1))
                ymax = max(0, min(ymax, 224 - 1))
                
                # Appned and shit to image:
                class_probs = y_pred[i, j, 5:]
                class_idx = np.argmax(class_probs)
                score = conf * class_probs[class_idx]
                
                boxes.append([xmin, ymin, xmax, ymax])
                classes.append(class_idx)
                scores.append(score)
                grid_positions.append((i, j))
    
    return classes, scores, boxes, grid_positions

# Visualization function to show ground truth and predicted classes with bounding boxes
def predict_and_draw(image_path, img, y_true_img):
    img = img / 255.0
    input_tensor = np.expand_dims(img, 0)

    img_actual = img.copy()
    img_predicted = img.copy()

    # Draw ground truth bounding boxes:
    for i in range(grid_size):
        for j in range(grid_size):
            if y_true_img[i, j, 0] > 0:
                x_cell, y_cell, w, h = y_true_img[i, j, 1:5]
                x = (j + x_cell) / grid_size
                y = (i + y_cell) / grid_size
                xmin = (x - w / 2) * 224
                ymin = (y - h / 2) * 224
                xmax = (x + w / 2) * 224
                ymax = (y + h / 2) * 224
                
                xmin = max(0, min(xmin, 224 - 1))
                ymin = max(0, min(ymin, 224 - 1))
                xmax = max(0, min(xmax, 224 - 1))
                ymax = max(0, min(ymax, 224 - 1))
                
                class_probs = y_true_img[i, j, 5:]
                class_idx = np.argmax(class_probs)
                class_name = subset[class_idx]
                cv2.rectangle(img_actual, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                cv2.putText(img_actual, class_name, (int(xmin), int(ymin) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    
    # Predict and get all predictions:
    y_pred = model.predict(input_tensor, verbose=0)
    classes, scores, boxes, grid_positions = predictions(y_pred)

    # Print predicted classes:
    print("Predicted classes for this image:")
    if len(classes) == 0:
        print("No predictions above confidence 50%.")
    else:
        for cls, score, (grid_y, grid_x) in zip(classes, scores, grid_positions):
            class_name = subset[cls]
            print(f"  Grid cell ({grid_y}, {grid_x}): {class_name} (confidence: {score:.2f})")

    # Draw predicted bounding boxes (blue because i like it :D)
    for box, score, cls in zip(boxes, scores, classes):
        xmin, ymin, xmax, ymax = box
        class_name = subset[cls]
        label = f"{class_name}: {score:.2f}"
        cv2.rectangle(img_predicted, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
        cv2.putText(img_predicted, label, (int(xmin), int(ymin) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Combine ground truth and predicted images side by side:
    # The jpegs are formatted in BGR (BLUE, GREEN, RED) that is why the.
    combined_img = np.hstack((img_actual, img_predicted))
    cv2.putText(combined_img, "Ground Truth", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(combined_img, "Predictions", (234, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    return combined_img

# Select 10 random images
num_images_to_show = 20
if len(images) < num_images_to_show:
    num_images_to_show = len(images)
    print(f"Only {num_images_to_show} images available, showing all.")
else:
    print(f"Selecting {num_images_to_show} random images to display.")

# Create a list of indices and shuffle them
indices = list(range(len(images)))
random.shuffle(indices)
selected_indices = indices[:num_images_to_show]

# Display the selected images
for idx in selected_indices:
    img_path = image_paths[idx]
    img = (images[idx] * 255).astype(np.uint8)
    y_true_img = y_true[idx]
    
    print(f"\nProcessing {img_path}")
    combined_img = predict_and_draw(img_path, img, y_true_img)
    
    # Display the combined image
    cv2.imshow(f"Image {idx}: Ground Truth vs Predictions", combined_img)
    cv2.waitKey(0)
    cv2.destroyWindow(f"Image {idx}: Ground Truth vs Predictions")

cv2.destroyAllWindows()
