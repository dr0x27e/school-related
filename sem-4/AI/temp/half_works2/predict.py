import xml.etree.ElementTree as ET
import tensorflow as tf
import numpy as np
import json
import cv2
import os
import random

# Directories
annon_dir = "VOC2012_train_val/Annotations"
images_dir = "VOC2012_train_val/JPEGImages"

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

subset = config["classes"]
num_classes = len(subset)
class_index = {c: i for i, c in enumerate(subset)}

# Grid parameters
grid_size = 7
output_filters = 1 + num_classes  # Only confidence + class probabilities

# Load image paths
with open("image_paths.json", "r") as f:
    image_paths = json.load(f)
print(f"Loaded {len(image_paths)} image paths")

# Load and preprocess data (same as main.py)
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
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            
            cell_x = int(center_x * grid_size)
            cell_y = int(center_y * grid_size)
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

# Load the trained model
model = tf.keras.models.load_model("yolo_model.keras", custom_objects={"yolo_loss": lambda y_true, y_pred: y_true})
print("Model loaded successfully")

# Function to extract predictions (no boxes)
def yolo_to_predictions(y_pred, conf_threshold=0.5):
    classes = []
    scores = []
    grid_positions = []
    
    y_pred = y_pred[0]  # Remove batch dimension
    
    for i in range(grid_size):
        for j in range(grid_size):
            conf = y_pred[i, j, 0]
            if conf > conf_threshold:
                class_probs = y_pred[i, j, 1:]
                class_idx = np.argmax(class_probs)
                score = conf * class_probs[class_idx]
                classes.append(class_idx)
                scores.append(score)
                grid_positions.append((i, j))
    
    return classes, scores, grid_positions

# Visualization function to show ground truth and predicted classes
def predict_and_draw(image_path, img, y_true_img, conf_threshold=0.4):
    img_rgb = img / 255.0
    input_tensor = np.expand_dims(img_rgb, 0)

    img_actual = img.copy()
    img_predicted = img.copy()

    # Draw ground truth grid cells (green dots)
    for i in range(grid_size):
        for j in range(grid_size):
            if y_true_img[i, j, 0] > 0:
                class_probs = y_true_img[i, j, 1:]
                class_idx = np.argmax(class_probs)
                class_name = subset[class_idx]
                # Compute the center of the grid cell in image coordinates
                cell_center_x = (j + 0.5) * (224 / grid_size)
                cell_center_y = (i + 0.5) * (224 / grid_size)
                # Draw a green dot at the center of the grid cell
                cv2.circle(img_actual, (int(cell_center_x), int(cell_center_y)), 5, (0, 255, 0), -1)
                cv2.putText(img_actual, class_name, (int(cell_center_x) + 10, int(cell_center_y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Predict and get all predictions
    y_pred = model.predict(input_tensor, verbose=0)
    classes, scores, grid_positions = yolo_to_predictions(y_pred, conf_threshold=conf_threshold)

    # Print predicted classes in the terminal
    print("Predicted classes for this image:")
    if len(classes) == 0:
        print("  No predictions above confidence threshold.")
    else:
        for cls, score, (grid_y, grid_x) in zip(classes, scores, grid_positions):
            class_name = subset[cls]
            print(f"  Grid cell ({grid_y}, {grid_x}): {class_name} (confidence: {score:.2f})")

    # Draw predicted grid cells (red dots)
    for (grid_y, grid_x), score, cls in zip(grid_positions, scores, classes):
        class_name = subset[cls]
        # Compute the center of the grid cell in image coordinates
        cell_center_x = (grid_x + 0.5) * (224 / grid_size)
        cell_center_y = (grid_y + 0.5) * (224 / grid_size)
        # Draw a red dot at the center of the grid cell
        cv2.circle(img_predicted, (int(cell_center_x), int(cell_center_y)), 5, (0, 0, 255), -1)
        cv2.putText(img_predicted, f"{class_name}: {score:.2f}", (int(cell_center_x) + 10, int(cell_center_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Combine ground truth and predicted images side by side
    combined_img = np.hstack((img_actual, img_predicted))
    cv2.putText(combined_img, "Ground Truth", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(combined_img, "Predictions", (234, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
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
    combined_img = predict_and_draw(img_path, img, y_true_img, conf_threshold=0.5)
    
    # Display the combined image
    cv2.imshow(f"Image {idx}: Ground Truth vs Predictions", combined_img)
    cv2.waitKey(0)  # Wait for a key press to move to the next image
    cv2.destroyWindow(f"Image {idx}: Ground Truth vs Predictions")

cv2.destroyAllWindows()
