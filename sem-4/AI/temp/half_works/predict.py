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

subset = config["classes"]  # ["dog", "cat", "car"]
num_classes = len(subset)
class_index = {c: i for i, c in enumerate(subset)}

# Grid parameters
grid_size = 7
output_filters = 5 + num_classes

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
            # Original coordinates (normalized to [0, 1] in 224x224 image)
            xmin = float(bndbox.find("xmin").text) * scale_x / 224
            ymin = float(bndbox.find("ymin").text) * scale_y / 224
            xmax = float(bndbox.find("xmax").text) * scale_x / 224
            ymax = float(bndbox.find("ymax").text) * scale_y / 224
            
            # Compute center, width, and height
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin
            
            # Compute the grid cell
            cell_x = int(center_x * grid_size)
            cell_y = int(center_y * grid_size)
            
            if 0 <= cell_x < grid_size and 0 <= cell_y < grid_size:
                # Adjust x, y to be relative to the grid cell
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

# Function to extract all bounding box predictions
def yolo_to_boxes(y_pred, img_width=224, img_height=224, conf_threshold=0.5):
    boxes = []
    scores = []
    classes = []
    
    y_pred = y_pred[0]  # Remove batch dimension
    
    for i in range(grid_size):
        for j in range(grid_size):
            conf = y_pred[i, j, 0]
            if conf > conf_threshold:
                # Predicted x, y, w, h
                x_cell, y_cell, w, h = y_pred[i, j, 1:5]
                
                # Convert x, y from cell-relative to image-relative
                x = (j + x_cell) / grid_size  # x in [0, 1]
                y = (i + y_cell) / grid_size  # y in [0, 1]
                
                # Convert to xmin, ymin, xmax, ymax
                xmin = (x - w / 2) * img_width
                ymin = (y - h / 2) * img_height
                xmax = (x + w / 2) * img_width
                ymax = (y + h / 2) * img_height
                
                # Clip to image boundaries
                xmin = max(0, min(xmin, img_width - 1))
                ymin = max(0, min(ymin, img_height - 1))
                xmax = max(0, min(xmax, img_width - 1))
                ymax = max(0, min(ymax, img_height - 1))
                
                class_probs = y_pred[i, j, 5:]
                class_idx = np.argmax(class_probs)
                score = conf * class_probs[class_idx]
                boxes.append([xmin, ymin, xmax, ymax])
                scores.append(score)
                classes.append(class_idx)
    
    return np.array(boxes), np.array(scores), np.array(classes)

# Visualization function to plot all predictions and print predicted classes
def predict_and_draw(image_path, img, y_true_img, conf_threshold=0.5):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_tensor = np.expand_dims(img_rgb, 0)

    img_actual = img.copy()
    img_predicted = img.copy()

    # Draw ground truth boxes (green)
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

    # Predict and get all bounding boxes
    y_pred = model.predict(input_tensor, verbose=0)
    boxes, scores, classes = yolo_to_boxes(y_pred, conf_threshold=conf_threshold)

    # Print predicted classes in the terminal
    print("Predicted classes for this image:")
    if len(classes) == 0:
        print("  No predictions above confidence threshold.")
    else:
        for cls, score in zip(classes, scores):
            class_name = subset[cls]
            print(f"  {class_name}: {score:.2f}")

    # Draw all predicted boxes (red)
    for box, score, cls in zip(boxes, scores, classes):
        xmin, ymin, xmax, ymax = box
        class_name = subset[cls]
        label = f"{class_name}: {score:.2f}"
        cv2.rectangle(img_predicted, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
        cv2.putText(img_predicted, label, (int(xmin), int(ymin) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the images
    cv2.imshow("Ground Truth", img_actual)
    cv2.imshow("All Predictions", img_predicted)
    cv2.waitKey(0)

# Interactive loop to visualize predictions
running = ''
for idx, (img_path, img, y_true_img) in enumerate(zip(image_paths, images, y_true), 1):
    if running == 'q' or idx > 5:
        break
    print(f"\nProcessing {img_path}")
    predict_and_draw(img_path, (img * 255).astype(np.uint8), y_true_img, conf_threshold=0.5)
    running = input("Next picture (Enter/q): ")

cv2.destroyAllWindows()
