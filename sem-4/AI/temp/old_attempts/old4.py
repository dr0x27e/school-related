import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import json
import cv2
import os


# Directories:
annon_dir  = "VOC2012_train_val/Annotations"
images_dir = "VOC2012_train_val/JPEGImages"

# Loading the config:
with open("config.json", "r") as f:
    config = json.load(f)

subset = config["classes"]
num_classes = len(subset)

# Directory to turn classes into indexes for one hot encoding:
class_index = {c: i for i, c in enumerate(subset)}


# Creating a list that holds all the classes and bounding boxes per image:
annotations = {}

for xml_file in os.listdir(annon_dir):
    tree = ET.parse(os.path.join(annon_dir, xml_file))
    root = tree.getroot()
    filename = os.path.join(images_dir, root.find("filename").text)
    
    # We need to resize the bounding boxes since we will resize the images
    # therfore we need to find the scale used to resize the images and use
    # this scale to scale the bounding boxes:
    width  = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    # We normalize them at the same time:
    scale_x_normalize = 224 / width / 224
    scale_y_normalize = 224 / height / 224

    # Now we check if the image even at all has 1 of the classes from our subset:
    objects = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name in subset:
            bndbox = obj.find("bndbox")
            box = np.array([
                float(bndbox.find("xmin").text) * scale_x_normalize,
                float(bndbox.find("ymin").text) * scale_y_normalize,
                float(bndbox.find("xmax").text) * scale_x_normalize,
                float(bndbox.find("ymax").text) * scale_y_normalize
            ])

            objects.append([
                tf.keras.utils.to_categorical(class_index[class_name], num_classes),
                box
            ])
    
    # Check if the image had any classes from our subclass
    if objects:
        annotations[filename] = np.array(objects)


# Creating an lazy list with resized images:
def data_generator():
    for image in annotations.keys():
        img = cv2.imread(image)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        yield img

lazy_images = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32)
    )
).repeat()

# Fetching all the one hot encoded classes and their bounding boxes:


# Function for creating CNN:
def create_cnn():
    print("... Creating custom CNN model")
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x) 
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    # Dropout: Sets x% of the prior Neuron layer into 0, this stops it from relying
    # too much on a few neurons. (after backpropagation the neurons get set to their
    # old value again)
    x = tf.keras.layers.Dropout(0.5)(x)         # For a more robust model
    class_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='class_output')(x)
    box_output = tf.keras.layers.Dense(4, activation='sigmoid', name='box_output')(x) 
    return tf.keras.Model(inputs, [class_output, box_output])


model = create_cnn()
model.compile(
    optimizer = "adam",
    loss = {"class_output": "categorical_crossentropy", "box_output": "mse"},
    # Lowered class weight as before boxes was improving to slowly.
    loss_weights={"class_output": 0.5, "box_output": 1.0},
    metrics={"class_output": "accuracy", "box_output": "mae"}
)

# Training the CNN
print("... Training the model")
history = model.fit(
    x = lazy_images,
    y = list(annotations.values()),
    epochs=2,
    steps_per_epoch = len(annotations.keys())
)

print("... Svaing the model ...")
# Saving the model so that we can use it to display predicted and acutal images:
model.save("sub_CNN.keras")
print("Model saved (sub_CNN.keras)")


# plotting the loss curve:
def plot_loss_curves(history):
    print("Plotting loss curves:")
    epochs = range(1, len(history.history['loss']) + 1)
    
    # Total loss
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, history.history['loss'], label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Class and Box losses
    plt.subplot(2, 1, 2)
    plt.plot(epochs, history.history['class_output_loss'], label='Train Class Loss')
    plt.plot(epochs, history.history['val_class_output_loss'], label='Val Class Loss')
    plt.plot(epochs, history.history['box_output_loss'], label='Train Box Loss')
    plt.plot(epochs, history.history['val_box_output_loss'], label='Val Box Loss')
    plt.title('Class and Box Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_loss_curves(history)

