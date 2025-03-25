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

# xml files:
xml_files = os.listdir(annon_dir)

# Loading config to fetch subset:
with open("config.json", "r") as f:
    config = json.load(f)

# Subset:
subset      = config["classes"]
num_classes = len(subset)

# turning classes to indecies to later encode to categorical:
class_index = {c: i for i, c in enumerate(subset)}

# I am planning to create a lazy list for to speare my pc from having to
# load all the images in RAM, as my laptop cant handle it. But with a lazy
# list comes a problem. A lazy list requires a consistent output, this is
# fine for classification tasks where you for instance classify numbers.
# this is becuase there are only 10 classes to choose from and this is
# consistant, but here there can be anything from 1 to 100 dogs to 
# classify.

# Therefore my plan is to have a lazy list for the Images and an eager
# list / directory for the annotations as this is a much smaller subset
# of data compared to the images.

# My original plan was to not even use a lazy list for the images but just
# have a list of path's to the images, using the directory as a sort of
# database. But since reading from disk (IO) is slow, this would have been
# our bottle neck. By effectivly using prefetch in our lazy list we can fetch
# the images before they get used effectivly eliminating IO time.

# Prefetching annotations:
annotations = {}

print("... Fetching annotations")

for xml_file in xml_files:
    tree = ET.parse(os.path.join(annon_dir, xml_file))
    root = tree.getroot()

    filename = root.find("filename").text
    img_path = os.path.join(images_dir, filename)

    # Finding the scale of the images, so that we can
    # scale down / up the bounding boxes as they will not
    # change when we later resize the images.
    width  = int(root.find("size/width").text)
    height = int(root.find("size/height").text)
    
    # Scaling and normalizing.
    scale_x_normalize = 224 / width / 224
    scale_y_normalize = 224 / height / 224
    
    # Fetching the classes:
    objects = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text

        # Only care about it if it is in our subset:
        if class_name in subset:
            class_idx = class_index[class_name]
            bndbox    = obj.find("bndbox")
            box = [
                float(bndbox.find("xmin").text) * scale_x_normalize,
                float(bndbox.find("ymin").text) * scale_y_normalize,
                float(bndbox.find("xmax").text) * scale_x_normalize,
                float(bndbox.find("ymax").text) * scale_y_normalize
            ]
            objects.append((class_idx, box))

    # Only store a image path if it has classes from our subset:
    if objects:
        annotations[img_path] = objects


# By only storing the image paths with classes from our subset
# We can speed up the lazy list and make sure it does not need
# to check images that dont contain the right classes:
image_paths = list(annotations.keys())

print("... Splitting dataset")

# Splitting the data into validation and training data:
# We do this so that we can check how well it does against data
# that it has not trained on, making sure that it does not get overfitted.
val_split = int(0.2 * len(image_paths))
val_paths = image_paths[:val_split]
train_paths = image_paths[val_split:]
val_annotations = {k: annotations[k] for k in val_paths}
train_annotations = {k: annotations[k] for k in train_paths}

# Generator function for the lazy lists:
def gen(annots):
    for img_path in annots:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0   # Normalizing.
        for class_idx, box in annots[img_path]:
            yield img 
                       

# Generating lazy list for training dataset and validation dataset.
train_dataset = tf.data.Dataset.from_generator(
    lambda: gen(train_annotations), # Generator function
    output_signature=(              # Output must be consistent
        tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32)
    )
).batch(32).repeat().prefetch(tf.data.AUTOTUNE)

# Added repeat, since i came across a problem where the data generator got "exhausted"
# (stopped generating data) (couldn't handle more than 2 epochs) now with .repeat
# if it ever reaches its end it will begin from the beginning again.

val_dataset = tf.data.Dataset.from_generator(
    lambda: gen(val_annotations), 
    output_signature=(
        tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32)
    )
).batch(32).repeat().prefetch(tf.data.AUTOTUNE)


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
# We need to add steps per epoch since the Model doesn't actually know the size
# of the dataset since it is in a lazy list. We do this by going through all
# image's objects "annotations.calues()" then taking the length aka how many
# of these objects there are in this picture, then we sum all of them.
# Finally we divide by 32 as this is the batch size we put inside the lazy
# lists example: We have 12000 images and each image has an average of 3 obj's
# then we will get 36000 different objects but we generate them in batches of 32
# so we then divide by 32 which is 1125. this represents the lazy lists length.
print("... Training the model")
history = model.fit(
    train_dataset, 
    validation_data=val_dataset, 
    epochs=2, 
    steps_per_epoch=sum(len(objs) for objs in train_annotations.values()) // 32,
    validation_steps=sum(len(objs) for objs in val_annotations.values()) // 32,
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
