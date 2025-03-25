import xml.etree.ElementTree as ET
import tensorflow as tf
import numpy as np
import cv2
import os


# Directories:
annon_dir  = "VOC2012_train_val/Annotations"
images_dir = "VOC2012_train_val/JPEGImages"

# xml files:
xml_files = os.listdir(annon_dir)

# Subset:
subset      = ["dog", "cat", "car", "person"]
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

    scale_x = 224 / width
    scale_y = 224 / height
    
    # Fetching the classes:
    objects = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text

        # Only care about it if it is in our subset:
        if class_name in subset:
            class_idx = class_index[class_name]
            bndbox    = obj.find("bndbox")
            box = [
                float(bndbox.find("xmin").text) * scale_x / 224,
                float(bndbox.find("ymin").text) * scale_y / 224,
                float(bndbox.find("xmax").text) * scale_x / 224,
                float(bndbox.find("ymax").text) * scale_y / 224
            ]
            objects.append((class_idx, box))

    # Only store a image path if it has classes from our subset:
    if objects:
        annotations[img_path] = objects


# By only storing the image paths with classes from our subset
# We can speed up the lazy list and make sure it does not need
# to check images that dont contain the right classes:
image_paths = list(annotations.keys())


# Creating the lazy list generator that yields 1 class at a time:
def data_generator():
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        
        # Normalizing the pixel values:
        img = img / 255.0

        # Going through all the classes in the pictures:
        for classidx, box in annotations[img_path]:
            class_label = tf.keras.utils.to_categorical(class_idx, num_classes)
            yield img, (class_label, np.array(box, dtype=np.float32))


# Creating the lazy list:
dataset = tf.data.Dataset.from_generator(
    data_generator,     # Generator.
    output_signature=(  # Requires a consistent output shape.
        tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
        (tf.TensorSpec(shape=(num_classes,), dtype=tf.float32),
         tf.TensorSpec(shape=(4,), dtype=tf.float32))
    )
).batch(32).repeat().prefetch(tf.data.AUTOTUNE)


# Function for creating CNN:
def create_cnn():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    class_output = tf.keras.layers.Dense(4, activation='softmax', name='class_output')(x)
    box_output = tf.keras.layers.Dense(4, activation='sigmoid', name='box_output')(x) 
    return tf.keras.Model(inputs, [class_output, box_output])


model = create_cnn()
model.compile(
    optimizer = "adam",
    loss = {"class_output": "categorical_crossentropy", "box_output": "mse"},
    loss_weights={"class_output": 0.4, "box_output": 1.0},
    metrics={"class_output": "accuracy"}
)

# Training the CNN
# We need to add steps per epoch since the Model doesn't actually know the size
# of the dataset since it is in a lazy list.
steps = sum(len(objs) for objs in annotations.values()) // 32
model.fit(dataset, epochs=2, steps_per_epoch=steps)
