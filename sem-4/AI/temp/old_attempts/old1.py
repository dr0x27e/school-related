import xml.etree.ElementTree as ET
import tensorflow as tf
import numpy as np
import cv2
import os


# Directories:
annon_dir  = "Annotations"
images_dir = "JPEGImages"

# Getting all the XML file names:
# We will uses these in the generator function for our lazy list
xml_files = os.listdir(annon_dir)

# Subset of classes:
subset      = ["dog", "cat", "car", "person"]
num_classes = len(subset)

# We need a max number of objects that we can have each in each image
# This is becuase if we want to read the xml files lazily (which I have to 
# since I dont have enough RAM otherwise) we need to use:
# tf.data.Dataset.from_generator, and this demands a fixed "output_signature"
# this forces us to predefine a tensor shape.
# If we wanted to we could search through all objects in the xml files and find
# what the most amount of objects there are in 1 file (I did this and its 112) 
# this could work, but this will pad most images with TONS of 
# zeros which will slow training significantly.

# Solutions:
# We could solve this with the use of pandas.
# Unfortunatly this is not allowed in the task, or so i atleast belive since it
# is not stated that we can use it.

# I have already done this however, the solution would be this:
# First we parse the images that are in the Annotations .xml files "<filename>".
# Then we save these resized images into a new directory. This new directory
# will then act as a sort of database.
# We also from the xml files create a dataframe (this is where pandas comes in)
# this dataframe (csv) holds: | image_file | classes | bounding boxes |
# here we use the image_file column as a foregin key to query into our "data base"
# saving us from having all the images loaded into RAM at once and eliminating
# the need for a lazy list with a constant output.

max_objects = 3

# Making a dictionary that turns the class strings into indecies
class_idx = {c: i for i, c in enumerate(subset)}


# Building a data generator function for our lazy list:
def data_generator():
    for xml_file in xml_files:
        # Parsing the xml:
        tree = ET.parse(os.path.join(annon_dir, xml_file))
        root = tree.getroot()
        
        # We need to scale the image and bounding boxes:
        width  = int(root.find("size/width").text)
        height = int(root.find("size/height").text)
        # Further than just scaling the bounding boxes to fit the
        # downscaled / upscaled image we need to normalize them:
        scale_x = (224 / width)
        scale_y = (224 / height)
        
        # We check the xml objects (classes) to see
        # if there are any of the subclasses that we want:
        count = 0

        objects = root.findall("object")
        class_labels = np.zeros((max_objects, num_classes), dtype=np.float32)
        boxes        = np.zeros((max_objects, 4), dtype=np.float32)
        
        # enumerating through each class:
        for obj in objects:
            class_name  = obj.find("name").text
            
            # Check if the class is in our subset:
            if class_name in subset:
                # Cap at max_objects
                if count >= max_objects: break

                class_index = class_idx[class_name]
                # Encode the class_indecies to tensorflow format
                # e.g., 3 -> [0 0 0 1] meaning person in our example.
                class_labels[count] = tf.keras.utils.to_categorical(
                    class_index, num_classes
                )

                # Fetching the bounding boxes:
                bndbox = obj.find("bndbox")
                xmin   = float(bndbox.find("xmin").text) * scale_x / 224
                ymin   = float(bndbox.find("ymin").text) * scale_y / 224
                xmax   = float(bndbox.find("xmax").text) * scale_x / 224
                ymax   = float(bndbox.find("ymax").text) * scale_y / 224
                
                # Appending the bounding box:
                boxes[count] = [xmin, ymin, xmax, ymax]

                count = count + 1

        
        # Only yeild if we found an image with the subset classes:
        if count > 0:
            # Fetching the corresponding jpg image:
            filename = root.find("filename").text


            # Loading and processing the image:
            img_path = os.path.join(images_dir, filename)
            img = cv2.imread(img_path)

            # Scaling the image to 224 x 224
            img = cv2.resize(img, (224, 224))

            # Normalizing the rgb values:
            img = img / 255

            yield img, (class_labels, boxes)


# Creating the lazy list (dataset):
dataset = tf.data.Dataset.from_generator(
    data_generator,     # Generator function.
    output_signature=(  # Requires consistant output shape.
        tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
        (tf.TensorSpec(shape=(max_objects, num_classes), dtype=tf.float32),
         tf.TensorSpec(shape=(max_objects, 4), dtype=tf.float32))
    )
# Pipelining for maximum speed.
).batch(32).prefetch(tf.data.AUTOTUNE)

# Build CNN from scratch
def create_cnn():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    # Convolutional layers
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # 112x112
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # 56x56
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # 28x28
    
    # Flatten and dense layers
    x = tf.keras.layers.Flatten()(x)  # 28x28x128 = 100352 features
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    
    # Output heads
    class_output = tf.keras.layers.Dense(max_objects * num_classes, activation='softmax')(x)
    class_output = tf.keras.layers.Reshape((max_objects, num_classes), name='class_output')(class_output)
    
    box_output = tf.keras.layers.Dense(max_objects * 4, activation='sigmoid')(x)
    box_output = tf.keras.layers.Reshape((max_objects, 4), name='box_output')(box_output)
    
    return tf.keras.Model(inputs, [class_output, box_output])

# Create and compile model
model = create_cnn()
model.compile(optimizer='adam',
              loss={'class_output': 'categorical_crossentropy', 'box_output': 'mse'},
              loss_weights={'class_output': 1.0, 'box_output': 0.1},
              metrics={'class_output': 'accuracy'})

# Train
model.fit(dataset, epochs=2, verbose=2)
