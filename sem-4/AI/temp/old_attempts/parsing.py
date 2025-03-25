import os
import cv2
import xml.etree.ElementTree as et
import pandas as pd

# Directories:
annon_dir      = "Annotations"
images_dir     = "JPEGImages"
scaled_img_dir = "scaled_images"


# Class subset:
subset = ["car", "person", "dog", "cat"]


xml_file_names = os.listdir(annon_dir)
data = []

print("... Parsing data")

for xml_file in xml_file_names:
    # Fetching the current xml file:
    tree = et.parse(os.path.join(annon_dir, xml_file))
    # Fetching root:
    root = tree.getroot()
    
    # Fetching the fields:
    filename        = root.find("filename").text
    size            = root.find("size")
    original_width  = int(size.find("width").text)
    original_height = int(size.find("height").text)

    # Finding the scale / differance between the original image
    # size and the desired image size, so that we can scale the
    # bounding boxes to the right size:
    scale_x = 224 / original_width
    scale_y = 224 / original_height
    
    # Creating the image path:
    image_path   = os.path.join(images_dir, filename)
    image        = cv2.imread(image_path)
    scaled_image = image.resize((224, 224))
    
    # Creating the scaled image dir path:
    scaled_dir   = os.path.join(scaled_img_dir, filename)
    
    count = 0

    # Iterating through each class:
    for obj in root.findall("object"):
        # Fetching class:
        class_name = obj.find("name").text
        
        # Checking if the class is in the subset:
        if (class_name in subset):
            count += 1

            # Fetching bounding box:
            bndbox = obj.find("bndbox")
        
            # Fetching bounding box coordinates and scaling.
            # Then scaling again to 0 -> 1
            xmin = float(bndbox.find("xmin").text) * scale_x / 224
            ymin = float(bndbox.find("ymin").text) * scale_y / 224
            xmax = float(bndbox.find("xmax").text) * scale_x / 224
            ymax = float(bndbox.find("ymax").text) * scale_y / 224
        

            # Appending the class, its bounding box and its parent image file name. to data:
            data.append({
                "image_path": scaled_dir,
                "class":      class_name,
                "xmin":       xmin,
                "ymin":       ymin,
                "xmax":       xmax,
                "ymax":       ymax,
            })

    # Saving image if it contains 1 of the subsets:
    if count > 0:
        cv2.imwrite(scaled_dir, image)

print("... Creating dataframe")

# Creating a dataframe and storing it:
df = pd.DataFrame(data)
df.to_csv("annotations.csv", index=False)

print("Dataframe created and saved\n")
