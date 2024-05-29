"""
This script is for cropping the pascal dataset using the annotations to be 
loaded and used as support/prompt image during the training
"""

import os
import xml.etree.ElementTree as ET
import cv2
import csv
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np
from torchvision import transforms

def crop_image_and_save_details(image_path, xml_path, output_dir, csv_path):
    # Read the image and XML annotation
    image = cv2.imread(image_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    

    

    # Initialize variables for image details
    image_name = os.path.basename(image_path)
    class_name = None
    difficulty_level = None

    # Extract image details from annotations
    for i, obj in enumerate(root.findall('object')):
        bndbox = obj.find('bndbox')
        try:
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
        except:
            breakpoint()

        class_name = obj.find('name').text
        difficulty_level = obj.find('difficult').text

        # Crop the image to the bounding box
        img_rgb = image[:, :, [2, 1, 0]]
        cropped_image = img_rgb[ymin:ymax, xmin:xmax]
        np_image = np.array(cropped_image)
        tensor = torch.from_numpy(np_image)
        tensor = tensor.permute(2, 0, 1)

        tensor_resized = transforms.Resize(510, interpolation=Image.BICUBIC, max_size=512, antialias=True)(tensor)

        #tensor_resized = tensor_resized.permute(2, 0, 1)
        cropped_image = transforms.ToPILImage()(tensor_resized)
        #cropped_image = cropped_image.convert('RGB')
        # np_resized_image = tensor_resized.numpy()
        # cropped_image = cv2.cvtColor(np_resized_image, )##cv2.COLOR_RGB2BGR)       #cropped_image





        # Save the cropped image to the output directory
        output_filename = os.path.basename(image_path).split('.')[0] + '_' + str(i) + '_' + class_name + '.jpg'
        output_path = os.path.join(output_dir, output_filename)
        cropped_image.save(output_path)
        ##cv2.imwrite(output_path, cropped_image)

        # Write image details to CSV file
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([image_name, class_name, difficulty_level, output_path])

# Specify the input directories
image_dir = '/home/bibahaduri/dataset/VOC2007_test/JPEGImages'
xml_dir = '/home/bibahaduri/dataset/VOC2007_test/Annotations'

# Specify the output directory and CSV file path
output_dir = '/home/bibahaduri/dataset/cropped_pascalvoc_test'
csv_path = '/home/bibahaduri/dataset/cropped_pascalvoc_test/image_details_test.csv'

# Create the output directory and CSV file if they don't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image Name', 'Class Name', 'Difficulty Level', 'Cropped Image Path'])

# Crop all images in the input directory and store their details in CSV
for image_filename in tqdm(os.listdir(image_dir), ):
    image_path = os.path.join(image_dir, image_filename)
    xml_filename = image_filename.split('.')[0] + '.xml'
    xml_path = os.path.join(xml_dir, xml_filename)

    crop_image_and_save_details(image_path, xml_path, output_dir, csv_path)