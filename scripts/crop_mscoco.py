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
import math
from pycocotools.coco import COCO
import pickle


def crop_image_and_save_details(image_path, coco_annots, output_dir, csv_path, pkl_annot):
    """
    coco_annots: dictionary of coco annotations
    pkl_annot:  annotation loaded from pkl file includes

    """
    image = cv2.imread(image_path)

    

    

    # Initialize variables for image details
    image_name = os.path.basename(image_path)
    class_name = None
    for i, obj in enumerate(coco_annots):
        if i in pkl_annot:
            if pkl_annot[i]['iou'] > 0.5 and pkl_annot[i]['score'] > 0.7:
                #if obj['iscrowd'] == 0:
                xmin = math.ceil(obj['bbox'][0])
                ymin = math.ceil(obj['bbox'][1])
                xmax = xmin + math.ceil(obj['bbox'][2])
                ymax = ymin + math.ceil(obj['bbox'][3])


                class_name = obj['category_id']
                difficulty_level = obj['iscrowd']
                annot_id = obj['id']
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
                output_filename = os.path.basename(image_path).split('.')[0] + '_' + str(i) + '_' + str(class_name) + '.jpg'
                output_path = os.path.join(output_dir, output_filename)
                cropped_image.save(output_path)
                ##cv2.imwrite(output_path, cropped_image)

                # Write image details to CSV file
                with open(csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([image_name, class_name,difficulty_level,annot_id , output_path])
        #     else:
        #         continue
        # else:
        #     continue



# Specify the input directories
image_dir = '/home/bibahaduri/data/coco/val2017'
xml_dir = '/home/bibahaduri/dataset/VOC2007_test/Annotations'

# Specify the output directory and CSV file path
output_dir = '/home/bibahaduri/data/coco/coco_crop_test_id'
csv_path = '/home/bibahaduri/data/coco/coco_crop_test_id/image_details.csv'
pkl_path = '/home/bibahaduri/data/coco/coco_val2017_e2e_mask_rcnn_R_101_FPN_1x_caffe2.pkl'
coco_instances_path = '/home/bibahaduri/data/coco/annotations/instances_val2017.json'
# Create the output directory and CSV file if they don't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image Name', 'Class Name','Difficulty Level' , 'Annot Id','Cropped Image Path'])

# Crop all images in the input directory and store their details in CSV
with open(pkl_path, 'rb') as f:
    imgs_ref = pickle.load(f)
  # Adjust the path as needed
coco = COCO(coco_instances_path)
images = coco.imgs

for image_id, image_info in tqdm(images.items()):
    image_path = os.path.join(image_dir, image_info['file_name'])
    ann_ids = coco.getAnnIds(imgIds=image_id)
    coco_annots = coco.loadAnns(ann_ids)
    pkl_annots = imgs_ref[image_id]
    crop_image_and_save_details(image_path, coco_annots, output_dir, csv_path, pkl_annots)