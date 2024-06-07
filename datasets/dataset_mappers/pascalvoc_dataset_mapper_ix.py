# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py

## shape_sample and its query used for sampling prompts were removed

import copy
import os
# import logging

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
import pandas as pd
# from detectron2.config import configurable
import datasets.transforms as T
from torchvision.transforms import InterpolationMode
from torchvision import transforms
from detectron2.structures import BitMasks, Boxes, Instances
from detectron2.data import MetadataCatalog, Metadata
import random
import pickle

# from utils import prompt_engineering
# from model.utils import configurable##, PASCAL_CLASSES
# from registration.register_pascalvoc_eval import PASCAL_CLASSES
# from ..visual_sampler import build_shape_sampler
# TODO:  a background should be added to pascal_classes based on SEEM codes for segmentation!!
__all__ = ["PascalVOCSegDatasetMapperIX"]

#from model.utils.config import configurable
PASCAL_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def make_self_det_transforms(training):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # The image of ImageNet is relatively small.
    scales = [400, 480, 512, 544]#, 576, 608, 640]##[320, 336, 352, 368, 400, 416, 432, 448, 464, 480]

    if training == True:
        # return T.Compose([
        #     # T.RandomHorizontalFlip(), HorizontalFlip may cause the pretext too difficult, so we remove it
        #     T.RandomResize(scales, max_size=512),
        #     normalize,
        # ])
        return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=512),  #642
                    T.Compose(
                        [
                            T.RandomResize([400, 480, 512]), #500, 600
                            T.RandomSizeCrop(384, 512), #384, 600
                            T.RandomResize(scales, max_size=512), #642
                        ]
                    ),
                ),
                normalize,
            ])

    if training == False:
        return T.Compose([
            T.RandomResize([512], max_size=512), #480    642
            normalize,
        ])


def get_query_transforms(training):
    if training == True:
        # SimCLR style augmentation
        return transforms.Compose([
            transforms.Resize((300, 300)),  ##512, 512
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=InterpolationMode.BILINEAR, fill=0),
            transforms.RandomGrayscale(p=0.4),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(),  HorizontalFlip may cause the pretext too difficult, so we remove it
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    if training == False:
        return transforms.Compose([
            transforms.Resize((300, 300)), #512, 512
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

# This is specifically designed for the COCO dataset.
class PascalVOCSegDatasetMapperIX:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    ##@configurable
    def __init__(
            self,
            is_train=True,
            dataset_name='',
            min_size_test=None,
            max_size_test=None,
            evalall=False,
            grounding=False,
            coco_split=None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.is_train = is_train
        self.dataset_name = dataset_name
        self.min_size_test = min_size_test
        self.max_size_test = max_size_test
        self.evalall = evalall
        self.coco_split = coco_split
        # self.base_ind = [1,2,3,4,5,6,8,10,11,12,13,14,15,17,18,19]
        # self.base_classes=[ "bicycle", "bird", "boat", "bottle", "bus", "car",
        #                     "chair", "diningtable", "dog", "horse", "motorbike", "person",
        #                         "pottedplant", "sofa", "train", "tvmonitor"]
        #self.ref_coco = pickle.load(open('/home/bibahaduri/data/coco/id_ref_instances_val2017.pkl', 'rb'), encoding='utf-8')
        if "pascalvoc_" in self.dataset_name:
            self.base_classes=[ "bicycle", "bird", "boat", "bottle", "bus", "car",
                                "chair", "diningtable", "dog", "horse", "motorbike", "person",
                                    "pottedplant", "sofa", "train", "tvmonitor"]
            if self.is_train:
                self.base_ind = [1,2,3,4,5,6,8,10,11,12,13,14,15,17,18,19]
            else:
                self.base_ind = [0, 7, 9, 16]
            
        elif "coco_" in self.dataset_name:
            if self.is_train:
                if self.coco_split == 1:
                    self.base_ind = [2, 3, 4, 6, 7, 8, 10, 11, 13, 15, 16, 17, 19, 20, 21, 23, 24, 25, 28, 
                                31, 32, 34, 35, 36, 38, 39, 40, 42, 43, 44, 47, 48, 49, 51, 52, 53, 55, 
                                56, 57, 59, 60, 61, 63, 64, 65, 70, 72, 73, 75, 76, 77, 79, 80, 81, 84, 
                                85, 86, 88, 89, 90]
                elif self.coco_split == 2:
                    self.base_ind = [1, 3, 4, 5, 7, 8, 9, 11, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 27, 
                                      31, 32, 33, 35, 36, 37, 39, 40, 41, 43, 44, 46, 48, 49, 50, 52, 53, 
                                      54, 56, 57, 58, 60, 61, 62, 64, 65, 67, 72, 73, 74, 76, 77, 78, 80, 
                                      81, 82, 85, 86, 87, 89, 90]
                elif self.coco_split == 3:
                    self.base_ind = [1, 2, 4, 5, 6, 8, 9, 10, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 27, 
                                      28, 32, 33, 34, 36, 37, 38, 40, 41, 42, 44, 46, 47, 49, 50, 51, 53, 
                                      54, 55, 57, 58, 59, 61, 62, 63, 65, 67, 70, 73, 74, 75, 77, 78, 79, 
                                      81, 82, 84, 86, 87, 88, 90]
                elif self.coco_split == 4:
                    self.base_ind = [1, 2, 3, 5, 6, 7, 9, 10, 11, 14, 15, 16, 18, 19, 20, 22, 23, 24, 27, 28, 
                                      31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47, 48, 50, 51, 52, 54, 55, 
                                      56, 58, 59, 60, 62, 63, 64, 67, 70, 72, 74, 75, 76, 78, 79, 80, 82, 84, 
                                      85, 87, 88, 89]
                else:
                    assert False, f"coco split {self.coco_split} is not known"
                # mode of group 4 should be 0 in register


            else:
                if self.coco_split == 1:
                    self.base_ind = [1, 5, 9, 14, 18, 22, 27, 33, 37, 41, 46, 50, 54, 58, 62, 67, 74, 78, 82, 87]
                elif self.coco_split == 2:
                    self.base_ind = [2, 6, 10, 15, 19, 23, 28, 34, 38, 42, 47, 51, 55, 59, 63, 70, 75, 79, 84, 88]
                elif self.coco_split == 3:
                    self.base_ind = [3, 7, 11, 16, 20, 24, 31, 35, 39, 43, 48, 52, 56, 60, 64, 72, 76, 80, 85, 89]
                elif self.coco_split == 4:
                    self.base_ind = [4, 8, 13, 17, 21, 25, 32, 36, 40, 44, 49, 53, 57, 61, 65, 73, 77, 81, 86, 90]
                else:
                    assert False, f"coco split {self.coco_split} is not known"
        elif "dota" in self.dataset_name:
            if self.is_train:
                self.base_ind = [1,2,4,6,7,8,9,10,11,12,13,14,16]
            else:
                self.base_ind = [3,5,15]




        self.transform_img = make_self_det_transforms(self.is_train)#transforms.Compose(t)
        self.query_transform = get_query_transforms(self.is_train)
        self.ignore_id = 220
        if "pascalvoc" in self.dataset_name:
            if self.is_train:
                file_path = 'data/pascalvoc/cropped_pascalvoc/image_details.csv'
            else:
                file_path = 'data/pascalvoc/cropped_pascalvoc_test/image_details_test.csv'
        elif "coco" in self.dataset_name:
            if self.is_train:
                file_path = 'data/coco/coco_crop/image_details.csv'
            else:
                file_path = 'data/coco/coco_crop_test/image_details.csv'
        elif "dota" in self.dataset_name:
            if self.is_train:
                file_path = '/home/bibahaduri/dota_dataset/coco/coco_crop/image_details.csv'
            else:
                file_path = '/home/bibahaduri/dota_dataset/coco/coco_crop_test/image_details.csv'
        self.prmpt_df = pd.read_csv(file_path)
        self.epoch = 0
        self.query_position = 0
        self.probability()
        if grounding:
            def _setattr(self, name, value):
                object.__setattr__(self, name, value)

            Metadata.__setattr__ = _setattr
            MetadataCatalog.get(dataset_name).evaluator_type = "interactive_grounding"

    @classmethod
    def from_config(cls, cfg, is_train=True, dataset_name=''):
        ret = {
            "is_train": is_train,
            "dataset_name": dataset_name,
            "min_size_test": cfg['INPUT']['MIN_SIZE_TEST'],
            "max_size_test": cfg['INPUT']['MAX_SIZE_TEST'],
            "grounding": cfg['STROKE_SAMPLER']['EVAL']['GROUNDING'],
        }
        return ret

    def get_pascal_labels(self, ):
        """Load the mapping that associates pascal classes with label colors
        Returns:
            np.ndarray with dimensions (21, 3)
        """
        return np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
        )

    def get_prompts(self, classes, image_id):
        #TODO  random_class should be put inside the class_choices
        # with torch.profiler.profile(profile_memory=True) as prof:
        # if len(classes) == 0 :
        #     classes = range(20)
        classes = list(set(classes))
        
        if not self.is_train and self.evalall:## and False:
            prmpts = []
            cls2idx = {}
            random.seed(image_id)
            classes = list(set(classes))
            for i, cls in enumerate(classes):
                if "pascalvoc" in self.dataset_name:
                    filtered_df = self.prmpt_df[(self.prmpt_df['Difficulty Level'] == 0 )& (self.prmpt_df['Class Name'] == PASCAL_CLASSES[cls])]
                else:
                    filtered_df = self.prmpt_df[self.prmpt_df['Class Name'] == cls] ##PASCAL_CLASSES[cls]
                
                if len(filtered_df) < 5:
                    # Repeat rows from the DataFrame until its length becomes 5
                    repeat_times = (5 // len(filtered_df)) + 1
                    filtered_df = pd.concat([filtered_df] * repeat_times, ignore_index=True)
                l = list(range(len(filtered_df)))
                random.shuffle(l)
                random_indices = l[:5]##random.sample(range(len(filtered_df)), 5) ##(filtered_df.index, 5)

                selected_rows = filtered_df.iloc[random_indices]
                # selected_rows = filtered_df.iloc[:5]
                for _, selected_row in selected_rows.iterrows():
                    prmpt_path = selected_row["Cropped Image Path"]
                    prmpt = Image.open(prmpt_path).convert('RGB')
                    prmpts.append(prmpt)
                cls2idx[cls] =i
            return prmpts, cls2idx
        classes = np.array(classes)
        classes = np.unique(classes)
        if len(classes)==1:
            random_class = classes[0]
        else:
            p = []
            for i in classes:
                p.append(self.show_time[i])
            p = np.array(p)
            p /= p.sum()
            if self.is_train:
                random_class  = np.random.choice(classes,1,p=p)[0]
            else:
                #random.seed(image_id)
                random_class = classes[0] ##random.choice(classes)
        #random_class = random.choice(classes)
        image_name = os.path.basename(image_id)##image_id + '.jpg'
        if self.is_train:
            if "pascalvoc" in self.dataset_name:
                filtered_df = self.prmpt_df[(self.prmpt_df['Difficulty Level'] == 0 )& (self.prmpt_df['Class Name'] == PASCAL_CLASSES[random_class])]
            else:
                filtered_df = self.prmpt_df[(self.prmpt_df['Image Name'] != image_name) & (self.prmpt_df['Class Name'] == random_class)]##PASCAL_CLASSES[random_class])]  ##(self.prmpt_df['Image Name'] == image_name) &
        else:
            if "pascalvoc" in self.dataset_name:
                filtered_df = self.prmpt_df[(self.prmpt_df['Difficulty Level'] == 0 )& (self.prmpt_df['Class Name'] == PASCAL_CLASSES[random_class])]
            else:
                filtered_df = self.prmpt_df[self.prmpt_df['Class Name'] == random_class]##PASCAL_CLASSES[random_class]

        # Check if there are rows that satisfy the condition
        if not filtered_df.empty:
            # Select one row randomly from the filtered DataFrame
            if self.is_train:
                random_row = random.choice(filtered_df.index)
            else:
                random_row = filtered_df.index[0]
            # Get the selected row as a DataFrame
            selected_row = self.prmpt_df.loc[random_row]  #iloc[0]##
            prmpt_path = selected_row["Cropped Image Path"]

            prmpt = Image.open(prmpt_path).convert('RGB')

            # prmpt =np.load(prmpt_path)
            # prmpt = torch.from_numpy(prmpt)

        
        # class_choices.append(random_class)
        # condition = b['labels'] == random_class
        # b['labels'] = b['labels'][condition]
        # b['boxes'] = b['boxes'][condition]
        #prmpts = nested_tensor_from_tensor_list(prmpts)
        return prmpt, random_class

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
                (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        file_name = dataset_dict['file_name']
        image = Image.open(file_name).convert('RGB')

        dataset_dict['width'] = image.size[0]
        dataset_dict['height'] = image.size[1]

       
        # image = image.permute(2, 0, 1)

        class_ids = [ann_dict['category_id'] for ann_dict in dataset_dict['annotations']]
        query, query_class = self.get_prompts(class_ids, file_name) ##dataset_dict['image_id'])
        if not self.is_train and self.evalall:## and False:
            for i in range(len(query)):
                query[i] = self.query_transform(query[i])
            target = {}
            target['boxes'] = torch.tensor([data_dict['bbox'] for data_dict in dataset_dict['annotations']])
            target['labels'] = torch.tensor(class_ids)
            image, target = self.transform_img(image, target)
            
            dataset_dict['cls2idx'] = query_class
        else:

            target = {}
            target['boxes'] = torch.tensor([data_dict['bbox'] for data_dict in dataset_dict['annotations']])
            target['labels'] = torch.tensor(class_ids)
            image, target = self.transform_img(image, target)
            # dataset_dict['orig_classes'] = target['labels']
            # dataset_dict['orig_boxes'] = target['boxes']
            condition = target['labels'] == query_class
            target['labels'] = target['labels'][condition]
            if self.is_train:
                target['labels'].fill_(1)
            target['boxes'] = target['boxes'][condition]
            query = self.query_transform(query)

        dataset_dict['size'] = target['size']
        dataset_dict['orig_size'] = torch.tensor([dataset_dict['height'], dataset_dict['width']])
        


        instances = Instances(image.shape[-2:])
        mask_instances = Instances(image.shape[-2:])
        if 'inst_name' in dataset_dict.keys():
            inst_name = dataset_dict['inst_name']
            instances_mask = cv2.imread(inst_name)
            instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2GRAY).astype(np.int32)

            objects_ids = dataset_dict['objects_ids']
            instances_mask_byid = [(instances_mask == idx).astype(np.int16) for idx in objects_ids]

            semseg_name = dataset_dict['semseg_name']
            #semseg = self.encode_segmap(cv2.imread(semseg_name)[:, :, ::-1])
            # #class_names = [PASCAL_CLASSES[np.unique(semseg[instances_mask_byid[i].astype(np.bool)])[0].astype(
            # np.int32)-1] for i in range(len(instances_mask_byid))]

            _, h, w = image.shape
            masks = BitMasks(torch.stack([torch.from_numpy(
                cv2.resize(m.astype(np.float), (w, h), interpolation=cv2.INTER_CUBIC).astype(np.bool)
            ) for m in instances_mask_byid]))
            mask_instances.gt_masks = masks
            # instances.gt_boxes = masks.get_bounding_boxes()

            for i in range(len(instances_mask_byid)):
                instances_mask_byid[i][instances_mask == self.ignore_id] = -1
            gt_masks_orisize = torch.stack([torch.from_numpy(m) for m in instances_mask_byid])
            dataset_dict['gt_masks_orisize'] = gt_masks_orisize  # (nm,h,w)
            

        instances.gt_boxes = target['boxes']#torch.tensor([data_dict['bbox'] for data_dict in dataset_dict['annotations']])
        instances.gt_classes = target['labels']##torch.tensor(class_ids)
            

        # # To have information about the classes existing in this image for use in getting apropriate prompt/support
        # image
        # classes_info = {}
        # for item in dataset_dict['annotations']:
        #     if item['category_id'] not in classes_info:
        #         classes_info[item['category_id']] = 0
        #     classes_info[item['category_id']] += 1

        dataset_dict['instances'] = instances  # gt_masks, gt_boxes
        dataset_dict['mask_instances'] = mask_instances
        dataset_dict['image'] = image  # (3,h,w)
        dataset_dict['query'] = query
        ##dataset_dict['classes'] = class_names  # [prompt_engineering(x, topk=1, suffix='.') for x in class_names]
        # dataset_dict['classes_info'] = classes_info
        return dataset_dict
    
    def update_epoch(self, epoch):
        self.epoch = epoch


    def probability(self):
        show_time = {}

        if "pascalvoc" in self.dataset_name:
            for i, cls in enumerate(PASCAL_CLASSES):
                # if cls in self.base_classes:
                show_time[i] = len(self.prmpt_df[self.prmpt_df['Class Name'] == cls])
                # else:
                #     show_time[i] =0
        else: ##if "coco" in self.dataset_name:
            for i in self.base_ind:
                show_time[i] = len(self.prmpt_df[self.prmpt_df['Class Name'] == i])

        for i in self.base_ind:
            show_time[i] = 1/show_time[i]


        sum_prob = sum(show_time.values())

        for i in self.base_ind:
            show_time[i] = show_time[i]/sum_prob

        self.show_time = show_time
