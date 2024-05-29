# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py

## shape_sample and its query used for sampling prompts were removed

import copy
# import logging

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
import random
# from detectron2.config import configurable
import datasets.transforms as T
from util.box_ops import box_cxcywh_to_xyxy
from torchvision import transforms
from detectron2.structures import BitMasks, Boxes, Instances
from detectron2.data import MetadataCatalog, Metadata

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


def get_random_patch_from_img(img, min_pixel=8):
    """
    :param img: original image
    :param min_pixel: min pixels of the query patch
    :return: query_patch,x,y,w,h
    """
    w, h = img.size
    min_w, max_w = min_pixel, w - min_pixel
    min_h, max_h = min_pixel, h - min_pixel
    sw, sh = np.random.randint(min_w, max_w + 1), np.random.randint(min_h, max_h + 1)
    x, y = np.random.randint(w - sw) if sw != w else 0, np.random.randint(h - sh) if sh != h else 0
    patch = img.crop((x, y, x + sw, y + sh))
    return patch, x, y, sw, sh

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
    scales = [400, 480, 512, 544, 576, 608, 640]###[320, 336, 352, 368, 400, 416, 432, 448, 464, 480]

    if training == True:
        return T.Compose([
            # T.RandomHorizontalFlip(), HorizontalFlip may cause the pretext too difficult, so we remove it
            T.RandomResize(scales, max_size=642),
            normalize,
        ])

    if training == False:
        return T.Compose([
            T.RandomResize([480], max_size=512),
            normalize,
        ])


def get_query_transforms(training):
    if training == True:
        # SimCLR style augmentation
        return transforms.Compose([
            transforms.Resize(510, max_size=512),##(512, 512)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(),  HorizontalFlip may cause the pretext too difficult, so we remove it
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    if training == False:
        return transforms.Compose([
            transforms.Resize(510, max_size=512),##(512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    #raise ValueError(f'unknown {image_set}')


# This is specifically designed for the COCO dataset.
class PascalVOCPretrainDatasetMapperIX:
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
            grounding=False,
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

        # t = []
        # t.append(transforms.Resize(self.min_size_test, interpolation=Image.BICUBIC, max_size=max_size_test, antialias=True))
        # t.append(transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD))
        self.transform_img = make_self_det_transforms(self.is_train)#transforms.Compose(t)
        self.query_transform = get_query_transforms(self.is_train)
        self.ignore_id = 220

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

        query, x, y, sw, sh = get_random_patch_from_img(image)
        #something nonsense
        # if not self.is_train:
        target = {}
        boxes = [[x, y, x + sw, y + sh]]
        
        orig_boxes = torch.tensor([data_dict['bbox'] for data_dict in dataset_dict['annotations']])
        orig_classes = torch.tensor([ann_dict['category_id'] for ann_dict in dataset_dict['annotations']])
        orig_classes_zeros = torch.zeros_like(orig_classes)
        class_ids = [1]#[ann_dict['category_id'] for ann_dict in dataset_dict['annotations']]    
        target['boxes'] = torch.cat((torch.tensor(boxes), orig_boxes ), dim = 0)
        target['labels'] = torch.cat((torch.tensor(class_ids), orig_classes_zeros ),dim=0)
        
        image, target = self.transform_img(image, target)
        #target['boxes'] = box_cxcywh_to_xyxy(target['boxes']) * torch.tensor([target['size'][1], target['size'][0], target['size'][1], target['size'][0]])
        dataset_dict['size'] = target['size']
        dataset_dict['orig_size'] = torch.tensor([dataset_dict['height'], dataset_dict['width']])
        query = self.query_transform(query)
        # image = image.permute(2, 0, 1)
        
        instances = Instances(image.shape[-2:])
        
            

        instances.gt_boxes = target['boxes']
        instances.gt_classes = target['labels']
        dataset_dict['orig_classes'] = torch.cat((torch.tensor([-1]), orig_classes))    ##original classes but just for contrastive loss  -1 is pushed at start just as placeholder for prompt
            

        # # To have information about the classes existing in this image for use in getting apropriate prompt/support
        # image
        classes_info = {}
        for item in dataset_dict['annotations']:
            if item['category_id'] not in classes_info:
                classes_info[item['category_id']] = 0
            classes_info[item['category_id']] += 1

        dataset_dict['instances'] = instances  # gt_masks, gt_boxes
        #dataset_dict['mask_instances'] = mask_instances
        dataset_dict['image'] = image  # (3,h,w)
        dataset_dict['query'] = query
        ##dataset_dict['classes'] = class_names  # [prompt_engineering(x, topk=1, suffix='.') for x in class_names]
        dataset_dict['classes_info'] = classes_info
        return dataset_dict
