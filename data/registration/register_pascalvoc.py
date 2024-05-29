# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import os
import glob

from typing import List, Tuple, Union
import xml.etree.ElementTree as ET
import logging
import pycocotools.mask as mask_util

import cv2
import numpy as np
from scipy.io import loadmat

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

__all__ = ["load_pascalvoc_instances", "register_pascalvoc_context"]

PASCAL_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)


PASCAL_VOC_BASE_CLASSES = (
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "chair",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sofa",
        "train",
        "tvmonitor",
        )

PASCAL_VOC_NOVEL_CLASSES = ("aeroplane", "cat",  "cow", "sheep", )


def get_labels_with_sizes(x):
    obj_sizes = np.bincount(x.flatten())
    labels = np.nonzero(obj_sizes)[0].tolist()
    labels = [x for x in labels if x != 0]
    return labels, obj_sizes[labels].tolist()

logger = logging.getLogger(__name__)
def load_pascalvoc_instances(name: str, dirname: str, mode: str, split: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=str)

    if mode == "Base":
        classes = PASCAL_VOC_BASE_CLASSES
    elif mode == "Novel":
        classes = PASCAL_VOC_NOVEL_CLASSES
    else:
        classes = PASCAL_CLASSES

    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")
        inst_path = os.path.join(dirname, "SegmentationObject", "{}.png".format(fileid))
        semseg_path = os.path.join(dirname, "SegmentationClass", "{}.png".format(fileid))

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        # check and see if instance segmentation also exists and append it to the annotations
        if os.path.exists(inst_path):
            instances_mask = cv2.imread(inst_path)
            instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2GRAY).astype(np.int32)

            objects_ids = np.unique(instances_mask)
            objects_ids = [x for x in objects_ids if x != 0 and x != 220]

            slice_size = 5
            for i in range(0, len(objects_ids), slice_size):
                r2 = {
                    "inst_name": inst_path,
                    "semseg_name": semseg_path,
                    "objects_ids": objects_ids[i:i+slice_size],
                }
                r.update(r2)
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            difficult = obj.find("difficult").text
            if int(difficult) == 1:
                continue
            if cls not in classes:
                continue
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": PASCAL_CLASSES.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        #
        if len(instances) > 0:
            r["annotations"] = instances
            dicts.append(r)
        # elif split == "val":
        #     r["annotations"] = instances
        #     dicts.append(r)
    return dicts


def load_coco_json(
    json_file, image_root, mode=None, classes=None, dataset_name=None, extra_annotation_keys=None
):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    
    json_file = PathManager.get_local_path(json_file)
    
    coco_api = COCO(json_file)
    
    logger.info(
        "Loading {} annotations.".format(json_file)
    )

    id_map = None
    cat_ids = sorted(coco_api.getCatIds())
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map
    
    cats = coco_api.loadCats(cat_ids)
    # The categories in a custom json file may not be sorted.
    thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        #meta.thing_classes = thing_classes

    regular_indices = list(range(1, len(cat_ids) + 1))

        # Create a mapping dictionary
    cls_ind_to_cat_id = {coco_id: index for coco_id, index in zip(regular_indices, cat_ids)}
    if mode != None:
        if mode == 4:
            mode = 0
        if classes == 'Base':
            list_class = [cls_ind_to_cat_id[cat] for cat in range(1,81) if cat%4 != mode]
        elif classes == 'Novel':
            list_class = [cls_ind_to_cat_id[cat] for cat in range(1,81) if cat%4 == mode]
        else:
            list_class = [cls_ind_to_cat_id[cat] for cat in range(1,81)]
        

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(
            ann_ids
        ), "Annotation ids in '{}' are not unique!".format(json_file)

    imgs_anns = list(zip(imgs, anns))
    logger.info(
        "Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file)
    )

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (
        extra_annotation_keys or []
    )

    num_instances_without_valid_segmentation = 0
    skipped = 0
    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        flag = True
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id
            if anno.get("iscrowd", 0) == 1:
                continue
            if mode != None:
                if anno["category_id"] not in list_class:
                    flag = False
                    continue ##break     #
            assert (
                anno.get("ignore", 0) == 0
            ), '"ignore" in COCO json file is not supported.'
            bbox = anno['bbox']
            bbox[0] = bbox[0] - 1
            bbox[1] = bbox[1] - 1
            bbox[2] += bbox[0] 
            bbox[3] += bbox[1]
            obj = {"category_id": anno['category_id'], "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            #{key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [
                        poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6
                    ]
                    # if len(segm) == 0:
                    #     num_instances_without_valid_segmentation += 1
                    #     continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            objs.append(obj)
        record["annotations"] = objs
        if len(objs) > 0:#     flag == True:   #
            dataset_dicts.append(record)
        else:
            skipped += 1

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )
    if skipped > 0:
        logger.warning(
            "Filtered out {} instances including novel classes. ".format(
                skipped
            )
        )
    
    return dataset_dicts




def load_dota_json(
    json_file, image_root, classes=None, dataset_name=None, extra_annotation_keys=None
):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    
    json_file = PathManager.get_local_path(json_file)
    
    coco_api = COCO(json_file)
    
    logger.info(
        "Loading {} annotations.".format(json_file)
    )

    id_map = None
    cat_ids = sorted(coco_api.getCatIds())
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map
    
    cats = coco_api.loadCats(cat_ids)
    # The categories in a custom json file may not be sorted.
    thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        #meta.thing_classes = thing_classes

    regular_indices = list(range(1, len(cat_ids) + 1))

        # Create a mapping dictionary
    cls_ind_to_cat_id = {coco_id: index for coco_id, index in zip(regular_indices, cat_ids)}
    if classes == 'Base':
        list_class = [1,2,4,6,7,8,9,10,11,12,13,14,16]
    elif classes == 'Novel':
        list_class = [3,5,15]
    else:
        list_class = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(
            ann_ids
        ), "Annotation ids in '{}' are not unique!".format(json_file)

    imgs_anns = list(zip(imgs, anns))
    logger.info(
        "Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file)
    )

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (
        extra_annotation_keys or []
    )

    num_instances_without_valid_segmentation = 0
    skipped = 0
    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        flag = True
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id
            if anno.get("iscrowd", 0) == 1:
                continue
            if anno["category_id"] not in list_class:
                flag = False
                continue ##break     #
            assert (
                anno.get("ignore", 0) == 0
            ), '"ignore" in COCO json file is not supported.'
            bbox = anno['bbox']
            bbox[2] += bbox[0] 
            bbox[3] += bbox[1]
            obj = {"category_id": anno['category_id'], "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            #{key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [
                        poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6
                    ]
                    # if len(segm) == 0:
                    #     num_instances_without_valid_segmentation += 1
                    #     continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            objs.append(obj)
        record["annotations"] = objs
        if len(objs) > 0:#     flag == True:   #
            dataset_dicts.append(record)
        else:
            skipped += 1

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )
    if skipped > 0:
        logger.warning(
            "Filtered out {} instances including novel classes. ".format(
                skipped
            )
        )
    
    return dataset_dicts




def register_pascalvoc_context(name, dirname, mode, split):
    DatasetCatalog.register("{}_{}".format(name, mode), lambda: load_pascalvoc_instances(name, dirname, mode, split))
    MetadataCatalog.get("{}_{}".format(name, mode)).set(
        dirname=dirname,
        thing_dataset_id_to_contiguous_id={},
    )


def register_all_sbd(root):
    SPLITS = [
        #("pascalvoc_val", "PascalVOC", "Point", "val"),
        #("pascalvoc_val", "PascalVOC", "Scribble", "val"),
        #("pascalvoc_val", "PascalVOC", "Polygon", "val"),
        ("pascalvoc_val", "PascalVOC", "Base", "val"),
        #("pascalvoc_val", "PascalVOC", "Novel", "val"),
        ("pascalvoc_train_2012", "PascalVOC", "Base", "trainval"), ##**
        ("pascalvoc_train_2007", "VOC2007", "Base", "trainval"),   ##**
        ("pascalvoc_uptrain", "PascalVOC", "All", "train"),
        ("pascalvoc_upval", "PascalVOC", "All", "val"),
        ("pascalvoc_uptrain", "PascalVOC", "Base", "train"),
        ("pascalvoc_upval", "PascalVOC", "Base", "val"),
        ("pascalvoc_testup", "VOC2007_test","Base", "test"),   ##**
        ("pascalvoc_test", "VOC2007_test","Novel", "test"),
        ("pascalvoc_test", "VOC2007_test","all", "test"),
    ]

    for name, dirname, mode, split in SPLITS:
        register_pascalvoc_context(name, os.path.join(root, dirname), mode, split)
        MetadataCatalog.get("{}_{}".format(name, mode)).evaluator_type = "bbox"

def register_coco_context(name, mode, classes, annotations, image_root):
    DatasetCatalog.register("{}_{}_{}".format(name, mode, classes), lambda: load_coco_json(annotations, image_root, mode, classes))
    MetadataCatalog.get("{}_{}_{}".format(name, mode, classes)).set(
        dirname=image_root,
        thing_dataset_id_to_contiguous_id={},
    )


def register_all_coco(root):
    SPLITS = [
        ("coco_train", "train2017", 1, "Base", "instances_train2017"),
        ("coco_val", "val2017", 1, "Novel", "instances_val2017"), ##**
        ("coco_val", "val2017", 1, "Base", "instances_val2017"),
        ("coco_train", "train2017", 2, "Base", "instances_train2017"),
        ("coco_val", "val2017", 2, "Novel", "instances_val2017"),
        ("coco_val", "val2017", 2, "Base", "instances_val2017"),
        ("coco_train", "train2017", 3, "Base", "instances_train2017"),
        ("coco_val", "val2017", 3, "Novel", "instances_val2017"),
        ("coco_train", "train2017", 4, "Base", "instances_train2017"),
        ("coco_val", "val2017", 4, "Novel", "instances_val2017"),
        ("coco_val", "val2017", 4, "all", "instances_val2017"),
        ("coco_val", "val2017", 1, "all", "instances_val2017"),
        ("coco_val", "val2017", 2, "all", "instances_val2017"),
        ("coco_val", "val2017", 3, "all", "instances_val2017"),
    ]

    for name, dirname, mode, classes, split in SPLITS:
        register_coco_context(name, mode, classes, os.path.join(root, 'annotations', split +'.json'), os.path.join(root, dirname))
        MetadataCatalog.get("{}_{}_{}".format(name, mode, classes)).evaluator_type = "bbox"



def register_dota_context(name, classes, annotations, image_root):
    DatasetCatalog.register("{}_{}".format(name, classes), lambda: load_dota_json(annotations, image_root, classes))
    MetadataCatalog.get("{}_{}".format(name, classes)).set(
        dirname=image_root,
        thing_dataset_id_to_contiguous_id={},
    )

def register_all_dota(root):
    SPLITS = [
        #("pascalvoc_val", "PascalVOC", "Point", "val"),
        #("pascalvoc_val", "PascalVOC", "Scribble", "val"),
        #("pascalvoc_val", "PascalVOC", "Polygon", "val"),
        ("dota_train", "train2017", "Base", "instances_train2017"),
        #("pascalvoc_val", "PascalVOC", "Novel", "val"),
        ("dota_val", "test2017", "Novel", "instances_test2017"), ##**
        # ("coco_val", "val2017", 1, "Base", "instances_val2017"),
        # ("coco_train", "train2017", 2, "Base", "instances_train2017"),
        # ("coco_val", "val2017", 2, "Novel", "instances_val2017"),
        # ("coco_train", "train2017", 3, "Base", "instances_train2017"),
        # ("coco_val", "val2017", 3, "Novel", "instances_val2017"),
        # ("coco_train", "train2017", 0, "Base", "instances_train2017"),
        # ("coco_val", "val2017", 0, "Novel", "instances_val2017"),
        # ("coco_val", "val2017", 1, "novel", "instances_val2017"),
        # ("coco_val", "val2017", 2, "novel", "instances_val2017"),
    ]

    for name, dirname, classes, split in SPLITS:
        register_dota_context(name, classes, os.path.join(root, 'annotations', split +'.json'), os.path.join(root, dirname))
        MetadataCatalog.get("{}_{}".format(name, classes)).evaluator_type = "bbox"


_root = os.getenv("DATASET_VOC", "data/pascalvoc")
register_all_sbd(_root)
_root= os.getenv("DATASET_COCO",  'data/coco')
register_all_coco(_root)
_root= os.getenv("DATASET_DOTA",  '/home/bibahaduri/dota_dataset/coco')
register_all_dota(_root)
