# ------------------------------------------------------------------------
# IA-DETR
# Copyright (c) 2024 l2TI lab - USPN.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.

import os
import itertools
import logging
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.utils.data
import torch.utils.data as torchdata

import detectron2.utils.comm as comm
from detectron2.data.build import (
    build_batch_data_loader,
    load_proposals_into_dataset,
    trivial_batch_collator,
)
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import InferenceSampler, TrainingSampler
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    verify_results,
)
from .utils import box_ops
from .evaluation.voc_evaluator import PascalVOCDetectionEvaluator
from .evaluation.bbox_evaluator import BboxEvaluator
from fvcore.common.config import CfgNode

from .dataset_mappers import *
from .dataset_mappers.pascalvoc_dataset_mapper_ix import PascalVOCSegDatasetMapperIX
from .dataset_mappers.pascalvoc_dataset_mapper_up import PascalVOCPretrainDatasetMapperIX
from .evaluation.instance_evaluation import InstanceSegEvaluator
import sys
sys.path.append("/home/bibahaduri/plain_detr/util")
from util.misc import nested_tensor_from_tensor_list, nested_tensor_from_2tensor_lists



class JointLoader(torchdata.IterableDataset):
    def __init__(self, loaders, key_dataset):
        dataset_names = []
        for key, loader in loaders.items():
            name = "{}".format(key) #.split('_')[0]
            setattr(self, name, loader)
            dataset_names += [name]
        self.dataset_names = dataset_names
        self.key_dataset = key_dataset
    
    def __iter__(self):
        for batch in zip(*[getattr(self, name) for name in self.dataset_names]):
            print("batch=", batch)
            yield {key: batch[i] for i, key in enumerate(self.dataset_names)}

    def __len__(self):
        length =0
        for loader in self.dataset_names:
            length += len(getattr(self, loader).dataset.dataset)
        return length##len(getattr(self, self.key_dataset))

def filter_images_with_only_crowd_annotations(dataset_dicts, dataset_names):
    """
    Filter out images with none annotations or only crowd annotations
    (i.e., images without non-crowd annotations).
    A common training-time preprocessing on COCO dataset.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.

    Returns:
        list[dict]: the same format, but filtered.
    """
    num_before = len(dataset_dicts)

    def valid(anns):
        for ann in anns:
            if isinstance(ann, list):
                for instance in ann:
                    if instance.get("iscrowd", 0) == 0:
                        return True
            else:
                if ann.get("iscrowd", 0) == 0:
                    return True
        return False

    dataset_dicts = [x for x in dataset_dicts if valid(x["annotations"])]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images with no usable annotations. {} images left.".format(
            num_before - num_after, num_after
        )
    )
    return dataset_dicts


def get_detection_dataset_dicts(
    dataset_names, filter_empty=True, proposal_files=None
):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        dataset_names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `dataset_names`.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    assert len(dataset_names)

    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    if proposal_files is not None:
        assert len(dataset_names) == len(proposal_files)
        # load precomputed proposals from proposal files
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = "annotations" in dataset_dicts[0]
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts, dataset_names)

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(dataset_names))
    return dataset_dicts


def _test_loader_from_config(cfg, dataset_name, mapper=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    dataset = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,
        proposal_files=None,
    )
    if mapper is None:
        mapper_cfg = CfgNode({'INPUT': cfg['INPUT'], 'MODEL': cfg['MODEL'], 'DATASETS': cfg['DATASETS']})
        mapper = DatasetMapper(mapper_cfg, False)
    assert cfg['TEST']['BATCH_SIZE_TOTAL'] % get_world_size() == 0, "Evaluation total batchsize is not divisible by gpu number"
    batch_size = cfg['TEST']['BATCH_SIZE_TOTAL'] // get_world_size()

    return {
        "dataset": dataset,
        "mapper": mapper,
        "num_workers": cfg['DATALOADER']['NUM_WORKERS'],
        "sampler": InferenceSampler(len(dataset)),
        "batch_size": batch_size,
    }



def build_detection_test_loader(
    dataset: Union[List[Any], torchdata.Dataset],
    *,
    mapper: Callable[[Dict[str, Any]], Any],
    sampler: Optional[torchdata.Sampler] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
) -> torchdata.DataLoader:
    """
    Similar to `build_detection_train_loader`, with default batch size = 1,
    and sampler = :class:`InferenceSampler`. This sampler coordinates all workers
    to produce the exact set of all samples.

    Args:
        dataset: a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable). They can be obtained
            by using :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper: a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           When using cfg, the default choice is ``DatasetMapper(cfg, is_train=False)``.
        sampler: a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`InferenceSampler`,
            which splits the dataset across all workers. Sampler must be None
            if `dataset` is iterable.
        batch_size: the batch size of the data loader to be created.
            Default to 1 image per worker since this is the standard when reporting
            inference time in papers.
        num_workers: number of parallel data loading workers
        collate_fn: same as the argument of `torch.utils.data.DataLoader`.
            Defaults to do no collation and return a list of data.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))

        # or, instantiate with a CfgNode:
        data_loader = build_detection_test_loader(cfg, "my_test")
    """
    
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = InferenceSampler(len(dataset))
    
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        collate_fn= batch_collator_eval ##trivial_batch_collator # if collate_fn is None else collate_fn,
    )


def _train_loader_from_config(cfg, dataset_name, mapper, *, dataset=None, sampler=None):
    cfg_datasets = cfg['DATASETS']
    cfg_dataloader = cfg['DATALOADER']
    
    if dataset is None:
        dataset = get_detection_dataset_dicts(
            dataset_name,
            filter_empty=cfg_dataloader['FILTER_EMPTY_ANNOTATIONS'],
            proposal_files=cfg_datasets['PROPOSAL_FILES_TRAIN'] if cfg_dataloader['LOAD_PROPOSALS'] else None,
        )

    if mapper is None:
        # TODO: Hack to work arround may be not the good way to solv
        mapper_cfg = CfgNode({'INPUT': cfg['INPUT'], 'MODEL': cfg['MODEL'], 'DATASETS': cfg['DATASETS']})
        mapper = DatasetMapper(mapper_cfg, True)

    if sampler is None:
        sampler_name = cfg_dataloader['SAMPLER_TRAIN']
        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        sampler = TrainingSampler(len(dataset))

    return {
        "dataset": dataset,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": cfg['TRAIN']['BATCH_SIZE_TOTAL'],
        "aspect_ratio_grouping": cfg_dataloader['ASPECT_RATIO_GROUPING'],
        "num_workers": cfg_dataloader['NUM_WORKERS'],
    }



def build_detection_train_loader(
    dataset, *, mapper, sampler=None, total_batch_size, aspect_ratio_grouping=True, num_workers=0
):
    """
    Build a dataloader for object detection with some default features.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that
            produces indices to be applied on ``dataset``.
            Default to :class:`TrainingSampler`, which coordinates a random shuffle
            sequence across all workers.
        total_batch_size (int): total batch size across all workers. Batching
            simply puts data into a list.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers

    Returns:
        torch.utils.data.DataLoader: a dataloader. Each output from it is a
            ``list[mapped_element]`` of length ``total_batch_size / num_workers``,
            where ``mapped_element`` is produced by the ``mapper``.
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=False, ##aspect_ratio_grouping,
        num_workers=num_workers,
        collate_fn = batch_collator_train
    )




def build_eval_dataloader(cfg, dataset_names):
    dataloaders = []
    mappers = []
    for dataset_name in dataset_names:##cfg['DATASETS']['TEST']:
        cfg = cfg
        # adjust mapper according to dataset
        if "pascalvoc_" in dataset_name or "coco_" in dataset_name or "dota_" in dataset_name:
            if cfg.upretrain == True:
                mapper = PascalVOCPretrainDatasetMapperIX(is_train=False, dataset_name= dataset_name, min_size_test= 517, max_size_test= 518)
            else:
                mapper = PascalVOCSegDatasetMapperIX(is_train=False, dataset_name= dataset_name, min_size_test= 517, max_size_test= 518, evalall = cfg.eval, coco_split=cfg.coco_split)
        else:
            mapper = None
        dataset = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,
        proposal_files=None,
        )
        sampler =  InferenceSampler(len(dataset))
        batch_size = cfg.batch_size
        mappers.append(mapper)
        dataloaders += [build_detection_test_loader(dataset=dataset, mapper=mapper, sampler=sampler, batch_size=batch_size, num_workers=cfg.num_workers)]
    return dataloaders, mappers


def build_train_dataloader(cfg, dataset_names):
    dataset_names = dataset_names##cfg['DATASETS']['TRAIN']
    
    loaders = {}
    datasets = []
    mappers = []
    for dataset_name in dataset_names:
        if "pascalvoc_" in dataset_name or "coco_" or "dota_" in dataset_name:
            if cfg.upretrain == True:
                mapper = PascalVOCPretrainDatasetMapperIX(dataset_name= dataset_name, min_size_test= 517, max_size_test= 518)
            else:
                mapper = PascalVOCSegDatasetMapperIX(dataset_name= dataset_name, min_size_test= 517, max_size_test= 518, evalall = cfg.eval, coco_split=cfg.coco_split)
            dataset = None
            if dataset is None:
                dataset = get_detection_dataset_dicts(
                    dataset_name,
                    filter_empty=False,
                    proposal_files=None,
                )
            datasets.append(dataset)
            mappers.append(mapper)

    dataset = torchdata.ConcatDataset(datasets)
    sampler = None
    if sampler is None:
        sampler_name = "TrainingSampler"
        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        sampler = TrainingSampler(len(dataset))
    loaders[dataset_name] = build_detection_train_loader(dataset= dataset, mapper=mapper, sampler=sampler, total_batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    
        # else:
        #     mapper = None
        #     loaders[dataset_name] = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
    # breakpoint()
    if len(loaders) == 1 and not False:
        return list(loaders.values())[0], mappers
    else:
        return JointLoader(loaders, key_dataset='pascalvoc'), mappers##cfg['LOADER'].get('KEY_DATASET', 'pascal'))

    
def build_evaluator(cfg, dataset_name, device, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each
    builtin dataset. For your own dataset, you can simply create an
    evaluator manually in your script and do not have to worry about the
    hacky if-else logic here.
    """
    # if output_folder is None:
    #     output_folder = os.path.join(cfg["SAVE_DIR"], "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

    # semantic segmentation
    # if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
    #     evaluator_list.append(
    #         SemSegEvaluator(
    #             dataset_name,
    #             distributed=True,
    #             output_dir=output_folder,
    #         )
    #     )
    # instance segmentation
    if evaluator_type == "coco":
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    
    if evaluator_type == "bbox":
        evaluator_list.append(BboxEvaluator(dataset_name, output_dir=output_folder, device = device))
        ##evaluator_list.append(PascalVOCDetectionEvaluator(dataset_name, output_dir=output_folder))

    
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
        
    
    return DatasetEvaluators(evaluator_list)

def batch_collator_eval(batch):
    """
    A batch collator that does nothing.
    """
    #['file_name', 'image_id', 'height', 'width', 'annotations', 'instances', 'mask_instances', 'image', 'classes_info'])
    
    #Instances(num_instances=1, image_height=388, image_width=518, fields=[gt_boxes: tensor([[  0.,  31., 343., 336.]]), gt_classes: tensor([7])])
    samples = []
    targets = []
    queries = []
    for b in batch:
        samples.append(b['image'])
        tgt_dict ={}
        if 'size' not in b:
            tgt_dict['size'] = torch.tensor([b['width'], b['height']])
        else:
            tgt_dict['size'] = b['size']
        tgt_dict['orig_size'] = b['orig_size']##torch.tensor([b['height'], b['width']])
        tgt_dict['image_id'] = b['image_id']
        tgt_dict['labels'] = b['instances'].gt_classes
        tgt_dict['boxes'] = b['instances'].gt_boxes
        if 'orig_classes' in b:
            tgt_dict['classes'] = b['orig_classes']
        if 'orig_boxes' in b:
            tgt_dict['orig_boxes'] = b['orig_boxes']
        #tgt_dict['classes_info'] = torch.tensor(list(b['classes_info'].keys()))
        if 'cls2idx' in b:
            tgt_dict['cls2idx'] = b['cls2idx']
        targets.append(tgt_dict)
        if 'query' in b:
            if isinstance(b['query'], list):
                queries.extend(b['query'])
            else:
                queries.append(b['query'])
        
    
    if len(queries) > 0:
        samples, queries = nested_tensor_from_2tensor_lists(samples, queries)
        del batch      # to not cause memory issues in multiprocessing
        return (samples, queries, targets)
    samples = nested_tensor_from_tensor_list(samples)
    del batch      # to not cause memory issues in multiprocessing
    return (samples, targets)

def batch_collator_train(batch):
    """
    A batch collator that does nothing.
    """
    #['file_name', 'image_id', 'height', 'width', 'annotations', 'instances', 'mask_instances', 'image', 'classes_info'])
    
    #Instances(num_instances=1, image_height=388, image_width=518, fields=[gt_boxes: tensor([[  0.,  31., 343., 336.]]), gt_classes: tensor([7])])
    samples = []
    targets = []
    queries = []
    for b in batch:
        samples.append(b['image'])
        tgt_dict ={}
        if 'size' not in b:
            tgt_dict['size'] = torch.tensor([b['width'], b['height']])
        else:
            tgt_dict['size'] = b['size']
        ##ratio = torch.tensor([b['width'], b['height'], b['width'], b['height']])
        tgt_dict['orig_size'] = b['orig_size']##torch.tensor([b['height'], b['width']])
        tgt_dict['image_id'] = b['image_id']
        tgt_dict['labels'] = b['instances'].gt_classes
        tgt_dict['boxes'] = b['instances'].gt_boxes##box_ops.box_xyxy_to_cxcywh(b['instances'].gt_boxes) / ratio
        if 'orig_classes' in b:
            tgt_dict['classes'] = b['orig_classes']
        if 'orig_boxes' in b:
            tgt_dict['orig_boxes'] = b['orig_boxes']
        #tgt_dict['classes_info'] = torch.tensor(list(b['classes_info'].keys()))
        targets.append(tgt_dict)
        if 'query' in b:
            queries.append(b['query'])
    if len(queries) > 0:
        samples, queries = nested_tensor_from_2tensor_lists(samples, queries)
        del batch      # to not cause memory issues in multiprocessing
        return (samples, queries, targets)
    samples = nested_tensor_from_tensor_list(samples)
    del batch      # to not cause memory issues in multiprocessing
    return (samples, targets)

