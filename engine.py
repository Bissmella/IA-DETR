# ------------------------------------------------------------------------
# Plain-DETR
# Copyright (c) 2023 Xi'an Jiaotong University & Microsoft Research Asia.
# Licensed under The MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import copy

import wandb
import torch
import util.misc as utils
from datasets.build import build_evaluator
from datasets.data_prefetcher import data_prefetcher
from torch import distributed as dist
from datasets import PASCAL_VOC_BASE_CLASSES, PASCAL_VOC_NOVEL_CLASSES
import gc
import logging
from pycocotools.coco import COCO

logger = logging.getLogger(__name__)
def train_hybrid(outputs, targets, k_one2many, criterion, lambda_one2many,upretrain):
    # one-to-one-loss
    loss_dict = criterion(outputs, targets)
    multi_targets = copy.deepcopy(targets)
    # repeat the targets
    for target in multi_targets:
        target["boxes"] = target["boxes"].repeat(k_one2many, 1)
        target["labels"] = target["labels"].repeat(k_one2many)
        if upretrain:
            target["classes"] = target["classes"].repeat(k_one2many)

    outputs_one2many = dict()
    outputs_one2many["pred_logits"] = outputs["pred_logits_one2many"]
    outputs_one2many["pred_boxes"] = outputs["pred_boxes_one2many"]
    if upretrain:
        outputs_one2many['pred_features'] = outputs["pred_features_one2many"]
    outputs_one2many["aux_outputs"] = outputs["aux_outputs_one2many"]
    if "pred_boxes_old_one2many" in outputs.keys():
        outputs_one2many["pred_boxes_old"] = outputs["pred_boxes_old_one2many"]
        outputs_one2many["pred_deltas"] = outputs["pred_deltas_one2many"]

    # one-to-many loss
    loss_dict_one2many = criterion(outputs_one2many, multi_targets)
    for key, value in loss_dict_one2many.items():
        if key + "_one2many" in loss_dict.keys():
            loss_dict[key + "_one2many"] += value * lambda_one2many
        else:
            loss_dict[key + "_one2many"] = value * lambda_one2many
    return loss_dict

#**
import random
from PIL import Image
from util.misc import NestedTensor, nested_tensor_from_tensor_list
from util.constants import PASCAL_CLASSES
from torchvision import transforms
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
PASCAL_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
            "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
def get_prompts(prmpt_df, batch, device):
    prmpts = []
    #TODO  random_class should be put inside the class_choices
    class_choices = []
    # with torch.profiler.profile(profile_memory=True) as prof:
    for b in batch:
        classes = b['classes_info'].tolist()
        #TODO  case where not ground truth class
        if len(classes) == 0 :
            classes = range(20)

        random_class = random.choice(classes)
        image_name = b['image_id'] + '.jpg'
        filtered_df = prmpt_df[(prmpt_df['Image Name'] == image_name) & (prmpt_df['Class Name'] == PASCAL_CLASSES[random_class])]
        # Check if there are rows that satisfy the condition
        if not filtered_df.empty:
            # Select one row randomly from the filtered DataFrame
            random_row = random.choice(filtered_df.index)
            # Get the selected row as a DataFrame
            selected_row = prmpt_df.iloc[0]#loc[random_row]  #iloc[0]##
            prmpt_path = selected_row["Cropped Image Path"]

            prmpt = Image.open(prmpt_path)
            prmpt = transforms.ToTensor()(prmpt)

            ##prmpt = prmpt / 255.0
            #prmpt = transforms.Resize(517, interpolation=Image.BICUBIC, max_size=518, antialias=True)(prmpt)
            prmpt = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)(prmpt)

            # prmpt =np.load(prmpt_path)
            # prmpt = torch.from_numpy(prmpt)
            prmpts.append(prmpt.to(device))
        
        class_choices.append(random_class)
        condition = b['labels'] == random_class
        b['labels'] = b['labels'][condition]
        b['boxes'] = b['boxes'][condition]
    prmpts = nested_tensor_from_tensor_list(prmpts)
    return prmpts, batch

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    lr_scheduler: torch.optim.lr_scheduler,
    max_norm: float = 0,
    k_one2many: int = 1,
    lambda_one2many: float = 1.0,
    use_wandb: bool = False,
    use_fp16: bool = False,
    scaler: torch.cuda.amp.GradScaler = None,
    epoch_iter = None,
    upretrain = False,
):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    metric_logger.add_meter(
        "grad_norm", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=False)
    samples, prmpts, targets = prefetcher.next()
    #prmpts, targets = get_prompts(df, targets, device)
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for idx in metric_logger.log_every(range(epoch_iter), print_freq, header):
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_fp16):
            outputs = model(samples, prmpts)

            if k_one2many > 0:
                loss_dict = train_hybrid(
                    outputs, targets, k_one2many, criterion, lambda_one2many, upretrain
                )
            else:
                loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        if use_fp16:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            if max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters(), norm_type=2)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters(), norm_type=2)
            optimizer.step()

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled,
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, prmpts, targets = prefetcher.next()
        lr_scheduler.step()

        if use_wandb and idx % print_freq == 0 and dist.get_rank() == 0:
            log_data = dict(
                loss=loss_value,
                lr=optimizer.param_groups[0]["lr"],
                grad_norm=grad_total_norm,
                **loss_dict_reduced_scaled
            )
            log_data = {"train/"+k: v for k, v in log_data.items()}
            wandb.log(data=log_data, step=(epoch * len(data_loader) + idx))
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model,
    criterion,
    postprocessors,
    data_loader,
    device,
    output_dir,
    step,
    use_wandb=False,
    reparam=False,
    df = None,
):
    # (hack) disable the one-to-many branch queries
    # save them frist
    save_num_queries = model.module.num_queries
    save_two_stage_num_proposals = model.module.transformer.two_stage_num_proposals
    model.module.num_queries = model.module.num_queries_one2one
    model.module.transformer.two_stage_num_proposals = model.module.num_queries

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    #coco_evaluator = CocoEvaluator(base_ds, iou_types)
    bbox_evaluator = build_evaluator(None, "pascalvoc_val_Base", device, output_dir)   #None is for cfg required in build_evaluator
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    
    
    for samples, prmpts, targets in metric_logger.log_every(data_loader[0], 10, header):
        samples = samples.to(device)

        targets = [{k: v.to(device) if k not in ["image_id", "cls2idx"] else v for k, v in t.items()} for t in targets]
        prmpts = prmpts.to(device)# , targets = get_prompts(df, targets, device)
        outputs = model(samples, prmpts)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        if reparam:
            results = postprocessors["bbox"](outputs, target_sizes, orig_target_sizes)
        else:
            results = postprocessors["bbox"](outputs, orig_target_sizes)
        if "segm" in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors["segm"](
                results, outputs, orig_target_sizes, target_sizes
            )
        # res = {
        #     target["image_id"].item(): output
        #     for target, output in zip(targets, results)
        # }
        if bbox_evaluator is not None:
            bbox_evaluator.process(targets, results)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](
                outputs, target_sizes, orig_target_sizes
            )
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # if bbox_evaluator is not None:
    #     coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()
    results, maps = bbox_evaluator.evaluate()

    # recover the model parameters for next training epoch
    model.module.num_queries = save_num_queries
    model.module.transformer.two_stage_num_proposals = save_two_stage_num_proposals
    test_metrics = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return {**test_metrics, **results, **maps} 


def evaluateall(    model,
    criterion,
    postprocessors,
    data_loader,
    device,
    output_dir,
    step,
    use_wandb=False,
    reparam=False,
    mapper = None,
    dataset_file = None,
):
    # (hack) disable the one-to-many branch queries
    # save them frist
    save_num_queries = model.module.num_queries
    save_two_stage_num_proposals = model.module.transformer.two_stage_num_proposals
    model.module.num_queries = model.module.num_queries_one2one
    model.module.transformer.two_stage_num_proposals = model.module.num_queries

    model.eval()
    criterion.eval()

    unseen_classes = mapper.base_ind

    #creating id to category maps
    if dataset_file == 'coco':
        coco = COCO('data/coco/annotations/instances_val2017.json')
        categories = coco.loadCats(coco.getCatIds())
        id_to_category_map = {category['id']: category['name'] for category in categories}
    elif dataset_file == 'pascalvoc':
        id_to_category_map = {category[id]: category for id, category in enumerate(PASCAL_CLASSES)}
    elif dataset_file == 'dota':
        id_to_category_map = {i : i for i in range(18)}  
    else:
        assert False, "Not implemented!"
    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter(
    #     "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    # )
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())

    bbox_evaluator = build_evaluator(None, "coco_train_1_Base", device, output_dir)

    panoptic_evaluator = None
    
    overall_results = {}
    for samples, prmpts, targets in metric_logger.log_every(data_loader[0], 10, header):
        samples = samples.to(device)

        targets = [{k: v.to(device) if k not in ["image_id", "cls2idx"] else v for k, v in t.items()} for t in targets]
        targets = targets[0]
        prmpts = prmpts.to(device)# , targets = get_prompts(df, targets, device)

        samples.tensors = torch.cat((samples.tensors, prmpts.tensors))
        samples.mask = torch.cat((samples.mask, prmpts.mask))
        srcs, masks, pos, query_embeds, self_attn_mask, prmpts, prmpt_masks, prmpt_pos = model.module.encode(samples)

        
        indices = torch.tensor([0,1,2,3,4]) #indices for selecting the queries related to a class   #HACK hard coded tobe enhanced
        
        for cls in targets['cls2idx'].keys():
            n_indices = indices + (targets['cls2idx'][cls] * 5)  ##indices for selecting the queries related to current class   #HACK hard coded tobe enhanced
            n_indices = n_indices.to(device)

            queries = prmpts[n_indices]   ##[i] for i in n_indices]
            queries_masks = prmpt_masks[n_indices]   ##[i] for i in n_indices]
            queries_pos = prmpt_pos[n_indices]   ##[i] for i in n_indices]
            srcs_n = [srcs.repeat(queries.shape[0], 1, 1, 1)]
            masks_n = [masks.repeat(queries.shape[0], 1, 1)]
            pos_n = [pos.repeat(queries.shape[0], 1, 1, 1)]
            outputs = model.module.decode(srcs_n, masks_n, pos_n, query_embeds, self_attn_mask, [queries], [queries_masks], [queries_pos])
            n_targets = copy.deepcopy(targets)
            condition = n_targets['labels'] == cls
            n_targets['labels'] = n_targets['labels'][condition]
            n_targets['labels'].fill_(1)
            n_targets['boxes'] = n_targets['boxes'][condition]
            n_targets = [n_targets for i in range(queries.shape[0])]
            orig_target_sizes = torch.stack([t["orig_size"] for t in n_targets], dim=0)
            target_sizes = torch.stack([t["size"] for t in n_targets], dim=0)
            if reparam:
                results = postprocessors["bbox"](outputs, target_sizes, orig_target_sizes)
            else:
                results = postprocessors["bbox"](outputs, orig_target_sizes)
            if "segm" in postprocessors.keys():
                target_sizes = torch.stack([t["size"] for t in n_targets], dim=0)
                results = postprocessors["segm"](
                    results, outputs, orig_target_sizes, target_sizes
                )
            
            if bbox_evaluator is not None:
                bbox_evaluator.process(n_targets, results)
                results, maps = bbox_evaluator.evaluate()
                bbox_evaluator.reset()
            if cls in overall_results:
                overall_results[cls].append(results['map50'])
            else:
                overall_results[cls] = [results['map50']]
            
        
        

    results_seen = {}
    results_unseen = {}
    for key, value in overall_results.items():
        # Calculate the mean of the list
        mean = sum(value) / len(value)
        
        # Update the list in the dictionary with the mean
        if key in unseen_classes:
            results_unseen[id_to_category_map[key]] = mean
        else:
            results_seen[id_to_category_map[key]] = mean
    print("results on seen classes: ")
    print("mean AP50 on seen classes: ",sum(results_seen.values())/len(results_seen.values()) )
    print(results_seen)
    print("*************************")
    print("results on unseen classes: ")
    print("mean AP50 on seen classes: ",sum(results_unseen.values())/len(results_unseen.values()) )
    print(results_unseen)
    #print(results)
    return {**results} ##, coco_evaluator


