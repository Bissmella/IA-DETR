# ------------------------------------------------------------------------
# IA-DETR
# Copyright (c) 2024 l2TI lab - USPN.
# Licensed under The MIT License [see LICENSE for details]
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


import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os

import math
import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from datasets.build import build_evaluator, build_eval_dataloader, build_train_dataloader
from engine import evaluate, evaluateall, train_one_epoch
from models import build_model
from torch import distributed as dist
import wandb
import pandas as pd
torch.multiprocessing.set_sharing_strategy('file_system')

def get_args_parser():
    parser = argparse.ArgumentParser("Deformable DETR Detector", add_help=False)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument(
        "--lr_backbone_names", default=["backbone.0"], type=str, nargs="+"
    )
    parser.add_argument("--lr_backbone", default=2e-5, type=float)
    parser.add_argument(
        "--lr_linear_proj_names",
        default=["reference_points", "sampling_offsets"],
        type=str,
        nargs="+",
    )
    parser.add_argument("--lr_linear_proj_mult", default=0.1, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr_drop", default=40, type=int)
    parser.add_argument("--lr_drop_epochs", default=None, type=int, nargs="+")
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )
    parser.add_argument("--warmup", default=0, type=int)

    parser.add_argument("--sgd", action="store_true")

    # * Modern DETR tricks
    # Deformable DETR tricks
    parser.add_argument("--with_box_refine", default=False, action="store_true")
    parser.add_argument("--two_stage", default=False, action="store_true")
    # DINO DETR tricks
    parser.add_argument("--mixed_selection", action="store_true", default=False)
    parser.add_argument("--look_forward_twice", action="store_true", default=False)
    # Hybrid Matching tricks
    parser.add_argument("--k_one2many", default=5, type=int)
    parser.add_argument("--lambda_one2many", default=1.0, type=float)
    parser.add_argument(
        "--num_queries_one2one",
        default=300,
        type=int,
        help="Number of query slots for one-to-one matching",
    )
    parser.add_argument(
        "--num_queries_one2many",
        default=0,
        type=int,
        help="Number of query slots for one-to-many matchining",
    )
    # Absolute coordinates & box regression reparameterization
    parser.add_argument(
        "--reparam",
        action="store_true",
        help="If true, we use absolute coordindates & reparameterization for bounding boxes",
    )

    # Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )

    # * Backbone
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned", "sine_unnorm"),
        help="Type of positional embedding to use on top of the image features",
    )
    parser.add_argument(
        "--position_embedding_scale",
        default=2 * np.pi,
        type=float,
        help="position / size * scale",
    )
    parser.add_argument(
        "--num_feature_levels", default=1, type=int, help="number of feature levels"
    )
    # swin backbone
    parser.add_argument(
        "--pretrained_backbone_path",
        default="./swin_tiny_patch4_window7_224.pkl",
        type=str,
    )
    parser.add_argument("--drop_path_rate", default=0.1, type=float)
    # upsample backbone output features
    parser.add_argument(
        "--upsample_backbone_output",
        action="store_true",
        help="If true, we upsample the backbone output feature to the target stride"
    )
    parser.add_argument(
        "--upsample_stride",
        default=16,
        type=int,
        help="Target stride for upsampling backbone output feature"
    )

    # * Transformer
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument("--norm_type", default="pre_norm", type=str)
    parser.add_argument("--auto_resume", action="store_true", default=False)
    parser.add_argument("--not_auto_resume", action="store_false", dest="auto_resume")
    # * dev: proposals
    parser.add_argument("--proposal_feature_levels", default=1, type=int)
    parser.add_argument("--proposal_in_stride", default=8, type=int)
    parser.add_argument("--proposal_tgt_strides", default=[8, 16, 32, 64], type=int, nargs="+")
    # * dev decoder: global decoder
    parser.add_argument("--decoder_type", default="deform", type=str)
    parser.add_argument("--decoder_use_checkpoint", default=False, action="store_true")
    parser.add_argument("--decoder_rpe_hidden_dim", default=512, type=int)
    parser.add_argument("--decoder_rpe_type", default="linear", type=str)
    # weight decay mult
    parser.add_argument(
        "--wd_norm_names",
        default=["norm", "bias", "rpb_mlp", "cpb_mlp", "logit_scale", "relative_position_bias_table",
                 "level_embed", "reference_points", "sampling_offsets", "rel_pos"],
        type=str,
        nargs="+"
    )
    parser.add_argument("--wd_norm_mult", default=1.0, type=float)
    parser.add_argument("--use_layerwise_decay", action="store_true", default=False)
    parser.add_argument("--lr_decay_rate", default=1.0, type=float)

    # * Segmentation
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Train segmentation head if the flag is provided",
    )

    # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )

    # * Matcher
    parser.add_argument(
        "--set_cost_class",
        default=2,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=5,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou",
        default=2,
        type=float,
        help="giou box coefficient in the matching cost",
    )

    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--cls_loss_coef", default=2, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--focal_alpha", default=0.25, type=float)

    # dataset parameters
    parser.add_argument("--dataset_file", default="pascal")
    parser.add_argument("--coco_split", default=1, help="coco split for training or evaluation")
    parser.add_argument("--coco_path", default="./data/coco", type=str)
    parser.add_argument("--coco_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")

    parser.add_argument(
        "--output_dir", default="exps/ex1", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument(
        "--cache_mode",
        default=False,
        action="store_true",
        help="whether to cache images on memory",
    )

    # * eval technologies
    parser.add_argument("--eval", action="store_true")
    # topk for eval
    parser.add_argument("--topk", default=100, type=int)

    parser.add_argument("--upretrain", action="store_true", default = False)

    # * training technologies
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--use_checkpoint", default=False, action="store_true")

    # * logging technologies
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_name", type=str)
    return parser


def main(args):
    if args.device == "cuda":
        utils.init_distributed_mode(args)
    else:
        args.distributed = False
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    output_dir = Path(args.output_dir)
    if args.auto_resume:
        if os.path.exists(output_dir / f'rng_state_{utils.get_rank()}.pth'):
                    rng_state_dict = torch.load(output_dir / f'rng_state_{utils.get_rank()}.pth', map_location='cpu')
                    torch.set_rng_state(rng_state_dict['cpu_rng_state'])
                    torch.cuda.set_rng_state(rng_state_dict['gpu_rng_state'])
                    np.random.set_state(rng_state_dict['numpy_rng_state'])
                    random.setstate(rng_state_dict['py_rng_state'])

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    if args.upretrain:
        for key, param in model.named_parameters():
            if key.startswith("backbone.0.net"):
                param.requires_grad = False


    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(model_without_ddp)
    print("number of params:", n_parameters)

    if args.use_fp16:
        scaler = torch.cuda.amp.GradScaler()


    if args.upretrain:
        if args.dataset_file == "pascalvoc":
            data_loader_train, mappers = build_train_dataloader(args, ["pascalvoc_train_2007_Base","pascalvoc_train_2012_Base"])
            data_loader_val, _ = build_eval_dataloader(args, ["pascalvoc_testup_Base",])
        elif args.dataset_file == "coco":
            data_loader_train, mappers = build_train_dataloader(args, [f"coco_train_{args.coco_split}_Base",]) 
            data_loader_val, _ = build_eval_dataloader(args, [f"coco_val_{args.coco_split}_Novel",])
    else:
        if args.dataset_file =="pascalvoc":
            data_loader_train, mappers = build_train_dataloader(args, ["pascalvoc_train_2007_Base","pascalvoc_train_2012_Base",])
            data_loader_val, mapper = build_eval_dataloader(args, ["pascalvoc_test_Novel",])
        elif args.dataset_file == "coco":
            data_loader_train, mappers = build_train_dataloader(args, [f"coco_train_{args.coco_split}_Base",]) 
            data_loader_val, mapper = build_eval_dataloader(args, [f"coco_val_{args.coco_split}_novel",])
        elif args.dataset_file == "dota":
            data_loader_train, mappers = build_train_dataloader(args, ["dota_train_Base",]) 
            data_loader_val, mapper = build_eval_dataloader(args, ["dota_val_Novel",])
            


    prmpt_df = None##pd.read_csv(file_path)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    #breakpoint()
    param_dicts = utils.get_param_dict(model_without_ddp, args, use_layerwise_decay=args.use_layerwise_decay)
    if args.sgd:
        optimizer = torch.optim.SGD(
            param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            param_dicts, lr=args.lr, weight_decay=args.weight_decay
        )

    # TODO: is there any more elegant way to print the param groups?
    name_dicts = utils.get_param_dict(model_without_ddp, args, return_name=True, use_layerwise_decay=args.use_layerwise_decay)
    if args.use_layerwise_decay:
        for i, name_dict in enumerate(name_dicts):
            print(f"Group-{i} {json.dumps(name_dict, indent=2)}")
    else:
        for i, name_dict in enumerate(name_dicts):
            print(f"Group-{i} lr: {name_dict['lr']} wd: {name_dict['weight_decay']}")
            print(json.dumps(name_dict["params"], indent=2))
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    #breakpoint()
    print("dataset_len=",len(data_loader_train.dataset.dataset)) ##
    epoch_iter = len(data_loader_train.dataset.dataset) // args.batch_size
    if args.warmup:
        lambda0 = lambda cur_iter: cur_iter / args.warmup if cur_iter < args.warmup else (0.1 if cur_iter > args.lr_drop * epoch_iter else 1)
    else:
        lambda0 = lambda cur_iter: 0.1 if cur_iter > args.lr_drop * epoch_iter else 1
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True, device_ids=[args.gpu])
        model_without_ddp = model.module


    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location="cpu")
        model_without_ddp.detr.load_state_dict(checkpoint["model"])

    if args.use_wandb and dist.get_rank() == 0:
        wandb.init(
            entity=args.wandb_entity,
            project='Plain-DETR',
            id=args.wandb_name,  # set id as wandb_name for resume
            name=args.wandb_name,
        )

    if args.auto_resume:
        resume_from = utils.find_latest_checkpoint(output_dir)
        if resume_from is not None:
            print(f'Use autoresume, overwrite args.resume with {resume_from}')
            args.resume = resume_from
        else:
            print(f'Use autoresume, but can not find checkpoint in {output_dir}')
    if args.resume and os.path.exists(args.resume):
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")

        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
            checkpoint["model"], strict=False
        )
        if len(missing_keys) > 0:
            print("Missing Keys: {}".format(missing_keys))

        if len(unexpected_keys) > 0:
            print("Unexpected Keys: {}".format(unexpected_keys))
        if (
            not args.eval
            and args.auto_resume
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            import copy

            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint["optimizer"])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg["lr"] = pg_old["lr"]
                pg["initial_lr"] = pg_old["initial_lr"]

            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            print(
                # For LambdaLR, the lambda funcs are not been stored in state_dict, see
                # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#LambdaLR.state_dict
                "Warning: lr scheduler has been resumed from checkpoint, but the lambda funcs are not been stored in state_dict. \n"
                "So the new lr schedule would override the resumed lr schedule."
            )
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint["epoch"] + 1

            if args.use_fp16 and "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
            
        # check the resumed model
        if not args.eval:
            test_stats = evaluate(
                model,
                criterion,
                postprocessors,
                data_loader_val,
                device,
                args.output_dir,
                step=args.start_epoch * math.ceil(len(data_loader_train.dataset.dataset) / args.batch_size),
                use_wandb=args.use_wandb,
                reparam=args.reparam,
                df = prmpt_df,
            )

    if args.eval:
        assert args.batch_size == 1, "evaluation is done only with batch size of 1"
        test_stats = evaluateall(
            model,
            criterion,
            postprocessors,
            data_loader_val,
            device,
            args.output_dir,
            step=args.start_epoch * math.ceil(len(data_loader_train.dataset.dataset) / args.batch_size),
            use_wandb=args.use_wandb,
            reparam=args.reparam,
            mapper= mapper[0],
            dataset_file= args.dataset_file,
        )

        return
    if args.upretrain:
        for key, param in model.module.named_parameters(): ##model.module   TODO  just for cpu usage change it back
            if key.startswith("backbone.0.net"):
                param.requires_grad = False
    print("Start training")
    start_time = time.time()
    #breakpoint()
    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     sampler_train.set_epoch(epoch)
        if not args.upretrain:
            if epoch >= 2:
                for key, param in model.module.named_parameters(): ##model.module   TODO  just for cpu usage change it back
                    if key.startswith("backbone.0.net"):
                        param.requires_grad = False
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            lr_scheduler,
            args.clip_max_norm,
            k_one2many=args.k_one2many,
            lambda_one2many=args.lambda_one2many,
            use_wandb=args.use_wandb,
            use_fp16=args.use_fp16,
            scaler=scaler if args.use_fp16 else None,
            epoch_iter = epoch_iter,
            upretrain = args.upretrain,
        )
        if not args.upretrain:
            for mapper in mappers:
                mapper.update_epoch(epoch +1)
        if args.output_dir:
            checkpoint_paths = []#output_dir / "checkpoint.pth"]
            # extra checkpoint before LR drop and every 5 epochs
            checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")#{epoch:04}.pth")   ##{epoch:04}.pth"){epoch:04}
            for checkpoint_path in checkpoint_paths:
                save_dict = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                }
                if args.use_fp16:
                    save_dict["scaler"] = scaler.state_dict()
                utils.save_on_master(
                    save_dict,
                    checkpoint_path,
                )
                rng_state_dict = {
                    'cpu_rng_state': torch.get_rng_state(),
                    'gpu_rng_state': torch.cuda.get_rng_state(),
                    'numpy_rng_state': np.random.get_state(),
                    'py_rng_state': random.getstate()
                }
                torch.save(rng_state_dict, output_dir / f'rng_state_{utils.get_rank()}.pth')
        if not args.upretrain:
            test_stats = evaluate(
                model,
                criterion,
                postprocessors,
                data_loader_val,
                device,
                args.output_dir,
                step=(epoch + 1) * math.ceil(len(data_loader_train.dataset.dataset) / args.batch_size),
                use_wandb=args.use_wandb,
                reparam=args.reparam,
                df = prmpt_df,
            )
        else:
            test_stats = {}
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }
        rng_state_dict = {
                    'cpu_rng_state': torch.get_rng_state(),
                    'gpu_rng_state': torch.cuda.get_rng_state(),
                    'numpy_rng_state': np.random.get_state(),
                    'py_rng_state': random.getstate()
                }
        torch.save(rng_state_dict, output_dir / f'rng_state_{utils.get_rank()}.pth')
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            coco_evaluator = None
            if coco_evaluator is not None:
                (output_dir / "eval").mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ["latest.pth"]
                    if epoch % 50 == 0:
                        filenames.append(f"{epoch:03}.pth")
                    for name in filenames:
                        torch.save(
                            coco_evaluator.coco_eval["bbox"].eval,
                            output_dir / "eval" / name,
                        )

                areaRngLbl = ['', '50', '75', 's', 'm', 'l']
                msg = "copypaste: "
                for label in areaRngLbl:
                    msg += f"AP{label} "
                for ap in coco_evaluator.coco_eval["bbox"].stats[:len(areaRngLbl)]:
                    msg += "{:.3f} ".format(ap)
                print(msg)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Deformable DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
