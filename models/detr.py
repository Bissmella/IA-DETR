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

"""
Deformable DETR model and criterion classes.
"""
import gc
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (
    NestedTensor,
    nested_tensor_from_tensor_list,
    accuracy,
    get_world_size,
    interpolate,
    is_dist_avail_and_initialized,
    inverse_sigmoid,
    _get_clones,
)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (
    DETRsegm,
    PostProcessPanoptic,
    PostProcessSegm,
    dice_loss,
    sigmoid_focal_loss,
)
from .transformer import build_transformer
import copy


class PlainDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_feature_levels,
        aux_loss=True,
        with_box_refine=False,
        two_stage=False,
        num_queries_one2one=300,
        num_queries_one2many=0,
        mixed_selection=False,
        pre_train = False,
        feature_dim=128,
    ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
            num_queries_one2one: number of object queries for one-to-one matching part
            num_queries_one2many: number of object queries for one-to-many matching part
            mixed_selection: a trick for Deformable DETR two stage

        """
        super().__init__()
        num_queries = num_queries_one2one + num_queries_one2many
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        if pre_train:
            self.feature_embed = MLP(hidden_dim, hidden_dim, feature_dim, 3)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim  * 2)## * 2)
        elif mixed_selection:
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            )
        ])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.pre_train = pre_train

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        if pre_train:
            for layer in self.feature_embed.layers:
                nn.init.xavier_uniform_(layer.weight.data, gain=1)
                nn.init.constant_(layer.bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (
            (transformer.decoder.num_layers + 1)
            if two_stage
            else transformer.decoder.num_layers
        )
        
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            if pre_train:
                self.feature_embed = _get_clones(self.feature_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList(
                [self.class_embed for _ in range(num_pred)]
            )
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        self.num_queries_one2one = num_queries_one2one
        self.mixed_selection = mixed_selection

    def forward(self, samples: NestedTensor, prompts):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        #** TODO add same thing for the prompt
        
        prompts, prmpt_pos = self.backbone(prompts)
        
        prmpts = []
        prmpt_masks = []
        for l, feat in enumerate(prompts):
            prmpt, mask = feat.decompose()
            prmpts.append(self.input_proj[l](prmpt))
            prmpt_masks.append(mask)
            assert mask is not None
        #**

        query_embeds = None
        if not self.two_stage or self.mixed_selection:
            query_embeds = self.query_embed.weight[0: self.num_queries, :]

        # make attn mask
        """ attention mask to prevent information leakage
        """
        self_attn_mask = (
            torch.zeros([self.num_queries, self.num_queries, ]).bool().to(src.device)
        )
        self_attn_mask[self.num_queries_one2one:, 0: self.num_queries_one2one, ] = True
        self_attn_mask[0: self.num_queries_one2one, self.num_queries_one2one:, ] = True

        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
            enc_outputs_delta,
            output_proposals,
            max_shape
        ) = self.transformer(srcs, masks, pos, query_embeds, self_attn_mask, prmpts, prmpt_masks, prmpt_pos)

        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_classes_one2many = []
        outputs_coords_one2many = []
        if self.pre_train:
            outputs_features_one2one = []
            outputs_features_one2many = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            if self.pre_train:
                outputs_feature = self.feature_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()

            outputs_classes_one2one.append(outputs_class[:, 0: self.num_queries_one2one])
            outputs_classes_one2many.append(outputs_class[:, self.num_queries_one2one:])

            if self.pre_train:
                outputs_features_one2one.append(outputs_feature[:, 0: self.num_queries_one2one])
                outputs_features_one2many.append(outputs_feature[:, self.num_queries_one2one:])

            outputs_coords_one2one.append(outputs_coord[:, 0: self.num_queries_one2one])
            outputs_coords_one2many.append(outputs_coord[:, self.num_queries_one2one:])

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)

        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)

        out = {
            "pred_logits": outputs_classes_one2one[-1],
            "pred_boxes": outputs_coords_one2one[-1],
            "pred_logits_one2many": outputs_classes_one2many[-1],
            "pred_boxes_one2many": outputs_coords_one2many[-1],
        }
        if self.pre_train:
            out['pred_features'] = outputs_features_one2one[-1]
            out['pred_features_one2many'] = outputs_features_one2many[-1]

        if not self.pre_train:
            outputs_features_one2many = None
            outputs_features_one2one = None
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                outputs_classes_one2one, outputs_coords_one2one, outputs_features_one2one
            )
            out["aux_outputs_one2many"] = self._set_aux_loss(
                outputs_classes_one2many, outputs_coords_one2many, outputs_features_one2many
            )

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
            }
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, output_features = None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if output_features ==None:  #in case of fine-tuning where there is not output_features for contrastive loss
            return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
        else:
            return [
                {"pred_logits": a, "pred_boxes": b, "pred_features": c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], output_features[:-1])
            ]
    @torch.no_grad()
    def encode(self, samples):
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None



        query_embeds = None
        if not self.two_stage or self.mixed_selection:
            query_embeds = self.query_embed.weight[0: self.num_queries, :]

        # make attn mask
        """ attention mask to prevent information leakage
        """
        self_attn_mask = (
            torch.zeros([self.num_queries, self.num_queries, ]).bool().to(src.device)
        )
        self_attn_mask[self.num_queries_one2one:, 0: self.num_queries_one2one, ] = True
        self_attn_mask[0: self.num_queries_one2one, self.num_queries_one2one:, ] = True
        # del samples, features
        # torch.cuda.empty_cache()
        # gc.collect()
        
        return srcs[0][0].unsqueeze(0), masks[0][0].unsqueeze(0), pos[0][0].unsqueeze(0), query_embeds, self_attn_mask, srcs[0][1:], masks[0][1:], pos[0][1:]
    
    @torch.no_grad()
    def decode(self, srcs, masks, pos, query_embeds, self_attn_mask, prmpts, prmpt_masks, prmpt_pos):
        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
            enc_outputs_delta,
            output_proposals,
            max_shape
        ) = self.transformer(srcs, masks, pos, query_embeds, self_attn_mask, prmpts, prmpt_masks, prmpt_pos)

        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_classes_one2many = []
        outputs_coords_one2many = []
        if self.pre_train:
            outputs_features_one2one = []
            outputs_features_one2many = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            if self.pre_train:
                outputs_feature = self.feature_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()

            outputs_classes_one2one.append(outputs_class[:, 0: self.num_queries_one2one])
            outputs_classes_one2many.append(outputs_class[:, self.num_queries_one2one:])

            if self.pre_train:
                outputs_features_one2one.append(outputs_feature[:, 0: self.num_queries_one2one])
                outputs_features_one2many.append(outputs_feature[:, self.num_queries_one2one:])

            outputs_coords_one2one.append(outputs_coord[:, 0: self.num_queries_one2one])
            outputs_coords_one2many.append(outputs_coord[:, self.num_queries_one2one:])

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)

        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)

        out = {
            "pred_logits": outputs_classes_one2one[-1],
            "pred_boxes": outputs_coords_one2one[-1],
            "pred_logits_one2many": outputs_classes_one2many[-1],
            "pred_boxes_one2many": outputs_coords_one2many[-1],
        }
        if self.pre_train:
            out['pred_features'] = outputs_features_one2one[-1]
            out['pred_features_one2many'] = outputs_features_one2many[-1]

        if not self.pre_train:
            outputs_features_one2many = None
            outputs_features_one2one = None
        # if self.aux_loss:
        #     out["aux_outputs"] = self._set_aux_loss(
        #         outputs_classes_one2one, outputs_coords_one2one, outputs_features_one2one
        #     )
        #     out["aux_outputs_one2many"] = self._set_aux_loss(
        #         outputs_classes_one2many, outputs_coords_one2many, outputs_features_one2many
        #     )

        # if self.two_stage:
        #     enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        #     out["enc_outputs"] = {
        #         "pred_logits": enc_outputs_class,
        #         "pred_boxes": enc_outputs_coord,
        #     }
        # del srcs, masks, pos, query_embeds, self_attn_mask, prmpts, prmpt_masks, prmpt_pos
        # torch.cuda.empty_cache()
        # gc.collect()
        return out





class PlainDETRReParam(PlainDETR):
    def forward(self, samples: NestedTensor, prompts):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        #** TODO add same thing for the prompt
        prompts, prmpt_pos = self.backbone(prompts)
        prmpts = []
        prmpt_masks = []
        for l, feat in enumerate(prompts):
            prmpt, mask = feat.decompose()
            prmpts.append(self.input_proj[l](prmpt))
            prmpt_masks.append(mask)
            assert mask is not None
        #**

        query_embeds = None
        if not self.two_stage or self.mixed_selection:
            query_embeds = self.query_embed.weight[0: self.num_queries, :]

        # make attn mask
        """ attention mask to prevent information leakage
        """
        self_attn_mask = (
            torch.zeros([self.num_queries, self.num_queries, ]).bool().to(src.device)
        )
        self_attn_mask[self.num_queries_one2one:, 0: self.num_queries_one2one, ] = True
        self_attn_mask[0: self.num_queries_one2one, self.num_queries_one2one:, ] = True

        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
            enc_outputs_delta,
            output_proposals,
            max_shape
        ) = self.transformer(srcs, masks, pos, query_embeds, self_attn_mask, prmpts, prmpt_masks, prmpt_pos)

        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_classes_one2many = []
        outputs_coords_one2many = []

        outputs_coords_old_one2one = []
        outputs_deltas_one2one = []
        outputs_coords_old_one2many = []
        outputs_deltas_one2many = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                outputs_coord = box_ops.box_xyxy_to_cxcywh(box_ops.delta2bbox(
                    reference, tmp, max_shape
                ))
            else:
                raise NotImplementedError

            outputs_classes_one2one.append(outputs_class[:, 0: self.num_queries_one2one])
            outputs_classes_one2many.append(outputs_class[:, self.num_queries_one2one:])

            outputs_coords_one2one.append(outputs_coord[:, 0: self.num_queries_one2one])
            outputs_coords_one2many.append(outputs_coord[:, self.num_queries_one2one:])

            outputs_coords_old_one2one.append(reference[:, :self.num_queries_one2one])
            outputs_coords_old_one2many.append(reference[:, self.num_queries_one2one:])
            outputs_deltas_one2one.append(tmp[:, :self.num_queries_one2one])
            outputs_deltas_one2many.append(tmp[:, self.num_queries_one2one:])

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)

        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)

        out = {
            "pred_logits": outputs_classes_one2one[-1],
            "pred_boxes": outputs_coords_one2one[-1],
            "pred_logits_one2many": outputs_classes_one2many[-1],
            "pred_boxes_one2many": outputs_coords_one2many[-1],

            "pred_boxes_old": outputs_coords_old_one2one[-1],
            "pred_deltas": outputs_deltas_one2one[-1],
            "pred_boxes_old_one2many": outputs_coords_old_one2many[-1],
            "pred_deltas_one2many": outputs_deltas_one2many[-1],
        }

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                outputs_classes_one2one, outputs_coords_one2one, outputs_coords_old_one2one, outputs_deltas_one2one
            )
            out["aux_outputs_one2many"] = self._set_aux_loss(
                outputs_classes_one2many, outputs_coords_one2many, outputs_coords_old_one2many, outputs_deltas_one2many
            )

        if self.two_stage:
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord_unact,
                "pred_boxes_old": output_proposals,
                "pred_deltas": enc_outputs_delta,
            }
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_coord_old, outputs_deltas):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b, "pred_boxes_old": c, "pred_deltas": d, }
            for a, b, c, d in zip(outputs_class[:-1], outputs_coord[:-1], outputs_coord_old[:-1], outputs_deltas[:-1])
        ]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, reparam=False, upretrain=False):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
            loss_bbox_type: how to perform loss_bbox
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.loss_bbox_type = 'l1' if (not reparam) else 'reparam'
        self.con_temperature = 0.5
        self.upretrain = upretrain

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True, enc=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        ).long()
        
        #** TODO  only part allowed to change for letting specific class in based on class_choice
        if not self.upretrain:
            if enc == False:
                target_classes_o.fill_(1)
        #**
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        

        target_classes[idx] = target_classes_o

        # if enc == True and False:   #in the encoder some of top classes are exempted from the classification loss
        #     mask = torch.zeros_like(src_logits, dtype=torch.bool)
        #     mask[idx] = 1
        #     src_masked = src_logits.masked_fill(mask, float('-inf'))
        #     top5 = torch.topk(src_masked[..., 0], k=5, dim=1)[1]
        #     mask_target = torch.zeros_like(target_classes, dtype=torch.bool)
        #     mask_target.scatter_(1, top5, 1)
        #     target_classes = target_classes.masked_fill(mask_target, 0)
            #target_classes[top5] = 0

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )

        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]

        loss_ce = (
            sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                num_boxes,
                alpha=self.focal_alpha,
                gamma=2,
            )
            * src_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, enc = False):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        if self.loss_bbox_type == "l1":
            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        elif self.loss_bbox_type == "reparam":
            src_deltas = outputs["pred_deltas"][idx]
            src_boxes_old = outputs["pred_boxes_old"][idx]
            target_deltas = box_ops.bbox2delta(src_boxes_old, target_boxes)
            loss_bbox = F.l1_loss(src_deltas, target_deltas, reduction="none")
        else:
            raise NotImplementedError

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses
    
    def con_loss_calc(self, query, key, negatives, reduction_override=None, avg_factor=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else 'mean')##self.reduction)

        num_query = query.size(0)
        num_key = key.size(0)

        # query: (m, feat_dim)
        # key  : (n, feat_dim)
        # neg  : (k, feat_dim)
        query = F.normalize(query)
        key = F.normalize(key)
        neg = F.normalize(negatives.detach())

        key = key.unsqueeze_(1)                              # (n, feat_dim) => (n, 1, feat_dim)
        neg = neg.unsqueeze_(0).expand(num_key, -1, -1)      # (k, feat_dim) => (1, k, feat_dim) => (n, k, feat_dim)
        feats = torch.cat([key, neg], dim=1)                 # (n, 1, feat_dim) + (n, k, feat_dim)  => (n, 1+k, feat_dim)

        query = query.unsqueeze(0).expand(num_key, -1, -1)   # (m, feat_dim) => (n, m, feat_dim)
        logits = torch.bmm(query, feats.permute(0, 2, 1))    # (n, m, feat_dim) @ (n, feat_dim, 1+k) => (n, m, 1+k)
        logits = logits.reshape(num_query*num_key, -1)       # (n, m, 1+k) => (n*m, 1+k)
        logits = logits / self.con_temperature

        labels = torch.zeros((num_query*num_key, ), dtype=torch.long).to(query.device)
        loss = F.cross_entropy(logits, labels)
        return loss ##* self.loss_weight
    
    def loss_contrastive(self, outputs, targets, indices, num_boxes, log=True, kone2many =1):
        """
        contrastive loss gotten from https://github.com/liming-ai/AlignDet/blob/master/AlignDet/models/losses/contrastive_loss.py

        constrastive loss is calculated only during pre-training on objects of interest (authentic objects other than the randomly queried objects)

        """
        outputs_feats = outputs['pred_features']
        ooi_indices = []
        tens1_c = 0
        tens2_c = 0
        for i, (tens1, tens2) in enumerate(indices):
            tens1 = tens1 + tens1_c
            tens2 = tens2 + tens2_c
            tens2_c = tens2_c + len(targets[i]["classes"])
            tens1_c = tens1_c + outputs_feats.shape[1]
            ooi_indices.append((tens1, tens2))##\[tens1[:id_max], tens1[id_max + 1:]]), torch.cat([tens2[:id_max], tens2[id_max + 1:]])))
        
        tensor1s = [t[0] for t in ooi_indices]
        tensor2s = [t[1] for t in ooi_indices]

        pred_indices = torch.cat(tensor1s, dim=0).to(outputs_feats.device)
        lbl_indices = torch.cat(tensor2s, dim=0).to(outputs_feats.device)

        
        target_classes = torch.cat([t["classes"] for t in targets]).to(outputs_feats.device)
        #print("lbl indices ", lbl_indices)
        #print("target classes ", target_classes)
        online_labels = target_classes[lbl_indices]
        outputs_features = outputs_feats.flatten(0, 1)
        ##copy rest from align Det
        loss_con_ = torch.tensor(0.).to(outputs_feats.device)
        num_valid_labels = 0
        # print("online labels ", online_labels.dtype)
        # print("online labels ", online_labels)
        # uniques = torch.unique(online_labels)
        # print("uniques ", uniques.shape)
        for label in torch.unique(online_labels):
            # ignore the background class
            if label == -1: #  -1 is place holder for prompt   self.num_classes:
                continue

            #                    label                      sample           #
            query_inds = (online_labels == label) #* (online_label_weights > 0)
            #key_inds   = (target_labels == label) * (target_label_weights > 0)
            online_neg_inds = (online_labels != label) #* (online_label_weights > 0)
            #target_neg_inds = (target_labels != label) * (target_label_weights > 0)

            num_valid_labels += 1

            query_inds = pred_indices[query_inds]
            query = outputs_features[query_inds]
            key = query
            neg_inds = pred_indices[online_neg_inds]
            neg = outputs_features[neg_inds]
            
            loss_con_ = loss_con_ + self.con_loss_calc(query, key, neg)
        losses = {}
        if num_valid_labels>0:
            losses['loss_con'] = loss_con_ / num_valid_labels   #TODO  change 'loss_cls' to something else
        else:
            losses['loss_con'] = loss_con_
        return losses



    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(
            [t["masks"] for t in targets]
        ).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(
            src_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
            "contrastive": self.loss_contrastive,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs" and k != "enc_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)
            )

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs["log"] = False
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                # bt["labels"] = bt["classes"]
                # bt["boxes"] = bt["orig_boxes"]
                bt["labels"] = torch.zeros_like(bt["labels"])
            indices = self.matcher(enc_outputs, bin_targets, encoder = True)
            for loss in self.losses:
                if loss == "masks":
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == "labels":
                    # Logging is enabled only for the last layer
                    kwargs["log"] = False
                    kwargs["enc"] = True
                if loss == "contrastive":
                    #no contrastive loss for encoder output
                    continue
                if loss == "boxes":
                    kwargs["enc"] = True
                l_dict = self.get_loss(
                    loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs
                )
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, topk=100, reparam=False):
        super().__init__()
        self.topk = topk
        self.reparam = reparam
        print("topk for eval:", self.topk)

    @torch.no_grad()
    def forward(self, outputs, target_sizes, original_target_sizes=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        assert not self.reparam or original_target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), self.topk, dim=1
        )
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        if self.reparam:
            img_h, img_w = img_h[:, None, None], img_w[:, None, None]  # [BS, 1, 1]
            boxes[..., 0::2].clamp_(min=torch.zeros_like(img_w), max=img_w)
            boxes[..., 1::2].clamp_(min=torch.zeros_like(img_h), max=img_h)
            scale_h, scale_w = (original_target_sizes / target_sizes).unbind(1)
            scale_fct = torch.stack([scale_w, scale_h, scale_w, scale_h], dim=1)
        else:
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [
            {"scores": s, "labels": l, "boxes": b}
            for s, l, b in zip(scores, labels, boxes)
        ]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 3 #if args.dataset_file != "coco" else 91
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model_class = PlainDETR if (not args.reparam) else PlainDETRReParam
    model = model_class(
        backbone,
        transformer,
        num_classes=num_classes,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        num_queries_one2one=args.num_queries_one2one,
        num_queries_one2many=args.num_queries_one2many,
        mixed_selection=args.mixed_selection,
        pre_train=args.upretrain
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {
        "loss_ce": args.cls_loss_coef,
        "loss_bbox": args.bbox_loss_coef,
        "loss_giou": args.giou_loss_coef,
    }
    if args.upretrain:
        weight_dict['loss_con'] = 2  #TODO contrastive loss coefficient hardcoded to 2
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    new_dict = dict()
    for key, value in weight_dict.items():
        new_dict[key] = value
        new_dict[key + "_one2many"] = value
    weight_dict = new_dict

    losses = ["labels", "boxes", "cardinality"]
    if args.upretrain:
        losses += ["contrastive"]
    if args.masks:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(
        num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha, reparam=args.reparam, upretrain=args.upretrain
    )
    criterion.to(device)
    postprocessors = {"bbox": PostProcess(topk=args.topk, reparam=args.reparam)}
    if args.masks:
        postprocessors["segm"] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(
                is_thing_map, threshold=0.85
            )

    return model, criterion, postprocessors
