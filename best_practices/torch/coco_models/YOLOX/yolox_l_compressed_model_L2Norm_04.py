#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

import torch

from yolox.exp import Exp as MyExp
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.models.losses import IOUloss
from yolox.utils import bboxes_iou, meshgrid

from loguru import logger

class RetrainUtils(nn.Module):
    def __init__(self, dtype=torch.float16):
        super().__init__()
        self.device = 'cuda'
        self.strides = [8, 16, 32]
        self.in_channels = [256, 512, 1024]
        self.n_anchors = 1
        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.grids = [torch.zeros(1).cuda()] * len(self.in_channels)
        # self.num_classes = 20
        self.num_classes = 80
        self.decode_in_inference = True
    def get_outputs_for_train(self, outputs):
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        all_outputs = []

        for k, output in enumerate(outputs):
            stride_this_level = self.strides[k]
            # (reg_output, obj_output, cls_output) = torch.split(output,[4,1,20],dim=1)
            # reg_output = reg_output.clone()
            reg_output = output[:,:4,:,:].clone()
            output, grid = self.get_output_and_grid(
                output, k, stride_this_level, output.type()
            )
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.zeros(1, grid.shape[1])
                .fill_(stride_this_level)
                .type(output.type())
            )
            if self.use_l1:
                batch_size = reg_output.shape[0]
                hsize, wsize = reg_output.shape[-2:]
                reg_output = reg_output.view(
                    batch_size, self.n_anchors, 4, hsize, wsize
                )
                reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                    batch_size, -1, 4
                )
                origin_preds.append(reg_output.clone())
            all_outputs.append(output)
        return x_shifts, y_shifts, expanded_strides, torch.cat(all_outputs,1), origin_preds, torch.float16


    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:] # 다 다름 ex)[] 64 64 / 32 32 /16 16 ]/ 76 76 ...
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def get_losses(self, x_shifts, y_shifts, expanded_strides, labels, outputs, origin_preds, dtype):
        bbox_preds = outputs[:, :, :4]  # [batch, 10710, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, 10710, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, 10710, 11]

        n_label = (labels.sum(dim=2) > 0).sum(dim=1)
        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)
        y_shifts = torch.cat(y_shifts, 1)
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)
        
        cls_targets = []
        reg_targets = []
        l1_targets = [] 
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(n_label[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes)) # new_zeros -> 안의 size만큼 0으로 채우고, dtype와 device는 target tensor를 따른다.
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                # print(f"labels : {labels.type()}")
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes, # preds와 매칭된 classes들 -> [num_fg_img]
                        fg_mask, # grid의 중심이 gt_bbox안에 들어오거나, cneter_radius안에 들어오는 anchor index [True or False] -> [10710] foreground
                        pred_ious_this_matching, # preds중에서 gt와 매칭된 애들과 그 애들의 ious -> [num_fg_img]
                        matched_gt_inds, # 매칭된 gt들의 indexes -> [num_fg_img]
                        num_fg_img, # pred들 중 gt와 매칭이 된 애들의 갯수
                    ) = self.get_assignments(
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        "cpu"
                    )
                torch.cuda.empty_cache()
                num_fg += num_fg_img
                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1) # [num_fg_img, 11] * [num_fg_img, 1] = [num_fg_img, 11]
                obj_target = fg_mask.unsqueeze(-1).to(self.device) # [10710, 1]
                reg_target = gt_bboxes_per_image[matched_gt_inds] # [num_fg_img, 4]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target) # [num_fg_img, 11]
            reg_targets.append(reg_target) # [num_fg_img, 4]
            obj_targets.append(obj_target.type(dtype)) # [10710*batch, 1]
            fg_masks.append(fg_mask) # [10710]
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0) # [num_fg ,11]            
        reg_targets = torch.cat(reg_targets, 0) # [num_fg ,4]
        obj_targets = torch.cat(obj_targets, 0).to(self.device) # [10710*batch ,1]
        fg_masks = torch.cat(fg_masks, 0) # [10710*batch]
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)
        # print(f"iou : {bbox_preds.view(-1, 4)[fg_masks].size()}, {reg_targets.size()}")
        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        # print(f"size : {bbox_preds.size()} bbox_preds : {bbox_preds}")
        # print(f"size : {reg_targets.size()} reg_targets : {reg_targets}")
        # print(f"obj : {obj_preds.view(-1, 1).size()}, {obj_targets.size()}")
        # print(f"obj : {obj_preds.get_device()}, {obj_targets.get_device()}")
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        # print(f"cls : {cls_preds.view(-1, self.num_classes)[fg_masks].size()}, {cls_targets.size()}")
        # print(f"cls_preds : {cls_preds.type()}, cls_targets : {cls_targets.type()}")
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg
        # print(f"size : {cls_preds.size()} cls_preds : {cls_preds}")
        # print(f"size : {cls_targets.size()} cls_targets : {cls_targets}")
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0
        reg_weight = 5.0
        # print(f"loss_iou : {loss_iou}, loss_obj : {loss_obj}, loss_cls : {loss_cls}")
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1)
        )


    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx, 
        num_gt, 
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes, # [num_gt]
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds,
        labels,
        mode="gpu",
    ):
        # if mode == "cpu":
            # print("------------CPU Mode for This Batch-------------")
            # gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            # bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            # gt_classes = gt_classes.cpu().float()
            # expanded_strides = expanded_strides.cpu().float()
            # x_shifts = x_shifts.cpu()
            # y_shifts = y_shifts.cpu()
        gt_bboxes_per_image = gt_bboxes_per_image.to(self.device)
        bboxes_preds_per_image = bboxes_preds_per_image.to(self.device)
        gt_classes = gt_classes.to(self.device)
        expanded_strides = expanded_strides.to(self.device)
        x_shifts = x_shifts.to(self.device)
        y_shifts = y_shifts.to(self.device)
        
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt
        )
        
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask] # [fg_num, 4]
        cls_preds_ = cls_preds[batch_idx][fg_mask] # [fg_num, 11]
        obj_preds_ = obj_preds[batch_idx][fg_mask] # [fg_num, 1]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        # if mode == "cpu":
        #     gt_bboxes_per_image = gt_bboxes_per_image.cpu()
        #     bboxes_preds_per_image = bboxes_preds_per_image.cpu()
        # print(f"gt_bboxes : {gt_bboxes_per_image.type()}, preds_bboxes : {bboxes_preds_per_image.type()}") # FloatTensor, FloatTensor
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1) # [num_gt, 1, 11]
            .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        # if mode == "cpu":
        #     cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()
        cls_preds_ = cls_preds_.to(self.device)
        obj_preds_ = obj_preds_.to(self.device)

        # with torch.cuda.amp.autocast(enabled=False):
            # cls_preds : [fg_num, 11]
            # obj_preds : [fg_num, 1]
        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
                # cls_preds_ : [num_gt, fg_num, 11]
                # gt_cls_per_image : [num_gt, fg_num. 11]
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)

        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        # if mode == "cpu":
        #     gt_matched_classes = gt_matched_classes.cuda()
        #     fg_mask = fg_mask.cuda()
        #     pred_ious_this_matching = pred_ious_this_matching.cuda()
        #     matched_gt_inds = matched_gt_inds.cuda()
        gt_matched_classes = gt_matched_classes.to(self.device)
        fg_mask = fg_mask.to(self.device)
        pred_ious_this_matching = pred_ious_this_matching.to(self.device)
        matched_gt_inds = matched_gt_inds.to(self.device)

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
    ):
        gt_bboxes_per_image = gt_bboxes_per_image.to(self.device)
        expanded_strides_per_image = expanded_strides[0].to(self.device)
        # print(f"stride : {expanded_strides_per_image.get_device()}, x_shifts[0] : {x_shifts[0].get_device()}")
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0]  * expanded_strides_per_image
        
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        # 각 grid의 center로 부터 거리인듯? -> 각 object와 각 grid의 center 간의 거리 [num_fg, num_anchors]
        # print(f"x_centers_per_image : {x_centers_per_image.size()}, gt_bboxes_per_image_l : {gt_bboxes_per_image_l.size()}")
        # print(f"x_centers : {x_centers_per_image.get_device()}, gt_bboxes : {gt_bboxes_per_image.get_device()}")
        b_l = x_centers_per_image - gt_bboxes_per_image_l 
        b_r = gt_bboxes_per_image_r - x_centers_per_image 
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bboxes_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)
        # [num_gt, 10710, 4] -> 각 object의 각 grid의 center(anchor)의 left, right, top, bottom 거리

        is_in_boxes = bboxes_deltas.min(dim=-1).values > 0.0 # [num_gt, 10710]
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0 # [10710], anchor가 모든 gt box들중 하나의 gt box안에라도 들어있는 애들 -> True else False

        center_radius = 2.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors # [num_gt, 10710]
        ) - center_radius * expanded_strides_per_image.unsqueeze(0) # [1, 10710]
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)


        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0 # [10710], anchor가 모든 gt center box들중 하나의 gt center box안에라도 들어있는 애들 -> True else False

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center
    
    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # cost : [num_gt, fg_num], pair_wise_ious : [num_gt, fg_num], gt_classes : [num_gt], fg_mask : [10710]
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious # [num_gt, fg_num]
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1)) # -> iou를 기반으로 k개를 고를떄 한 gt당 k개만 예측하자 이말인가?
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1) # [num_gt, n_candidate_k]
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1) # top-k개 고른 iou의 합을 기반으로 최종적으로 몇개의 cost를 가지고 계산할지 결정한다.?
        dynamic_ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1
        
        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0) # 각 anchor마다 matching되는 gt의 갯수 [fg_num]
        if (anchor_matching_gt > 1).sum() > 0: # gt와 1개 이상 matching되는 anchor가 1개 이상일때
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0) # matching되는 anchor중 gt와 cost가 가장 작은 gt의 idx를 고른다.
            # cost_argmin : 
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0 # -> anchor중 위의 조건을 만족하는 gt와 매칭이되면 True else False [num_fg]
        num_fg = fg_mask_inboxes.sum().item() # fg_mask_inboxes중 True의 갯수

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]

        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs
    def forward(self, preds, labels, inps):
        if self.training:
            labels = labels.to('cuda')
            x_shifts, y_shifts, expanded_strides, outputs, origin_preds, dtype = self.get_outputs_for_train(preds)
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.get_losses(x_shifts, y_shifts, expanded_strides, labels, outputs, origin_preds, dtype)
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
            return outputs        
        else:
            outputs = []
            for pred in preds:
                (reg_output, obj_output, cls_output) = torch.split(pred,[4,1,80],dim=1)
                output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()],1)
                outputs.append(output)
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=output[0].type())
            else:
                return outputs         
            
class compressed_YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, pt_model, head_utils):
        super().__init__()
        self.backbone = pt_model
        self.head = head_utils

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        preds = self.backbone(x)
        if self.training:
            assert targets is not None
            outputs = self.head(preds,targets,x)
        else:
            outputs = self.head(preds, targets, x)
        return outputs
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        self.max_epochs = 150
        self.basic_lr_per_img = 0.001 / 64
    def get_model(self):
        
        model = torch.load('../YOLOX_l_compressed_model_L2Norm_04.pt')
        retrain_utils = RetrainUtils()
        self.model =  compressed_YOLOX(model, retrain_utils)
        return self.model