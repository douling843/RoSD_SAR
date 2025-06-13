# -*- coding: utf-8 -*-
import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin

from mmdet.models.losses import accuracy
import math
import mmdet.core.bbox.iou_calculators.iou2d_calculator as iou

import copy
import numpy as np
import cv2
import mmcv
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json

@HEADS.register_module()
class StandardRoIHeadOAMILhrsidmy(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)  amil.py

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """  


        def max_iou_sum(suggestion_boxes, bboxes_clean):
            """
            计算每次最大的IoU，并将其累加
            suggestion_boxes: 4个水平建议框的坐标，格式为[[x1, y1, x2, y2], ...]
            ground_truth_boxes: 3个ground-truth框的坐标，格式为[[x1, y1, x2, y2], ...]
            """
            suggestion_boxes_cpu = suggestion_boxes.cpu()
#             ground_truth_boxes = ground_truth_boxes.cpu()
            # 初始化总的IoU和
            total_iou = 0
            suggestion_boxes_cpu = np.array(suggestion_boxes_cpu)
            for proposal in suggestion_boxes_cpu:
                    # 计算与所有ground-truth的IoU
                ious = [calculate_iou(proposal, bbox) for bbox in bboxes_clean]
                    # 找到最大的IoU
                max_iou = max(ious)
                total_iou += max_iou 
            return total_iou



        def calculate_iou(box1, box2):
                    # box1 和 box2 为[左上顶点(x1, y1), 右下顶点(x2, y2)]的形式
                    # 计算交集的坐标
            x_left = max(box1[0], box2[0])
            y_top = max(box1[1], box2[1])
            x_right = min(box1[2], box2[2])
            y_bottom = min(box1[3], box2[3])

                    # 如果没有交集，返回0
            if x_left >= x_right or y_top >= y_bottom:
                return 0

                    # 计算交集的面积
            intersection_area = (x_right - x_left) * (y_bottom - y_top)

                    # 计算两个框的面积
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

                    # 计算并集的面积
            union_area = box1_area + box2_area - intersection_area

                    # 计算IoU
            iou = intersection_area / union_area
            return iou
        
        
        def find_bboxes_in_coco(img_metas, annotation_file):
            """
            根据图像名称，从COCO格式的标注文件中获取该图片对应的bbox标注。

            :param img_metas: dict，包含图像的元信息，例如 {'filename': './path/to/image.jpg', 'ori_filename': 'image.jpg'}
            :param annotation_file: str，COCO格式的标注文件路径，例如 'train.json'
            :return: list，包含该图像对应的所有bbox标注
            """
            # 读取COCO格式的标注文件
            with open(annotation_file, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)

            # 提取images和annotations字段
            images = coco_data.get('images', [])
            annotations = coco_data.get('annotations', [])

            # 获取图像原始名称（如'000394.jpg'）
            target_filename = img_metas['ori_filename']
            # 找到目标图像的ID
            image_id = None
            for image in images:
                if image['file_name'] == target_filename:
                    image_id = image['id']
                    break

            if image_id is None:
                print(f"Error: 图像 {target_filename} 在标注文件中未找到！")
                return []

            # 根据图像ID提取对应的bbox标注
            bboxes = []
            for annotation in annotations:
                if annotation['image_id'] == image_id:
                    bboxes.append(annotation['bbox'])

            return bboxes




        def get_sampling_results(img_metas, proposal_list, gt_bboxes, gt_bboxes_ignore):
            num_imgs = len(img_metas)

            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
            return sampling_results

        '''
        Object-Aware Multiple Instance Learning
        '''



        losses = dict()
        

        if self.with_bbox or self.with_mask:

            sampling_results = get_sampling_results(img_metas, proposal_list, gt_bboxes, gt_bboxes_ignore)

        # FasterRCNN second stage forward
        bbox_results, rois, bbox_targets = self._bbox_forward_train(x, sampling_results, gt_bboxes, gt_labels, img_metas, **kwargs)   ####在189行

        # apply OA-MIL
        loss_oamil, pseudo_bbox_targets = self._oamil(bbox_targets, bbox_results, rois, x, img_metas)     #####212行   loss_oamil, pseudo_bbox_targets = self._oamil(bbox_targets, bbox_results, rois, x) 

        # compute classification and localization loss
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'], bbox_results['bbox_pred'], rois, *bbox_targets, pseudo_bbox_targets=pseudo_bbox_targets)
        losses.update(loss_bbox)
        losses.update(loss_oamil)

        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        i_extractor.num_inputs= 4
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)

        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)     # type='Shared2FCBBoxHeadOAMIL',

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas, **kwargs):
        """Run forward function and compute loss for box head during training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])

        bbox_results = self._bbox_forward(x, rois)
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg, **kwargs)

        return bbox_results, rois, bbox_targets

    def _oamil(self, bbox_targets, bbox_results, rois, x, img_metas):
        """
        Procedure:
            1. perform instance selection
            2. compute instance selection loss
        """
        loss_bbox = dict()

        # lambda controls whether to perform OA-MIL
        if self.bbox_head.oamil_lambda > 0:              ##### 在配置文件中 oamil_lambda=0.1,   # for 40% noise 
            '''
            1. Perform instance selection
            '''
            # get bbox targets
            labels, cur_bbox_targets = bbox_targets[0], bbox_targets[2]

            pos_inds = (labels >= 0) & (labels < self.bbox_head.num_classes)
#
            pos_labels = labels[pos_inds.type(torch.bool)]
#
            pos_bbox_targets = cur_bbox_targets[pos_inds.type(torch.bool)]
#
            noisy_gt_boxes = self.bbox_head.bbox_coder.decode(rois[:, 1:][pos_inds.type(torch.bool)], pos_bbox_targets)
#
            uniq_inst, pos_indices = torch.unique(noisy_gt_boxes.sum(dim=1), sorted=True, return_inverse=True)
#
            oaie_scores_list, pseudo_bbox_targets = self._instance_selection(bbox_results, labels, rois, x, cur_bbox_targets, img_metas)

            '''
            2. Compute instance selection loss
            '''
            inst_scores_list = []
            for confidence_scores in oaie_scores_list:
                inst_scores = self._get_instance_cls_scores(pos_labels, confidence_scores, uniq_inst, pos_indices)
                inst_scores_list.append(inst_scores)
            
            if len(inst_scores_list) > 1:
                if self.bbox_head.oaie_type == 'refine':
                    loss_bbox['loss_oais'] = (1-inst_scores_list[0]) + (1-sum(inst_scores_list[1:])/self.bbox_head.oaie_num)*self.bbox_head.oaie_coef
                elif self.bbox_head.oaie_type == 'random':
                    loss_bbox['loss_oais'] = 1 - sum(inst_scores_list)/(self.bbox_head.oaie_num+1)
            else:
                loss_bbox['loss_oais'] = 1-inst_scores_list[0]
            loss_bbox['loss_oais'] *= self.bbox_head.oamil_lambda
        else:
            pseudo_bbox_targets = None

        return loss_bbox, pseudo_bbox_targets

    def _instance_selection(self, bbox_results, labels, rois, x, cur_bbox_targets, img_metas):
        """
        Procedure of instance selection:
            1. construct object bags 
            2. apply instance selector (OA-IE is optional in step 2)
            3. get best selected instances using Eq. (4)
        """

        '''
        1. Construct object bags
        '''
        # get indices of object bags from noisy gt
        pos_inds = (labels >= 0) & (labels < self.bbox_head.num_classes)

        inds = torch.ones(pos_inds.sum()).cuda()

        pos_bbox_targets = cur_bbox_targets[pos_inds.type(torch.bool)]

        
        noisy_gt_boxes = self.bbox_head.bbox_coder.decode(rois[:, 1:][pos_inds.type(torch.bool)], pos_bbox_targets)

        
        uniq_inst, pos_indices = torch.unique(noisy_gt_boxes.sum(dim=1), sorted=True, return_inverse=True)

        
        object_bag_indices = [torch.where(pos_indices == inst)[0] for inst in torch.unique(pos_indices)]

        

        new_bbox_results = {}
        new_bbox_results['cls_score'], new_bbox_results['bbox_pred'] = bbox_results['cls_score'], bbox_results['bbox_pred']

        new_bbox_results['bbox_pred'] = new_bbox_results['bbox_pred'].view(new_bbox_results['bbox_pred'].size(0), -1, 4)[pos_inds.type(torch.bool)]

        
        '''
        2. Apply instance selector
            - OA-IE is optional in this step
        '''

        oaie_bboxes_list, oaie_scores_list = [], []
        iter_num = self.bbox_head.oaie_num+1 if self.bbox_head.oaie_flag and self._epoch+1 >= self.bbox_head.oaie_epoch else 1    # self.bbox_head.oaie_num=4, oaie_flag=True, oaie_epoch=9

        for i in range(iter_num):
            # get prediction of each instance
            bbox_pred = new_bbox_results['bbox_pred']

            
            inds = torch.ones(bbox_pred.size(0)).type(torch.bool).cuda()

            pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)[inds, labels[pos_inds.type(torch.bool)]]



            if i == 0 or self.bbox_head.oaie_type == 'random':

                new_pred_boxes = self.bbox_head.bbox_coder.decode(rois[:, 1:][pos_inds.type(torch.bool)], pos_bbox_pred)

            else:
                new_pred_boxes = self.bbox_head.bbox_coder.decode(new_roi[:, 1:], pos_bbox_pred)
            
            new_roi = rois[pos_inds.type(torch.bool)].clone()

            new_roi[:,1:] = new_pred_boxes
            oaie_bboxes_list.append(new_pred_boxes)

            new_bbox_results = self._bbox_forward(x, new_roi)      ######  176行

            confidence_scores = torch.softmax(new_bbox_results['cls_score'], dim=1)[inds.type(torch.bool), labels[pos_inds.type(torch.bool)]]

            
            oaie_scores_list.append(confidence_scores)

        '''
        3. Get best selected instances
        '''
        # perform OA-IS
        oais_flag = self.bbox_head.oais_flag and (self._epoch+1 >= self.bbox_head.oais_epoch)   ### oais_flag=1   oais_epoch=2

        if oais_flag:

            pseudo_bbox_targets = torch.zeros_like(cur_bbox_targets[pos_inds.type(torch.bool)]).cuda()

            pseudo_gt_boxes = torch.zeros_like(cur_bbox_targets[pos_inds.type(torch.bool)]).cuda()

            new_pred_boxes, confidence_scores = oaie_bboxes_list[0], oaie_scores_list[0]

                object_bag_boxes = new_pred_boxes[index]   




                object_bag_scores = confidence_scores[index]

                


                noise_factor = 0.4
                object_bag_scores_sum = sum(object_bag_scores)
                object_bag_scores_all = object_bag_scores.view(-1, 1)/object_bag_scores_sum
                object_bag_boxes_weight = object_bag_boxes * object_bag_scores_all
                box2_col_sums = object_bag_boxes_weight.sum(dim=0)
                box2_col_sums = box2_col_sums.clamp(min=0.0)
                best_selected_instance = noise_factor * box2_col_sums + noisy_gt_boxes[index[0]].view(1, -1) * (1 - noise_factor)


#####################################################################################################################################################我的第三个想法逐渐线性
#                 noise_factor = 0.4
# # #                 num_rows = object_bag_boxes.size(0)
# # #                 if num_rows <= 1:
# # #                     best_instance = object_bag_boxes.clone()
# # #                     best_score = object_bag_scores.clone()
# # #                     best_instance = best_instance.clamp(min=0.0)
# # #                     best_selected_instance = 0.4 * best_instance.detach() + noisy_gt_boxes[index[0]].view(1, -1) * (1 - 0.4)

# # #                 else:
#                 diji_epoch = self._epoch
#                 object_bag_boxes, object_bag_scores = self.remove_lowest_20_percent(object_bag_boxes, object_bag_scores, diji_epoch)
#                 object_bag_scores_sum = sum(object_bag_scores)
#                 object_bag_scores_all = object_bag_scores.view(-1, 1)/object_bag_scores_sum
#                 object_bag_boxes_weight = object_bag_boxes * object_bag_scores_all
#                 box2_col_sums = object_bag_boxes_weight.sum(dim=0)
#                 box2_col_sums = box2_col_sums.clamp(min=0.0).view(1, -1)
#                 best_selected_instance = noise_factor * box2_col_sums + noisy_gt_boxes[index[0]].view(1, -1) * (1 - noise_factor)



                # reset grount-truth according to best selected instance
                pseudo_gt_targets = self.bbox_head.bbox_coder.encode(rois[:, 1:][pos_inds.type(torch.bool)][index], best_selected_instance.repeat(len(index), 1))
                pseudo_bbox_targets[index] = pseudo_gt_targets
                pseudo_gt_boxes[index] = best_selected_instance.repeat(len(index), 1)
        else:
            pseudo_bbox_targets = None

        return oaie_scores_list, pseudo_bbox_targets
    

    def remove_lowest_20_percent(self, data_tensor, score_tensor, iteration):

        num_rows = data_tensor.size(0)

        # if num_rows <= 1:
        #     return data_tensor, score_tensor
        # num_remove = int(num_rows * 0.2)
        # if num_remove == 0:
        #     break
        num_remove = min(int((num_rows / 12) * iteration), num_rows-1)

        # 找到分数最低的行的索引
        if num_remove == 0:
            data_tensor2 = data_tensor.clone()
            score_tensor2 = score_tensor.clone()
        else:
            _, indices = torch.topk(score_tensor, num_remove, largest=False)

        # 移除分数最低的行
            mask = torch.ones(num_rows, dtype=torch.bool)
            mask[indices] = False
            data_tensor2 = data_tensor[mask]
            score_tensor2 = score_tensor[mask]

        return data_tensor2, score_tensor2
             #########################################################################

    def _get_instance_cls_scores(self, pos_labels, confidence_scores, uniq_inst, pos_indices):
        """Compute confidence scores of object bags in training."""
        # instance-level confidence scores
        inst_labels = []
        inst_scores = []
        for inst in torch.unique(pos_indices):
            inst_inds = torch.where(pos_indices == inst)[0]
            inst_scores.append(confidence_scores[inst_inds].max().view(-1))
            inst_labels.append(pos_labels[inst_inds[0]].view(-1))

        inst_labels = torch.cat(inst_labels, dim=0)
        inst_scores = torch.cat(inst_scores, dim=0)

        # class-level confidence scores
        cls_scores = 0
        for cls in torch.unique(inst_labels):
            cls_inds = torch.where(inst_labels == cls)[0]
            cls_scores += inst_scores[cls_inds].mean()
        cls_scores /= len(torch.unique(inst_labels))
        return cls_scores
    
    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            if pos_rois.shape[0] == 0:
                return dict(loss_mask=None)
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)
            if pos_inds.shape[0] == 0:
                return dict(loss_mask=None)
            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        if self.test_cfg is None:
            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_metas, proposal_list, None, rescale=rescale)
            det_bboxes = [boxes.cpu().numpy() for boxes in det_bboxes]
            det_labels = [labels.cpu().numpy() for labels in det_labels]
            return det_bboxes, det_labels

        else:
            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
            bbox_results = [
                bbox2result(det_bboxes[i], det_labels[i],
                            self.bbox_head.num_classes)
                for i in range(len(det_bboxes))
            ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]
