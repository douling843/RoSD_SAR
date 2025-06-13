import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin

from mmdet.models.losses import accuracy
import math
import mmdet.core.bbox.iou_calculators.iou2d_calculator as iou

import copy
import numpy as np    #######################我添加的
import cv2            #######################我添加的
import mmcv            #######################我添加的
import os              #######################我添加的
from pathlib import Path   #######################我添加的
from PIL import Image, ImageDraw, ImageFont   #######################我添加的
import json   #######################我添加的

@HEADS.register_module()
class StandardRoIHeadOAMIL(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
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
        self.bbox_head = build_head(bbox_head)

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
        
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            sampling_results = get_sampling_results(img_metas, proposal_list, gt_bboxes, gt_bboxes_ignore)

        # FasterRCNN second stage forward
        bbox_results, rois, bbox_targets = self._bbox_forward_train(x, sampling_results, gt_bboxes, gt_labels, img_metas, **kwargs)

        # apply OA-MIL
        loss_oamil, pseudo_bbox_targets = self._oamil(bbox_targets, bbox_results, rois, x)

        # compute classification and localization loss
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'], bbox_results['bbox_pred'], rois, *bbox_targets, pseudo_bbox_targets=pseudo_bbox_targets)
        losses.update(loss_bbox)
        losses.update(loss_oamil)

        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        cls_score, bbox_pred = self.bbox_head(bbox_feats)
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

    def _oamil(self, bbox_targets, bbox_results, rois, x):
        """
        Procedure:
            1. perform instance selection
            2. compute instance selection loss
        """
        loss_bbox = dict()

        # lambda controls whether to perform OA-MIL
        if self.bbox_head.oamil_lambda > 0:
            '''
            1. Perform instance selection
            '''
            # get bbox targets
            labels, cur_bbox_targets = bbox_targets[0], bbox_targets[2]
            pos_inds = (labels >= 0) & (labels < self.bbox_head.num_classes)
            pos_labels = labels[pos_inds.type(torch.bool)]

            # get indices of unique gt boxes
            pos_bbox_targets = cur_bbox_targets[pos_inds.type(torch.bool)]
            noisy_gt_boxes = self.bbox_head.bbox_coder.decode(rois[:, 1:][pos_inds.type(torch.bool)], pos_bbox_targets)
            uniq_inst, pos_indices = torch.unique(noisy_gt_boxes.sum(dim=1), sorted=True, return_inverse=True)
            
            # perform Object-Aware Instance Selection (OA-IS)
            oaie_scores_list, pseudo_bbox_targets = self._instance_selection(bbox_results, labels, rois, x, cur_bbox_targets)

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

    def _instance_selection(self, bbox_results, labels, rois, x, cur_bbox_targets):
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

        # initialize bbox results for object bags
        new_bbox_results = {}
        new_bbox_results['cls_score'], new_bbox_results['bbox_pred'] = bbox_results['cls_score'], bbox_results['bbox_pred']
        
        # keep positive instances
        new_bbox_results['bbox_pred'] = new_bbox_results['bbox_pred'].view(new_bbox_results['bbox_pred'].size(0), -1, 4)[pos_inds.type(torch.bool)]
        
        '''
        2. Apply instance selector
            - OA-IE is optional in this step
        '''
        # The number of iteration depends on whether to perform Object-Aware Instance Extension (OA-IE)
        #   - iter num=1 if w/o OA-IE 
        #   - iter num=N+1 if with OA-IE, where N is the number of OA-IE
        oaie_bboxes_list, oaie_scores_list = [], []
        iter_num = self.bbox_head.oaie_num+1 if self.bbox_head.oaie_flag and self._epoch+1 >= self.bbox_head.oaie_epoch else 1
        for i in range(iter_num):
            # get prediction of each instance
            bbox_pred = new_bbox_results['bbox_pred']
            inds = torch.ones(bbox_pred.size(0)).type(torch.bool).cuda()
            pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)[inds, labels[pos_inds.type(torch.bool)]]

            # decode prediction of each instance
            if i == 0 or self.bbox_head.oaie_type == 'random':
                new_pred_boxes = self.bbox_head.bbox_coder.decode(rois[:, 1:][pos_inds.type(torch.bool)], pos_bbox_pred)
            else:
                new_pred_boxes = self.bbox_head.bbox_coder.decode(new_roi[:, 1:], pos_bbox_pred)
            
            new_roi = rois[pos_inds.type(torch.bool)].clone()
            new_roi[:,1:] = new_pred_boxes
            oaie_bboxes_list.append(new_pred_boxes)

            # apply instance selector and output confidence scores
            # NOTE: instance selector shares the same parameters with classifier
            new_bbox_results = self._bbox_forward(x, new_roi)
            confidence_scores = torch.softmax(new_bbox_results['cls_score'], dim=1)[inds.type(torch.bool), labels[pos_inds.type(torch.bool)]]
            oaie_scores_list.append(confidence_scores)

        '''
        3. Get best selected instances
        '''
        # perform OA-IS
        oais_flag = self.bbox_head.oais_flag and (self._epoch+1 >= self.bbox_head.oais_epoch)
        if oais_flag:
            # to save best selected instances
            pseudo_bbox_targets = torch.zeros_like(cur_bbox_targets[pos_inds.type(torch.bool)]).cuda()
            pseudo_gt_boxes = torch.zeros_like(cur_bbox_targets[pos_inds.type(torch.bool)]).cuda()
            
            new_pred_boxes, confidence_scores = oaie_bboxes_list[0], oaie_scores_list[0]
            # iterate over each object bag
            for index in object_bag_indices:
                # get confidence score of each instance in an object bag
                object_bag_boxes = new_pred_boxes[index]
                object_bag_scores = confidence_scores[index]
                
                
                #########################################################################################################################
#                 # ############################################################################################################################################    这是计算当前这个包属于哪个图片        
#                 def find_matching_image(object_bag_boxes, img_metas):
#                     """
#                     输入：
#                     object_bag_boxes: 目标检测框坐标 tensor([x_min, y_min, x_max, y_max])
#                     img_metas: COCO格式的图片元数据列表

#                     输出：
#                     matching_filename: 匹配的图片文件名
#                     """
#                     # 将检测框转换为numpy数组便于计算
#                     det_box = object_bag_boxes.cpu().detach().numpy()

#                     # 初始化最小距离和匹配文件名
#                     min_distance = float('inf')
#                     matching_filename = None

#                     # 遍历每张图片的元数据
#                     for img_meta in img_metas:
#                         # 获取ground truth框
#                         gt_bboxes = img_meta['gt_bboxes'].cpu().detach().numpy()
#                         filename = img_meta['filename']
#                         scale_factor = img_meta['scale_factor']
#                         flip = img_meta['flip']
#                         ori_shape = img_meta['ori_shape']
#                         img_shape = img_meta['img_shape']

#                         # 对每个ground truth框进行比较
#                         for gt_box in gt_bboxes:
# #                             # 如果图片被翻转，需要调整坐标
# #                             if flip and img_meta['flip_direction'] == 'horizontal':
# #                                 gt_box_adjusted = gt_box.copy()
# #                                 gt_box_adjusted[0] = img_shape[1] - gt_box[2]  # x_min = width - x_max
# #                                 gt_box_adjusted[2] = img_shape[1] - gt_box[0]  # x_max = width - x_min
# #                             else:
# #                                 gt_box_adjusted = gt_box.copy()

#                             # 计算检测框和ground truth框的中心点距离
#                             det_center = np.array([(det_box[0] + det_box[2]) / 2, 
#                                                  (det_box[1] + det_box[3]) / 2])
#                             gt_box_adjusted = gt_box.copy()
#                             gt_center = np.array([(gt_box_adjusted[0] + gt_box_adjusted[2]) / 2,
#                                                 (gt_box_adjusted[1] + gt_box_adjusted[3]) / 2])

#                             # 计算欧几里得距离
#                             distance = np.linalg.norm(det_center - gt_center)

#                             # 更新最小距离和对应文件名
#                             if distance < min_distance:
#                                 min_distance = distance
#                                 matching_filename = filename

#                     return matching_filename

#                 object_bag_box = object_bag_boxes[0]
#                 matching_filename = find_matching_image(object_bag_box, img_metas)       #######找到了匹配的那个图片：例如 ：./data/ssdd/ssdd/images/train/000554.jpg

#  ################################################################################################################################               
#                 def draw_boxes_on_image(image_path, object_boxes, scores, img_metas, output_dir, bboxes_clean, best_selected_instance):             ########画框和得分
#                     """
#                     在图片上绘制检测框和ground truth框，并保存新图片

#                     参数：
#                     image_path: 输入图片路径
#                     object_boxes: 检测框坐标
#                     scores: 检测框得分
#                     img_metas: 图片元数据
#                     output_dir: 输出文件夹
#                     """
#                     # 读取图片
#                     image = cv2.imread(image_path)
#                     _0, width_, _1 = image.shape
#                     if image is None:
#                         raise ValueError(f"无法读取图片: {image_path}")

#                     # 获取对应的img_meta
#                     target_meta = None
#                     for meta in img_metas:
#                         if meta['filename'] == image_path:
#                             target_meta = meta
#                             break

#                     if target_meta is None:
#                         raise ValueError(f"未找到匹配的img_meta: {image_path}")

#                     # 获取ground truth框
#                     gt_bboxes = target_meta['gt_bboxes'].cpu().detach().numpy()
#                     scale_factor = torch.from_numpy(target_meta['scale_factor'])
#                     gt_bboxes = gt_bboxes / scale_factor
#                     best_instance = best_selected_instance.detach().clone()
#                     best_instance = best_instance.cpu().detach().numpy() / scale_factor

#                     # 绘制检测框（橘黄色）
#                     for i, box in enumerate(object_boxes.cpu().detach().numpy()):
#                         box = box / scale_factor    
#                         x_min, y_min, x_max, y_max = map(int, box)
#                         if target_meta['flip'] and target_meta['flip_direction'] == 'horizontal':
#                             x_min = int((width_ - box[2]).item())  # x_min = width - x_max                         
#                             x_max = int((width_ - box[0]).item())  # x_max = width - x_min
#                         else:
#                             x_min, y_min, x_max, y_max = map(int, box)
#                         # 橘黄色 (BGR格式: 0, 165, 255)
#                         cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 165, 255), 2)
#                         # 添加得分标签
#                         score = scores[i].item()
#                         label = f"{score:.2f}"
#                         cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 
#                                    0.5, (0, 165, 255), 1)

#                     # 绘制ground truth框（红色）
#                     for box in gt_bboxes:
#                         x_min, y_min, x_max, y_max = map(int, box)
#                         if target_meta['flip'] and target_meta['flip_direction'] == 'horizontal':
#                             x_min = int((width_ - box[2]).item())  # x_min = width - x_max
#                             x_max = int((width_ - box[0]).item())  # x_max = width - x_min
#                         else:
#                             x_min, y_min, x_max, y_max = map(int, box)
                        
#                         # 红色 (BGR格式: 0, 0, 255)
#                         cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                    
#                     # 绘制bboxes_clean框（绿色）
#                     for box in bboxes_clean:
#                         x_min, y_min, x_max, y_max = map(int, box)                 ##############################bboxes_clean没有翻转   也不需要进行/ scale_factor
#                         cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        
#                     # 绘制best_instance框（蓝色）      青色
#                     for box in best_instance:
#                         x_min, y_min, x_max, y_max = map(int, box)
#                         if target_meta['flip'] and target_meta['flip_direction'] == 'horizontal':
#                             x_min = int((width_ - box[2]).item())  # x_min = width - x_max
#                             x_max = int((width_ - box[0]).item())  # x_max = width - x_min
#                         else:
#                             x_min, y_min, x_max, y_max = map(int, box)
#                         # 红色 (BGR格式: 0, 0, 255)
#                         cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (196, 206, 62), 2)
                    
#                     #计算bboxes_clean 和 best_instance的IoU
#                     def calculate_iou1(box1, box2):
#                         if len(box1.shape) > 1 and box1.shape[0] > 1:
#                             box1 = box1[0]
#                         box1_cpu = box1.reshape(-1, 4)    ####   bboxes_clean
#                         box2_cpu = box2.cpu().numpy()
#                         x_min = box2_cpu[0][0]
#                         y_min = box2_cpu[0][1]
#                         x_max = box2_cpu[0][2]
#                         y_max = box2_cpu[0][3]
#                         if target_meta['flip'] and target_meta['flip_direction'] == 'horizontal':
#                             x_min = int(width_ - box2_cpu[0][2])
#                             x_max = int(width_ - box2_cpu[0][0])
                                                       
#                         x_left = max(box1_cpu[0][0], x_min)
#                         y_top = max(box1_cpu[0][1], y_min)
#                         x_right = min(box1_cpu[0][2], x_max)
#                         y_bottom = min(box1_cpu[0][3], y_max)

#                         if x_left >= x_right or y_top >= y_bottom:
#                             return 0

#                         intersection_area = (x_right - x_left) * (y_bottom - y_top)
#                         box1_area = (box1_cpu[0][2] - box1_cpu[0][0]) * (box1_cpu[0][3] - box1_cpu[0][1])
#                         box2_area = (x_max - x_min) * (y_max - y_min)
#                         union_area = box1_area + box2_area - intersection_area
#                         iou = intersection_area / union_area
#                         return iou
                
#                     IoU1 = calculate_iou1(bboxes_clean, best_instance)
                    
                    
#                     #计算bboxes_clean 和 GT的IoU
#                     def calculate_iou2(box1, box2):
#                         if len(box1.shape) > 1 and box1.shape[0] > 1:
#                             box1 = box1[0]
#                         box1_cpu = box1.reshape(-1, 4)    ####   bboxes_clean
#                         if len(box2.shape) > 1 and box2.shape[0] > 1:
#                             box2 = box2[0]
#                         box2_cpu = box2.reshape(-1, 4) 
#                         x_min = box2_cpu[0][0]
#                         y_min = box2_cpu[0][1]
#                         x_max = box2_cpu[0][2]
#                         y_max = box2_cpu[0][3]
#                         if target_meta['flip'] and target_meta['flip_direction'] == 'horizontal':
#                             x_min = int(width_ - box2_cpu[0][2])
#                             x_max = int(width_ - box2_cpu[0][0])
                                                       
#                         x_left = max(box1_cpu[0][0], x_min)
#                         y_top = max(box1_cpu[0][1], y_min)
#                         x_right = min(box1_cpu[0][2], x_max)
#                         y_bottom = min(box1_cpu[0][3], y_max)

#                         if x_left >= x_right or y_top >= y_bottom:
#                             return 0

#                         intersection_area = (x_right - x_left) * (y_bottom - y_top)
#                         box1_area = (box1_cpu[0][2] - box1_cpu[0][0]) * (box1_cpu[0][3] - box1_cpu[0][1])
#                         box2_area = (x_max - x_min) * (y_max - y_min)
#                         union_area = box1_area + box2_area - intersection_area
#                         iou = intersection_area / union_area
#                         return iou
#                     IoU2 = calculate_iou2(bboxes_clean, gt_bboxes)
                    
#                     max_score_idx = torch.argmax(scores)
#                     corresponding_box = object_boxes[max_score_idx]
#                     corresponding_box = corresponding_box.reshape(-1, 4) 
#                     corresponding_box = corresponding_box.cpu().detach().numpy()
#                     corresponding_box = corresponding_box / scale_factor
                    
#                     IoU3 = calculate_iou1(bboxes_clean, corresponding_box)
                    
#                     # 转换为PIL图像格式，以便添加文本
#                     pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#                     draw = ImageDraw.Draw(pil_image)
#                     # 字体设置 (可以根据自己的系统选择字体)
#                     font = ImageFont.truetype("/root/.local/share/fonts/DejaVu Sans Mono Bold for Powerline.ttf", 15)  # 指定其他字体font_path="/usr/share/fonts/arial.ttf", font_size = 20) 
#                      # 设置文本内容
#                     text1 = f"IoU: {IoU1:.2f},"     ######text = f"All bboxes: {pos_bboxes.size(0)}. Average IoU: {average_iou.item():.2f}"
#                     # 设置文本位置
#                     text_position1 = (image.shape[1] - 300, 5)  # 右上角，适当调整位置
#                     text_color1 = (62, 206, 196)  # 蓝色 (0, 0, 255)    青色 (62, 206, 196)
#                      # 添加文本到图片
#                     draw.text(text_position1, text1, font=font, fill=text_color1)
#                     text2 = f"IoU: {IoU3:.2f},"
#                     text_position2 = (image.shape[1] - 200, 5) 
#                     text_color2 = (255, 165, 0)  # 红色  (255, 0, 0)
#                     draw.text(text_position2, text2, font=font, fill=text_color2)
#                     text3 = f"IoU: {IoU2:.2f}."
#                     text_position3 = (image.shape[1] - 100, 5) 
#                     text_color3 = (255, 0, 0)  # 橘黄色   (255, 165, 0)
#                     draw.text(text_position3, text3, font=font, fill=text_color3)
                    
#                     # 转换回OpenCV图像格式
#                     image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    
#                     # 创建输出文件夹
#                     if not os.path.exists(output_dir):
#                         os.makedirs(output_dir)

#                     # 保存新图片
#                     output_filename = os.path.join(output_dir, os.path.basename(image_path))
#                     cv2.imwrite(output_filename, image)
########################################################################################################################################################
                
                
                
                
                
                
                ###########################################################################################################################
                
                
                
                
                

                # get best instance in each object bag
                _, max_inds = torch.max(object_bag_scores, dim=0)
                best_score = object_bag_scores[max_inds].clone()
                best_instance = object_bag_boxes[max_inds].clone()
                best_instance = best_instance.clamp(min=0.0).view(1, -1)

                # modify noisy gt using Eq. (4)
                phi = ((best_score.detach())**self.bbox_head.oais_gamma).clamp(max=self.bbox_head.oais_theta)
                best_selected_instance = best_instance.detach() * phi + noisy_gt_boxes[index[0]].view(1, -1) * (1 - phi)
                
                
                ###################################################################################################################################
#                    ################################################################################################################    画框和得分
#                 if self._epoch == 11:
#                     output_dir = '/workspace/OA-MIL/vis_sampling/boxes_score/hla04/epoch11'
#                     annotation_file = './data/ssdd/ssdd/annotations/train.json'      ####OA-MIL/data/ssdd/ssdd/annotations/train.json
#                     def get_bboxes_from_coco(annotation_file, image_path):
#                         """
#                         从COCO标注文件中提取指定图片的bounding box信息

#                         参数：
#                         annotation_file: COCO格式的标注文件路径
#                         image_path: 目标图片的路径

#                         返回：
#                         image_info: 图片的基本信息（id, width, height等）
#                         bboxes: 该图片的bounding box列表，每个bbox包含[x_min, y_min, width, height]
#                         """
#                         # 检查文件是否存在
#                         if not os.path.exists(annotation_file):
#                             raise FileNotFoundError(f"标注文件不存在: {annotation_file}")

#                         # 读取COCO标注文件
#                         with open(annotation_file, 'r') as f:
#                             coco_data = json.load(f)

#                         # 提取图片信息
#                         images = coco_data['images']
#                         target_image_info = None

#                         # 根据文件名查找对应图片信息
#                         for img in images:
#                             # COCO中的filename可能只包含文件名，也可能是完整路径，视数据集而定
#                             # 这里假设filename是完整路径，如果不是，需要调整匹配逻辑
#                             if img['file_name'] == image_path.split("/")[-1]:
#                                 target_image_info = img
#                                 break

#                         if target_image_info is None:
#                             raise ValueError(f"未在标注文件中找到图片: {image_path}")

#                         # 获取图片ID
#                         image_id = target_image_info['id']

#                         # 提取该图片的所有bounding box
#                         annotations = coco_data['annotations']
#                         bboxes = []

#                         for ann in annotations:
#                             if ann['image_id'] == image_id:
#                                 # COCO格式的bbox是[x_min, y_min, width, height]
#                                 bbox = ann['bbox']
#                                 bboxes.append(bbox)

#                         if not bboxes:
#                             print(f"警告: 图片 {image_path} 在标注文件中没有对应的bounding box")

#                         return  bboxes
#                     bboxes_clean = get_bboxes_from_coco(annotation_file, matching_filename)    ###get_bboxes_from_coco
#                     bboxes_clean = torch.tensor(bboxes_clean, dtype=torch.float32)      ####数据类型转成tensor
#                     wh = bboxes_clean[:, 2:]
#                     bboxes_clean[:, 2:] =  bboxes_clean[:, :2] + wh  ###右下角=左上角+wh
#                     bboxes_clean=bboxes_clean.cpu().detach().numpy() 
#                     draw_boxes_on_image(matching_filename, object_bag_boxes, object_bag_scores, img_metas, output_dir, bboxes_clean, best_selected_instance)
                ####################################################################################################################
                
                ########################################################################################################################################

                # reset grount-truth according to best selected instance
                pseudo_gt_targets = self.bbox_head.bbox_coder.encode(rois[:, 1:][pos_inds.type(torch.bool)][index], best_selected_instance.repeat(len(index), 1))
                pseudo_bbox_targets[index] = pseudo_gt_targets
                pseudo_gt_boxes[index] = best_selected_instance.repeat(len(index), 1)
        else:
            pseudo_bbox_targets = None

        return oaie_scores_list, pseudo_bbox_targets
    
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
