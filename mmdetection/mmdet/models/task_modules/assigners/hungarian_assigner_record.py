# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import torch
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from mmdet.registry import TASK_UTILS
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
import os.path as osp
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from mmdet.structures.bbox import bbox_overlaps


@TASK_UTILS.register_module()
class HungarianAssigner_record(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of some components.
    For DETR the costs are weighted sum of classification cost, regression L1
    cost and regression iou cost. The targets don't include the no_object, so
    generally there are more predictions than targets. After the one-to-one
    matching, the un-matched are treated as backgrounds. Thus each query
    prediction will be assigned with `0` or a positive integer indicating the
    ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        match_costs (:obj:`ConfigDict` or dict or \
            List[Union[:obj:`ConfigDict`, dict]]): Match cost configs.
    """

    def __init__(
        self, match_costs: Union[List[Union[dict, ConfigDict]], dict,
                                 ConfigDict],
            record_txt=None
    ) -> None:

        if isinstance(match_costs, dict):
            match_costs = [match_costs]
        elif isinstance(match_costs, list):
            assert len(match_costs) > 0, \
                'match_costs must not be a empty list.'

        self.match_costs = [
            TASK_UTILS.build(match_cost) for match_cost in match_costs
        ]

        self.record_txt = record_txt

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               img_meta: Optional[dict] = None,
               layer_num = None,
               **kwargs) -> AssignResult:
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places. It may includes ``masks``, with shape
                (n, h, w) or (n, l).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                ``labels``, with shape (k, ) and ``masks``, with shape
                (k, h, w) or (k, l).
            img_meta (dict): Image information.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert isinstance(gt_instances.labels, Tensor)
        num_gts, num_preds = len(gt_instances), len(pred_instances)
        gt_labels = gt_instances.labels
        device = gt_labels.device

        # 1. assign -1 by default
        assigned_gt_inds = torch.full((num_preds, ),
                                      -1,
                                      dtype=torch.long,
                                      device=device)
        assigned_labels = torch.full((num_preds, ),
                                     -1,
                                     dtype=torch.long,
                                     device=device)

        if num_gts == 0 or num_preds == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels)

        # 2. compute weighted cost
        cost_list = []
        for match_cost in self.match_costs:
            cost = match_cost(
                pred_instances=pred_instances,
                gt_instances=gt_instances,
                img_meta=img_meta)
            cost_list.append(cost)
        cost = torch.stack(cost_list).sum(dim=0)

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        # --------------- only output gt_num and high_neg_query_num ----------------------
        recored_txt = True
        if recored_txt:
            iou_threshold = 0.3
            small_object = 32 * 32
            medium_object = 96 * 96
            aera = (gt_instances.bboxes[:, -1] - gt_instances.bboxes[:, 1]) * (
                    gt_instances.bboxes[:, -2] - gt_instances.bboxes[:, 0])
            overlaps = bbox_overlaps(pred_instances.bboxes, gt_instances.bboxes, mode='iou', is_aligned=False)

            ###### record decoder match iou under different object size #########
            matched_iou = overlaps[matched_row_inds, matched_col_inds]
            matched_iou_list = matched_iou.cpu().tolist()
            aera = aera[matched_col_inds]
            if self.record_txt and layer_num is not None:
                txt_out_dir = './work_dirs/RSVG_out_imshow/'
                img_name = img_meta['img_path'].split('/')[-1] + ' '
                if not os.path.exists(txt_out_dir):
                    os.mkdir(txt_out_dir)
                # out_str_small = 'small_object_iou: '
                # out_str_medium = 'medium_object_iou: '
                # out_str_large = 'large_object_iou: '
                # for i, iou in enumerate(matched_iou_list):
                #     size = aera[i]
                #     if size < small_object:
                #         out_str_small += str(round(iou, 4)) + ' '
                #     elif size > medium_object:
                #         out_str_large += str(round(iou, 4)) + ' '
                #     else:
                #         out_str_medium += str(round(iou, 4)) + ' '
                # out_str = img_name + out_str_small + out_str_medium + out_str_large + '\n'
                miss_obj_num = str((matched_iou==0).sum().item())
                all_obj_num = str((matched_iou.shape[0]))
                out_str = img_name + miss_obj_num + ' ' + all_obj_num + '\n'
                with open(osp.join(txt_out_dir, 'layer_' + str(layer_num) + '_' + self.record_txt), 'a') as f:
                    f.write(out_str)
            # ===================================================================
            # max_overlaps, argmax_overlaps = overlaps.max(dim=1)
            # # neg = max_overlaps[assigned_gt_inds == 0].clone()
            # pos_iou = max_overlaps[matched_row_inds].clone()
            # aera = aera[matched_col_inds]
            # medium_sign = torch.zeros(aera.shape, device=aera.device)
            # medium_sign[aera > medium_object] = 1
            # medium_sign[aera < small_object] = 1
            # small_iou = pos_iou[aera < small_object]
            # medium_iou = pos_iou[medium_sign == 0]
            # large_iou = pos_iou[aera > medium_object]

            bbox_preds_assign = pred_instances.bboxes[matched_row_inds]

            if matched_row_inds.shape[0] != 0 and pred_instances.bboxes.shape[0] < 5000:
                imshow = False
                output = True
                out_dir = './work_dirs/assign_results/'
                if not osp.exists(out_dir):
                    os.mkdir(out_dir)
                imshow_bbox8(img_meta['img_path'], img_meta['flip_direction'],
                             gt_instances.bboxes, bbox_preds_assign, pred_instances.bboxes,
                             None, imshow, output, out_dir)
                # if self.record_txt:
                #     txt_out_dir = './work_dirs/record_txt/'
                #     if not osp.exists(txt_out_dir):
                #         os.mkdir(txt_out_dir)
                #     if small_iou.shape[0] != 0:
                #         small_iou_mean = small_iou.mean().item()
                #     else:
                #         small_iou_mean = -1
                #     if medium_iou.shape[0] != 0:
                #         medium_iou_mean = medium_iou.mean().item()
                #     else:
                #         medium_iou_mean = -1
                #     if large_iou.shape[0] != 0:
                #         large_iou_mean = large_iou.mean().item()
                #     else:
                #         large_iou_mean = -1
                #     out_str = ''
                #     out_str += str(gt_instances.bboxes.shape[0]) + ' ' + str(small_iou.shape[0]) + \
                #                ' ' + str(medium_iou.shape[0]) + ' ' + str(large_iou.shape[0]) + ' ' + \
                #                str(small_iou[small_iou < iou_threshold].shape[0]) + ' ' + \
                #                str(medium_iou[medium_iou < iou_threshold].shape[0]) + ' ' + \
                #                str(large_iou[large_iou < iou_threshold].shape[0]) + ' ' + \
                #                str(small_iou_mean) + ' ' + str(medium_iou_mean) + ' ' + str(large_iou_mean) + '\n'
                #     with open(osp.join(txt_out_dir, self.record_txt), 'a') as f:
                #         f.write(out_str)

        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels)

def bbox528_numpy(bbox5):

    x, y, w, h, angle = bbox5[..., :1], bbox5[..., 1:2],\
                        bbox5[..., 2:3], bbox5[..., 3:4],\
                        -bbox5[..., 4:5]
    w_x, h_y = w * np.cos(angle), w * np.sin(angle)
    x_lc, x_rc, y_lc, y_rc = x - w_x / 2, x + w_x / 2, y + h_y / 2, y - h_y / 2
    x1, y1 = x_lc - h * np.sin(angle) / 2, y_lc - h * np.cos(angle) / 2
    x4, y4 = x_lc + h * np.sin(angle) / 2, y_lc + h * np.cos(angle) / 2
    x2, y2 = x_rc - h * np.sin(angle) / 2, y_rc - h * np.cos(angle) / 2
    x3, y3 = x_rc + h * np.sin(angle) / 2, y_rc + h * np.cos(angle) / 2
    return np.concatenate((x1,y1,x2,y2,x3,y3,x4,y4),axis=-1)

def imshow_bbox8(img_dir, flip_direction, gt_bbox, bbox_pred, bbox_preds_all=None, high_quality_neg=None, imshow=True, output=False, out_dir = None):
    if not imshow and not output:
        return
    elif not imshow and output:
        P = random.random()
        # if gt_bbox.shape[0] > 550:
        #     img = draw_line(img_dir, flip_direction, bbox_preds_all, gt_bbox, bbox_pred, high_quality_neg)
        #     img_name = img_dir.split('/')[-1].split('.')[0] + '_' + str(bbox_pred.shape[0]) + '_' + str(
        #         high_quality_neg.shape[0]) + '.jpg'
        #     cv2.imwrite(osp.join(out_dir, img_name), img)
        # elif P <= 0.0001:
        # if gt_bbox.shape[0] > 20 and P <= 0.001:
        if P <= 0.0003:
            img = draw_line(img_dir, flip_direction, bbox_preds_all, gt_bbox, bbox_pred, high_quality_neg)
            img_name = img_dir.split('/')[-1].split('.')[0] + '.jpg'
            cv2.imwrite(osp.join(out_dir, img_name), img)

    elif imshow and not output:
        img = draw_line(img_dir, flip_direction, bbox_preds_all, gt_bbox, bbox_pred, high_quality_neg)
        plt.imshow(img)
        plt.show()
    elif imshow and output:
        img = draw_line(img_dir, flip_direction, bbox_preds_all, gt_bbox, bbox_pred, high_quality_neg)
        plt.imshow(img)
        plt.show()
        img_name = img_dir.split('/')[-1].split('.')[0] + '.jpg'
        cv2.imwrite(osp.join(out_dir, img_name), img)


def draw_line(img_dir, flip_direction, bbox_preds_all, gt_bbox, bbox_pred, high_quality_neg):
    img = cv2.imread(img_dir)
    if flip_direction != None:
        import mmcv
        img = mmcv.imflip(img, flip_direction).astype(np.uint8).copy()
    thickness_ = 2
    if bbox_preds_all is not None:
        for line in bbox_preds_all.tolist():
            bbox8 = line
            points = [(int(bbox8[0]), int(bbox8[1])), (int(bbox8[2]), int(bbox8[1])),
                  (int(bbox8[2]), int(bbox8[3])), (int(bbox8[0]), int(bbox8[3]))]
            cv2.line(img, points[0], points[1], color=(0, 0, 100), thickness=thickness_)
            cv2.line(img, points[1], points[2], color=(0, 0, 100), thickness=thickness_)
            cv2.line(img, points[2], points[3], color=(0, 0, 100), thickness=thickness_)
            cv2.line(img, points[3], points[0], color=(0, 0, 100), thickness=thickness_)
    if high_quality_neg is not None:
        if high_quality_neg.shape[0] != 0:
            for line in high_quality_neg.tolist():
                bbox8 = line
                points = [(int(bbox8[0]), int(bbox8[1])), (int(bbox8[2]), int(bbox8[1])),
                  (int(bbox8[2]), int(bbox8[3])), (int(bbox8[0]), int(bbox8[3]))]
                cv2.line(img, points[0], points[1], color=(255, 255, 255), thickness=thickness_)
                cv2.line(img, points[1], points[2], color=(255, 255, 255), thickness=thickness_)
                cv2.line(img, points[2], points[3], color=(255, 255, 255), thickness=thickness_)
                cv2.line(img, points[3], points[0], color=(255, 255, 255), thickness=thickness_)
    for line in gt_bbox.tolist():
        bbox8 = line
        points = [(int(bbox8[0]), int(bbox8[1])), (int(bbox8[2]), int(bbox8[1])),
                  (int(bbox8[2]), int(bbox8[3])), (int(bbox8[0]), int(bbox8[3]))]
        cv2.line(img, points[0], points[1], color=(0, 255, 0), thickness=thickness_)
        cv2.line(img, points[1], points[2], color=(0, 255, 0), thickness=thickness_)
        cv2.line(img, points[2], points[3], color=(0, 255, 0), thickness=thickness_)
        cv2.line(img, points[3], points[0], color=(0, 255, 0), thickness=thickness_)
    for line in bbox_pred.tolist():
        bbox8 = line
        points = [(int(bbox8[0]), int(bbox8[1])), (int(bbox8[2]), int(bbox8[1])),
                  (int(bbox8[2]), int(bbox8[3])), (int(bbox8[0]), int(bbox8[3]))]
        cv2.line(img, points[0], points[1], color=(255, 0, 0), thickness=thickness_)
        cv2.line(img, points[1], points[2], color=(255, 0, 0), thickness=thickness_)
        cv2.line(img, points[2], points[3], color=(255, 0, 0), thickness=thickness_)
        cv2.line(img, points[3], points[0], color=(255, 0, 0), thickness=thickness_)
    return img
