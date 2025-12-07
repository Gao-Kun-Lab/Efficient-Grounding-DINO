# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
from mmengine.evaluator import BaseMetric

from mmdet.registry import METRICS

def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)

def xywhc2xyxy(bbox):
    '''
    :param bbox: left top corner point with wh : (x,y,w,h)
    :return: left top point and right down point : (x, y, x, y)
    '''
    x1, y1 = bbox[..., 0], bbox[..., 1]
    x2, y2 = bbox[..., 0] + bbox[..., 2], bbox[..., 1] + bbox[..., 3]
    return torch.stack((x1, y1, x2, y2), dim=-1)


@METRICS.register_module()
class RSVGMetric_topk(BaseMetric):
    """Referring Expression Metric."""

    def __init__(self, metric: Sequence = ('cIoU', 'mIoU', 'Pr@0.5', 'Pr@0.6', 'Pr@0.7', 'Pr@0.8', 'Pr@0.9'), **kwargs):
        super().__init__(**kwargs)
        assert set(metric).issubset(['cIoU', 'mIoU', 'Pr@0.5', 'Pr@0.6', 'Pr@0.7', 'Pr@0.8', 'Pr@0.9']), \
            f'Only support cIoU, mIoU, Pr@0.5, Pr@0.6, Pr@0.7, Pr@0.8, Pr@0.9, but got {metric}'
        assert len(metric) > 0, 'metrics should not be empty'
        self.metrics = metric

    def compute_iou(self, pred_bbox: torch.Tensor,
                    gt_bbox: torch.Tensor) -> tuple:

        area1 = (pred_bbox[..., 2] - pred_bbox[..., 0]) * (
                pred_bbox[..., 3] - pred_bbox[..., 1])
        area2 = (gt_bbox[..., 2] - gt_bbox[..., 0]) * (
                gt_bbox[..., 3] - gt_bbox[..., 1])

        lt = torch.max(pred_bbox[..., :, None, :2],
                       gt_bbox[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(pred_bbox[..., :, None, 2:],
                       gt_bbox[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        union = area1[..., None] + area2[..., None, :] - overlap
        return overlap, union

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_label = data_sample['pred_instances']['bboxes']
            pred_score = data_sample['pred_instances']['scores']
            pred_label = pred_label[pred_score.sort(descending=True)[1]]
            # pred_label = pred_label[pred_score.sort(descending=True)[1][0]][None]
            # label = data_sample['gt_masks'].to_tensor(
            #     pred_label.dtype, pred_label.device).bool()
            label = data_sample['gt_instances']['bboxes']
            # calculate iou
            overlap, union = self.compute_iou(pred_label, label)

            bs = len(pred_label)
            # bs = len(pred_label)
            gt_num = len(label)
            Pr2, Pr3, Pr4, Pr5, Pr6, Pr7, Pr8, Pr9 = 0, 0, 0, 0, 0, 0, 0, 0
            Pr5_2, Pr5_3, Pr5_4, Pr5_5, Pr5_6, Pr5_7, Pr5_8, Pr5_9 = 0, 0, 0, 0, 0, 0, 0, 0

            TP, FP, FN = 0, 0, 0
            redundancy, bg = 0, 0
            if overlap.shape[0] == 0:
                FN += gt_num
                return self.results.append((overlap.sum(), union.sum(), torch.tensor(0, device=label.device), gt_num,
                                            Pr2, Pr3, Pr4, Pr5, Pr6, Pr7, Pr8, Pr9,
                                            Pr5_2, Pr5_3, Pr5_4, Pr5_5, Pr5_6, Pr5_7, Pr5_8, Pr5_9,
                                            gt_num, TP, FP, FN, redundancy, bg))
            iou = overlap.reshape(bs, -1).sum(-1) * 1.0 / union.reshape(
                bs, -1).sum(-1)
            iou = torch.nan_to_num_(iou, nan=0.0)
            overlap = overlap[iou.sort(descending=True)[1]][0]
            union = union[iou.sort(descending=True)[1]][0]

            if iou.shape[0] > 0:
                iou_max = iou[0].max()
                # iou_max = iou[pred_score.sort(descending=True)[1]][0]
                # Pr@0.2
                if iou_max > 0.2:
                    Pr2 += 1
                # Pr@0.3
                if iou_max > 0.3:
                    Pr3 += 1
                # Pr@0.4
                if iou_max > 0.4:
                    Pr4 += 1
                # Pr@0.5
                if iou_max > 0.5:
                    Pr5 += 1
                # Pr@0.6
                if iou_max > 0.6:
                    Pr6 += 1
                # Pr@0.7
                if iou_max > 0.7:
                    Pr7 += 1
                # Pr@0.8
                if iou_max > 0.8:
                    Pr8 += 1
                # Pr@0.9
                if iou_max > 0.9:
                    Pr9 += 1
                iou5 = iou[:5].max()
                # Pr5@0.2
                if iou5 > 0.2:
                    Pr5_2 += 1
                # Pr5@0.3
                if iou5 > 0.3:
                    Pr5_3 += 1
                # Pr5@0.4
                if iou5 > 0.4:
                    Pr5_4 += 1
                # Pr5@0.5
                if iou5 > 0.5:
                    Pr5_5 += 1
                # Pr5@0.6
                if iou5 > 0.6:
                    Pr5_6 += 1
                # Pr5@0.7
                if iou5 > 0.7:
                    Pr5_7 += 1
                # Pr5@0.8
                if iou5 > 0.8:
                    Pr5_8 += 1
                # Pr5@0.9
                if iou5 > 0.9:
                    Pr5_9 += 1
                # TP, NP
                if iou_max > 0.5:
                    TP += 1
                else:
                    FN += 1
                if iou.shape[0] > 1:
                    iou_FN = iou[1:]
                    FP += iou_FN.shape[0]
                    redundancy += iou_FN[iou_FN > 0.5].shape[0]
                    bg += iou_FN[iou_FN <= 0.5].shape[0]

            self.results.append((overlap.sum(), union.sum(), iou.sum(), gt_num, Pr2, Pr3, Pr4,
                                 Pr5, Pr6, Pr7, Pr8, Pr9, Pr5_2, Pr5_3, Pr5_4, Pr5_5, Pr5_6, Pr5_7, Pr5_8, Pr5_9,
                                 gt_num, TP, FP, FN, redundancy, bg))

    def compute_metrics(self, results: list) -> dict:
        results = tuple(zip(*results))
        # assert len(results) == 10
        cum_i = sum(results[0])
        cum_u = sum(results[1])
        iou = sum(results[2])
        seg_total = sum(results[3])

        cum_Pr2 = sum(results[4])
        cum_Pr3 = sum(results[5])
        cum_Pr4 = sum(results[6])
        cum_Pr5 = sum(results[7])
        cum_Pr6 = sum(results[8])
        cum_Pr7 = sum(results[9])
        cum_Pr8 = sum(results[10])
        cum_Pr9 = sum(results[11])
        cum_Pr5_2 = sum(results[12])
        cum_Pr5_3 = sum(results[13])
        cum_Pr5_4 = sum(results[14])
        cum_Pr5_5 = sum(results[15])
        cum_Pr5_6 = sum(results[16])
        cum_Pr5_7 = sum(results[17])
        cum_Pr5_8 = sum(results[18])
        cum_Pr5_9 = sum(results[19])
        cum_gt_total = sum(results[20])
        TP = sum(results[21])
        FP = sum(results[22])
        FN = sum(results[23])
        redundancy = sum(results[24])
        bg = sum(results[25])
        total = TP + FP + FN

        metrics = {}
        # if 'cIoU' in self.metrics:
        #     metrics['cIoU'] = cum_i * 100 / cum_u
        # if 'mIoU' in self.metrics:
        #     metrics['mIoU'] = iou * 100 / seg_total
        # if 'Pr@0.2' in self.metrics:
        metrics['Pr1@0.2'] = cum_Pr2 * 100 / cum_gt_total
        metrics['Pr5@0.2'] = cum_Pr5_2 * 100 / cum_gt_total
        # if 'Pr@0.3' in self.metrics:
        metrics['Pr1@0.3'] = cum_Pr3 * 100 / cum_gt_total
        metrics['Pr5@0.3'] = cum_Pr5_3 * 100 / cum_gt_total
        # if 'Pr@0.4' in self.metrics:
        metrics['Pr1@0.4'] = cum_Pr4 * 100 / cum_gt_total
        metrics['Pr5@0.4'] = cum_Pr5_4 * 100 / cum_gt_total
        if 'Pr@0.5' in self.metrics:
            metrics['Pr1@0.5'] = cum_Pr5 * 100 / cum_gt_total
            metrics['Pr5@0.5'] = cum_Pr5_5 * 100 / cum_gt_total
        if 'Pr@0.6' in self.metrics:
            metrics['Pr1@0.6'] = cum_Pr6 * 100 / cum_gt_total
            metrics['Pr5@0.6'] = cum_Pr5_6 * 100 / cum_gt_total
        if 'Pr@0.7' in self.metrics:
            metrics['Pr1@0.7'] = cum_Pr7 * 100 / cum_gt_total
            metrics['Pr5@0.7'] = cum_Pr5_7 * 100 / cum_gt_total
        if 'Pr@0.8' in self.metrics:
            metrics['Pr1@0.8'] = cum_Pr8 * 100 / cum_gt_total
            metrics['Pr5@0.8'] = cum_Pr5_8 * 100 / cum_gt_total
        if 'Pr@0.9' in self.metrics:
            metrics['Pr1@0.9'] = cum_Pr9 * 100 / cum_gt_total
            metrics['Pr5@0.9'] = cum_Pr5_9 * 100 / cum_gt_total

        metrics['TP1@0.5'] = TP
        metrics['FP1@0.5'] = FP
        metrics['FN1@0.5'] = FN
        # metrics['redundancy@0.5'] = redundancy
        metrics['bg1@0.5'] = bg
        metrics['total1@0.5'] = total
        # metrics['cum_gt_total'] = cum_gt_total

        return metrics
