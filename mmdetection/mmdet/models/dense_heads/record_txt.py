import cv2
import random
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
from mmdet.structures.bbox import bbox_overlaps, bbox_cxcywh_to_xyxy


def record_iou_every_layer(encoder_bboxs, decoder_bboxs, gt_bbox, batch_img_metas, iou_mode='iou', txt_name=None):
    bs = encoder_bboxs.shape[0]
    txt_dir = './work_dirs/record_txt/'
    # factors = torch.cat(factors, 0)

    for j in range(bs):
        img_h, img_w, = batch_img_metas[j]['img_shape']
        factor = encoder_bboxs[j].new_tensor([img_w, img_h, img_w,
                                       img_h])
        # scale_factor = encoder_bboxs[j].new_tensor(batch_img_metas[j]['scale_factor']).repeat(1, 2)

        encoder_bbox = bbox_cxcywh_to_xyxy(encoder_bboxs[j]) * factor
        encoder_bbox[:, 0::2].clamp_(min=0, max=img_h)
        encoder_bbox[:, 1::2].clamp_(min=0, max=img_w)
        # imshow_bbox8(batch_img_metas[j]['img_path'], batch_img_metas[j]['flip_direction'],
        #              gt_bbox[j]['bboxes'], encoder_bbox)
        for i, decoder_bbox in enumerate(decoder_bboxs[:, j, ...]):
            decoder_bbox = bbox_cxcywh_to_xyxy(decoder_bbox) * factor
            decoder_bbox[:, 0::2].clamp_(min=0, max=img_h)
            decoder_bbox[:, 1::2].clamp_(min=0, max=img_w)

            # imshow_bbox8(batch_img_metas[j]['img_path'], batch_img_metas[j]['flip_direction'],
            #              gt_bbox[j]['bboxes'], decoder_bbox)

            overlaps = bbox_overlaps(decoder_bbox, encoder_bbox, mode=iou_mode)
            ious = overlaps.diag()
            with open(osp.join(txt_dir, txt_name + '_encoder_decoder_iou_simple_train.txt'), 'a') as f:
                out_str = ''
                out_str += batch_img_metas[j]['img_path'].split('/')[-1] + ' ' + str(i+1) + ' ' + str(round(ious.mean().item(), 4)) + ' ' + \
                           str(round(ious.max().item(), 4)) + ' ' + str(round(ious.min().item(), 4)) + '\n'
                f.write(out_str)
                f.close()

            with open(osp.join(txt_dir, txt_name + '_encoder_decoder_iou_all_train.txt'), 'a') as f:
                out_str_ = batch_img_metas[j]['img_path'].split('/')[-1] + ' ' + str(i + 1)
                for iou in ious:
                    out_str_ += str(round(iou.item(), 4)) + ' '
                out_str_ += '\n'
                f.write(out_str_)
                f.close()
            # print(1)



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
        if P <= 0.0001:
            img = draw_line(img_dir, flip_direction, bbox_preds_all, gt_bbox, bbox_pred, high_quality_neg)
            img_name = img_dir.split('/')[-1].split('.')[0] + '_' + str(bbox_pred.shape[0]) + '_' + str(high_quality_neg.shape[0]) + '.jpg'
            cv2.imwrite(osp.join(out_dir, img_name), img)

    elif imshow and not output:
        img = draw_line(img_dir, flip_direction, bbox_preds_all, gt_bbox, bbox_pred, high_quality_neg)
        plt.imshow(img)
        plt.show()
    elif imshow and output:
        img = draw_line(img_dir, flip_direction, bbox_preds_all, gt_bbox, bbox_pred, high_quality_neg)
        plt.imshow(img)
        plt.show()
        img_name = img_dir.split('/')[-1].split('.')[0] + '_' + str(bbox_pred.shape[0]) + '_' + str(high_quality_neg.shape[0]) + '.jpg'
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