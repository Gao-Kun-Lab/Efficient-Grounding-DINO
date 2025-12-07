# Copyright (c) senior. All rights reserved.
import collections
import os
import os.path as osp
import random
from typing import Dict, List
import xml.etree.ElementTree as ET
import mmengine
from mmengine.dataset import BaseDataset
import numpy as np
from mmdet.registry import DATASETS
import cv2


@DATASETS.register_module()
class RSVG_HRDataset(BaseDataset):
    """Visual Grounding in Remote Sensing dataset.

    The `Refcoco` and `Refcoco+` dataset is based on
    `ReferItGame: Referring to Objects in Photographs of Natural Scenes
    <http://tamaraberg.com/papers/referit.pdf>`_.

    The `Refcocog` dataset is based on
    `Generation and Comprehension of Unambiguous Object Descriptions
    <https://arxiv.org/abs/1511.02283>`_.

    Args:
        ann_file (str): Annotation file path.
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (str): Prefix for training data.
        split_file (str): Split file path.
        split (str): Split name. Defaults to 'train'.
        text_mode (str): Text mode. Defaults to 'random'.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """
    METAINFO = {
        'classes':
            ('Expressway-Service-area', 'Expressway-toll-station', 'airplane', 'airport', 'baseballfield',
             'basketballcourt', 'bridge', 'chimney', 'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass',
             'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
            [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
             (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
             (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
             (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
             (0, 82, 0), (120, 166, 157)]
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 split_file: str,
                 data_prefix: Dict,
                 split: str = 'train',
                 **kwargs):
        self.split_file = split_file
        self.split = split

        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            **kwargs,
        )

    def _join_prefix(self):
        if isinstance(self.split_file, list):
            assert len(self.split_file) > 0, 'there is no file in split file'
            split_file_list = []
            for file in self.split_file:
                if not mmengine.is_abs(file) and file:
                    split_file_list.append(osp.join(self.data_root, file))
            self.split_file = split_file_list
        else:
            if not mmengine.is_abs(self.split_file) and self.split_file:
                self.split_file = osp.join(self.data_root, self.split_file)

        return super()._join_prefix()

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        data_list = []
        img_prefix = self.data_prefix['img_path']

        files = open(self.split_file, "r").readlines()

        for img_id, anno in enumerate(files):
            anno_list = anno.strip('\n').split(',')
            img_name = anno_list[0]
            xmin_gt, ymin_gt, xmax_gt, ymax_gt = float(anno_list[1]), float(anno_list[2]), float(anno_list[3]), float(anno_list[4])
            text = anno_list[-1]
            imageFile = osp.join(img_prefix, img_name)
            img = cv2.imread(imageFile)
            height, width, _ = img.shape
            box = np.array([xmin_gt, ymin_gt, xmax_gt, ymax_gt], dtype=np.float32)
            instances = []
            bbox_label = 0
            ins = [{
                'bbox': box,
                'bbox_label': bbox_label,
                'ignore_flag': 0
            }]
            instances.extend(ins)
            data_info = {
                'img_path': imageFile,
                'width': width,
                'height': height,
                'img_id': img_id,
                'instances': instances,
                'text': text
            }
            data_list.append(data_info)

        if len(data_list) == 0:
            raise ValueError(f'No sample in split "{self.split}".')

        return data_list
