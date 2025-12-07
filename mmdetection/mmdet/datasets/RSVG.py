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


@DATASETS.register_module()
class RSVGDataset(BaseDataset):
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
        if isinstance(self.split_file, list):
            for splits in self.split_file:
                self.splits = open(splits, 'r')
                self.instances = sorted([os.path.join(self.ann_file, file_name) for file_name in os.listdir(self.ann_file)])
                img_prefix = self.data_prefix['img_path']

                count = 0
                Index = [int(index.strip('\n')) for index in self.splits.readlines()]

                classes = self.METAINFO['classes']
                for anno_path in self.instances:
                    root = ET.parse(anno_path).getroot()
                    size = root.find('size')
                    width = int(size.find('width').text)
                    height = int(size.find('height').text)
                    for member in root.findall('object'):
                        if count in Index:
                            imageFile = img_prefix + root.find("./filename").text
                            class_name = member.find('name').text.replace('-', ' ')
                            bbox_label = 0
                            # bbox_label = classes.index(class_name)

                            img_id = root.find("./filename").text.split('.')[0]
                            box = np.array([int(member[2][0].text), int(member[2][1].text), int(member[2][2].text),
                                            int(member[2][3].text)], dtype=np.float32)
                            bbox_size = (box[2] - box[0]) * (box[3] - box[1])
                            if bbox_size < 32:
                                print(imageFile)
                            text = member[3].text
                            instances = []
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
                        count += 1
        else:
            self.splits = open(self.split_file, 'r')
            self.instances = sorted([os.path.join(self.ann_file, file_name) for file_name in os.listdir(self.ann_file)])
            img_prefix = self.data_prefix['img_path']

            count = 0
            Index = [int(index.strip('\n')) for index in self.splits.readlines()]


            classes = self.METAINFO['classes']
            for anno_path in self.instances:
                root = ET.parse(anno_path).getroot()
                size = root.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)
                for member in root.findall('object'):
                    if count in Index:
                        imageFile = img_prefix + root.find("./filename").text
                        class_name = member.find('name').text.replace('-', ' ')
                        bbox_label = 0
                        # bbox_label = classes.index(class_name)

                        img_id = root.find("./filename").text.split('.')[0]
                        box = np.array([int(member[2][0].text), int(member[2][1].text), int(member[2][2].text), int(member[2][3].text)],dtype=np.float32)
                        text = member[3].text
                        instances = []
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
                    count += 1

        if len(data_list) == 0:
            raise ValueError(f'No sample in split "{self.split}".')

        return data_list
