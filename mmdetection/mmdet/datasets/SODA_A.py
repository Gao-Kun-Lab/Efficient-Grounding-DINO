from .coco import CocoDataset
import numpy as np
from mmdet.registry import DATASETS


@DATASETS.register_module()
class SODAADataset(CocoDataset):
    METAINFO = {
        'classes':
        ('airplane', 'helicopter', 'small-vehicle', 'large-vehicle',
               'ship', 'container', 'storage-tank', 'swimming-pool',
               'windmill'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192)]
    }