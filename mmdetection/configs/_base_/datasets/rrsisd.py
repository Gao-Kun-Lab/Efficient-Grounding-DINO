# dataset settings
dataset_type = 'rrsisdDataset'
data_root = '/data1/detection_data/rrsisd/'

backend_args = None

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(
        type='LoadAnnotations',
        with_mask=False,
        with_bbox=True,
        with_seg=False,
        # with_label=False
    ),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'gt_masks', 'text'))
]


train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/rrsisd/JPEGImages/'),
        ann_file='rrsisd/instances.json',
        split_file='rrsisd/refs(unc).p',
        split=['train', 'val'],
        text_mode='select_first',
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/rrsisd/JPEGImages/'),
        ann_file='rrsisd/instances.json',
        split_file='rrsisd/refs(unc).p',
        split=['test'],  # or 'testB'
        text_mode='select_first',
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/rrsisd/JPEGImages/'),
        ann_file='rrsisd/instances.json',
        split_file='rrsisd/refs(unc).p',
        split=['test'],  # or 'testB'
        text_mode='select_first',
        pipeline=test_pipeline))

val_evaluator = [dict(type='RSVGMetric', metric=['cIoU', 'mIoU', 'Pr@0.5', 'Pr@0.6', 'Pr@0.7', 'Pr@0.8', 'Pr@0.9']),
                 dict(type='RSVGMetric_topk', metric=['cIoU', 'mIoU', 'Pr@0.5', 'Pr@0.6', 'Pr@0.7', 'Pr@0.8', 'Pr@0.9']),
                 dict(type='VOCMetric', metric='mAP', eval_mode='11points')]
test_evaluator = val_evaluator
