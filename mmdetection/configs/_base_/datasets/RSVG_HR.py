# dataset settings
dataset_type = 'RSVG_HRDataset'
data_root = '/data1/detection_data/rsvg/RSVG-HR/'

backend_args = None

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='LoadAnnotations',
        with_mask=False,
        with_bbox=True,
        with_seg=False,
        # with_label=True,
    ),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'gt_masks', 'text'))
]


train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', prob=0.),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file='',
                    split_file='rsvg_hr_train.txt',
                    split='trainval',
                    data_prefix=dict(img_path='images/'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    pipeline=train_pipeline,)
        ))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/'),
        ann_file='',
        split_file='rsvg_hr_test.txt',
        split='test',
        pipeline=test_pipeline))

# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(img_path='JPEGImages/'),
#         ann_file='Annotations',
#         split_file='val.txt',
#         split='val',
#         pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/'),
        ann_file='',
        split_file='rsvg_hr_test.txt',
        split='test',
        pipeline=test_pipeline))

val_evaluator = [dict(type='RSVGMetric_top1', metric=['Pr@0.5', 'Pr@0.6', 'Pr@0.7', 'Pr@0.8', 'Pr@0.9', 'meanIoU', 'cumIoU'])]
# val_evaluator = [dict(type='RSVGMetric', metric=['cIoU', 'mIoU', 'Pr@0.5', 'Pr@0.6', 'Pr@0.7', 'Pr@0.8', 'Pr@0.9']),
#                  dict(type='RSVGMetric_topk', metric=['cIoU', 'mIoU', 'Pr@0.5', 'Pr@0.6', 'Pr@0.7', 'Pr@0.8', 'Pr@0.9']),
#                  dict(type='VOCMetric', metric='mAP', eval_mode='11points')
#                  ]
test_evaluator = val_evaluator
