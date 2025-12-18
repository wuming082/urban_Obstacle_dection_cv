backend_args = None
data_root = '/root/autodl-fs/yolo_train/YOLO_DATASET/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=10,
        max_keep_ckpts=2,
        rule='greater',
        save_best='coco/bbox_mAP_50',
        save_last=True,
        save_optimizer=False,
        save_param_scheduler=False,
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'DEBUG'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
metainfo = dict(
    classes=(
        'red',
        'green',
        'crosswalk',
        'blind',
        'pothole',
    ),
    palette=[
        (
            255,
            0,
            0,
        ),
        (
            0,
            255,
            0,
        ),
        (
            255,
            255,
            0,
        ),
        (
            0,
            0,
            255,
        ),
        (
            128,
            0,
            128,
        ),
    ])
model = dict(
    backbone=dict(
        frozen_stages=-1,
        init_cfg=None,
        out_indices=(
            1,
            2,
            4,
            6,
        ),
        type='MobileNetV2',
        widen_factor=1.0),
    bbox_head=dict(
        anchor_generator=dict(
            octave_base_scale=4,
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales_per_octave=3,
            strides=[
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        num_classes=5,
        stacked_convs=4,
        type='RetinaHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        add_extra_convs='on_output',
        in_channels=[
            24,
            32,
            96,
            320,
        ],
        num_outs=4,
        out_channels=256,
        relu_before_extra_convs=True,
        start_level=0,
        type='FPN'),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.5, type='nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            ignore_iof_thr=-1,
            min_pos_iou=0,
            neg_iou_thr=0.4,
            pos_iou_thr=0.5,
            type='MaxIoUAssigner'),
        debug=False,
        pos_weight=-1),
    type='SingleStageDetector')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.015, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=100,
        gamma=0.1,
        milestones=[
            80,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='coco_framework/common/val.json',
        backend_args=None,
        data_prefix=dict(img='images/'),
        data_root='/root/autodl-fs/yolo_train/YOLO_DATASET/',
        metainfo=dict(
            classes=(
                'red',
                'green',
                'crosswalk',
                'blind',
                'pothole',
            ),
            palette=[
                (
                    255,
                    0,
                    0,
                ),
                (
                    0,
                    255,
                    0,
                ),
                (
                    255,
                    255,
                    0,
                ),
                (
                    0,
                    0,
                    255,
                ),
                (
                    128,
                    0,
                    128,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/root/autodl-fs/yolo_train/YOLO_DATASET/coco_framework/common/val.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=500, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=3,
    dataset=dict(
        ann_file='coco_framework/common/train.json',
        backend_args=None,
        data_prefix=dict(img='images/'),
        data_root='/root/autodl-fs/yolo_train/YOLO_DATASET/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(
            classes=(
                'red',
                'green',
                'crosswalk',
                'blind',
                'pothole',
            ),
            palette=[
                (
                    255,
                    0,
                    0,
                ),
                (
                    0,
                    255,
                    0,
                ),
                (
                    255,
                    255,
                    0,
                ),
                (
                    0,
                    0,
                    255,
                ),
                (
                    128,
                    0,
                    128,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='coco_framework/common/val.json',
        backend_args=None,
        data_prefix=dict(img='images/'),
        data_root='/root/autodl-fs/yolo_train/YOLO_DATASET/',
        metainfo=dict(
            classes=(
                'red',
                'green',
                'crosswalk',
                'blind',
                'pothole',
            ),
            palette=[
                (
                    255,
                    0,
                    0,
                ),
                (
                    0,
                    255,
                    0,
                ),
                (
                    255,
                    255,
                    0,
                ),
                (
                    0,
                    0,
                    255,
                ),
                (
                    128,
                    0,
                    128,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/root/autodl-fs/yolo_train/YOLO_DATASET/coco_framework/common/val.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/root/autodl-fs/output_train/ssd_mobilenet_fpn_500e_1x8b_coco_1v/'
