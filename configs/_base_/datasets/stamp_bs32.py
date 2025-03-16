# dataset settings
dataset_type = 'STAMP'
data_preprocessor = dict(
    num_classes=9,
    mean=[234.30045203, 225.71031344, 235.95306603],
    std=[8.5827689, 35.25244448,38.0740568 ],
    to_rgb=True,
)
fold = '2'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root='/home/wjx/data/dataset/STAMP/processed',
        split='train',
        fold = fold,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root='/home/wjx/data/dataset/STAMP/processed',
        split='test',
        fold = fold,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = val_dataloader

# calculate precision_recall_f1 and mAP
val_evaluator = dict(type='Accuracy', topk=(1, 3))
test_dataloader = val_dataloader
test_evaluator = val_evaluator
