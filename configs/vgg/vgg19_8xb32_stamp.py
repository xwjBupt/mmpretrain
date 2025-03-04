_base_ = [
    '../_base_/models/vgg19bn.py',
    '../_base_/datasets/stamp_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py',
]

model = dict(
    backbone=dict(
        type='VGG', depth=19, norm_cfg=dict(type='BN'), num_classes=9),
        head=dict(
                topk=(1, 3),
    )
)
# schedule settings
optim_wrapper = dict(optimizer=dict(lr=0.01))

visualizer=dict(
        type='Visualizer',
        vis_backends=[
            dict(
                type='WandbVisBackend',
                init_kwargs=dict(project='STAMP')
            ),
        ],
    )
