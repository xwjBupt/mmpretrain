_base_ = [
    '../_base_/models/resnet18.py', '../_base_/datasets/stamp_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

model = dict(
    head=dict(
        num_classes=9,
        topk=(1, 3),
    ))

visualizer=dict(
        type='Visualizer',
        vis_backends=[
            dict(
                type='WandbVisBackend',
                init_kwargs=dict(project='STAMP')
            ),
        ],
    )
