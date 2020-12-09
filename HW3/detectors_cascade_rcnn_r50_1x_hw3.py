_base_ = ['detectors_cascade_rcnn_r50_1x_coco.py']

model = dict(
    roi_head=dict(
        type='CascadeRoIHead',
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                num_classes=20),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=20),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=20),
        ],
        mask_head=dict(
            type='FCNMaskHead',
            num_classes=20))
    )

dataset_type = 'COCODataset'
classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

data = dict(
    # samples_per_gpu=1,
    # workers_per_gpu=1,
    train=dict(
        # type=dataset_type,
        ann_file='/media/intel/G/Hung/VRDL/HW3/dataset/pascal_train.json',
        img_prefix='/media/intel/G/Hung/VRDL/HW3/dataset/train_images/',
        classes=classes,
        ),
    val=dict(
        # type=dataset_type,
        ann_file='/media/intel/G/Hung/VRDL/HW3/dataset/pascal_train.json',
        img_prefix='/media/intel/G/Hung/VRDL/HW3/dataset/train_images/',
        classes=classes),
    test=dict(
        # type=dataset_type,
        ann_file='/media/intel/G/Hung/VRDL/HW3/dataset/test.json',
        img_prefix='/media/intel/G/Hung/VRDL/HW3/dataset/test_images/',
        classes=classes))
