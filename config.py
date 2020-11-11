import os
import sys
import torch
import torchvision.transforms as transforms

# config dict for specific dataset
def load_config(numcls=196, backbone='resnet50'):
    config = {'numcls': numcls,
              'backbone': backbone,}

    return config

# data transforms
def load_data_transforms(mode, resize_reso=512, crop_reso=448):

    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    if mode == 'train':
        data_transforms = transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.RandomCrop((crop_reso, crop_reso)),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.126, saturation=0.5),
            transforms.ToTensor(),
            normalize,
        ])
    elif mode == 'test':
        data_transforms = transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.CenterCrop((crop_reso, crop_reso)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        sys.exit(0)

    return data_transforms
