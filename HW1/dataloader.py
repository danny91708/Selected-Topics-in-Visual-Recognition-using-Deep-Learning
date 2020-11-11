import os
import sys
import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import csv
import glob

from config import load_data_transforms

class CarDataset(Dataset):
    def __init__(self, root, mode, transforms):
        super(CarDataset, self).__init__()
        self.root = root
        self.mode = mode
        self.transforms = transforms

        self.class_dict = {} # {class name: index number, ...} 
        self.image_label_list = self.prepare_data(self.root, self.mode)
        
        '''
        # print
        for idx, (name, label) in enumerate(self.image_label_list):
            print('index: {0}, {1}, label: {2}'.format(idx, name, label))
        '''
    def __len__(self):
        return len(self.image_label_list)

    # index range from [0, __len__()-1]
    def __getitem__(self, index):
        (anchor_path, anchor_label) = self.image_label_list[index]

        # convert gray-scale image to RGB image
        anchor = Image.open(anchor_path).convert('RGB')

        anchor = self.transforms(anchor)

        if self.mode == 'train':
            sample = {'anchor': anchor,
                      'label': anchor_label,
                      'path': anchor_path,
                     }
        # Test data don't have labels
        elif self.mode == 'test':
            sample = {'anchor': anchor,
                      'path': anchor_path,
                     }

        return sample

    def prepare_data(self, root, mode):
        # turn the class names into numbers and store into class_dict
        image_label = []
        
        if mode == 'train':
            with open(os.path.join(root, 'training_labels.csv'), 'r') as f:
                label_idx = 0
                for idx, line in enumerate(f.readlines()):
                    if idx == 0: continue # idx = 0 is header

                    line = line.strip('\n')
                    img_num, name = line.split(',', 1)

                    if not name in self.class_dict: 
                        self.class_dict[name] = label_idx
                        label_idx += 1

                    image_name = img_num
                    image_path = os.path.join(root, 'training_data/training_data/' + image_name + '.jpg')
                    label = self.class_dict[name]
                    image_label.append((image_path, label))
                    # print(img_num, name, self.class_dict[name])
                    # print(image_path, label)
                    # print()
            print("> There are", len(self.class_dict), "classes...")

        # No label
        if mode == 'test': 
            folder_path = os.path.join(root, 'testing_data/testing_data/')
            image_path = glob.glob(os.path.join(folder_path, '*.jpg'))
            for img in image_path:
                image_label.append((img, None))

        
        print('> Found {0} {1} images...'.format(len(image_label), mode))


        
        return image_label

    def get_class_dict():
        return self.class_dict

'''
if __name__ == '__main__':
    data_transforms = load_data_transforms('train')
    train_set = CarDataset('dataset', 'train', data_transforms)
    test_set = CarDataset('dataset', 'test', data_transforms)

    batch_size = 4
    # test for DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    
    class_dict = train_set.class_dict
    print(len(class_dict))

    # unit test
    for sample in test_loader:
        image = sample['anchor']
        # label = sample['label']

        print(image.size())
        # print(label.size())
        #print(label)

        print('image type:', image.type())
        # print('label type:', label.type())
        break

    # # test for DataLoader, collate_fn
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_4train)
    # # unit test
    # for sample in train_loader:
    #     # sample is the return of collate_fn
    #     print('\n\n\n')
    #     print(len(sample))
    #     print(sample[batch_size-1])
    #     break
'''