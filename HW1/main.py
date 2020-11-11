import os
import datetime
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

from loadmodels import Block
from config import load_data_transforms
from dataloader import CarDataset
from trainmodels import train

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def parse():
    parser = argparse.ArgumentParser(description='hyper parameters')
    parser.add_argument('--root', dest='root', default='dataset', type=str,
                        help='path of dataset root')
    parser.add_argument('--model', dest='model_dir', default='.', type=str,
                        help='path of model directory')
    parser.add_argument('--epoch', dest='epoch_num', default=100, type=int,
                        help='total epoch number')
    parser.add_argument('--start_epoch', dest='start_epoch', default=1, type=int,
                        help='starting epoch number')
    parser.add_argument('--batch', dest='train_batch', default=39, type=int,
                        help='training batch size')
    parser.add_argument('--tb', dest='test_batch', default=1, type=int,
                        help='testing batch size')
    parser.add_argument('--vb', dest='val_batch', default=512, type=int,
                        help='validation batch size')
    parser.add_argument('--lr', dest='base_lr', default=0.001, type=float,
                        help='base learning rate')
    parser.add_argument('--decay_step', dest='decay_step', default=50, type=int,
                        help='lr decay steps for lr_scheduler')
    parser.add_argument('--cls_lr_ratio', dest='cls_lr_ratio', default=10.0, type=float,
                        help='lr for cls_layer = base_lr * cls_lr_ratio')
    parser.add_argument('--numcls', dest='numcls', default=196, type=int,
                        help='number of output classes')
    parser.add_argument('--backbone', dest='backbone', default='resnet50', type=str,
                        help='backbone of convolutional neural network architecture')
    parser.add_argument('--resize', dest='resize_reso', default=512, type=int,
                        help='resize resolution')
    parser.add_argument('--crop', dest='crop_reso', default=448, type=int,
                        help='crop resolution')
    parser.add_argument('--describe', dest='describe', default='Cars_settings', type=str,
                        help='description of this training model')

    args = parser.parse_args()
    return args

def set_cuda_device(net, if_grad=True):
    for param in net.parameters():
        param.requires_grad = if_grad

    net = net.to(torch.device('cuda:0'))
    net = nn.DataParallel(net, device_ids=[0, 1, 2])
    return net

def weight_init(m):
    nn.init.kaiming_normal_(m.weight.data)

def print_args(args, file_name):
    print('args:')
    #print(args)
    f = open(file_name, 'w')
    args_str = str(args)
    start_idx = args_str.find('(')+1
    end_idx = args_str.find(')')

    while True:
        comma_idx = args_str.find(',', start_idx)
        content = args_str[start_idx:end_idx] if comma_idx == -1 else args_str[start_idx:comma_idx]
        f.write(content+'\n')
        print(content)

        if comma_idx != -1:
            start_idx = comma_idx+2
        else:
            break
    f.close()
    print('')
    return

def main():
    args = parse()
    # use datetime to name the model file
    time = datetime.datetime.now()
    model_name = '{0}_{1}{2}{3}'.format(args.describe, time.month, time.day, time.hour)
    model_path = os.path.join(args.model_dir, model_name)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    print_args(args, model_path+'.txt')

    train_transforms = load_data_transforms('train', resize_reso=args.resize_reso, crop_reso=args.crop_reso)
    test_transforms = load_data_transforms('test', resize_reso=args.resize_reso, crop_reso=args.crop_reso)

    data_loader = {}

    train_set = CarDataset(args.root, 'train', train_transforms)
    class_dict = train_set.class_dict
    test_set = CarDataset(args.root, 'test', test_transforms)

    train_loader = DataLoader(train_set,\
                              batch_size=args.train_batch,\
                              shuffle=True,\
                              num_workers=8,\
                              drop_last=True,\
                              pin_memory=False)
    setattr(train_loader, 'total_item_len', len(train_set))
    setattr(train_loader, 'numcls', args.numcls)
    #setattr(train_loader, 'batch_size', args.train_batch)

    test_loader = DataLoader(test_set,\
                             batch_size=args.test_batch,\
                             shuffle=False,\
                             num_workers=8,\
                             drop_last=False,\
                             pin_memory=False)
    setattr(test_loader, 'total_item_len', len(test_set))
    setattr(test_loader, 'numcls', args.numcls)
    #setattr(test_loader, 'batch_size', args.test_batch)

    data_loader['train'] = train_loader
    data_loader['test'] = test_loader



    # set current cuda device
    torch.cuda.set_device(0)
    print('CUDA device count:', torch.cuda.device_count(), 'GPUs')
    print('Current cuda device: GPU', torch.cuda.current_device(), '\n')
    cudnn.benchmark = True



    block_main = Block()

    block_main = set_cuda_device(block_main, if_grad=True)

    base_lr = args.base_lr
    cls_lr_ratio = args.cls_lr_ratio

    # prefix '.module' stands for 'nn.DataParallel'
    cls_optimizer = optim.SGD([{'params': block_main.module.cls_net.parameters()},
                               {'params': block_main.module.fc_cls.parameters(), 'lr': base_lr*cls_lr_ratio},
                               ], lr=base_lr, momentum=0.9, weight_decay=1e-4)

    #cls_lr_scheduler = lr_scheduler.StepLR(cls_optimizer, step_size=args.decay_step, gamma=0.1)

    train(block_main,
          args.start_epoch,
          args.epoch_num,
          data_loader,
          cls_optimizer,
          model_path,
          class_dict)

if __name__ == '__main__':
    main()
