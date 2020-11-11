import torch
import torch.nn as nn
import numpy as np
import random
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import csv
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(block_main,
          start_epoch,
          epoch_num,
          data_loader,
          cls_optimizer,
          model_path,
          class_dict,
          checkpoint=50
          ):

    train_loader = data_loader['train']
    train_batch = train_loader.batch_size

    avgpool = nn.AdaptiveAvgPool2d(output_size=1)
    #maxpool = nn.AdaptiveMaxPool2d(output_size=1)
    ce_loss_func = nn.CrossEntropyLoss()
    d = nn.PairwiseDistance(p=2.0, keepdim=False)
    #iteration = (train_loader.total_item_len // train_batch) if (train_loader.total_item_len % train_batch) == 0 else (train_loader.total_item_len // train_batch)+1
    iteration = (train_loader.total_item_len // train_batch)

    init_lr = []
    for n in range(len(cls_optimizer.param_groups)):
        init_lr.append(cls_optimizer.param_groups[n]['lr'])

    epoch_acc = 0.0
    best_acc = 0.0

    epoch_att_acc = 0.0
    best_att_acc = 0.0

    for epoch in range(start_epoch, epoch_num+1):
        block_main.train()

        for idx_batch, sample_batch in enumerate(train_loader):

            anchor, label = sample_batch['anchor'], sample_batch['label']
            #anchor_path = sample_batch['path']
            assert list(label.size()) == [train_batch]
            #assert len(anchor_path) == train_batch

            anchor = anchor.to(device)
            label = label.to(device)

            cls_optimizer.zero_grad()
            out, out_CBAM = block_main(anchor)

            ce_loss = ce_loss_func(out, label)
            ce_loss_CBAM = ce_loss_func(out_CBAM, label)
            cls_loss = ce_loss + ce_loss_CBAM

            cls_loss.backward()
            cls_optimizer.step()

            if ((idx_batch+1) % checkpoint == 0) or ((idx_batch+1) == iteration):
                print('epoch: {0} - iteration: {1}/{2}'.format(epoch, idx_batch+1, iteration))
                print('cls_loss: {0}'.format(cls_loss.item()))


        # Save models
        torch.save(block_main.state_dict(), 'checkpoint/epoch_{0}_checkpoint.pth'.format(epoch))
        print('model saved successfully at epoch {0}'.format(epoch))

        # Create the test labels in the csv file
        test(block_main, data_loader, class_dict, epoch)


        print('cls_optimizer lr: [', end='')
        for n in range(len(cls_optimizer.param_groups)):
            print(cls_optimizer.param_groups[n]['lr'], end=' ')
        print(']')
        print('')

        #cls_lr_scheduler.step(epoch)
        for n in range(len(cls_optimizer.param_groups)):
            cls_optimizer.param_groups[n]['lr'] = cosine_anneal_schedule(epoch, epoch_num, init_lr[n])

        del(cls_loss)
        torch.cuda.empty_cache()


def test(block_main,
         data_loader,
         class_dict,
         epoch
         ):

    test_loader = data_loader['test']
    test_batch = test_loader.batch_size

    block_main.eval()

    with open('predict_csv/predict_epoch_{0}.csv'.format(epoch), 'w', newline='') as f:
        with open('predict_csv/predict_epoch_{0}_combined.csv'.format(epoch), 'w', newline='') as g:
            writer_f = csv.writer(f)
            writer_g = csv.writer(g)
            writer_f.writerow(['id', 'label'])
            writer_g.writerow(['id', 'label'])

            for idx_batch, sample_batch in enumerate(test_loader):
                anchor = sample_batch['anchor']
                anchor = anchor.to(device)
                out, out_CBAM = block_main(anchor)

                img_name = os.path.split(sample_batch['path'][0])[1][:-4]

                calLabel_write(out, class_dict, writer_f, img_name)
                calLabel_write(out + out_CBAM, class_dict, writer_g, img_name)


            del(out)
            del(out_CBAM)
            torch.cuda.empty_cache()

    print('Write the csv file successfully')


def calLabel_write(out, class_dict, writer, img_name):
    out = out.to(torch.device('cpu'))
    _, out_label = torch.max(out, dim=1)
    out_label = list(class_dict.keys())[list(class_dict.values()).index(out_label)]
    writer.writerow([img_name, out_label])
