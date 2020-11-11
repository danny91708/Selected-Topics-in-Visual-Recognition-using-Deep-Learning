import torch
import torch.nn as nn
import numpy as np
import random
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import csv
import os
#from sklearn.metrics import pairwise_distances

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def triplet_loss(anchor, pos, neg, margin=1.0):
    # L2-norm
    anchor = anchor / torch.norm(anchor, p=2.0, dim=1, keepdim=True)
    pos = pos / torch.norm(pos, p=2.0, dim=1, keepdim=True)
    neg = neg / torch.norm(neg, p=2.0, dim=1, keepdim=True)
    # calculate distance
    d = nn.PairwiseDistance(p=2.0, keepdim=False)
    distance = d(anchor, pos) - d(anchor, neg) + margin
    loss = torch.mean(torch.max(distance, torch.zeros_like(distance)))
    return loss

def mutual_attention_loss(cls_integral_fmaps):
    avgpool = nn.AdaptiveAvgPool2d(output_size=1)

    attention_num = len(cls_integral_fmaps)

    mutual_loss = 0
    total = 0

    for i in range(attention_num):
        for j in range(i+1, attention_num):
            # (batch_size, 2048, 1, 1)
            ith_obj_part = avgpool(cls_integral_fmaps[i])
            jth_obj_part = avgpool(cls_integral_fmaps[j])

            ith_obj_part = ith_obj_part.view(ith_obj_part.size(0), -1)
            jth_obj_part = jth_obj_part.view(jth_obj_part.size(0), -1)

            each_loss = torch.mean(ith_obj_part*jth_obj_part, dim=1)
            batch_loss = torch.mean(each_loss, dim=0)

            total += 1
            mutual_loss += batch_loss

    return (mutual_loss / total)

def hard_negative_mining(org_anchor, org_pos, label):
    # org_anchor, org_pos size: [batch_size, 512]
    hard_neg = torch.zeros_like(org_pos, device=device)
    # L2-norm
    anchor = org_anchor / torch.norm(org_anchor, p=2.0, dim=1, keepdim=True)
    pos = org_pos / torch.norm(org_pos, p=2.0, dim=1, keepdim=True)

    anchor_np = anchor.data.cpu().numpy()
    pos_np = pos.data.cpu().numpy()

    distance = pairwise_distances(anchor_np, pos_np)
    distance = torch.from_numpy(distance)
    #assert list(distance.size()) == [org_anchor.size(0), org_pos.size(0)]

    values, indices = torch.topk(distance, k=org_pos.size(0), dim=1, largest=False)

    for i, index in enumerate(indices):
        k=0
        now = index[k].item()

        while label[now].item() == label[i].item():
            k = k + 1
            now = index[k].item()
            # we don't consider the condition if labels in one batch size are all the same

        hard_neg[i] = org_pos[now]

    return hard_neg

def jigsaw_generator(images, l, n, image_size=448):

    halfsize = image_size // n
    rounds = n ** 2
    jigsaws = images.clone()

    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0:halfsize, 0:halfsize].clone()
        jigsaws[..., 0:halfsize, 0:halfsize] = jigsaws[..., x * halfsize:(x + 1) * halfsize, y * halfsize:(y + 1) * halfsize].clone()
        jigsaws[..., x * halfsize:(x + 1) * halfsize, y * halfsize:(y + 1) * halfsize] = temp

    return jigsaws

def attention_cropper(anchor, batch_att_map, padding_ratio=0.1, std_ratio=0.0):
    batch, fix_reso = anchor.size(0), anchor.size(-1)

    # batch_att_map: (batch, 2048, 16, 16)
    batch_att_map = torch.mean(batch_att_map, dim=1, keepdim=True)
    batch_att_map = F.interpolate(batch_att_map, size=(fix_reso, fix_reso), mode='bilinear', align_corners=False)
    # batch_att_map: (batch, 1, fix_reso, fix_reso)

    # anchor: (batch, 3, fix_reso, fix_reso)
    crop_images = []



    ratio = 0



    for idx in range(batch):
        # att_map: (fix_reso, fix_reso)
        att_map = batch_att_map[idx, 0].clone()

        max_value = torch.max(att_map)
        min_value = torch.min(att_map)
        # normalize torch.float32 to [0, 1]
        att_map = (att_map - min_value) / (max_value - min_value + 0.000001)
        # convert [0, 1] to [0, 255]
        att_map = att_map * 255

        mean_value = torch.mean(att_map)
        std_value = torch.std(att_map)

        crop_mask = torch.ge(att_map, mean_value + (std_ratio * std_value))

        nonzero_indices = torch.nonzero(crop_mask, as_tuple=False)

        height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * fix_reso), 0)
        height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * fix_reso), fix_reso)
        width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * fix_reso), 0)
        width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * fix_reso), fix_reso)


        # Ratio cropping
        height = height_max - height_min
        width = width_max - width_min

        if width / height > height / width:
            ratio += width / height
            # print("width ratio=", width / height)
        else:
            ratio += height / width
            # print("heigh ratio=", height / width)




        # (1, 3, fix_reso, fix_reso)
        crop_images.append(F.interpolate(anchor[idx:idx+1, :, height_min:height_max, width_min:width_max], size=(fix_reso, fix_reso), mode='bilinear', align_corners=False))


    ratio = ratio / batch
    print("ratio=", ratio)


    crop_images = torch.cat(crop_images, dim=0)

    return crop_images

def mixup_data(anchor, label, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = anchor.size(0)

    index = torch.randperm(batch_size)
    index = index.to(device)

    mixed_anchor = lam * anchor + (1 - lam) * anchor[index]
    label_a, label_b = label, label[index]
    return mixed_anchor, label_a, label_b, lam

def mixup_criterion(criterion, pred, label_a, label_b, lam):
    return lam * criterion(pred, label_a) + (1 - lam) * criterion(pred, label_b)

def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # (t - 1) is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)

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

    n = 4

    for epoch in range(start_epoch, epoch_num+1):
        block_main.train()

        for idx_batch, sample_batch in enumerate(train_loader):

            anchor, label = sample_batch['anchor'], sample_batch['label']
            #anchor_path = sample_batch['path']
            assert list(label.size()) == [train_batch]
            #assert len(anchor_path) == train_batch

            anchor = anchor.to(device)
            label = label.to(device)

            l = []
            for x in range(n):
                for y in range(n):
                    l.append([x, y])

            random.shuffle(l)

            #-----------------------------------------------org-----------------------------------------------
            cls_optimizer.zero_grad()

            # mixed_anchor, label_a, label_b, lam = mixup_data(anchor, label, alpha=1.0)
            # _, _, _, _, mixed_out = block_main(mixed_anchor)

            # return masked_fmaps_prior, masked_fmaps_secondary, out_prior, out_secondary, out
            # masked_fmaps_prior, _, out_prior, _, out = block_main(anchor)
            out, out_CBAM = block_main(anchor)

            # masked_fmaps_prior: (batch, 2048, 16, 16)
            # anchor_crop = attention_cropper(anchor, masked_fmaps_prior.detach())

            # _, _, _, out_secondary, out_crop = block_main(anchor_crop)
            ce_loss = ce_loss_func(out, label)
            ce_loss_CBAM = ce_loss_func(out_CBAM, label)
            cls_loss = ce_loss + ce_loss_CBAM
            # ce_loss_prior = ce_loss_func(out_prior, label)
            # ce_loss_crop = ce_loss_func(out_crop, label)
            # ce_loss_secondary = ce_loss_func(out_secondary, label)
            # mixup_ce_loss = mixup_criterion(ce_loss_func, mixed_out, label_a, label_b, lam)

            # cls_loss = ce_loss + ce_loss_prior + ce_loss_crop + ce_loss_secondary + mixup_ce_loss

            cls_loss.backward()
            cls_optimizer.step()

            #-----------------------------------------------jigsaw-----------------------------------------------
            # cls_optimizer.zero_grad()

            # jigsaw = jigsaw_generator(anchor, l, n=n)
            # jigsaw_crop = jigsaw_generator(anchor_crop, l, n=n)

            # _, _, jigsaw_out_prior, _, jigsaw_out = block_main(jigsaw)
            # _, _, _, jigsaw_out_secondary, jigsaw_out_crop = block_main(jigsaw_crop)


            # ce_loss_jigsaw = ce_loss_func(jigsaw_out, label)
            # ce_loss_jigsaw_crop = ce_loss_func(jigsaw_out_crop, label)

            # cls_loss_jigsaw = ce_loss_jigsaw + ce_loss_jigsaw_crop

            # cls_loss_jigsaw.backward()
            # cls_optimizer.step()


            if ((idx_batch+1) % checkpoint == 0) or ((idx_batch+1) == iteration):
                print('epoch: {0} - iteration: {1}/{2}'.format(epoch, idx_batch+1, iteration))

                print('cls_loss: {0}'.format(cls_loss.item()))
                # print('cls_loss_jigsaw: {0}'.format(cls_loss_jigsaw.item()))

                # print('acc / best_acc: {0}/{1}'.format(epoch_acc, best_acc))
                # print('att_acc / best_att_acc: {0}/{1}'.format(epoch_att_acc, best_att_acc))
                #print('cls lr scheduler: {0}'.format(cls_lr_scheduler.get_lr()))
                

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



    


    # # save models
    # torch.save(block_main.state_dict(), 'cars.pth'.format(epoch))
    # print('model saved successfully')


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

    # return (correct / total), (correct_combine_all / total)

def calLabel_write(out, class_dict, writer, img_name):
    out = out.to(torch.device('cpu'))
    _, out_label = torch.max(out, dim=1)
    out_label = list(class_dict.keys())[list(class_dict.values()).index(out_label)]
    writer.writerow([img_name, out_label])
