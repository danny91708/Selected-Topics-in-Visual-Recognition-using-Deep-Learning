import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from config import load_config

pretrained_models = {'resnet50' : 'pretrained_model/resnet50-19c8e357.pth',
                     'resnet101': 'pretrained_model/resnet101-5d3b4d8f.pth',}

class ClsNet(nn.Module):
    def __init__(self, config, verbose=False):
        super(ClsNet, self).__init__()
        self.num_classes = config['numcls']
        self.backbone_arch = config['backbone']
        print('backbone architecture:', self.backbone_arch)

        if self.backbone_arch in dir(models):
            self.model = getattr(models, self.backbone_arch)()
            if self.backbone_arch in pretrained_models:
                self.model.load_state_dict(torch.load(pretrained_models[self.backbone_arch]))
                print('load pretrained model', self.backbone_arch, 'successfully')
            if verbose:
                print(self.model)

        if self.backbone_arch == 'resnet50' or self.backbone_arch == 'resnet18':
            self.model = nn.Sequential(*list(self.model.children())[:-2])

    def forward(self, x):
        cls_fmaps = self.model(x)

        return cls_fmaps


class CBAM(nn.Module):
    def __init__(self, in_channel, reduction_ratio=16, dilation=1):
        super(CBAM, self).__init__()
        self.hid_channel = in_channel // reduction_ratio
        self.dilation = dilation

        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.globalMaxPool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP.
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=self.hid_channel),
            nn.ReLU(),
            nn.Linear(in_features=self.hid_channel, out_features=in_channel)
        )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, dilation=self.dilation, bias=False)

    def forward(self, x):
        ''' Channel attention '''
        avgOut = self.globalAvgPool(x)
        avgOut = avgOut.view(avgOut.size(0), -1)
        avgOut = self.mlp(avgOut)

        maxOut = self.globalMaxPool(x)
        maxOut = maxOut.view(maxOut.size(0), -1)
        maxOut = self.mlp(maxOut)
        # sigmoid(MLP(AvgPool(F)) + MLP(MaxPool(F)))
        Mc = self.sigmoid(avgOut + maxOut)
        Mc = Mc.view(Mc.size(0), Mc.size(1), 1, 1)
        Mf1 = Mc * x

        ''' Spatial attention. '''
        # sigmoid(conv7x7( [AvgPool(F); MaxPool(F)]))
        maxOut = torch.max(Mf1, 1)[0].unsqueeze(1)
        avgOut = torch.mean(Mf1, 1).unsqueeze(1)
        Ms = torch.cat((maxOut, avgOut), dim=1)

        Ms = self.conv1(Ms)
        Ms = self.sigmoid(Ms)
        Ms = Ms.view(Ms.size(0), 1, Ms.size(2), Ms.size(3))
        Mf2 = Ms * Mf1
        return Mf2


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()

        self.cls_config = load_config(numcls=196, backbone='resnet50')
        self.cls_net = ClsNet(self.cls_config)
        
        

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        # resnet50 output channel: 2048
        # resnet18 output channel: 512

        self.CBAM = CBAM(2048)
        self.condense = nn.Sequential(
                                    nn.Conv2d(2048, 512, kernel_size=(4, 4), stride=(4, 4), padding=0, bias=False),
                                    nn.BatchNorm2d(512, eps=1e-05),
                                    nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 2), padding=0, bias=False),
                                    nn.BatchNorm2d(512, eps=1e-05),
                                    nn.AdaptiveAvgPool2d(output_size=1),
                                    nn.ReLU(inplace=True),
                                )

        self.fc_cls = nn.Linear(2048, 196, bias=False)
        self.fc_cls.apply(self.weight_init)

        self.fc_CBAM = nn.Linear(512, 196, bias=False)
        self.fc_CBAM.apply(self.weight_init)

    def forward(self, x):
        cls_fmaps = self.cls_net(x)
        # cls_fmaps[batch, 2048, 14, 14] (resnet50)

        cls_vec = self.avgpool(cls_fmaps)
        cls_vec = cls_vec.view(cls_vec.size(0), -1)
        out = self.fc_cls(cls_vec)

        cbam_fmaps = self.CBAM(cls_fmaps)
        # cbam_fmaps[batch, 2048, 14, 14] (resnet50)

        att_vec = self.condense(cbam_fmaps)
        att_vec = att_vec.view(att_vec.size(0), -1)
        out_CBAM = self.fc_CBAM(att_vec)

        return out, out_CBAM



    def weight_init(self, m):
        nn.init.kaiming_normal_(m.weight.data)
