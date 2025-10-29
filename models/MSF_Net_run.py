import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils
import numpy as np
import pandas as pd
from collections import OrderedDict
import torch.nn.functional as F
   

class FeatExtractor1(nn.Module):
    def __init__(self, batchNorm=False, c_in=3):
        super(FeatExtractor1, self).__init__()
        self.conv1 = model_utils.conv(batchNorm, c_in, 64,  k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 64,   128, k=3, stride=1, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 128,  96, k=3, stride=2, pad=1)
        self.conv4 = model_utils.conv(batchNorm, 96,  96, k=3, stride=1, pad=1)
      
        
    def forward(self, x):
        out=self.conv1(x)
        out=self.conv2(out)
        out=self.conv3(out)
        out=self.conv4(out)
        n, c, h, w = out.shape
        out = out.view(-1)
        return out, [n, c, h, w]
    
class FeatExtractor2(nn.Module):
    def __init__(self, batchNorm=False):
        super(FeatExtractor2, self).__init__()
        self.conv1 = model_utils.conv(batchNorm, 192, 192, k=3, stride=1, pad=1)
        self.conv2 = model_utils.deconv(192, 128)
        self.conv3 = model_utils.conv(batchNorm, 128,  96, k=3, stride=2, pad=1)
       
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        n, c, h, w = out.shape
        out   = out.view(-1)
        return out, [n, c, h, w]

class FeatExtractor3(nn.Module):
    def __init__(self, batchNorm=False):
        super(FeatExtractor3, self).__init__()
        self.conv1 = model_utils.conv(batchNorm, 192, 192, k=3, stride=1, pad=1)
        self.conv2 = model_utils.deconv(192, 128)
        self.conv3 = model_utils.conv(batchNorm, 128, 96, k=3, stride=2, pad=1)
       
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        n, c, h, w = out.shape
        out   = out.view(-1)
        return out, [n, c, h, w]


class MSF_Net(nn.Module):
    def __init__(self, fuse_type='max', batchNorm=False, c_in=3, other={}):
        super(MSF_Net, self).__init__()
        self.extractor1 = FeatExtractor1(batchNorm, c_in)
        self.extractor2 = FeatExtractor2(batchNorm)
        self.extractor3 = FeatExtractor3(batchNorm)
        self.regressor = Regressor(batchNorm)
        self.fusion = Fusion(96,96)
        self.c_in      = c_in
        self.fuse_type = fuse_type

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        img   = x[0]
        img_split = torch.split(img, 3, 1)
        if len(x) > 1: # Have lighting
            light = x[1]
            light_split = torch.split(light, 3, 1)

        feat_fused = torch.Tensor()
        for i in range(len(img_split)):
            net_in = img_split[i] if len(x) == 1 else torch.cat([img_split[i], light_split[i]], 1)
            # b,c,h/2,w/2
            feat, shape = self.extractor1(net_in)
            if i == 0:
                feat_fused = feat
            else:
                feat_fused, _ = torch.stack([feat_fused, feat], 1).max(1)
        feat_fused=feat_fused.view(shape[0], shape[1], shape[2], shape[3])

        feat_fused2 = torch.Tensor()
        for i in range(len(img_split)):
            net_in = img_split[i] if len(x) == 1 else torch.cat([img_split[i], light_split[i]], 1)
            feat, shape = self.extractor1(net_in)
            feat = feat.view(shape)
            featt = self.fusion(feat,feat_fused)
            featt = torch.cat((feat,featt),1)
            feat2, shape2 = self.extractor2(featt)
            if i == 0:
                feat_fused2 = feat2
            else:
                feat_fused2, _ = torch.stack([feat_fused2, feat2], 1).max(1)
        feat_fused2=feat_fused2.view(shape[0], shape[1], shape[2], shape[3])

        feat_fused3 = torch.Tensor()
        for i in range(len(img_split)):
            net_in = img_split[i] if len(x) == 1 else torch.cat([img_split[i], light_split[i]], 1)
            feat, shape = self.extractor1(net_in)
            feat = feat.view(shape)
            featt = self.fusion(feat,feat_fused)
            featt=torch.cat((feat,featt),1)
            feat2, shape2 = self.extractor2(featt)
            feat2 = feat2.view(shape2)
            featt = self.fusion(feat2,feat_fused2)
            featt = torch.cat((feat2,featt),1)
            feat3, shape3 = self.extractor3(featt)
            if i == 0:
                feat_fused3 = feat3
            else:
                feat_fused3, _ = torch.stack([feat_fused3, feat3], 1).max(1)
        feat_fused3=feat_fused3.view(shape[0], shape[1], shape[2], shape[3])

        normal = self.regressor(feat_fused3, shape2)
        
        return normal


class Regressor(nn.Module):
    def __init__(self, batchNorm=False): 
        super(Regressor, self).__init__()
        self.deconv1 = model_utils.conv(batchNorm, 96, 128,  k=3, stride=1, pad=1)
        self.deconv2 = model_utils.deconv(128, 64)
        self.deconv3 = model_utils.conv(batchNorm, 64,32,  k=3, stride=1, pad=1)
        self.est_normal=self._make_output(32, 3, k=3, stride=1, pad=1)


    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
               nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x, shape):
        x      = x.view(shape[0], shape[1], shape[2], shape[3])
        out    = self.deconv1(x)
        out    = self.deconv2(out)
        out    = self.deconv3(out)
        normal = self.est_normal(out)
        normal = torch.nn.functional.normalize(normal, 2, 1)
        return normal



class Fusion(nn.Module):
    def __init__(self, in_channels, mid_channels,  with_channel=True, BatchNorm=nn.BatchNorm2d):
        super(Fusion, self).__init__()
        self.with_channel = with_channel
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,kernel_size=1, bias=False),
            BatchNorm(mid_channels) 
            )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,kernel_size=1, bias=False),
            BatchNorm(mid_channels) 
            )

        self.gelu=nn.GELU()

    def forward(self, x, y):
        y = self.gelu(y)
        x = self.gelu(x)

        y_q = self.f_y(y)
        x_k = self.f_x(x)

        sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
        x = (1 - sim_map) * x + sim_map * y

        return x