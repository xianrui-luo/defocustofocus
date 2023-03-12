#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from base_model import BaseModel
from blocks import FeatureFusionBlock, Interpolate, _make_encoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class MidasNet(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, use_pretrained=False, features=256):

        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        # print("Loading weights:", path)

        super(MidasNet, self).__init__()

        # use_pretrained = False if path else True


        self.pretrained, self.scratch = _make_encoder(features, use_pretrained)
        
        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)



        self.scratch.output_conv = nn.Sequential(
            # nn.Conv2d(384, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            )

        self.scratch.output_classify=nn.Sequential(

            nn.Conv2d(32, 6, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Softmax(dim=1)
        )
        # self.scratch.output_classify=nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)


        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)



        path_4 = self.scratch.refinenet4(layer_4_rn)

        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)

        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)

        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # _,_,h,w=path_1.shape
        # id1,id2,id3,id4=self.spp(path_1)
        # id1=F.interpolate(id1,size=(h,w),mode='bilinear',align_corners=False)
        # id2=F.interpolate(id2,size=(h,w),mode='bilinear',align_corners=False)
        # id3=F.interpolate(id3,size=(h,w),mode='bilinear',align_corners=False)
        # id4=F.interpolate(id4,size=(h,w),mode='bilinear',align_corners=False)
        # tmp=torch.cat((path_1,id1,id2,id3,id4),1)

        # out = self.scratch.output_conv(tmp)
        out=self.scratch.output_conv(path_1)
        out=self.scratch.output_classify(out)

    
        # print(torch.max(residual))
        
        # out = out / torch.sum(out, dim=1, keepdim=True)
        # out = torch.exp(1 * out) / torch.sum(torch.exp(1 * out), dim=1, keepdim=True)
        # out[out > 0.5] = 1000
        # out = out / torch.sum(out, dim=1, keepdim=True)

        return out


class MidasNet_v1(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, use_pretrained=False, features=256):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        # print("Loading weights:", path)

        super(MidasNet_v1, self).__init__()

        # use_pretrained = False if path else True


        self.pretrained, self.scratch = _make_encoder(features, use_pretrained)
        
        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)



        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),

            nn.ReLU(True),
            nn.Conv2d(32,1,kernel_size=1,stride=1,padding=0),
            )


    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)


        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)



        path_4 = self.scratch.refinenet4(layer_4_rn)

        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)

        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)

        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        out=self.scratch.output_conv(path_1)


        return out



   

class blendnet2(BaseModel):

    def __init__(self):
        super(blendnet2, self).__init__()
        self.conv1=nn.Sequential(
            # nn.Conv2d(6,64,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(2,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
        )

        self.conv2=nn.Sequential(
            nn.Conv2d(64,16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(16,3,kernel_size=3,stride=1,padding=1)
            nn.Conv2d(16,1,kernel_size=3,stride=1,padding=1)
        )
    

    
 

    def forward(self,img,defocus,thres):
        n,c,h,w=img.shape
        defocus=F.interpolate(defocus,size=(h,w),mode='bilinear',align_corners=False)

        beta=defocus.flatten().reshape(defocus.shape[0], -1).min(dim=1)
        alpha=defocus.flatten().reshape(defocus.shape[0], -1).max(dim=1)
        
        beta=torch.amin(defocus,(1,2,3))
        alpha=torch.amax(defocus,(1,2,3))
        temp=(alpha-beta)*thres+beta
        temp1=thres*torch.ones_like(alpha)
        alpha=torch.where(beta>=thres,temp,temp1)
        alpha=alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        alpha=alpha.expand(n,1,h,w)
        
        a=torch.randn((n,1,h,w))
        ones=torch.ones_like(a).to(device)
        zeros=torch.zeros_like(a).to(device)
        bi_mask=torch.where(defocus<alpha,ones,zeros)

        Gray = 0.29900*img[:,0,:,:] + 0.58700*img[:,1,:,:] + 0.11400*img[:,2,:,:]
        Gray=Gray.unsqueeze(1)

        inversedefocus=1-defocus
        input_img=torch.cat((inversedefocus,Gray),1)

        tmp=self.conv1(input_img)

        softmask=self.conv2(tmp)

        return bi_mask,softmask



