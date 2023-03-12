#!/user/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from midas_net import MidasNet,BaseModel,MidasNet_v1,blendnet2
from blur_func import to_blur_single
import os




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device='cpu'


def resize_tensor(tensor, ratio, ensure_multiple_of=32):
    h, w = tensor.shape[2:]
    re_h = (np.round(h / ratio / ensure_multiple_of) * ensure_multiple_of).astype(int)
    re_w = (np.round(w / ratio / ensure_multiple_of) * ensure_multiple_of).astype(int)
    output_tensor = F.interpolate(tensor, size=(re_h, re_w), mode='bilinear', align_corners=False)
    return output_tensor
             


class SoftDiskBlur(nn.Module):
    def __init__(self, kernel_size):
        super(SoftDiskBlur, self).__init__()
        r = kernel_size // 2
        x_grid, y_grid = np.meshgrid(np.arange(-int(r), int(r)+1), np.arange(-int(r), int(r)+1))
        # kernel = (x_grid**2 + y_grid**2) <= r**2
        kernel = 0.5 + 0.5 * np.tanh(0.25 * (r**2 - x_grid**2 - y_grid**2) + 0.5)
        kernel = kernel.astype(np.float) / kernel.sum()
        kernel = torch.FloatTensor(kernel).expand(3, 1, kernel_size, kernel_size)
        self.pad = nn.ReflectionPad2d(r)  # mirror fill
        self.weight = kernel.to(device)
 
    def forward(self, x):
        out = self.pad(x)
        ch = x.shape[1]
        out = F.conv2d(out, self.weight[:ch], padding=0, groups=ch)
        return out



class RadianceModule(nn.Module):
    def __init__(self):
        super(RadianceModule, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 1)
        self.conv2 = nn.Conv2d(16, 16, 1)
        self.conv3 = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = torch.sigmoid(self.conv3(out))
        # out=self.conv3(out)
        return out




class BlendNet2(BaseModel):
    def __init__(self):
        super(BlendNet2,self).__init__()
        self.visual_names=['predicted','gt','mask','image','softmask']

        self.blend = blendnet2()



    def forward(self, pred,img,defocus,gt,thres):
        n,c,h,w=img.shape
        self.mask,self.softmask=self.blend(img,defocus,thres)
        self.gt=gt
        self.image=img
        self.predicted=F.interpolate(pred,size=(h,w),mode='bilinear',align_corners=False)

        return self.predicted,self.mask,self.softmask



class BokehNet2(BaseModel):
    def __init__(self):
        super(BokehNet2, self).__init__()
        self.visual_names=['predicted','gt','original','defocus'
        ]

        #3, 5, 7, 11, 15,19, 23, 27, 33, 39,45, 53, 61, 69
        self.blur_s=[
            None,
            SoftDiskBlur(3),
            SoftDiskBlur(5),
            SoftDiskBlur(7),
            SoftDiskBlur(11),
            SoftDiskBlur(15),
            SoftDiskBlur(19),
            SoftDiskBlur(23),
            SoftDiskBlur(27),
            SoftDiskBlur(33),
            SoftDiskBlur(39),
            SoftDiskBlur(45),
            SoftDiskBlur(53),
            SoftDiskBlur(61),
            SoftDiskBlur(69),
        ]
        


        # midas_pretrained = True if check_path is None else False
        midas_pretrained=True
        self.midas = MidasNet_v1(midas_pretrained)
        self.radiance = RadianceModule()



    def forward(self, image,gt):
        self.original=image
        self.defocus=self.midas(image)
        self.weight=self.radiance(image)



        mask=((image<0.99).sum(dim=1,keepdim=True)<3).float()
        self.weight=self.weight*(1-mask)+mask*3*(image)**5  
        radiance=image*self.weight
        self.predicted= to_blur_single(radiance, self.defocus, self.blur_s) / (to_blur_single(self.weight, self.defocus, self.blur_s)+1e-10)




        self.gt=gt

        return self.predicted,self.defocus



