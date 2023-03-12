import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import random
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from models import BokehNet2,resize_tensor,BlendNet2



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device='cpu'


def main():
      # checkpoint for defocus hallucination
    check_path ='.pth'
     # checkpoint for deep poisson fusion
    check_path1='.pth'

    #onestage
    prenet=BokehNet2()

    state_dict=torch.load(check_path)
    if 'optimizer' in state_dict:
        state_dict=state_dict['net']
    prenet.load_state_dict(state_dict)


    net=BlendNet2()
    state_dict = torch.load(check_path1)
    if 'optimizer' in state_dict:
        state_dict=state_dict['net']
    net.load_state_dict(state_dict)


    prenet.to(device)

    prenet.eval()
    net = net.to(device)
    net.eval()
    
    data_dir='/data7/RRBC2020/EBB!/TestBokehFree/'
    i=0

    

    with torch.no_grad():
        files=os.listdir(data_dir)
        files.sort(key= lambda x:int(x[:-4]))
        for file in files:

            image_path = os.path.join(data_dir,file)
            image = cv2.imread(image_path)
            h, w, _ = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            image = image.transpose((2, 0, 1))
            image = torch.from_numpy(image).unsqueeze(0).to(device)


            # start = time.time()
            h, w = image.shape[2:]
            shrink_scale = 2
            re_image = resize_tensor(image, shrink_scale, 128/shrink_scale)

            bokeh_pred, defocus= prenet.forward(re_image,None)


    
            bokeh_pred=F.interpolate(bokeh_pred,size=(h,w),mode='bilinear',align_corners=False)


            thres=1/8
            bokeh_pred,_,mask=net.forward(bokeh_pred,image,defocus,None,thres)
            mask=mask.clamp(0,1)          

            bokeh_pred=mask*image+(1-mask)*bokeh_pred
            bokeh_pred=bokeh_pred.clamp(0,1)



            bokeh_pred = bokeh_pred.squeeze().cpu().numpy().transpose(1, 2, 0) * 255.0
            bokeh_pred=cv2.cvtColor(bokeh_pred,cv2.COLOR_RGB2BGR)

            savedir='./test'
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            name1=str(i)+'.png'
            i=i+1

            cv2.imwrite(os.path.join(savedir,name1),bokeh_pred)



if __name__ == '__main__':
    main()
