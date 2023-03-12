#!/user/bin/env python3
# -*- coding: utf-8 -*-
import torch

def to_blur_single(image, defocus, blur_kernel, alpha=100):
    mask_stack=[]
    out_stack=[]
    kernel_num = len(blur_kernel)
    interval = 1 / (kernel_num - 1)
    res_s = torch.zeros_like(image)
    mask_s = torch.zeros_like(defocus)

    for i in range(kernel_num-1, -1, -1):
        d = i * interval
        # alpha=5
        mask = 1/2 + 1/2 * torch.tanh(alpha * (interval- torch.abs(defocus - d)))
        if i > 0:
            mask_b = blur_kernel[i](mask)
            res_b = blur_kernel[i](mask * image)
            out_stack.append(res_b)
            mask_stack.append(mask)
        else:
            mask_b = mask
            res_b = mask * image
            mask_stack.append(mask)
            out_stack.append(res_b)
        mask_s = mask_s * (1 - mask_b) + mask_b
        res_s = res_s * (1 - mask_b) + res_b

    res = res_s / (mask_s + 1e-10)

    # return res,mask_stack,out_stack
    return res
