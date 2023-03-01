'''
File: trt_infer_test.py
Project: jetson-converter
File Created: 2023-01-25 13:32:20
Author: sangminlee
-----
This script ...
Reference
...
'''
from torch2trt import TRTModule
import torch
import h5py
import numpy as np
from PIL import Image
from models.smpunet import SmpUnet
from collections import OrderedDict


class TensorRtInfer(object):
    def __init__(self, model_path: str, torch_model_ckpt: str, device: str = 'cuda', is_fp16: bool = False,
                 norm_01: bool = False):
        self.is_fp16 = is_fp16
        self.device = device
        self.norm_01 = norm_01
        self.model = TRTModule()
        self.torch_model_ckpt = torch_model_ckpt

        with open(model_path, 'rb') as f:
            binary_eingine = f.read()
            f.close()
        eng = {'engine': binary_eingine,
               'input_names': ['input_0'],
               'output_names': ['output_0']}
        self.model.load_state_dict(eng)
        self.model = self.model.eval().to(device)
        if is_fp16:
            self.model = self.model.half()

        self.torch_model = self.load_torch_model()

    def load_torch_model(self):
        torch_model = SmpUnet(in_channels=1,
                              classes=2,
                              encoder_weights='noisy-student',
                              encoder_name='timm-efficientnet-b3',
                              decoder_channels=[256, 128, 64, 32, 16]).eval().cuda().float()
        weight = torch.load(self.torch_model_ckpt, map_location='cuda:0')['state_dict']
        weight_new = OrderedDict()
        for key in weight.keys():
            if 'unet' in key:
                key_new = key.replace('unet.', '')
                weight_new[key_new] = weight[key]

        torch_model.load_state_dict(weight_new)
        return torch_model

    def load_test_set(self):
        h = h5py.File('test.h5', 'r')
        if self.norm_01:
            total_imgs = torch.from_numpy((np.array(h['images'])[:, :512, :] / 255. - 0.5) * 2.).float().cuda()
        else:
            total_imgs = torch.from_numpy(np.array(h['images'])[:, :512, :] / 255.).float().cuda()
        if self.is_fp16:
            return total_imgs.half()
        else:
            return total_imgs

    def __call__(self, *args, **kwargs):
        pred_list = []
        torch_pred_list = []
        total_imgs = self.load_test_set().contiguous()
        with torch.no_grad():
            for i in range(249):
                crt_img = total_imgs[i].unsqueeze(0).unsqueeze(0)
                pred, _, _, _, _, _ = self.model(crt_img)
                pred_torch, _, _, _, _, _ = self.torch_model(crt_img)
                pred_list.append(pred.detach().cpu().numpy())
                torch_pred_list.append(pred_torch.detach().cpu().numpy())

                img = Image.fromarray(np.uint8(
                    torch.where(torch.softmax(pred[0].float().detach().cpu(), dim=0)[1] > 0.5, 255, 0).numpy()))
                torch_img = Image.fromarray(np.uint8(
                    torch.where(torch.softmax(pred_torch[0].float().detach().cpu(), dim=0)[1] > 0.5, 255, 0).numpy()))
                out_img = Image.new('RGB', (1024, 512))
                out_img.paste(img, (0, 0))
                out_img.paste(torch_img, (512, 0))
                out_img.save('./%s_%d.png' % (kwargs['dtype'], i))
        np.save('./%s.npy' % kwargs['dtype'], np.array(pred_list))
