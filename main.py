'''
File: main.py
Project: jetson-converter
File Created: 2023-01-25 13:27:49
Author: sangminlee
-----
This script ...
Reference
...
'''
import torch
from torch2trt import torch2trt
from models.memoryunet import MemoryUnet
from models.smpunet import SmpUnet
import time
import h5py
from trt_infer_test import TensorRtInfer
from torch2trt.dataset import Dataset
from collections import OrderedDict


class CalibDataset(Dataset):
    def __init__(self, is_fp16: bool = False, norm_01: bool = False):
        h = h5py.File('test.h5')
        self.img = h['images']
        self.is_fp16 = is_fp16
        self.norm_01 = norm01

    def __len__(self):
        return len(self.img)

    def __getitem__(self, item):
        if self.norm_01:
            normed = torch.from_numpy((self.img[item, :512, :512] / 255. - 0.5) * 2.)
        else:
            normed = torch.from_numpy(self.img[item, :512, :512] / 255.)
        if self.is_fp16:
            return normed.unsqueeze(0).cuda().float().half()
        else:
            return normed.unsqueeze(0).cuda().float()


def convert_to_trt_lstm(ckpt_path: str, target_path: str, img_size: int = 512, is_fp16: bool = False,
                        norm_01: bool = False):
    assert target_path.endswith('.trt')
    assert ckpt_path.endswith('.ckpt')
    now = time.time()
    with torch.no_grad():
        model = MemoryUnet(img_size=img_size,
                           in_channels=1,
                           classes=2,
                           encoder_weights='noisy-student',
                           encoder_name='timm-efficientnet-b3',
                           decoder_channels=[256, 128, 64, 32, 16]).eval().cuda().float()
        weight = torch.load(ckpt_path, map_location='cuda:0')['state_dict']
        weight_new = OrderedDict()
        for key in weight.keys():
            if 'unet' in key:
                key_new = key.replace('unet.', '')
                weight_new[key_new] = weight[key]

        model.load_state_dict(weight_new)
        input = torch.zeros([1, 1, 1, img_size, img_size]).cuda().float().half()
        h1 = torch.zeros([1, 16, img_size, img_size]).cuda().float().half()
        h2 = torch.zeros([1, 16, img_size, img_size]).cuda().float().half()
        c1 = torch.zeros([1, 16, img_size, img_size]).cuda().float().half()
        c2 = torch.zeros([1, 16, img_size, img_size]).cuda().float().half()
        if is_fp16:
            model = model.half()
            input = input.half()
            h1 = h1.half()
            h2 = h2.half()
            c1 = c1.half()
            c2 = c2.half()

        model_trt = torch2trt(model, [input, h1, h2, c1, c2], fp16_mode=is_fp16)
        with open(target_path, 'wb') as f:
            f.write(model_trt.engine.serialize())
            f.close()

    print('Converted successfully. Time consumption: %.3f' % (time.time() - now))


def convert_to_trt(ckpt_path: str, target_path: str, img_size: int = 512, is_fp16: bool = False, norm_01: bool = False):
    assert target_path.endswith('.trt')
    assert ckpt_path.endswith('.ckpt')
    now = time.time()
    ds = CalibDataset(norm_01=norm_01)
    with torch.no_grad():
        model = SmpUnet(in_channels=1,
                        classes=2,
                        encoder_weights='noisy-student',
                        encoder_name='timm-efficientnet-b3',
                        decoder_channels=[256, 128, 64, 32, 16]).eval().cuda()

        weight = torch.load(ckpt_path, map_location='cuda:0')['state_dict']
        weight_new = OrderedDict()
        for key in weight.keys():
            if 'unet' in key:
                key_new = key.replace('unet.', '')
                weight_new[key_new] = weight[key]

        model.load_state_dict(weight_new)
        if is_fp16:
            model = model.half()
            input = ds[0].unsqueeze(0).half()

        model_trt = torch2trt(model, [input], fp16_mode=is_fp16)
        with open(target_path, 'wb') as f:
            f.write(model_trt.engine.serialize())
            f.close()

    print('Converted successfully. Time consumption: %.3f' % (time.time() - now))


def main():
    source_ckpt = 'model.ckpt'
    target_path = 'model.trt'
    is_fp16 = False
    img_size = 512
    norm_01 = False
    convert_to_trt(source_ckpt, target_path, img_size, is_fp16, norm_01)

    tester = TensorRtInfer(None, './model_benchmark/model_inference/fp16_feature3.trt', 'cuda', torch.float16)
    tester(dtype='fp16')


if __name__ == '__main__':
    main()
