'''
File: converter.py
Project: jetson-converter
File Created: 2023-03-08 13:42:30
Author: sangminlee
-----
This script ...
Reference
...
'''
import os
import time
from torch2trt.dataset import Dataset
import h5py
import torch
import cv2
import copy
from torch2trt import torch2trt
from torch2trt import TRTModule
import tensorrt as trt
from collections import OrderedDict


class CalibDataset(Dataset):
    def __init__(self, is_fp16: bool = False, norm_01: bool = False, img_size: int = 256):
        h = h5py.File('test.h5')
        self.img = h['images']
        self.is_fp16 = is_fp16
        self.norm_01 = norm_01
        self.img_size = img_size
        self.trt_model = None

    def __len__(self):
        return len(self.img)

    def __getitem__(self, item):
        crop_img = self.img[item, :512, :512]
        resized_img = cv2.resize(crop_img, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
        if self.norm_01:
            normed = torch.from_numpy((resized_img / 255. - 0.5) * 2.)
        else:
            normed = torch.from_numpy(resized_img / 255.)
        if self.is_fp16:
            return normed.unsqueeze(0).unsqueeze(0).cuda().float().half()
        else:
            return normed.unsqueeze(0).unsqueeze(0).cuda().float()


class Converter(object):
    def __init__(self, model: torch.nn.Module, source_ckpt: str, target_trt: str = None, img_size: int = 256,
                 is_fp16: bool = False, norm_01: bool = False):
        assert not is_fp16
        if target_trt is None:
            target_trt = source_ckpt.replace('.ckpt', '.trt')  # .replace('.pth', '.trt')

        self.model = model.eval().float().cuda()
        self.source_ckpt = source_ckpt
        self.target_trt = target_trt
        self.img_size = img_size
        self.is_fp16 = is_fp16
        self.norm_01 = norm_01
        ''' convert 시 초음파 영상을 input으로 넣어서 하면 더 잘될까 싶었지만, 딱히 그러지 않음. '''
        ''' int8로 만들게 되면 dataset 활용 하는게 무조건 좋음'''
        self.ds = CalibDataset(norm_01=self.norm_01, is_fp16=self.is_fp16, img_size=img_size)

        ckpt = torch.load(self.source_ckpt, map_location='cuda:0')

        if 'state_dict' in ckpt.keys():
            ''' lightning을 통해 학습하면서 만들어진 model의 ckpt는 state_dict라는 field에 모델 변수 저장 '''
            ckpt = torch.load(self.source_ckpt, map_location='cuda:0')['state_dict']
            state_dict = OrderedDict()
            for key in ckpt:
                ''' 그리고 unet이라는 모듈을 삽입하는 식으로 만들어졌기 때문에 key에서 unet.을 빼서 load 해줘야 key matching 됨 '''
                state_dict[key.replace('unet.', '')] = ckpt[key]
            self.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(ckpt)

        print('Torch model loaded from %s' % self.source_ckpt)
        if self.is_fp16:
            self.model = self.model.half()

    def convert(self, input_shape: list):
        now = time.time()
        ''' 값 다른 문제 해결할 때 추가해봤던 코드로 영향 없을 것 같음. '''
        model_to_save = copy.deepcopy(self.model).eval().float().cuda()
        with torch.no_grad():
            input_data = torch.ones(input_shape).cuda().float()
            if self.is_fp16:
                input_data = input_data.half()
            model_trt = torch2trt(model_to_save, [input_data],
                                  fp16_mode=self.is_fp16,
                                  use_onnx=True,  # 필수
                                  strict_type_constraints=True)
            with open(self.target_trt, 'wb') as f:
                f.write(model_trt.engine.serialize())
                f.close()
        print('TRT model saved at %s' % self.target_trt)
        print('Converted successfully. Time consumption: %.3f' % (time.time() - now))

    def test(self, input_bind_len: int = 1, output_bind_len: int = 1):
        torch.set_printoptions(precision=21)

        with torch.no_grad():
            self.trt_model = TRTModule()
            with open(self.target_trt, 'rb') as f:
                binary_eingine = f.read()
                f.close()

            eng = {'engine': binary_eingine,
                   'input_names': ['input_%d' % i for i in range(input_bind_len)],
                   'output_names': ['output_%d' % i for i in range(input_bind_len)]}

            self.trt_model.load_state_dict(eng)
            self.trt_model = self.trt_model.eval().cuda().float()
            print('TRT model loaded from %s' % self.target_trt)

            input_data = self.ds[0].clone().float().cuda()
            if self.is_fp16:
                input_data = input_data.half()

            trt_out = self.trt_model(input_data)
            torch_out = self.model(input_data)
            for i in range(output_bind_len):
                print('Tested on ultrasound data. idx of output is %d :' % i,
                      torch.max(torch.abs(torch_out[i] - trt_out[i])).item())
