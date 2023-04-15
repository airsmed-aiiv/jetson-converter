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
from models.test_model import TestModel
from models.smpunet import SmpUnet
from models.needle_multitasker import MultiTasker
from converter import Converter


def main():
    ''' 두희님게서 알려주신 torch에서 결과값이 달라지는 것을 해결해주는 Flag '''
    torch.backends.cudnn.benchmark = True

    model_type = 'multitasker'
    source_ckpt = 'top_view_model.ckpt'
    target_path = 'top_view_model_0407_b1.trt'
    img_size = 256
    is_fp16 = False
    norm_01 = False  # debug 용으로 크게 중요하지는 않음.

    if model_type == 'smpunet':
        model = SmpUnet(in_channels=11, encoder_name='timm-efficientnet-b3', classes=2,
                        encoder_weights='noisy-student', decoder_channels=[256, 128, 64, 32, 16])
    elif model_type == 'multitasker':
        model = MultiTasker(5)
    elif model_type == 'test':
        model = TestModel()
        torch.save(model.state_dict(), source_ckpt)
    else:
        raise NotImplementedError

    converter = Converter(model, source_ckpt, target_path, img_size=img_size, norm_01=norm_01, is_fp16=is_fp16)
    converter.convert(input_shape=[1, 5, 256, 256])
    converter.test(input_bind_len=1, output_bind_len=2)


if __name__ == '__main__':
    main()
