'''
File: needle_multitasker.py
Project: jetson-converter
File Created: 2023-03-30 19:54:53
Author: sangminlee
-----
This script ...
Reference
...
'''
'''
File: multi_tasker.py
Project: needle_calibration
File Created: 2023-03-08 15:37:51
Author: sangminlee
-----
This script ...
Reference
...
'''
import segmentation_models_pytorch as smp
import torch


class CustomUnet(smp.Unet):
    def __init__(self, **kwargs):
        super(CustomUnet, self).__init__(**kwargs)

    def forward(self, x):
        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks  # , features[-1]


class MultiTasker(torch.nn.Module):
    def __init__(self, in_channels: int = 3):
        super(MultiTasker, self).__init__()
        self.unet = smp.Unet(in_channels=in_channels, encoder_name='timm-efficientnet-b3', classes=2,
                             encoder_weights='noisy-student', decoder_channels=[256, 128, 64, 32, 16]
                             )
        self.gap = torch.nn.AvgPool2d(8)
        self.fc1 = torch.nn.Linear(384, 384)
        self.fc2 = torch.nn.Linear(384, 7 * 2)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(*features)

        masks = self.unet.segmentation_head(decoder_output)

        f = self.relu(self.fc1(self.gap(features[-1]).squeeze(2).squeeze(2)))
        coord = self.sigmoid(self.fc2(f)).reshape([-1, 7, 2])
        return masks, coord
