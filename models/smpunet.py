import segmentation_models_pytorch as smp


class SmpUnet(smp.Unet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        """Sequentially pass `x` through model`s encoder, decoder and heads"""

        features = self.encoder(x)

        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        return masks#, features[1], features[2], features[3], features[4], features[5]
