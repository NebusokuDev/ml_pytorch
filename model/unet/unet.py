from copy import deepcopy
from typing import Optional

from torch import Tensor
from torch.nn import Sigmoid, Module
from torch.nn.functional import interpolate

from model.unet.bottleneck import BottleNeck
from model.unet.decoder import Decoder
from model.unet.encoder import Encoder


class UNet(Module):
    def __init__(self,
                 encoder1: Optional[Module] = None,
                 encoder2: Optional[Module] = None,
                 encoder3: Optional[Module] = None,
                 encoder4: Optional[Module] = None,
                 decoder1: Optional[Module] = None,
                 decoder2: Optional[Module] = None,
                 decoder3: Optional[Module] = None,
                 decoder4: Optional[Module] = None,
                 bottleneck: Optional[Module] = None,
                 output: Optional[Module] = None,
                 ):
        super().__init__()

        self.encoder1 = encoder1 or Encoder(in_channels=3, out_channels=64)
        self.encoder2 = encoder2 or Encoder(in_channels=64, out_channels=128)
        self.encoder3 = encoder3 or Encoder(in_channels=128, out_channels=256)
        self.encoder4 = encoder4 or Encoder(in_channels=256, out_channels=512)

        self.bottleneck = bottleneck or BottleNeck(in_channels=512, out_channels=512)

        self.decoder1 = decoder1 or Decoder(in_channels=512, out_channels=256)
        self.decoder2 = decoder2 or Decoder(in_channels=256, out_channels=128)
        self.decoder3 = decoder3 or Decoder(in_channels=128, out_channels=64)
        self.decoder4 = decoder4 or Decoder(in_channels=64, out_channels=3)  # 出力チャンネル数を1に設定
        self.output = output or Sigmoid()

    class UNet(Module):
        # 省略（__init__は変わらず）

        def forward(self, input_value: Tensor):
            # 各エンコーダを適用
            enc1 = self.encoder1(input_value)
            enc2 = self.encoder2(enc1)
            enc3 = self.encoder3(enc2)
            enc4 = self.encoder4(enc3)

            # ボトルネックを適用
            bottleneck_output = self.bottleneck(enc4)

            # スキップ接続を使用して各デコーダを適用
            # enc3のサイズに合わせてdec1を補間
            dec1 = self.decoder1(bottleneck_output)
            dec1_resized = interpolate(dec1, size=enc3.shape[2:], mode="bilinear", align_corners=True)
            dec2 = self.decoder2(dec1_resized + enc3)  # スキップ接続

            # enc2のサイズに合わせてdec2を補間
            dec2_resized = interpolate(dec2, size=enc2.shape[2:], mode="bilinear", align_corners=True)
            dec3 = self.decoder3(dec2_resized + enc2)  # スキップ接続

            # enc1のサイズに合わせてdec3を補間
            dec3_resized = interpolate(dec3, size=enc1.shape[2:], mode="bilinear", align_corners=True)
            dec4 = self.decoder4(dec3_resized + enc1)  # スキップ接続

            if self.output is None:
                return dec4

            return self.output(dec4)  # 出力をSigmoidで活性化

    def copy_with(self,
                  encoder1: Optional[Module] = None,
                  encoder2: Optional[Module] = None,
                  encoder3: Optional[Module] = None,
                  encoder4: Optional[Module] = None,
                  decoder1: Optional[Module] = None,
                  decoder2: Optional[Module] = None,
                  decoder3: Optional[Module] = None,
                  decoder4: Optional[Module] = None,
                  bottleneck: Optional[Module] = None,
                  output: Optional[Module] = None, ) -> "UNet":
        encoder1 = encoder1 or deepcopy(self.encoder1)
        encoder2 = encoder2 or deepcopy(self.encoder2)
        encoder3 = encoder3 or deepcopy(self.encoder3)
        encoder4 = encoder4 or deepcopy(self.encoder4)
        decoder1 = decoder1 or deepcopy(self.decoder1)
        decoder2 = decoder2 or deepcopy(self.decoder2)
        decoder3 = decoder3 or deepcopy(self.decoder3)
        decoder4 = decoder4 or deepcopy(self.decoder4)
        bottleneck = bottleneck or deepcopy(self.bottleneck)
        output = output or deepcopy(self.output)

        return UNet(encoder1, encoder2, encoder3, encoder4, decoder1, decoder2, decoder3, decoder4, bottleneck, output)
