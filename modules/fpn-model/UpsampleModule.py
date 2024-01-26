from torch import nn
from SemibasicBlock import SemiBasicBlock

class UpsampleModule(nn.Module):
    def __init__(self, in_chs, decoder_chs, norm_layer):
        super(UpsampleModule, self).__init__()

        self.down_conv = nn.Conv2d(in_chs, decoder_chs, kernel_size=1, bias=False)
        self.down_bn = norm_layer(decoder_chs)
        downsample = nn.Sequential(
            self.down_conv,
            self.down_bn,
        )
        self.conv_enc = SemiBasicBlock(in_chs, decoder_chs, downsample=downsample, norm_layer=norm_layer)
        self.conv_out = SemiBasicBlock(decoder_chs, decoder_chs, norm_layer=norm_layer)
        self.conv_up = nn.ConvTranspose2d(decoder_chs, decoder_chs, kernel_size=2, stride=2, bias=False)

    def forward(self, enc, prev):
        enc = self.conv_up(prev) + self.conv_enc(enc)
        dec = self.conv_out(enc)
        return dec