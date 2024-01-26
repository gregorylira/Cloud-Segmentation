from torch import nn
from UpsampleModule import UpsampleModule

class FPNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(FPNHead, self).__init__()
        decoder_chs = out_channels
        layer_chs_list = [512, 256, 128, 64]  # Update the layer channels based on your input size

        self.conv_enc2dec = nn.Conv2d(layer_chs_list[0], decoder_chs, kernel_size=1, bias=False)
        self.bn_enc2dec = norm_layer(out_channels)
        self.relu_enc2dec = nn.ReLU(inplace=True)

        self.up3 = UpsampleModule(layer_chs_list[1], decoder_chs, norm_layer)
        self.up2 = UpsampleModule(layer_chs_list[2], decoder_chs, norm_layer)
        self.up1 = UpsampleModule(layer_chs_list[3], decoder_chs, norm_layer)

        self.conv_up0 = nn.ConvTranspose2d(decoder_chs, decoder_chs, kernel_size=2, stride=2, bias=False)
        self.conv_up1 = nn.ConvTranspose2d(decoder_chs, out_channels, kernel_size=2, stride=2, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, c1, c2, c3, c4):
        c4 = self.relu_enc2dec(self.bn_enc2dec(self.conv_enc2dec(c4)))
        c3 = self.up3(c3, c4)
        c2 = self.up2(c2, c3)
        c1 = self.up1(c1, c2)

        c1 = self.conv_up0(c1)
        c1 = self.conv_up1(c1)
        return c1