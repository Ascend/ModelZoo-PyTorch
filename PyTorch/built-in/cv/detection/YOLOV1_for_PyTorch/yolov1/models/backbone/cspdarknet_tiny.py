"""
    This is a CSPDarkNet-53 with LaekyReLU.
"""
import os
import torch
import torch.nn as nn


__all__ = ['cspdarkner53']


class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, act=True):
        super(Conv, self).__init__()
        if act:
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
                nn.BatchNorm2d(c2),
                nn.LeakyReLU(0.1, inplace=True)
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
                nn.BatchNorm2d(c2)
            )

    def forward(self, x):
        return self.convs(x)


class ResidualBlock(nn.Module):
    """
    basic residual block for CSP-Darknet
    """
    def __init__(self, in_ch):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv(in_ch, in_ch, k=1)
        self.conv2 = Conv(in_ch, in_ch, k=3, p=1, act=False)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        h = self.conv2(self.conv1(x))
        out = self.act(x + h)

        return out


class CSPStage(nn.Module):
    def __init__(self, c1, n=1):
        super(CSPStage, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = Conv(c1, c_, k=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(in_ch=c_) for _ in range(n)])
        self.cv3 = Conv(2 * c_, c1, k=1)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.res_blocks(self.cv2(x))

        return self.cv3(torch.cat([y1, y2], dim=1))


# CSPDarkNet-Tiny
class CSPDarknetTiny(nn.Module):
    """
    CSPDarknet_Tiny.
    """
    def __init__(self):
        super(CSPDarknetTiny, self).__init__()
            
        self.layer_1 = nn.Sequential(
            Conv(3, 16, k=3, p=1),      
            Conv(16, 32, k=3, p=1, s=2),
            CSPStage(c1=32, n=1)                       # p1/2
        )
        self.layer_2 = nn.Sequential(   
            Conv(32, 64, k=3, p=1, s=2),             
            CSPStage(c1=64, n=1)                      # P2/4
        )
        self.layer_3 = nn.Sequential(
            Conv(64, 128, k=3, p=1, s=2),             
            CSPStage(c1=128, n=1)                      # P3/8
        )
        self.layer_4 = nn.Sequential(
            Conv(128, 256, k=3, p=1, s=2),             
            CSPStage(c1=256, n=1)                      # P4/16
        )
        self.layer_5 = nn.Sequential(
            Conv(256, 512, k=3, p=1, s=2),             
            CSPStage(c1=512, n=1)                     # P5/32
        )


    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        return c3, c4, c5


def cspdarknet_tiny(pretrained=False, **kwargs):
    """Constructs a CSPDarknet53 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = CSPDarknetTiny()
    if pretrained:
        print('Loading the pretrained model ...')
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint = torch.load(path_to_dir + '/weights/cspdarknet_tiny/cspdarknet_tiny.pth', map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
    return model


if __name__ == '__main__':
    import time
    net = cspdarknet_tiny(pretrained=True)
    x = torch.randn(1, 3, 224, 224)
    t0 = time.time()
    y = net(x)
    t1 = time.time()
    print('Time: ', t1 - t0)