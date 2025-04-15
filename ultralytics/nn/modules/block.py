# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Block modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer

from ultralytics.nn.modules.conv import Conv, DWConv, GhostConv, LightConv, RepConv, MLPBlock, CBAM, PConv
from .transformer import TransformerBlock

__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost',
           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3')


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):  #protonet
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))

class Proto_new1(nn.Module):  #protonet
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = PConv(c1, c_)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = PConv(c_, c_)
        self.cv3 = PConv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))

class Proto_new2(nn.Module):  #protonet
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = PConv(c1, c_)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = PConv(c_, c_)
        self.cv3 = PConv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))

#修改后增加_end
class Proto4_new_add_dw(nn.Module):
    # YOLOv5 mask Proto module for segmentation models
    # SOLOv2思想
    def __init__(self, c_in=(), c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv0 = DWConv(c_in[0], c_in[0], k=3)
        self.cv0_ = Conv(c_in[0], c_, k=1)
        #self.up0 = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)

        self.cv1 = DWConv(c_in[1], c_in[1], k=3)
        self.cv1_ = Conv(c_in[1], c_, k=1)
        self.up1 = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv11 = DWConv(c_, c_, k=3)
        #self.up11 = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)

        self.cv2 = DWConv(c_in[2], c_in[2], k=3)
        self.cv2_ = Conv(c_in[2], c_, k=1)
        self.up2 = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv22 = DWConv(c_, c_, k=3)
        self.up22 = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv222 = DWConv(c_, c_, k=3)
        #self.up222 = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)

        self.before = nn.Sequential(
            nn.Conv2d(c_, 4*c_, 3,1,1), nn.LeakyReLU(inplace=True))
        self.upsample = nn.PixelShuffle(2)
        self.after1 = Conv(c_, c_, k=3)
        self.after2 = Conv(c_, c2)

    def forward(self, x):
        mask0 = self.cv0_(self.cv0(x[0]))
        mask1 = self.cv11(self.up1(self.cv1_(self.cv1(x[1]))))
        mask2 = self.cv222(self.up22(self.cv22(self.up2(self.cv2_(self.cv2(x[2]))))))
        out = self.upsample(self.before(mask2+mask1+mask0))
        out = self.after2(self.after1(out))

        return out
#后修改环节
class Proto3_solo(nn.Module):
    # YOLOv5 mask Proto module for segmentation models
    # SOLOv2思想 使用add
    def __init__(self, c_in=(), c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv2 = Conv(c_in[2], c_in[1], k=3)
        self.up2 = nn.ConvTranspose2d(c_in[1], c_in[1], 2, 2, 0, bias=True)
        self.cv1 = Conv(c_in[1], c_in[0], k=3)
        self.up1 = nn.ConvTranspose2d(c_in[0], c_in[0], 2, 2, 0, bias=True)
        self.cv0 = Conv(c_in[0], c_, k=3)
        self.up0 = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.pred = Conv(c_, c2, k=1)

    def forward(self, x):
        #print("use proto_solo")
        mask2 = self.up2(self.cv2(x[2]))
        mask1 = self.up1(self.cv1(mask2+x[1]))
        mask0 = self.up0(self.cv0(mask1+x[0]))
        out = self.pred(mask0)
        return out

class Proto4_solo(nn.Module):
    # YOLOv5 mask Proto module for segmentation models
    # SOLOv2思想
    def __init__(self, c_in=(), c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv0 = Conv(c_in[0], c_, k=3)
        self.up0 = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv1 = Conv(c_in[1], c_, k=3)
        self.up1 = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv11 = Conv(c_, c_, k=3)
        self.up11 = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv2 = Conv(c_in[2], c_, k=3)
        self.up2 = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv22 = Conv(c_, c_, k=3)
        self.up22 = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv222 = Conv(c_, c_, k=3)
        self.up222 = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.pred = Conv(c_, c2, k=1)

    def forward(self, x):
        #print("use proto_solo")
        mask0 = self.up0(self.cv0(x[0]))
        mask1 = self.up11(self.cv11(self.up1(self.cv1(x[1]))))
        mask2 = self.up222(self.cv222(self.up22(self.cv22(self.up2(self.cv2(x[2]))))))
        out = self.pred(mask2+mask1+mask0)
        return out

class Proto5_solo(nn.Module):
    # YOLOv5 mask Proto module for segmentation models
    # SOLOv2思想
    def __init__(self, c_in=(), c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv0 = PConv(c_in[0], c_)
        self.up0 = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv1 = PConv(c_in[1], c_)
        self.up1 = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv11 = PConv(c_, c_)
        self.up11 = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv2 = PConv(c_in[2], c_)
        self.up2 = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv22 = PConv(c_, c_)
        self.up22 = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv222 = PConv(c_, c_)
        self.up222 = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.pred = PConv(c_, c2)

    def forward(self, x):
        #print("use proto_solo")
        mask0 = self.up0(self.cv0(x[0]))
        mask1 = self.up11(self.cv11(self.up1(self.cv1(x[1]))))
        mask2 = self.up222(self.cv222(self.up22(self.cv22(self.up2(self.cv2(x[2]))))))
        out = self.pred(mask2+mask1+mask0)
        return out


#修改后增加_end

class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):  # ch_in, ch_out, number
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))

class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2f_Res2Block(C2f):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottle2neck(self.c, self.c, shortcut) for _ in range(n))



"""
class C2f_Res2Block(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottle2neck(self.c, self.c, shortcut, g) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
"""


class C2f_attention1(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.m1 = nn.ModuleList([CBAM(self.c)]) #添加在Bottleneck之前
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y1=self.m1[0](y[-1])
        y.extend(m(y1) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2f_attention2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.m1 = nn.ModuleList([CBAM((2 + n) * self.c)])  # 添加在cv2之前
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y1 = self.m1[0](torch.cat(y, 1))
        return self.cv2(y1)

class C2f_combine(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.m1 = nn.ModuleList([EMA_attention(self.c)])  # 添加在Bottleneck之前
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        #self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        #self.m = nn.ModuleList(GhostBottleneckV2(self.c, self.c, self.c) for _ in range(n))  #38、39
        self.m = nn.ModuleList(GhostBottleneckV3(self.c, self.c, self.c) for _ in range(n)) #40、41

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y1 = self.m1[0](y[-1])
        y.extend(m(y1) for m in self.m)
        return self.cv2(torch.cat(y, 1))

# #添加在Bottleneck之前
class C2f_attention3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.m1 = nn.ModuleList([EMA_attention(self.c)])  # 添加在Bottleneck之前
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y1 = self.m1[0](y[-1])
        y.extend(m(y1) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C2f_attention4(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.m1 = nn.ModuleList([EMA_attention((2 + n) * self.c)])  # 添加在最后一个卷积之前
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y1 = self.m1[0](torch.cat(y, 1))
        return self.cv2(y1)


class C2f_Res2Block2(C2f_attention3):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottle2neck(self.c, self.c, shortcut) for _ in range(n))


class C2f_Res2Block3(C2f_attention4):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottle2neck(self.c, self.c, shortcut) for _ in range(n))

#ODConv+ConNeXt_start
######################   ODConv+ConvNeXt  ####     start ###############################

# ------------------------------------ODConv start -------------------------------------
class ODConv2d_3rd(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 K=4, r=1 / 16, save_parameters=False,
                 padding_mode='zeros', device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.K = K
        self.r = r
        self.save_parameters = save_parameters

        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)

        del self.weight
        self.weight = nn.Parameter(torch.empty((
            K,
            out_channels,
            in_channels // groups,
            *self.kernel_size,
        ), **factory_kwargs))

        if bias:
            del self.bias
            self.bias = nn.Parameter(torch.empty(K, out_channels, **factory_kwargs))

        hidden_dim = max(int(in_channels * r), 16)  # 设置下限为16
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.reduction = nn.Linear(in_channels, hidden_dim)
        self.fc = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        # self.act = nn.SiLU(inplace=True)

        self.fc_f = nn.Linear(hidden_dim, out_channels)
        if not save_parameters or self.kernel_size[0] * self.kernel_size[1] > 1:
            self.fc_s = nn.Linear(hidden_dim, self.kernel_size[0] * self.kernel_size[1])
        if not save_parameters or in_channels // groups > 1:
            self.fc_c = nn.Linear(hidden_dim, in_channels // groups)
        if not save_parameters or K > 1:
            self.fc_w = nn.Linear(hidden_dim, K)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan_out = self.kernel_size[0] * self.kernel_size[1] * self.out_channels // self.groups
        for i in range(self.K):
            import math
            self.weight.data[i].normal_(0, math.sqrt(2.0 / fan_out))
        if self.bias is not None:
            self.bias.data.zero_()

    def extra_repr(self):
        return super().extra_repr() + f', K={self.K}, r={self.r:.4}'

    def get_weight_bias(self, context):
        B, C, H, W = context.shape

        if C != self.in_channels:
            raise ValueError(
                f"Expected context{[B, C, H, W]} to have {self.in_channels} channels, but got {C} channels instead")

        # x = self.gap(context).squeeze(-1).squeeze(-1)  # B, c_in
        # x = self.reduction(x)  # B, hidden_dim
        x = self.gap(context)
        x = self.fc(x)
        if x.size(0) > 1:
            x = self.bn(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.act(x)

        attn_f = self.fc_f(x).sigmoid()  # B, c_out
        attn = attn_f.view(B, 1, -1, 1, 1, 1)  # B, 1, c_out, 1, 1, 1
        if hasattr(self, 'fc_s'):
            attn_s = self.fc_s(x).sigmoid()  # B, k * k
            attn = attn * attn_s.view(B, 1, 1, 1, *self.kernel_size)  # B, 1, c_out, 1, k, k
        if hasattr(self, 'fc_c'):
            attn_c = self.fc_c(x).sigmoid()  # B, c_in // groups
            attn = attn * attn_c.view(B, 1, 1, -1, 1, 1)  # B, 1, c_out, c_in // groups, k, k
        if hasattr(self, 'fc_w'):
            attn_w = self.fc_w(x).softmax(-1)  # B, n
            attn = attn * attn_w.view(B, -1, 1, 1, 1, 1)  # B, n, c_out, c_in // groups, k, k

        weight = (attn * self.weight).sum(1)  # B, c_out, c_in // groups, k, k
        weight = weight.view(-1, self.in_channels // self.groups, *self.kernel_size)  # B * c_out, c_in // groups, k, k

        bias = None
        if self.bias is not None:
            if hasattr(self, 'fc_w'):
                bias = attn_w @ self.bias
            else:
                bias = self.bias.tile(B, 1)
            bias = bias.view(-1)  # B * c_out

        return weight, bias

    def forward(self, input, context=None):
        B, C, H, W = input.shape

        if C != self.in_channels:
            raise ValueError(
                f"Expected input{[B, C, H, W]} to have {self.in_channels} channels, but got {C} channels instead")

        weight, bias = self.get_weight_bias(context or input)

        output = nn.functional.conv2d(
            input.view(1, B * C, H, W), weight, bias,
            self.stride, self.padding, self.dilation, B * self.groups)  # 1, B * c_out, h_out, w_out
        output = output.view(B, self.out_channels, *output.shape[2:])

        return output

    def debug(self, input, context=None):
        B, C, H, W = input.shape

        if C != self.in_channels:
            raise ValueError(
                f"Expected input{[B, C, H, W]} to have {self.in_channels} channels, but got {C} channels instead")

        output_size = [
            ((H, W)[i] + 2 * self.padding[i] - self.dilation[i] * (self.kernel_size[i] - 1) - 1) // self.stride[i] + 1
            for i in range(2)
        ]

        weight, bias = self.get_weight_bias(context or input)

        weight = weight.view(B, self.groups, self.out_channels // self.groups,
                             -1)  # B, groups, c_out // groups, c_in // groups * k * k

        unfold = nn.functional.unfold(
            input, self.kernel_size, self.dilation, self.padding, self.stride)  # B, c_in * k * k, H_out * W_out
        unfold = unfold.view(B, self.groups, -1,
                             output_size[0] * output_size[1])  # B, groups, c_in // groups * k * k, H_out * W_out

        output = weight @ unfold  # B, groups, c_out // groups, H_out * W_out
        output = output.view(B, self.out_channels, *output_size)  # B, c_out, H_out * W_out

        if bias is not None:
            output = output + bias.view(B, self.out_channels, 1, 1)

        return output

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class ODConv_3rd(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, kerNums=1, g=1, p=None,
                 act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = ODConv2d_3rd(c1, c2, k, s, autopad(k, p), groups=g, K=kerNums)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


# ------------------------------------ODConv end -------------------------------------


# ------------------------------------Convnext start -------------------------------------
# ConvNextBlock
# ConvNextBlock
class ConvNextBlock(nn.Module):

    def __init__(self, inputdim, dim, drop_path=0., layer_scale_init_value=1e-6,
                 kersize=7):  # demo: [64, 64, 1]  1 denotes the number of repeats
        super().__init__()
        # 匹配yolov5配置文件加入outdim输出通道
        # self.flag = True if dim == outdim else False

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kersize, padding=kersize // 2, groups=dim)  # depthwise conv
        self.norm = LayerNorm_s(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath_ConvNext(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # if self.flag == False:
        #     raise ValueError(
        #         f"Expected input out to have {dim} channels, but got {outdim} channels instead")

        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm_s(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class DropPath_ConvNext(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath_ConvNext, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


# ------------------------------------Convnext end -------------------------------------

######################   ODConv+ConvNeXt  ####     END   by  AI&CV ###############################
#ODConv+ConNeXt_end


#EMA_start
###################### EMA_attention  ####     START   by  AI&CV  ###############################

import torch
from torch import nn


class EMA_attention(nn.Module):
    def __init__(self, channels, c2=None, factor=8):
        super(EMA_attention, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


###################### EMA_attention  ####     END   by  AI&CV  ###############################
#EMA_end


#DCNv3_start

###################### DCNV3  ####     start#############################

from ultralytics.nn.ops_dcnv3.modules import DCNv3


class DCNV3_YoLo(nn.Module):
    def __init__(self, inc, ouc, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()

        self.conv = Conv(inc, ouc, k=1)
        self.dcnv3 = DCNv3(ouc, kernel_size=k, stride=s, group=g, dilation=d)
        self.bn = nn.BatchNorm2d(ouc)
        self.act = Conv.default_act

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.dcnv3(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act(self.bn(x))
        return x


class Bottleneck_DCNV3(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = DCNV3_YoLo(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class DCNV3(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_DCNV3(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
###################### DCNV3  ####     end#############################

#DCNv3_end

class CAC2f(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.75):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g,  e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class CAC3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.8):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)  # to bottleneck
        self.cv2 = Conv(c1, c1 // 2, 1, 1)  # shortcut
        self.cv3 = Conv(c1 // 2 + c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


# 第三部分实验修改_start
import math

class Bottle2neck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, shortcut, baseWidth=26, scale=4):
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 96.0)))
        self.conv1 = Conv(inplanes, width * scale, k=1)

        convs = []
        for i in range(3):
            # convs.append(Conv(width*2, width*2, 3))
            if i == 1:
                convs.append(DWConv(width * 2, width * 2, 3))
            else:
                convs.append(ODConv(width * 2, width * 2, 3))
        self.convs = nn.ModuleList(convs)
        self.conv3 = Conv(width * scale * 2, planes * self.expansion, k=1, act=False)
        # self.conv3 = DCNv2(width*scale, planes * self.expansion, 1,1)

        self.silu = nn.SiLU(inplace=True)
        self.scale = scale
        self.width = width
        self.shortcut = shortcut

    def forward(self, x):

        if self.shortcut:
            residual = x
        out = self.conv1(x)
        spx = torch.split(out, self.width * 2, 1)
        sp = spx[0]
        out = spx[1]

        for i in range(3):
            sp = self.convs[i](sp)
            out = torch.cat((out, sp), 1)

        out = self.conv3(out)
        if self.shortcut:
            out += residual
        return self.silu(out)

class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, bias=False,norm_cfg=dict(type='BN2d', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        # self.bias = bias
        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=bias))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class Bottle3neck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, shortcut, baseWidth=26, scale=4, key_dim=16, num_heads=4,attn_ratio=2, activation=nn.ReLU):
        super(Bottle3neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 96.0)))
        self.conv1 = Conv(inplanes, width * scale, k=1)

        convs = []
        for i in range(3):
            # convs.append(Conv(width*2, width*2, 3))
            if i == 1:
                convs.append(DWConv(width * 2, width * 2, 3))
            else:
                convs.append(ODConv(width * 2, width * 2, 3))
        self.convs = nn.ModuleList(convs)
        self.conv3 = Conv(width * scale * 2, planes * self.expansion, k=1, act=False)
        # self.conv3 = DCNv2(width*scale, planes * self.expansion, 1,1)

        self.silu = nn.SiLU(inplace=True)
        self.scale = scale
        self.width = width
        self.shortcut = shortcut

        #修改_start
        self.num_heads = num_heads
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.to_q = Conv2d_BN(inplanes, nh_kd, 1)
        self.to_k = Conv2d_BN(inplanes, nh_kd, 1)
        self.to_v = Conv2d_BN(inplanes, self.dh, 1)
        self.act = activation()
        self.sigmoid = h_sigmoid()
        self.dwconv = Conv2d_BN(self.dh + 2 * self.nh_kd, 2 * self.nh_kd + self.dh, ks=3, stride=1, pad=1, dilation=1, groups=2 * self.nh_kd + self.dh)
        self.pwconv = Conv2d_BN(2 * self.nh_kd + self.dh, inplanes, ks=1)
        #修改_end

    def forward(self, x):

        # 修改部分_start
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        # detail enhance

        qkv = torch.cat([q, k, v], dim=1)
        qkv = self.act(self.dwconv(qkv))
        qkv = self.pwconv(qkv)
        # 修改部分_end

        if self.shortcut:
            residual = x
        out = self.conv1(x)
        spx = torch.split(out, self.width * 2, 1)
        sp = spx[0]
        out = spx[1]

        for i in range(3):
            sp = self.convs[i](sp)
            out = torch.cat((out, sp), 1)

        out = self.conv3(out)
        if self.shortcut:
            out += residual
        # return self.silu(out)  #修改前
        #方法一：
        xx = self.silu(out)
        xx = self.sigmoid(xx) * qkv
        #方法二：
        #xx = self.silu(out) * qkv
        return xx


class C2f_Res3Block(C2f):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottle3neck(self.c, self.c, shortcut) for _ in range(n))

class C2f_Res3Block2(C2f_attention3):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottle3neck(self.c, self.c, shortcut) for _ in range(n))

class C2f_Res3Block3(C2f_attention4):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottle3neck(self.c, self.c, shortcut) for _ in range(n))

"""
class C3_Res2Block(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottle2neck(c_, c_, shortcut) for _ in range(n)))
        #self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
"""

class C3_Res2Block(C3):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottle2neck(c_, c_, shortcut) for _ in range(n)))


class ODConv(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=nn.BatchNorm2d,
                 reduction=0.0625, kernel_num=1):
        padding = (kernel_size - 1) // 2
        super(ODConv, self).__init__(
            ODConv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups,
                     reduction=reduction, kernel_num=kernel_num),
            norm_layer(out_planes),
            nn.SiLU()
        )

class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,
    groups=1,
    reduction=0.0625,
    kernel_num=4,
    min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention
        self.bn_1 = nn.LayerNorm([attention_channel,1,1])
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn_1(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)

class ODConv2d(nn.Module):
    def __init__(self,
    in_planes,
    out_planes,
    kernel_size=3,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    reduction=0.0625,
    kernel_num=1):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):

        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)

# 第三部分实验修改_end

class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))

class C3_P(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(MLPBlock(c_, c_) for _ in range(n)))

class C2f_P(C2f):
    # C2f module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(MLPBlock(c_, c_) for _ in range(n)))

class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

#多余的实验
# 2020.11.06-Changed for building GhostNetV2
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from timm.models.registry import register_model

__all__ = ['ghostnetv2']


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostModuleV2(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=4, dw_size=3, stride=1, relu=True, mode=None, args=None):
        super(GhostModuleV2, self).__init__()
        self.mode = mode
        self.gate_fn = nn.Sigmoid()

        if self.mode in ['original']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = nn.Sequential(
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
        elif self.mode in ['attn']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = nn.Sequential(
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.short_conv = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.mode in ['original']:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, :self.oup, :, :]
        elif self.mode in ['attn']:
            res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, :self.oup, :, :] * F.interpolate(self.gate_fn(res), size=(out.shape[-2], out.shape[-1]),
                                                           mode='nearest')


class GhostBottleneckV2(nn.Module):

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=4, layer_id=0, args=None):  #修改ratio，原本为0
        super(GhostBottleneckV2, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        if layer_id <= 1:
            self.ghost1 = GhostModuleV2(in_chs, mid_chs, relu=True, mode='original', args=args)
        else:
            self.ghost1 = GhostModuleV2(in_chs, mid_chs, relu=True, mode='attn', args=args)

            # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        self.ghost2 = GhostModuleV2(mid_chs, out_chs, relu=False, mode='original', args=args)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x

class GhostBottleneckV3(nn.Module):

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=8, layer_id=0, args=None):
        super(GhostBottleneckV3, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        if layer_id <= 1:
            self.ghost1 = GhostModuleV2(in_chs, mid_chs, relu=True, mode='original', args=args)
        else:
            self.ghost1 = GhostModuleV2(in_chs, mid_chs, relu=True, mode='attn', args=args)

            # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            #self.se = ScConv(mid_chs, squeeze_radio=se_ratio) #修改B2
            #self.se=EMA_attention(mid_chs)   #修改2
            self.se=ContextAggregation(mid_chs)  #修改B3
            #self.se=MCALayer(mid_chs)  #修改4
        else:
            self.se = None

        self.ghost2 = GhostModuleV2(mid_chs, out_chs, relu=False, mode='original', args=args)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x

class C2f_GhostBottleneckV22(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(GhostBottleneckV2(self.c, self.c, self.c) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C2f_GhostBottleneckV3(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(GhostBottleneckV3(self.c, self.c, self.c) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class GhostNetV2(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width=1.0, dropout=0.2, block=GhostBottleneckV2, args=None):
        super(GhostNetV2, self).__init__()
        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        # block = block
        layer_id = 0
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                if block == GhostBottleneckV2:
                    layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                        se_ratio=se_ratio, layer_id=layer_id, args=args))
                input_channel = output_channel
                layer_id += 1
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


@register_model
def ghostnetv2(**kwargs):
    cfgs = [
        # k, t, c, SE, s
        [[3, 16, 16, 0, 1]],
        [[3, 48, 24, 0, 2]],
        [[3, 72, 24, 0, 1]],
        [[5, 72, 40, 0.25, 2]],
        [[5, 120, 40, 0.25, 1]],
        [[3, 240, 80, 0, 2]],
        [[3, 200, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
         ],
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
         ]
    ]
    return GhostNetV2(cfgs, num_classes=kwargs['num_classes'],
                      width=kwargs['width'],
                      dropout=kwargs['dropout'],
                      args=kwargs['args'])


import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8 mg
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0


class SqueezeExcite(nn.Module):
    def __init__(
            self,
            in_chs,
            se_ratio=0.25,
            reduced_base_chs=None,
            # act_layer=nn.ReLU,
            act_layer=nn.SiLU,
            gate_fn=hard_sigmoid,
            divisor=4,
            **_,
    ):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible(
            (reduced_base_chs or in_chs) * se_ratio, divisor,
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, act_layer=nn.SiLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(
            in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class RepGhostModule(nn.Module):
    def __init__(
            self, inp, oup, kernel_size=1, dw_size=3, stride=1, relu=True, deploy=False, reparam_bn=True,
            reparam_identity=False
    ):
        super(RepGhostModule, self).__init__()
        init_channels = oup
        new_channels = oup
        self.deploy = deploy

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False,
            ),
            nn.BatchNorm2d(init_channels),
            nn.SiLU(inplace=True) if relu else nn.Sequential(),
        )
        fusion_conv = []
        fusion_bn = []
        if not deploy and reparam_bn:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.BatchNorm2d(init_channels))
        if not deploy and reparam_identity:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.Identity())

        self.fusion_conv = nn.Sequential(*fusion_conv)
        self.fusion_bn = nn.Sequential(*fusion_bn)

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=deploy,
            ),
            nn.BatchNorm2d(new_channels) if not deploy else nn.Sequential(),
            # nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        if deploy:
            self.cheap_operation = self.cheap_operation[0]
        if relu:
            self.relu = nn.SiLU(inplace=False)
        else:
            self.relu = nn.Sequential()

    def forward(self, x):
        x1 = self.primary_conv(x)  # mg
        x2 = self.cheap_operation(x1)
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            x2 = x2 + bn(conv(x1))
        return self.relu(x2)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.cheap_operation[0], self.cheap_operation[1])
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            kernel, bias = self._fuse_bn_tensor(conv, bn, kernel3x3.shape[0], kernel3x3.device)
            kernel3x3 += self._pad_1x1_to_3x3_tensor(kernel)
            bias3x3 += bias
        return kernel3x3, bias3x3

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    @staticmethod
    def _fuse_bn_tensor(conv, bn, in_channels=None, device=None):
        in_channels = in_channels if in_channels else bn.running_mean.shape[0]
        device = device if device else bn.weight.device
        if isinstance(conv, nn.Conv2d):
            kernel = conv.weight
            assert conv.bias is None
        else:
            assert isinstance(conv, nn.Identity)
            kernel_value = np.zeros((in_channels, 1, 1, 1), dtype=np.float32)
            for i in range(in_channels):
                kernel_value[i, 0, 0, 0] = 1
            kernel = torch.from_numpy(kernel_value).to(device)

        if isinstance(bn, nn.BatchNorm2d):
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std
        assert isinstance(bn, nn.Identity)
        return kernel, torch.zeros(in_channels).to(kernel.device)

    def switch_to_deploy(self):
        if len(self.fusion_conv) == 0 and len(self.fusion_bn) == 0:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.cheap_operation = nn.Conv2d(in_channels=self.cheap_operation[0].in_channels,
                                         out_channels=self.cheap_operation[0].out_channels,
                                         kernel_size=self.cheap_operation[0].kernel_size,
                                         padding=self.cheap_operation[0].padding,
                                         dilation=self.cheap_operation[0].dilation,
                                         groups=self.cheap_operation[0].groups,
                                         bias=True)
        self.cheap_operation.weight.data = kernel
        self.cheap_operation.bias.data = bias
        self.__delattr__('fusion_conv')
        self.__delattr__('fusion_bn')
        self.fusion_conv = []
        self.fusion_bn = []
        self.deploy = True


class RepGhostBottleneck(nn.Module):
    """RepGhost bottleneck w/ optional SE"""

    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            dw_kernel_size=3,
            stride=1,
            se_ratio=0.0,
            shortcut=True,
            reparam=True,
            reparam_bn=True,
            reparam_identity=False,
            deploy=False,
    ):
        super(RepGhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride
        self.enable_shortcut = shortcut
        self.in_chs = in_chs
        self.out_chs = out_chs

        # Point-wise expansion
        self.ghost1 = RepGhostModule(
            in_chs,
            mid_chs,
            relu=True,
            reparam_bn=reparam and reparam_bn,
            reparam_identity=reparam and reparam_identity,
            deploy=deploy,
        )

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = RepGhostModule(
            mid_chs,
            out_chs,
            relu=False,
            reparam_bn=reparam and reparam_bn,
            reparam_identity=reparam and reparam_identity,
            deploy=deploy,
        )

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                ),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(
                    in_chs, out_chs, 1, stride=1,
                    padding=0, bias=False,
                ),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x
        x1 = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x1)
            x = self.bn_dw(x)
        else:
            x = x1

        if self.se is not None:
            x = self.se(x)

        # 2nd repghost bottleneck mg
        x = self.ghost2(x)
        if not self.enable_shortcut and self.in_chs == self.out_chs and self.stride == 1:
            return x
        return x + self.shortcut(residual)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class C2f_repghost(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  #
        self.m = nn.ModuleList(RepGhostBottleneck(self.c, self.c, self.c, dw_kernel_size=((3), (3))) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


import torch
from torch import nn
import math

__all__ = ['MCALayer', 'MCAGate']


class StdPool(nn.Module):
    def __init__(self):
        super(StdPool, self).__init__()

    def forward(self, x):
        b, c, _, _ = x.size()

        std = x.view(b, c, -1).std(dim=2, keepdim=True)
        std = std.reshape(b, c, 1, 1)

        return std


class MCAGate(nn.Module):
    def __init__(self, k_size, pool_types=['avg', 'std']):
        """Constructs a MCAGate module.
        Args:
            k_size: kernel size
            pool_types: pooling type. 'avg': average pooling, 'max': max pooling, 'std': standard deviation pooling.
        """
        super(MCAGate, self).__init__()

        self.pools = nn.ModuleList([])
        for pool_type in pool_types:
            if pool_type == 'avg':
                self.pools.append(nn.AdaptiveAvgPool2d(1))
            elif pool_type == 'max':
                self.pools.append(nn.AdaptiveMaxPool2d(1))
            elif pool_type == 'std':
                self.pools.append(StdPool())
            else:
                raise NotImplementedError

        self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), stride=1, padding=(0, (k_size - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

        self.weight = nn.Parameter(torch.rand(2))

    def forward(self, x):
        feats = [pool(x) for pool in self.pools]

        if len(feats) == 1:
            out = feats[0]
        elif len(feats) == 2:
            weight = torch.sigmoid(self.weight)
            out = 1 / 2 * (feats[0] + feats[1]) + weight[0] * feats[0] + weight[1] * feats[1]
        else:
            assert False, "Feature Extraction Exception!"

        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.conv(out)
        out = out.permute(0, 3, 2, 1).contiguous()

        out = self.sigmoid(out)
        out = out.expand_as(x)

        return x * out


class MCALayer(nn.Module):
    def __init__(self, inp, no_spatial=True):
        """Constructs a MCA module.
        Args:
            inp: Number of channels of the input feature maps
            no_spatial: whether to build channel dimension interactions
        """
        super(MCALayer, self).__init__()

        lambd = 1.5
        gamma = 1
        temp = round(abs((math.log2(inp) - gamma) / lambd))
        kernel = temp if temp % 2 else temp - 1

        self.h_cw = MCAGate(3)
        self.w_hc = MCAGate(3)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.c_hw = MCAGate(kernel)

    def forward(self, x):
        x_h = x.permute(0, 2, 1, 3).contiguous()
        x_h = self.h_cw(x_h)
        x_h = x_h.permute(0, 2, 1, 3).contiguous()

        x_w = x.permute(0, 3, 2, 1).contiguous()
        x_w = self.w_hc(x_w)
        x_w = x_w.permute(0, 3, 2, 1).contiguous()

        if not self.no_spatial:
            x_c = self.c_hw(x)
            x_out = 1 / 3 * (x_c + x_h + x_w)
        else:
            x_out = 1 / 2 * (x_h + x_w)

        return x_out

    '''
    Description: 
    Date: 2023-07-21 14:36:27
    LastEditTime: 2023-07-22 14:55:44
    FilePath: /chengdongzhou/ScConv.py
    '''
    import torch
    import torch.nn.functional as F
import torch.nn as nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
        # Pad to 'same' shape outputs
        if d > 1:
            k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
        return p

class GroupBatchnorm2d(nn.Module):
        def __init__(self, c_num: int,
                     group_num: int = 16,
                     eps: float = 1e-10
                     ):
            super(GroupBatchnorm2d, self).__init__()
            assert c_num >= group_num
            self.group_num = group_num
            self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))
            self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
            self.eps = eps

        def forward(self, x):
            N, C, H, W = x.size()
            x = x.view(N, self.group_num, -1)
            mean = x.mean(dim=2, keepdim=True)
            std = x.std(dim=2, keepdim=True)
            x = (x - mean) / (std + self.eps)
            x = x.view(N, C, H, W)
            return x * self.gamma + self.beta

class SRU(nn.Module):
        def __init__(self,
                     oup_channels: int,
                     group_num: int = 16,
                     gate_treshold: float = 0.5
                     ):
            super().__init__()

            self.gn = GroupBatchnorm2d(oup_channels, group_num=group_num)
            self.gate_treshold = gate_treshold
            self.sigomid = nn.Sigmoid()

        def forward(self, x):
            gn_x = self.gn(x)
            w_gamma = F.softmax(self.gn.gamma, dim=0)
            reweigts = self.sigomid(gn_x * w_gamma)
            # Gate
            info_mask = w_gamma > self.gate_treshold
            noninfo_mask = w_gamma <= self.gate_treshold
            x_1 = info_mask * reweigts * x
            x_2 = noninfo_mask * reweigts * x
            x = self.reconstruct(x_1, x_2)
            return x

        def reconstruct(self, x_1, x_2):
            x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
            x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
            return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)

class CRU(nn.Module):
        '''
        alpha: 0<alpha<1
        '''

        def __init__(self,
                     op_channel: int,
                     alpha: float = 1 / 2,
                     squeeze_radio: int = 2,
                     group_size: int = 2,
                     group_kernel_size: int = 3,
                     ):
            super().__init__()
            self.up_channel = up_channel = int(alpha * op_channel)
            self.low_channel = low_channel = op_channel - up_channel
            self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
            self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
            # up
            self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                                 padding=group_kernel_size // 2, groups=group_size)
            self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
            # low
            self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio,
                                  kernel_size=1, bias=False)
            self.advavg = nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            # Split
            up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
            up, low = self.squeeze1(up), self.squeeze2(low)
            # Transform
            Y1 = self.GWC(up) + self.PWC1(up)
            Y2 = torch.cat([self.PWC2(low), low], dim=1)
            # Fuse
            out = torch.cat([Y1, Y2], dim=1)
            out = F.softmax(self.advavg(out), dim=1) * out
            out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
            return out1 + out2

class ScConv(nn.Module):
        def __init__(self,
                     op_channel: int,
                     group_num: int = 16,
                     gate_treshold: float = 0.5,
                     alpha: float = 1 / 2,
                     squeeze_radio: int = 2,
                     group_size: int = 2,
                     k: int = 3,
                     ):
            super().__init__()
            self.SRU = SRU(op_channel,
                           group_num=group_num,
                           gate_treshold=gate_treshold)
            self.CRU = CRU(op_channel,
                           alpha=alpha,
                           squeeze_radio=squeeze_radio,
                           group_size=group_size,
                           group_kernel_size=k)

        def forward(self, x):
            x = self.SRU(x)
            x = self.CRU(x)
            return x

class C2f_ScConv(nn.Module):
        # CSP Bottleneck with 2 convolutions
        def __init__(self, c1, c2, n=1, shortcut=False, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__()
            self.c = int(c2 * e)  # hidden channels
            self.cv1 = Conv(c1, 2 * self.c, 1, 1)
            self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
            self.m = nn.ModuleList(ScConv(self.c) for _ in range(n))

        def forward(self, x):
            y = list(self.cv1(x).chunk(2, 1))
            y.extend(m(y[-1]) for m in self.m)
            return self.cv2(torch.cat(y, 1))

        def forward_split(self, x):
            y = list(self.cv1(x).split((self.c, self.c), 1))
            y.extend(m(y[-1]) for m in self.m)
            return self.cv2(torch.cat(y, 1))

if __name__ == '__main__':
        x = torch.randn(1, 32, 16, 16)
        model = ScConv(32)
        print(model(x).shape)


import torch
import torch.nn as nn
import numpy as np


class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn


class LSKblockAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKblock(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


import torch.nn as nn
import torch


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert (m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed

    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv


class RepViTBlock(nn.Module):
    def __init__(self, in1, inp, hidden_dim, oup, kernel_size=3, stride=2, use_se=0, use_hs=0):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        print(inp)
        print(hidden_dim)
        print(oup)
        assert (hidden_dim == 2 * inp)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
            ))
        else:
            assert (self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


###################### ContextAggregation  ####     START   by  AI&CV  ###############################

from mmcv.cnn import ConvModule
from mmengine.model import caffe2_xavier_init, constant_init


class ContextAggregation(nn.Module):
    """
    Context Aggregation Block.
    Args:
        in_channels (int): Number of input channels.
        reduction (int, optional): Channel reduction ratio. Default: 1.
        conv_cfg (dict or None, optional): Config dict for the convolution
            layer. Default: None.
    """

    def __init__(self, in_channels, reduction=1):
        super(ContextAggregation, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = max(in_channels // reduction, 1)

        conv_params = dict(kernel_size=1, act_cfg=None)

        self.a = ConvModule(in_channels, 1, **conv_params)
        self.k = ConvModule(in_channels, 1, **conv_params)
        self.v = ConvModule(in_channels, self.inter_channels, **conv_params)
        self.m = ConvModule(self.inter_channels, in_channels, **conv_params)

        self.init_weights()

    def init_weights(self):
        for m in (self.a, self.k, self.v):
            caffe2_xavier_init(m.conv)
        constant_init(self.m.conv, 0)

    def forward(self, x):
        # n, c = x.size(0)
        n = x.size(0)
        c = self.inter_channels
        # n, nH, nW, c = x.shape

        # a: [N, 1, H, W]
        a = self.a(x).sigmoid()

        # k: [N, 1, HW, 1]
        k = self.k(x).view(n, 1, -1, 1).softmax(2)

        # v: [N, 1, C, HW]
        v = self.v(x).view(n, 1, c, -1)

        # y: [N, C, 1, 1]
        y = torch.matmul(v, k).view(n, c, 1, 1)
        y = self.m(y) * a

        return x + y

###################### ContextAggregation  ####     END   by  AI&CV  ###############################

###################### BiFPN  ####     START   by  AI&CV  ###############################


class BiFPN_Concat2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)


class BiFPN_Concat3(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat3, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        return torch.cat(x, self.d)

###################### BiFPN  ####     END   by  AI&CV  ###############################

###################### GFPN  ####     strat   by  AI&CV  ###############################

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    '''Basic cell for rep-style block, including conv and bn'''
    result = nn.Sequential()
    result.add_module(
        'conv',
        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  groups=groups,
                  bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepConv(nn.Module):
    '''RepConv is a basic rep-style block, including training and deploy status
    Code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 deploy=False,
                 act='relu',
                 norm=None):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        if isinstance(act, str):
            self.nonlinearity = get_activation(act)
        else:
            self.nonlinearity = act

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = None
            self.rbr_dense = conv_bn(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   stride=stride,
                                   padding=padding_11,
                                   groups=groups)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(
            self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(
            kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)


def get_activation(name='silu', inplace=True):
    if name is None:
        return nn.Identity()

    if isinstance(name, str):
        if name == 'silu':
            module = nn.SiLU(inplace=inplace)
        elif name == 'relu':
            module = nn.ReLU(inplace=inplace)
        elif name == 'lrelu':
            module = nn.LeakyReLU(0.1, inplace=inplace)
        elif name == 'swish':
            module = Swish(inplace=inplace)
        elif name == 'hardsigmoid':
            module = nn.Hardsigmoid(inplace=inplace)
        elif name == 'identity':
            module = nn.Identity()
        else:
            raise AttributeError('Unsupported act type: {}'.format(name))
        return module

    elif isinstance(name, nn.Module):
        return name

    else:
        raise AttributeError('Unsupported act type: {}'.format(name))


def get_norm(name, out_channels, inplace=True):
    if name == 'bn':
        module = nn.BatchNorm2d(out_channels)
    else:
        raise NotImplementedError
    return module


class ConvBNAct(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
            self,
            in_channels,
            out_channels,
            ksize,
            stride=1,
            groups=1,
            bias=False,
            act='silu',
            norm='bn',
            reparam=False,
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        if norm is not None:
            self.bn = get_norm(norm, out_channels, inplace=True)
        if act is not None:
            self.act = get_activation(act, inplace=True)
        self.with_norm = norm is not None
        self.with_act = act is not None

    def forward(self, x):
        x = self.conv(x)
        if self.with_norm:
            x = self.bn(x)
        if self.with_act:
            x = self.act(x)
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))


class BasicBlock_3x3_Reverse(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_hidden_ratio,
                 ch_out,
                 act='relu',
                 shortcut=True):
        super(BasicBlock_3x3_Reverse, self).__init__()
        assert ch_in == ch_out
        ch_hidden = int(ch_in * ch_hidden_ratio)
        self.conv1 = ConvBNAct(ch_hidden, ch_out, 3, stride=1, act=act)
        self.conv2 = RepConv(ch_in, ch_hidden, 3, stride=1, act=act)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv2(x)
        y = self.conv1(y)
        if self.shortcut:
            return x + y
        else:
            return y


class SPP(nn.Module):
    def __init__(
            self,
            ch_in,
            ch_out,
            k,
            pool_size,
            act='swish',
    ):
        super(SPP, self).__init__()
        self.pool = []
        for i, size in enumerate(pool_size):
            pool = nn.MaxPool2d(kernel_size=size,
                                stride=1,
                                padding=size // 2,
                                ceil_mode=False)
            self.add_module('pool{}'.format(i), pool)
            self.pool.append(pool)
        self.conv = ConvBNAct(ch_in, ch_out, k, act=act)

    def forward(self, x):
        outs = [x]

        for pool in self.pool:
            outs.append(pool(x))
        y = torch.cat(outs, axis=1)

        y = self.conv(y)
        return y


class CSPStage(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 n=1,
                 block_fn='BasicBlock_3x3_Reverse',
                 ch_hidden_ratio=1.0,
                 act='silu',
                 spp=False):
        super(CSPStage, self).__init__()

        split_ratio = 2
        ch_first = int(ch_out // split_ratio)
        ch_mid = int(ch_out - ch_first)
        self.conv1 = ConvBNAct(ch_in, ch_first, 1, act=act)
        self.conv2 = ConvBNAct(ch_in, ch_mid, 1, act=act)
        self.convs = nn.Sequential()

        next_ch_in = ch_mid
        for i in range(n):
            if block_fn == 'BasicBlock_3x3_Reverse':
                self.convs.add_module(
                    str(i),
                    BasicBlock_3x3_Reverse(next_ch_in,
                                           ch_hidden_ratio,
                                           ch_mid,
                                           act=act,
                                           shortcut=True))
            else:
                raise NotImplementedError
            if i == (n - 1) // 2 and spp:
                self.convs.add_module(
                    'spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act))
            next_ch_in = ch_mid
        self.conv3 = ConvBNAct(ch_mid * n + ch_first, ch_out, 1, act=act)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)

        mid_out = [y1]
        for conv in self.convs:
            y2 = conv(y2)
            mid_out.append(y2)
        y = torch.cat(mid_out, axis=1)
        y = self.conv3(y)
        return y

###################### GPFN  ####     END   by  AI&CV  ###############################

###################### PolarizedSelfAttention     ####     start   by  AI&CV  ###############################
from torch import nn
from torch.nn import init


class PolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        self.softmax_channel = nn.Softmax(1)
        self.softmax_spatial = nn.Softmax(-1)
        self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()
        self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel-only Self-Attention
        channel_wv = self.ch_wv(x)  # bs,c//2,h,w
        channel_wq = self.ch_wq(x)  # bs,1,h,w
        channel_wv = channel_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        channel_wq = channel_wq.reshape(b, -1, 1)  # bs,h*w,1
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)  # bs,c//2,1,1
        channel_weight = self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))).permute(0, 2,
                                                                                                                 1).reshape(
            b, c, 1, 1)  # bs,c,1,1
        channel_out = channel_weight * x

        # Spatial-only Self-Attention
        spatial_wv = self.sp_wv(x)  # bs,c//2,h,w
        spatial_wq = self.sp_wq(x)  # bs,c//2,h,w
        spatial_wq = self.agp(spatial_wq)  # bs,c//2,1,1
        spatial_wv = spatial_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)  # bs,1,c//2
        spatial_wq = self.softmax_spatial(spatial_wq)
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # bs,1,h*w
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))  # bs,1,h,w
        spatial_out = spatial_weight * x
        out = spatial_out + channel_out
        return out

###################### PolarizedSelfAttention    ####     end   by  AI&CV  ###############################

######################  EVC  ####  AI&CV   start ###############################

# by AI&CV EVCBlock

# ecvblcok


import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function, Variable
from torch.nn import Module, parameter

import warnings


from torch.nn.modules.batchnorm import _BatchNorm
from functools import partial

from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


# LVC
class Encoding(nn.Module):
    def __init__(self, in_channels, num_codes):
        super(Encoding, self).__init__()
        # init codewords and smoothing factor
        self.in_channels, self.num_codes = in_channels, num_codes
        num_codes = 64
        std = 1. / ((num_codes * in_channels) ** 0.5)
        # [num_codes, channels]
        self.codewords = nn.Parameter(
            torch.empty(num_codes, in_channels, dtype=torch.float).uniform_(-std, std), requires_grad=True)
        # [num_codes]
        self.scale = nn.Parameter(torch.empty(num_codes, dtype=torch.float).uniform_(-1, 0), requires_grad=True)

    @staticmethod
    def scaled_l2(x, codewords, scale):
        num_codes, in_channels = codewords.size()
        b = x.size(0)
        expanded_x = x.unsqueeze(2).expand((b, x.size(1), num_codes, in_channels))

        reshaped_codewords = codewords.view((1, 1, num_codes, in_channels))

        reshaped_scale = scale.view((1, 1, num_codes))  # N, num_codes

        scaled_l2_norm = reshaped_scale * (expanded_x - reshaped_codewords).pow(2).sum(dim=3)
        return scaled_l2_norm

    @staticmethod
    def aggregate(assignment_weights, x, codewords):
        num_codes, in_channels = codewords.size()

        reshaped_codewords = codewords.view((1, 1, num_codes, in_channels))
        b = x.size(0)

        expanded_x = x.unsqueeze(2).expand((b, x.size(1), num_codes, in_channels))

        assignment_weights = assignment_weights.unsqueeze(3)  # b, N, num_codes,

        encoded_feat = (assignment_weights * (expanded_x - reshaped_codewords)).sum(1)
        return encoded_feat

    def forward(self, x):
        assert x.dim() == 4 and x.size(1) == self.in_channels
        b, in_channels, w, h = x.size()

        # [batch_size, height x width, channels]
        x = x.view(b, self.in_channels, -1).transpose(1, 2).contiguous()

        # assignment_weights: [batch_size, channels, num_codes]
        assignment_weights = F.softmax(self.scaled_l2(x, self.codewords, self.scale), dim=2)

        # aggregate
        encoded_feat = self.aggregate(assignment_weights, x, self.codewords)
        return encoded_feat


#  1*1 3*3 1*1
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        expansion = 4
        c = out_channels // expansion

        self.conv1 = nn.Conv2d(in_channels, c, kernel_size=1, stride=1, padding=0, bias=False)  # [64, 256, 1, 1]
        self.bn1 = norm_layer(c)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(c, c, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(c)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(c, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(out_channels)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.residual_bn = norm_layer(out_channels)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)  # if x_t_r is None else self.conv2(x + x_t_r)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class Mean(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions. Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""  # CBL

    def __init__(
            self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, k, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=k,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class LVCBlock(nn.Module):
    def __init__(self, c1, c2, num_codes, channel_ratio=0.25, base_channel=64):
        super(LVCBlock, self).__init__()
        self.c2 = c2
        self.num_codes = num_codes
        num_codes = 64

        self.conv_1 = ConvBlock(c1, c1, res_conv=True, stride=1)

        self.LVC = nn.Sequential(
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            Encoding(c1, num_codes=num_codes),
            nn.BatchNorm1d(num_codes),
            nn.ReLU(inplace=True),
            Mean(dim=1))
        self.fc = nn.Sequential(nn.Linear(c1, c1), nn.Sigmoid())

    def forward(self, x):
        x = self.conv_1(x, return_x_2=False)
        en = self.LVC(x)
        gam = self.fc(en)
        b, in_channels, _, _ = x.size()
        y = gam.view(b, in_channels, 1, 1)
        x = F.relu_(x + x * y)
        return x


# LightMLPBlock
class LightMLPBlock(nn.Module):
    def __init__(self, c1, c2, ksize=1, stride=1, act="silu",
                 mlp_ratio=4., drop=0., act_layer=nn.GELU,
                 use_layer_scale=True, layer_scale_init_value=1e-5, drop_path=0.,
                 norm_layer=GroupNorm):  # act_layer=nn.GELU,
        super().__init__()
        self.dw = DWConv(c1, c2, ksize=1, stride=1, act="silu")
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.c2 = c2

        self.norm1 = norm_layer(c1)
        self.norm2 = norm_layer(c1)

        mlp_hidden_dim = int(c1 * mlp_ratio)
        self.mlp = Mlp(in_features=c1, hidden_features=mlp_hidden_dim, act_layer=nn.GELU,
                       drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((c2)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((c2)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.dw(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.dw(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# EVCBlock
class EVCBlock(nn.Module):
    def __init__(self, c1, c2, channel_ratio=4, base_channel=16):
        super().__init__()
        expansion = 2
        ch = c2 * expansion
        # Stem stage: get the feature maps by conv block (copied form resnet.py) 进入conformer框架之前的处理
        self.conv1 = nn.Conv2d(c1, c1, kernel_size=7, stride=1, padding=3, bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(c1)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 1 / 4 [56, 56]

        # LVC
        self.lvc = LVCBlock(c1, c2, num_codes=64)  # c1值暂时未定
        # LightMLPBlock
        self.l_MLP = LightMLPBlock(c1, c2, ksize=1, stride=1, act="silu", act_layer=nn.GELU, mlp_ratio=4., drop=0.,
                                   use_layer_scale=True, layer_scale_init_value=1e-5, drop_path=0.,
                                   norm_layer=GroupNorm)
        self.cnv1 = nn.Conv2d(ch, c2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.maxpool(self.act1(self.bn1(self.conv1(x))))
        # LVCBlock
        x_lvc = self.lvc(x1)
        # LightMLPBlock
        x_lmlp = self.l_MLP(x1)
        # concat
        x = torch.cat((x_lvc, x_lmlp), dim=1)
        x = self.cnv1(x)
        return x

######################  EVC  ####  AI&CV   end ###############################


######################  SPD-Conv  ####     start ###############################

class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

######################  SPD-Conv  ####     start ###############################

"""
InceptionNeXt implementation, paper: https://arxiv.org/abs/2303.16900
Some code is borrowed from timm: https://github.com/huggingface/pytorch-image-models
"""

from functools import partial

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import checkpoint_seq
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """

    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        from timm.layers import to_2tuple
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class MlpHead(nn.Module):
    """ MLP classification head
    """

    def __init__(self, dim, num_classes=1000, mlp_ratio=3, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), drop=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.mean((2, 3))  # global average pooling
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class MetaNeXtBlock(nn.Module):
    """ MetaNeXtBlock Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            token_mixer=InceptionDWConv2d,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=ConvMlp,
            mlp_ratio=4,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,

    ):
        super().__init__()
        self.token_mixer = token_mixer(dim)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x


class MetaNeXtStage(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            ds_stride=2,
            depth=1,
            drop_path_rates=None,
            ls_init_value=1.0,
            act_layer=nn.GELU,
            norm_layer=nn.BatchNorm2d,
            mlp_ratio=4,
    ):
        super().__init__()
        self.grad_checkpointing = False
        if ds_stride > 1:
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                nn.Conv2d(in_chs, out_chs, kernel_size=ds_stride, stride=ds_stride),
            )
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(MetaNeXtBlock(
                dim=out_chs,
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratio,
            ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x