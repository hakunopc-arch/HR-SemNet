import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv
from .block import C2f, Bottleneck


class CARAFE(nn.Module):
    def __init__(self, inC, outC, kernel_size=3, up_factor=2):
        super(CARAFE, self).__init__()
        self.inc = inC
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(inC, inC // 4, 1)
        if self.kernel_size % 2 == 0:
            self.encoder = nn.Conv2d(inC // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                    self.kernel_size, 1, 0)
        else:
            self.encoder = nn.Conv2d(inC // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                    self.kernel_size, 1, self.kernel_size // 2)
        # self.out = DSConv(inC, outC, 3, 1)

    def forward(self, in_tensor):
        N, C, H, W = in_tensor.size()

        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        kernel_tensor = self.down(in_tensor)  # (N, Cm, H, W)
        if self.kernel_size % 2 == 0:
            kernel_tensor = F.pad(kernel_tensor, pad=(self.kernel_size // 2, self.kernel_size // 2 - 1,
                                              self.kernel_size // 2, self.kernel_size // 2 - 1),
                            mode='constant', value=0)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor) # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor) # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(N, self.kernel_size ** 2, H, W, self.up_factor ** 2) # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        if self.kernel_size % 2 == 0:
            in_tensor = F.pad(in_tensor, pad=(self.kernel_size // 2, self.kernel_size // 2 - 1,
                                            self.kernel_size // 2, self.kernel_size // 2 - 1),
                            mode='constant', value=0)
        else:
            in_tensor = F.pad(in_tensor, pad=(self.kernel_size // 2, self.kernel_size // 2,
                                            self.kernel_size // 2, self.kernel_size // 2),
                            mode='constant', value=0) # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        in_tensor = in_tensor.unfold(2, self.kernel_size, step=1) # (N, C, H, W+Kup//2+Kup//2, Kup)
        in_tensor = in_tensor.unfold(3, self.kernel_size, step=1) # (N, C, H, W, Kup, Kup)
        in_tensor = in_tensor.reshape(N, C, H, W, -1) # (N, C, H, W, Kup^2)
        in_tensor = in_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        out_tensor = torch.matmul(in_tensor, kernel_tensor)  # (N, H, W, C, S^2)
        out_tensor = out_tensor.reshape(N, H, W, -1)

        out_tensor = out_tensor.permute(0, 3, 1, 2)

        out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)

        # out_tensor = self.out(out_tensor)
        return out_tensor


class bilinear(nn.Module):
    def __init__(self, c1, c2, u=2):
        super().__init__()
        self.bool = False
        self.up = u
        if c1 != c2:
            self.bool = True
            self.cv = Conv(c1, c2, 1, 1)
        self.upsample = nn.Upsample(None, self.up, 'bilinear')

    def forward(self, x):
        if self.bool:
            x = self.cv(x)
        # _, _, H, W = x.size()
        # x = F.interpolate(
        #     x,
        #     size=(H*self.up, W*self.up),
        #     mode='bilinear',
        #     align_corners=True)
        x = self.upsample(x)

        return x


class LCSM(nn.Module):
    def __init__(self, c1, k=1, f=2):
        super().__init__()
        self.c = c1
        self.cm = c1 * f**2
        self.kernel = k
        self.factor = f
        self.cfm = CARAFE(self.c, self.c, self.kernel, self.factor)
        # self.cfm = nn.Upsample(None, self.factor, 'nearest')
        self.cvm = Conv(self.cm, self.c, 1, 1)

    def forward(self, x):
        xd = F.pixel_unshuffle(x, self.factor)
        xd = self.cvm(xd)
        xu = self.cfm(xd)
        return xu + x


class LCSMDense(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.l1 = LCSM(c1, 1, 2)
        self.l2 = LCSM(c1, 2, 4)
        self.l3 = LCSM(c1, 3, 8)

        self.cv = Conv(c1 * 4, c1, 1, 1)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x)
        x3 = self.l3(x)
        x = self.cv(torch.cat((x, x1, x2, x3), 1))

        return x


class CLfS3(nn.Module):
    def __init__(self, c1, c2, shortcut=False, g=1, e=0.5):
        super().__init__()

        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + 3) * self.c, c2, 1)
        self.m1 = Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
        self.m2 = Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
        self.m3 = Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)

        self.u2 = CARAFE(self.c, self.c, 1, 2)
        self.u3 = CARAFE(self.c, self.c, 2, 4)
        self.cvm2 = Conv(self.c * 4, self.c, 1, 1)
        self.cvm3 = Conv(self.c * 16, self.c, 1, 1)

        # self.l1 = LCSM(self.c, 1, 2)
        # self.l2 = LCSM(self.c, 2, 4)

    def forward(self, x):

        y, y0 = self.cv1(x).chunk(2, 1)
        y1 = self.m1(y0)

        y2 = torch.pixel_unshuffle(y1, 2)
        y2 = self.cvm2(y2)
        y2 = self.u2(y2) + y1
        y2 = self.m2(y2)
        # y2 = self.m2(self.l1(y1))

        y3 = torch.pixel_unshuffle(y2, 4)
        y3 = self.cvm3(y3)
        y3 = self.u3(y3) + y2
        y3 = self.m3(y3)
        # y3 = self.m3(self.l2(y2))

        return self.cv2(torch.cat((y, y0, y1, y2, y3), 1))


class CLfS4(nn.Module):
    def __init__(self, c1, c2, shortcut=False, g=1, e=0.5):
        super().__init__()

        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + 4) * self.c, c2, 1)
        self.m1 = Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
        self.m2 = Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
        self.m3 = Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
        self.m4 = Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)

        self.u2 = CARAFE(self.c, self.c, 1, 2)
        self.u3 = CARAFE(self.c, self.c, 2, 4)
        self.u4 = CARAFE(self.c, self.c, 3, 8)
        self.cvm2 = Conv(self.c * 4, self.c, 1, 1)
        self.cvm3 = Conv(self.c * 16, self.c, 1, 1)
        self.cvm4 = Conv(self.c * 64, self.c, 1, 1)

        # self.l1 = LCSM(self.c, 1, 2)
        # self.l2 = LCSM(self.c, 2, 4)
        # self.l3 = LCSM(self.c, 3, 8)

    def forward(self, x):

        y, y0 = self.cv1(x).chunk(2, 1)
        y1 = self.m1(y0)

        y2 = torch.pixel_unshuffle(y1, 2)
        y2 = self.cvm2(y2)
        y2 = self.m2(self.u2(y2) + y1)
        # y2 = self.m2(self.l1(y1))

        y3 = torch.pixel_unshuffle(y2, 4)
        y3 = self.cvm3(y3)
        y3 = self.m3(self.u3(y3) + y2)
        # y3 = self.m3(self.l2(y2))

        y4 = torch.pixel_unshuffle(y3, 8)
        y4 = self.cvm4(y4)
        y4 = self.m4(self.u4(y4) + y3)
        # y4 = self.m4(self.l3(y3))

        return self.cv2(torch.cat((y, y0, y1, y2, y3, y4), 1))


class CLfD(nn.Module):
    def __init__(self, c1, c2, shortcut=False, g=1, e=0.5):
        super().__init__()

        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + 3) * self.c, c2, 1)
        self.m1 = Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
        self.m2 = Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
        self.m3 = Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)

        self.l1 = LCSMDense(self.c)
        self.l2 = LCSMDense(self.c)
        self.l3 = LCSMDense(self.c)

    def forward(self, x):

        y, y0 = self.cv1(x).chunk(2, 1)

        y1 = self.m1(self.l1(y0))
        y2 = self.m2(self.l2(y1))
        y3 = self.m3(self.l3(y2))

        return self.cv2(torch.cat((y, y0, y1, y2, y3), 1))


