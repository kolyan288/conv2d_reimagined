import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


class Conv2dImg2Col(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        padding_mode="zeros",
    ):
        super(Conv2dImg2Col, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        kernel_h, kernel_w = self.kernel_size

        self.weight = Parameter(
            torch.Tensor(out_channels, in_channels, kernel_h, kernel_w)
        )

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self._img2col_conv(x, self.weight, self.bias, self.stride, self.padding)

    def _img2col_conv(self, X, weight, bias=None, stride=1, padding=0):
        N, C_in, H, width = X.shape
        C_out, _, kernel_h, kernel_w = weight.shape

        H_out = (H + 2 * padding - kernel_h) // stride + 1
        W_out = (width + 2 * padding - kernel_w) // stride + 1

        if padding > 0:
            if self.padding_mode == "zeros":
                X_padded = F.pad(X, (padding, padding, padding, padding))
            elif self.padding_mode == "reflect":
                X_padded = F.pad(
                    X, (padding, padding, padding, padding), mode="reflect"
                )
            elif self.padding_mode == "replicate":
                X_padded = F.pad(
                    X, (padding, padding, padding, padding), mode="replicate"
                )
            else:
                raise NotImplementedError(
                    f"Padding mode {self.padding_mode} not implemented"
                )
        else:
            X_padded = X

        patches = X_padded.unfold(2, kernel_h, stride).unfold(3, kernel_w, stride)

        patches_reshaped = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches_reshaped = patches_reshaped.view(N * H_out * W_out, -1)

        weight_reshaped = weight.view(C_out, -1)

        output = torch.mm(patches_reshaped, weight_reshaped.t())

        if bias is not None:
            output += bias.unsqueeze(0)

        output = output.view(N, H_out, W_out, C_out).permute(0, 3, 1, 2)

        return output

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != 0:
            s += ", padding={padding}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)
