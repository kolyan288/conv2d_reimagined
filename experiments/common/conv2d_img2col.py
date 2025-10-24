import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


class Img2ColConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, bias, stride, padding, padding_mode):
        ctx.save_for_backward(X, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.padding_mode = padding_mode

        N, C_in, H, W = X.shape
        C_out, _, kernel_h, kernel_w = weight.shape

        if isinstance(padding, int):
            ph, pw = padding, padding
        else:
            ph, pw = padding
        if isinstance(stride, int):
            sh, sw = stride, stride
        else:
            sh, sw = stride

        H_out = (H + 2 * ph - kernel_h) // sh + 1
        W_out = (W + 2 * pw - kernel_w) // sw + 1

        if ph > 0 or pw > 0:
            if padding_mode == "zeros":
                X_padded = F.pad(X, (pw, pw, ph, ph))
            elif padding_mode == "reflect":
                X_padded = F.pad(X, (pw, pw, ph, ph), mode="reflect")
            elif padding_mode == "replicate":
                X_padded = F.pad(X, (pw, pw, ph, ph), mode="replicate")
            else:
                raise NotImplementedError(
                    f"Padding mode {padding_mode} not implemented"
                )
        else:
            X_padded = X

        patches = X_padded.unfold(2, kernel_h, sh).unfold(3, kernel_w, sw)
        patches_reshaped = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches_reshaped = patches_reshaped.view(N * H_out * W_out, -1)

        weight_reshaped = weight.view(C_out, -1)
        output = torch.mm(patches_reshaped, weight_reshaped.t())

        if bias is not None:
            output += bias.unsqueeze(0)

        output = output.view(N, H_out, W_out, C_out).permute(0, 3, 1, 2)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        X, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        padding_mode = ctx.padding_mode

        N, C_in, H, W = X.shape
        C_out, _, kernel_h, kernel_w = weight.shape

        if isinstance(padding, int):
            ph, pw = padding, padding
        else:
            ph, pw = padding
        if isinstance(stride, int):
            sh, sw = stride, stride
        else:
            sh, sw = stride

        H_out = (H + 2 * ph - kernel_h) // sh + 1
        W_out = (W + 2 * pw - kernel_w) // sw + 1

        grad_output_reshaped = (
            grad_output.permute(0, 2, 3, 1).contiguous().view(N * H_out * W_out, C_out)
        )

        if bias is not None:
            grad_bias = grad_output.sum(dim=[0, 2, 3])
        else:
            grad_bias = None

        if ph > 0 or pw > 0:
            if padding_mode == "zeros":
                X_padded = F.pad(X, (pw, pw, ph, ph))
            elif padding_mode == "reflect":
                X_padded = F.pad(X, (pw, pw, ph, ph), mode="reflect")
            elif padding_mode == "replicate":
                X_padded = F.pad(X, (pw, pw, ph, ph), mode="replicate")
        else:
            X_padded = X

        patches = X_padded.unfold(2, kernel_h, sh).unfold(3, kernel_w, sw)
        patches_reshaped = (
            patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(N * H_out * W_out, -1)
        )

        grad_weight_reshaped = torch.mm(grad_output_reshaped.t(), patches_reshaped)
        grad_weight = grad_weight_reshaped.view(C_out, C_in, kernel_h, kernel_w)

        weight_reshaped = weight.view(C_out, -1)
        grad_patches = torch.mm(grad_output_reshaped, weight_reshaped)
        grad_patches_reshaped = grad_patches.view(
            N, H_out, W_out, C_in, kernel_h, kernel_w
        )
        grad_patches_permuted = grad_patches_reshaped.permute(
            0, 3, 1, 2, 4, 5
        ).contiguous()
        grad_patches_flat = grad_patches_permuted.view(
            N, C_in * kernel_h * kernel_w, H_out * W_out
        )

        fold = torch.nn.Fold(
            output_size=(X_padded.shape[2], X_padded.shape[3]),
            kernel_size=(kernel_h, kernel_w),
            stride=(sh, sw),
            padding=0,
            dilation=1,
        )
        grad_input_padded = fold(grad_patches_flat)

        if ph > 0 or pw > 0:
            grad_input = grad_input_padded[:, :, ph:-ph, pw:-pw]
        else:
            grad_input = grad_input_padded

        return grad_input, grad_weight, grad_bias, None, None, None


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

        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        if isinstance(self.padding, int):
            self.padding = (self.padding, self.padding)

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
        return Img2ColConvFunction.apply(
            x, self.weight, self.bias, self.stride, self.padding, self.padding_mode
        )

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0, 0):
            s += ", padding={padding}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)
