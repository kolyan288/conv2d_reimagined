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


def test_custom_conv_layer():
    print("=== Тестирование корректности прямого прохода ===")

    custom_conv = Conv2dImg2Col(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
    standard_conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)

    custom_conv.weight.data = standard_conv.weight.data.clone()
    custom_conv.bias.data = standard_conv.bias.data.clone()

    X = torch.randn(2, 3, 8, 8, requires_grad=True)

    output_custom = custom_conv(X)
    output_standard = standard_conv(X)

    print(f"Custom Conv2dImg2Col shape: {output_custom.shape}")
    print(f"Standard nn.Conv2d shape: {output_standard.shape}")
    print(
        f"Максимальное расхождение: {torch.max(torch.abs(output_custom - output_standard))}"
    )
    print(
        f"Результаты равны: {torch.allclose(output_custom, output_standard, atol=1e-6)}"
    )

    print("\n=== Тестирование обратного прохода и обучения ===")

    custom_conv_train = Conv2dImg2Col(
        3, 16, kernel_size=3, stride=1, padding=1, bias=True
    )
    standard_conv_train = nn.Conv2d(
        3, 16, kernel_size=3, stride=1, padding=1, bias=True
    )

    with torch.no_grad():
        custom_conv_train.weight.data = standard_conv_train.weight.data.clone()
        custom_conv_train.bias.data = standard_conv_train.bias.data.clone()

    X_train = torch.randn(2, 3, 8, 8, requires_grad=True)

    output_custom_train = custom_conv_train(X_train)
    output_standard_train = standard_conv_train(X_train)

    target = torch.randn_like(output_custom_train)

    optimizer_custom = torch.optim.SGD(custom_conv_train.parameters(), lr=0.01)
    loss_custom = F.mse_loss(output_custom_train, target)

    loss_custom.backward()

    print(
        f"Custom conv - Градиент weights: {custom_conv_train.weight.grad is not None}"
    )
    print(f"Custom conv - Градиент bias: {custom_conv_train.bias.grad is not None}")
    print(f"Custom conv - Градиент input: {X_train.grad is not None}")

    optimizer_custom.step()
    print("Custom conv - Оптимизация прошла успешно")

    optimizer_standard = torch.optim.SGD(standard_conv_train.parameters(), lr=0.01)
    loss_standard = F.mse_loss(output_standard_train, target)
    loss_standard.backward()

    print(
        f"Standard conv - Градиент weights: {standard_conv_train.weight.grad is not None}"
    )
    print(f"Standard conv - Градиент bias: {standard_conv_train.bias.grad is not None}")

    optimizer_standard.step()
    print("Standard conv - Оптимизация прошла успешно")

    print("\n=== Тестирование различных конфигураций ===")

    configs = [
        (3, 16, 3, 1, 0),
        (3, 16, 5, 2, 2),
        (3, 16, 1, 1, 0),
    ]

    for in_ch, out_ch, k, s, p in configs:
        custom_conv = Conv2dImg2Col(in_ch, out_ch, k, s, p, bias=True)
        standard_conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=True)

        with torch.no_grad():
            custom_conv.weight.data = standard_conv.weight.data.clone()
            custom_conv.bias.data = standard_conv.bias.data.clone()

        X_test = torch.randn(2, in_ch, 16, 16)
        out_custom = custom_conv(X_test)
        out_standard = standard_conv(X_test)

        is_equal = torch.allclose(out_custom, out_standard, atol=1e-6)
        print(f"Config in={in_ch}, out={out_ch}, k={k}, s={s}, p={p}: {is_equal}")

def test_torch_trace():
    model = nn.Sequential(
        Conv2dImg2Col(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
    )
    input = torch.randn(2, 3, 8, 8, requires_grad=True)
    torch.jit.trace(model, input)
    print(f"Trace passed .... \n")
    # torch.jit.script(model)
    # print(f"Scripting passed .... \n")

if __name__ == "__main__":
    test_torch_trace()
    test_custom_conv_layer()
    
