import torch
import torch.nn as nn
from torchvision.models import resnet18
from conv2d_img2col import Conv2dImg2Col


def replace_conv2d_with_custom(module, custom_conv_class=Conv2dImg2Col):

    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):

            custom_conv = custom_conv_class(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                bias=child.bias is not None,
                padding_mode=(
                    child.padding_mode if hasattr(child, "padding_mode") else "zeros"
                ),
            )

            with torch.no_grad():
                custom_conv.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    custom_conv.bias.data = child.bias.data.clone()

            setattr(module, name, custom_conv)
        else:
            replace_conv2d_with_custom(child, custom_conv_class)


model = resnet18(pretrained=True)
model.eval()

original_model = resnet18(pretrained=True)
original_model.eval()

replace_conv2d_with_custom(model)

print("=== Проверка замены слоев ===")
for name, module in model.named_modules():
    if isinstance(module, Conv2dImg2Col):
        print(f"Заменен слой: {name} -> {module}")

x = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    output_custom = model(x)
    output_original = original_model(x)

print(f"Shape custom model output: {output_custom.shape}")
print(f"Shape original model output: {output_original.shape}")
print(f"Outputs are close: {torch.allclose(output_custom, output_original, atol=1e-5)}")

# torch.save(model.state_dict(), 'resnet18_custom_conv.pth')
