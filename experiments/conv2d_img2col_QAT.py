import torch
import copy
from common.conv2d_img2col import Conv2dImg2Col
import torch.nn as nn
import torch.optim as optim
from torch.quantization.quantize_fx import prepare_qat_fx
from torch.quantization.quantize_fx import convert_fx
from torch.quantization.quantize_fx import prepare_fx
from torch.quantization.fuser_method_mappings import fuse_conv_bn
from torch.ao.quantization import get_default_qat_qconfig_mapping
from src.models.dummy import DummyModel

torch.quantization.fuser_method_mappings._DEFAULT_OP_LIST_TO_FUSER_METHOD.update(
    {(Conv2dImg2Col, nn.BatchNorm2d): fuse_conv_bn}
)
# from torch.quantization.quantization_mappings
QCONFIG_MAPPING = get_default_qat_qconfig_mapping("x86")
# QCONFIG_MAPPING = get_default_qat_qconfig_mapping("qnnpack")


class QAT(nn.Module):

    def __init__(self, model):
        super().__init__()
        print("Wrapping model with QuantStub and DeQuantStub ...")
        self.model = model
        self.quant = torch.quantization.QuantStub()
        self.de_quant = torch.quantization.DeQuantStub()

        # self.nc = self.model.head.nc
        # self.no = self.model.head.no
        # self.stride = self.model.stride

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)

        # for i in range(len(x)):
        #     x[i] = self.de_quant(x[i])
        return self.de_quant(x)



def setup_qat_for_model(model, example_input, config = None):
    """
    Apply QAT to the model with custom configuration
    """
    # model = QAT(model)
    # model.qconfig = torch.quantization.get_default_qconfig("x86")  # ("qnnpack")
    # torch.quantization.prepare_qat(model, inplace=True)
    model = prepare_fx(  # d prepare_fx  prepare_qat_fx
        model=model,
        qconfig_mapping=QCONFIG_MAPPING if config is None else config,
        example_inputs=(example_input, ),
    )
    return model


def train_step(model, data, target, optimizer, criterion):
    """
    Single training step
    """
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()


def create_dummy_dataloader(batch_size=32, num_batches=100, device="cuda:0"):
    for _ in range(num_batches):
        data = torch.randn(batch_size, 3, 32, 32).to(device)  # CIFAR-10 like
        target = torch.randint(0, 10, (batch_size,)).to(device)
        yield data, target


def export_onnx(model, input, suffix=""):
    with torch.no_grad():
        torch.onnx.export(model, input, f=f"model_{suffix}.onnx")


def train():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dummy model
    print("Creating dummy model with custom Conv2dImg2Col layers...")
    model = DummyModel().to(device)
    model.eval()

    for name, module in model.named_modules():
        if isinstance(module, Conv2dImg2ColWithBN):
            # Fuse conv and bn within the custom block
            torch.ao.quantization.fuse_modules(module, [["conv", "bn"]], inplace=True)
            print(f"Fused Conv-BN in block: {name}")

    torch.save(model.state_dict(), "dummy_model_fp32.pth")
    print(
        f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Apply QAT
    print("Applying Quantization-Aware Training...")
    model.train()
    example_input = next(iter(create_dummy_dataloader()))[0]
    model = setup_qat_for_model(model, example_input)
    model.train()

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("Starting QAT training...")
    num_epochs = 1

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(create_dummy_dataloader()):
            loss = train_step(model, data, target, optimizer, criterion)
            epoch_loss += loss
            num_batches += 1

            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss:.4f}")

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

    # Convert to quantized model
    print("Converting to quantized model...")
    model.eval()
    # Convert to quantized model
    save = copy.deepcopy(model)
    save.eval()
    save.to(torch.device("cpu"))
    save = convert_fx(save.cpu())
    # torch.ao.quantization.convert(save.cpu(), inplace=True)
    # torch.jit.script(save)
    example_input = data.to(torch.device("cpu"))
    save = torch.jit.trace(save, example_input)
    export_onnx(save, example_input, "traced")
    torch.jit.save(save, "model-jit-trace.pt")
    torch.cuda.empty_cache()
    print("Model quantization complete!")
    print("Models saved!")


if __name__ == "__main__":
    train()
