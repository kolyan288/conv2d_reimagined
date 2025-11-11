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
    import torch._dynamo as dynamo
    # very dirty trick to work with CamVid model =(
    old = dynamo.is_compiling
    dynamo.is_compiling = lambda : True

    model = prepare_fx( 
        model=model,
        qconfig_mapping=QCONFIG_MAPPING if config is None else config,
        example_inputs=(example_input, ),
    )
    dynamo.is_compiling = old
    return model
