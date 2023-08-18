import copy
from torch import nn
import torch

class QuantizedModel(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedModel, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = copy.deepcopy(model_fp32)
        self.model_fp32.eval()
        self.model_fp32.fuse_model()
        self.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x
    
    def quantize_qat(self, train_fn, train_args, eval_fn=None, eval_args=None, inplace=False):
        if not inplace:
            model = copy.deepcopy(self)
        model.train()
        torch.ao.quantization.prepare_qat(model, inplace=True)
        # Run qat
        train_fn(model, *train_args)
        # eval
        if eval_fn is not None:
            eval_result = eval_fn(model, *eval_args)
            print(eval_result)
        model.to('cpu')
        # Convert to quantized model
        torch.quantization.convert(model, inplace=True)
        model.eval()
        return model
    
    def quantize_ptq(self, run_fn, run_args, inplace=False):
        if not inplace:
            model = copy.deepcopy(self)
        torch.ao.quantization.prepare(model, inplace=True)
        # Calibrate with the dataset
        run_fn(model, *run_args)
        # Convert to quantized model
        torch.ao.quantization.convert(model.to('cpu'), inplace=True)
        return model