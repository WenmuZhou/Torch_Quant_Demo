import copy

import torch
from torch import nn
from torch.ao.quantization import quantize_fx


class QuantizedModel(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedModel, self).__init__()
        # FP32 model
        self.model_fp32 = copy.deepcopy(model_fp32)
        self.model_fp32.eval()
        self.qconfig = {"": torch.ao.quantization.get_default_qat_qconfig("x86")}

    def quantize_qat(
        self,
        train_fn,
        train_args,
        input_shape,
        eval_fn=None,
        eval_args=None,
        inplace=False,
        **kwargs
    ):
        if not inplace:
            model = copy.deepcopy(self.model_fp32)
        else:
            model = self.model_fp32
        model.train()
        example_inputs = torch.randn(input_shape)
        model = quantize_fx.prepare_qat_fx(model, self.qconfig, example_inputs)
        # Run qat
        train_fn(model, *train_args)
        # eval
        if eval_fn is not None:
            eval_result = eval_fn(model, *eval_args)
            print(eval_result)
        model.to("cpu")
        # Convert to quantized model
        model.eval()
        model = quantize_fx.convert_fx(model)
        return model

    def quantize_ptq(self, run_fn, run_args, input_shape, inplace=False, **kwargs):
        if not inplace:
            model = copy.deepcopy(self.model_fp32)
        else:
            model = self.model_fp32
        model.eval()
        example_inputs = torch.randn(input_shape)
        model = quantize_fx.prepare_fx(model, self.qconfig, example_inputs)
        # Calibrate with the dataset
        run_fn(model, *run_args)
        # Convert to quantized model
        model = quantize_fx.convert_fx(model.to("cpu"))
        return model
