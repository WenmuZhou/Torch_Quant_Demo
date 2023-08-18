import os
import sys
import time
import numpy as np
import copy

import torch
import torch.nn as nn

from models.mobilenetv2 import MobileNetV2
from models.resnet import resnet18, resnet50
from models.QuantizedModel import QuantizedModel

from utils import *
from data_loader import prepare_dataloader

def ptq(fp32_model, train_loader, test_loader, device, scripted_float_model_file):
    print('Post Training Quantization', '-'*10)
    quant_model = QuantizedModel(fp32_model)
    quantized_model = quant_model.quantize_ptq(evaluate, [test_loader, device])

    print("Size of model after quantization")
    print_size_of_model(quantized_model)

    eval_top1, eval_top5  = evaluate(quantized_model, test_loader, 'cpu')
    print(f'Evaluation accuracy, Top1: {eval_top1:.2f}, Top5: {eval_top5:.2f}')

    torch.jit.save(torch.jit.script(quantized_model), scripted_float_model_file)

def qat(fp32_model, train_loader, test_loader, device, scripted_float_model_file):
    print('Quantization-aware training', '-'*10)
    quant_model = QuantizedModel(fp32_model)

    quantized_model = quant_model.quantize_qat(train_model, [train_loader, test_loader, device, 0.0001, 10])
    
    eval_top1, eval_top5  = evaluate(quantized_model, test_loader, 'cpu')
    print(f'Evaluation accuracy, Top1: {eval_top1:.2f}, Top5: {eval_top5:.2f}')

    print("Size of model after quantization")
    print_size_of_model(quantized_model)

    torch.jit.save(torch.jit.script(quantized_model), scripted_float_model_file)

def main():
    saved_model_dir = 'output/'
    float_model_file = os.path.join(saved_model_dir, 'float32.pth')
    scripted_float_model_file = os.path.join(saved_model_dir, 'float32_scripted.pth')
    scripted_ptq_model_file = os.path.join(saved_model_dir, 'ptq_scripted_quantized.pth')
    scripted_qat_model_file = os.path.join(saved_model_dir, 'qat_scripted_quantized.pth')

    train_loader, test_loader = prepare_dataloader(num_workers=8, train_batch_size=512, eval_batch_size=256)
    cuda_device = torch.device("cuda:0")
    
    float_model = MobileNetV2(num_classes=10).to('cpu')
    # float_model = resnet50(num_classes=10).to('cpu')
    # Train model.
    if not os.path.exists(float_model_file):
        print("Training Model...")
        model = train_model(model=float_model, train_loader=train_loader, test_loader=test_loader,
                            device=cuda_device, learning_rate=0.01, num_epochs=10)
        # Save model.
        save_model(model, saved_model_dir, float_model_file)
    else:
        state_dict = torch.load(float_model_file)
        float_model.load_state_dict(state_dict)
        print('FP32 model already trained, directly load it.')

    print("Size of fp32 model")
    print_size_of_model(float_model)

    eval_top1, eval_top5  = evaluate(float_model, test_loader, cuda_device)
    print(f'Evaluation accuracy, {eval_top1:.2f}')
    torch.jit.save(torch.jit.script(float_model), scripted_float_model_file)

    ptq(float_model, train_loader, test_loader, cuda_device, scripted_ptq_model_file)
    qat(float_model, train_loader, test_loader, cuda_device, scripted_qat_model_file)

    run_benchmark(scripted_float_model_file, test_loader, 'cpu', 'fp32')
    run_benchmark(scripted_ptq_model_file, test_loader, 'cpu', 'ptq int8')
    run_benchmark(scripted_qat_model_file, test_loader, 'cpu', 'qat int8')

if __name__ == "__main__":
    torch.manual_seed(191009)
    main()