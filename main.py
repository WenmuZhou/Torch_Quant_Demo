import fire
import torch.cuda

from data_loader import prepare_dataloader
from utils import *


def ptq(
    quant_model, train_loader, test_loader, device, scripted_model_file, input_shape
):
    print("Post Training Quantization", "-" * 10)
    quantized_model = quant_model.quantize_ptq(
        evaluate, [train_loader, device], input_shape=input_shape
    )

    print("Size of model after quantization")
    print_size_of_model(quantized_model)

    eval_top1, eval_top5 = evaluate(quantized_model, test_loader, "cpu")
    print(f"Evaluation accuracy, Top1: {eval_top1:.2f}, Top5: {eval_top5:.2f}")

    to_onnx(quantized_model, input_shape, scripted_model_file + ".onnx")

    script_model = torch.jit.script(quantized_model)
    torch.jit.save(script_model, scripted_model_file)


def qat(
    quant_model,
    train_loader,
    test_loader,
    device,
    scripted_model_file,
    input_shape,
    num_epochs,
):
    print("Quantization-aware training", "-" * 10)
    quantized_model = quant_model.quantize_qat(
        train,
        [train_loader, test_loader, device, 0.01, num_epochs],
        input_shape=input_shape,
    )

    eval_top1, eval_top5 = evaluate(quantized_model, test_loader, "cpu")
    print(f"Evaluation accuracy, Top1: {eval_top1:.2f}, Top5: {eval_top5:.2f}")

    print("Size of model after quantization")
    print_size_of_model(quantized_model)

    to_onnx(quantized_model, input_shape, scripted_model_file + ".onnx")

    script_model = torch.jit.script(quantized_model)
    torch.jit.save(script_model, scripted_model_file)


def main(model_name="MobileNetV2", num_epochs=10, fx=True):
    if fx:
        from models_fx import build_model
        from models_fx.QuantizedModel import QuantizedModel
    else:
        from models_eager import build_model
        from models_eager.QuantizedModel import QuantizedModel
    saved_model_dir = "output/"
    float_model_file = os.path.join(saved_model_dir, "float32.pth")
    scripted_float_model_file = os.path.join(saved_model_dir, "float32_scripted.pth")
    scripted_ptq_model_file = os.path.join(
        saved_model_dir, "ptq_scripted_quantized.pth"
    )
    scripted_qat_model_file = os.path.join(
        saved_model_dir, "qat_scripted_quantized.pth"
    )

    train_loader, test_loader = prepare_dataloader(8, 512, 256)
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    input_shape = [1] + list(test_loader.dataset.__getitem__(0)[0].shape)

    float_model = build_model(model_name, num_classes=10).to("cpu")
    # Train model.
    if not os.path.exists(float_model_file):
        print("Training Model...")
        model = train(float_model, train_loader, test_loader, device, 0.01, num_epochs)
        # Save model.
        save_model(model, saved_model_dir, float_model_file)
    else:
        state_dict = torch.load(float_model_file)
        float_model.load_state_dict(state_dict)
        print("FP32 model already trained, directly load it.")

    print("Size of fp32 model")
    print_size_of_model(float_model)

    eval_top1, eval_top5 = evaluate(float_model, test_loader, device)
    print(f"Evaluation accuracy, {eval_top1:.2f}")
    to_onnx(float_model, input_shape, float_model_file + ".onnx")
    torch.jit.save(torch.jit.script(float_model), scripted_float_model_file)

    ptq(
        QuantizedModel(float_model, backend='x86'),
        train_loader,
        test_loader,
        device,
        scripted_ptq_model_file,
        input_shape,
    )
    qat(
        QuantizedModel(float_model, backend='x86'),
        train_loader,
        test_loader,
        device,
        scripted_qat_model_file,
        input_shape,
        10,
    )

    run_benchmark(scripted_float_model_file, test_loader, "cpu", "fp32")
    run_benchmark(scripted_ptq_model_file, test_loader, "cpu", "ptq int8")
    run_benchmark(scripted_qat_model_file, test_loader, "cpu", "qat int8")


if __name__ == "__main__":
    torch.manual_seed(191009)
    fire.Fire(main)
