import os
import time

import torch
from torch import nn, optim


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, data_loader, device):
    model.to(device)
    model.eval()
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    with torch.no_grad():
        for image, target in data_loader:
            image = image.to(device)
            target = target.to(device)
            output = model(image)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
    return top1.avg, top5.avg


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p") / 1e6)
    os.remove("temp.p")


def save_model(model, model_dir, model_filepath):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_filepath)


def train_one_epoch(model, criterion, optimizer, data_loader, device):
    model.train()
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    avgloss = AverageMeter("Loss", "1.5f")

    cnt = 0
    for image, target in data_loader:
        start_time = time.time()
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
    return avgloss.avg, top1.avg, top5.avg


def train(model, train_loader, test_loader, device, learning_rate=1e-1, num_epochs=200):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1, last_epoch=-1
    )

    for epoch in range(num_epochs):
        # Training
        train_loss, train_top1, train_top5 = train_one_epoch(
            model, criterion, optimizer, train_loader, device
        )
        # Evaluation
        eval_top1, eval_top5 = evaluate(model, test_loader, device)

        print(
            f"Epoch: {epoch:03d}/{num_epochs:03d} Train Loss: {train_loss:.3f} Train Acc: {train_top1:.3f} Eval Top1: {eval_top1:.3f} Eval Top5: {eval_top5:.3f} Lr: {scheduler.get_last_lr()[0]:.4f}"
        )

        # Set learning rate scheduler
        scheduler.step()

    return model


def run_benchmark(model_file, img_loader, device, desc="fp32"):
    elapsed = 0
    model = torch.jit.load(model_file, map_location=device)
    model.to(device)
    model.eval()
    num_batches = 100
    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(img_loader):
        if i < num_batches:
            start = time.time()
            output = model(images.to(device))
            end = time.time()
            elapsed = elapsed + (end - start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print(f"{desc} elapsed time: {elapsed / num_images * 1000:3.0f} ms")
    return elapsed


def to_onnx(model, input_shape, sava_path="model.onnx"):
    dummy_input = torch.randn(*input_shape, device="cpu")
    torch.onnx.export(
        model.to("cpu"),
        dummy_input,
        sava_path,
        input_names=["input"],
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size", 2: "in_width", 3: "int_height"},
            "output": {0: "batch_size", 2: "out_width", 3: "out_height"},
        },
    )


def script_to_onnx(model_path, input_shape, sava_path=None):
    model = torch.jit.load(model_path).to("cpu")
    dummy_input = torch.randn(*input_shape, device="cpu")
    if sava_path is None:
        sava_path = model_path + ".onnx"
    torch.onnx.export(
        model,
        dummy_input,
        sava_path,
        input_names=["input"],
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size", 2: "in_width", 3: "int_height"},
            "output": {0: "batch_size", 2: "out_width", 3: "out_height"},
        },
    )
