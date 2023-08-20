# Torch量化demo

## 实验说明

使用分类网络进行torch量化实验, 数据集为Cifar10

环境如下：

- torch: 2.0
- 显卡: Tesla T4

参数如下:

- 训练epoch:10
- QAT训练epoch：10
- 图片大小: 64*64

建议优先选择`torch.fx`进行量化，这种方式更友好

启动命令如下
```bash
python main.py resnet50 --fx=true
```
## 数据如下

| 网络         | 模式   | 精度(%) | 速度(CPU/ms) | 大小(MB) |
|------------|------|-------|------------|--------|
| MobileNetV2 | fp32 | 64.72 | 7          | 9.17   |
|            | PTQ  | 64.25 | 2          | 2.63   |
|            | QAT  | 66.89 | 2          | 2.63   |
| ResNet18   | fp32 | 70.75 | 5          | 44.79  |
|            | PTQ  | 70.79 | 3          | 11.31  |
|            | QAT  | 73.71 | 3          | 11.31  |
| ResNet50   | fp32 | 61.41 | 12         | 94.41  |
|            | PTQ  | 45.55 | 6          | 24.10  |
|            | QAT  | 64.54 | 6          | 24.10  |

## onnx推理

训练过程中会自动导出onnx模型，推理方式如下

```bash
python onnx_infer.py model.onnx 1,3,512,32
```
