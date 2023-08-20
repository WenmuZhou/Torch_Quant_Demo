# -*- coding: utf-8 -*-
# @Time    : 2023/8/20 15:09
# @Author  : zhoujun
import fire
import numpy as np
import onnxruntime


class ONNXModel:
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(
            onnx_path,
            providers=[
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def __call__(self, image_numpy):
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: image_numpy})
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        result = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return result


def infer(onnx_path, input_shape=[1, 3, 224, 224]):
    img = np.random.random(input_shape)
    print("input shape:", img.shape)
    onnx_engine = ONNXModel(onnx_path)
    out = onnx_engine(img.astype(np.float32))
    print(out)


if __name__ == "__main__":
    fire.Fire(infer)
