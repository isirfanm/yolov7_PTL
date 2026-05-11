import argparse

import torch

import tensorrt as trt
import os
# import pycuda.driver as cuda
# import pycuda.autoinit
import numpy as np
import cv2

# --- INT8 Calibrator ---


class YOLOv7EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, training_data_path, cache_file, batch_size=8, input_shape=(640, 640)):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.imgs = [os.path.join(training_data_path, f) for f in os.listdir(training_data_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.batch_count = len(self.imgs) // self.batch_size
        self.current_index = 0
        # # Allocate GPU memory for a single batch
        # self.device_input = cuda.mem_alloc(
        #     self.batch_size * 3 * input_shape[0] * input_shape[1] * 4)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= self.batch_count:
            return None

        batch_imgs = []
        for i in range(self.batch_size):
            img_path = self.imgs[self.current_index * self.batch_size + i]
            img = cv2.imread(img_path)
            img = cv2.resize(img, self.input_shape)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
            batch_imgs.append(np.ascontiguousarray(img))

        batch_data = np.concatenate(batch_imgs).ravel()
        # cuda.memcpy_htod(self.device_input, batch_data)
        self.current_index += 1
        # return [int(self.device_input)]

        # Allocate and copy
        batch_tensor = torch.from_numpy(batch_data).cuda()
        return [batch_tensor.data_ptr()]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# --- Engine Builder ---


def build_engine(onnx_file_path, engine_file_path, calibration_data_path, calib_cache="yolov7_int8.cache"):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)

    # Create Network and Parser
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    if not os.path.exists(onnx_file_path):
        print(f"File {onnx_file_path} not found.")
        return

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    config = builder.create_builder_config()
    # Workspace size: 1GB is usually the safe limit for Nano
    config.max_workspace_size = 1 << 30

    # Enable INT8 and set the calibrator
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = YOLOv7EntropyCalibrator(
        calibration_data_path, calib_cache)

    print("Building INT8 engine on Jetson Nano. This can take 10-30 minutes...")
    engine = builder.build_engine(network, config)

    if engine:
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        print(f"Success! Engine saved to {engine_file_path}")
    else:
        print("Failed to build engine.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='trt_int8_calib.py')
    parser.add_argument('--onnx', type=str, default='yolov7.onnx', help='ONNX file path')
    parser.add_argument('--engine', type=str, default='yolov7.engine', help='Engine file path')
    parser.add_argument('--data', type=str, default='dataset/images', help='Images path')
    parser.add_argument('--cache', type=str, default='yolov7_int8.cache', help='Calib cache file')
    opt = parser.parse_args()
    
    # ONNX_PATH = "quantization/model/yolov7-tiny-base.onnx"
    ONNX_PATH = opt.onnx
    # ENGINE_PATH = "quantization/model/yolov7-tiny-base_int8.engine"
    ENGINE_PATH = opt.engine
    # DATA_PATH = "../furnas_dataset_v0.07/split_dataset/val/images"
    DATA_PATH = opt.data
    CALIB_CACHE=opt.cache

    build_engine(ONNX_PATH, ENGINE_PATH, DATA_PATH, CALIB_CACHE)
