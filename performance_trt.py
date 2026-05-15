from typing import List

import cv2
import numpy as np
import psutil
import os
import time
import subprocess
import psutil
import glob
import argparse
import torch
import torch.nn as nn
from models.experimental import attempt_load
from utils.datasets import create_dataloader, letterbox
from utils.general import check_img_size, colorstr
from utils.torch_utils import select_device

import tensorrt as trt
import pycuda.driver as cuda


class TRTModelWrapper:
    def __init__(self, engine_path, device):
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

    def _allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # Pin host memory and allocate device VRAM buffers
            host_mem = cuda.pagelocked_empty(trt.volume(shape), dtype=dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))
            binding_dict = {"host": host_mem,
                            "device": device_mem, "shape": shape, "dtype": dtype}

            if self.engine.binding_is_input(binding):
                self.input_shape = shape
                inputs.append(binding_dict)
            else:
                outputs.append(binding_dict)

        return inputs, outputs, bindings, stream

    def __call__(self, im_tensor):
        """
        Adapts PyTorch DataLoader tensors into TensorRT memory bindings.
        Handles both FP32, FP16, and INT8 engines automatically at execution time.
        """
        # Ensure tensor is float32 on CPU as a contiguous numpy array
        if im_tensor.is_cuda:
            im_tensor = im_tensor.cpu()
        input_data = im_tensor.numpy().astype(np.float32)

        # 1. Fill Host Buffer
        np.copyto(self.inputs[0]["host"], input_data.ravel())

        # 2. Upload Host Memory to GPU Device Buffer
        cuda.memcpy_htod_async(
            self.inputs[0]["device"], self.inputs[0]["host"], self.stream)

        # 3. Compute Quantized Graph
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle)

        # 4. Download Device Memory Buffers back to Host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)

        self.stream.synchronize()

        # 5. Extract fixed shapes from end-to-end engine outputs
        # Struct: [num_dets, detection_boxes, detection_scores, detection_classes]
        num_dets = self.outputs[0]["host"].reshape(self.outputs[0]["shape"])
        boxes = self.outputs[1]["host"].reshape(self.outputs[1]["shape"])
        scores = self.outputs[2]["host"].reshape(self.outputs[2]["shape"])
        classes = self.outputs[3]["host"].reshape(self.outputs[3]["shape"])

        batch_size = input_data.shape[0]
        wrapped_predictions = []

        # Convert absolute model output metrics back to yolov7 test.py legacy format
        for b in range(batch_size):
            valid_count = int(num_dets[b][0] if len(
                num_dets.shape) > 1 else num_dets[b])
            if valid_count == 0:
                wrapped_predictions.append(
                    torch.zeros((0, 6), device=self.device))
                continue

            # Slice current batch element arrays
            b_boxes = boxes[b][:valid_count]      # [N, 4] -> (x1, y1, x2, y2)
            b_scores = scores[b][:valid_count]    # [N]
            b_classes = classes[b][:valid_count]  # [N]

            # Reconstruct legacy tracking metrics layout matrix: [x1, y1, x2, y2, conf, class_id]
            pred_matrix = np.zeros((valid_count, 6), dtype=np.float32)
            pred_matrix[:, :4] = b_boxes
            pred_matrix[:, 4] = b_scores
            pred_matrix[:, 5] = b_classes

            wrapped_predictions.append(
                torch.from_numpy(pred_matrix).to(self.device))

        # Return structured list mimicking post-NMS PyTorch tensors
        return [wrapped_predictions]


def start_power_monitor(power_file):
    power_monitor = subprocess.Popen(
        ['python3', 'power_trt.py', '--power_file', power_file])
    return power_monitor


def read_power_stats(power_file):
    time.sleep(1)  # make sure the power monitor file is ready
    value = 0
    with open(power_file, 'r') as f:
        value = float(f.read().strip())
    return value


def read_power_on_idle():
    power_monitor = start_power_monitor('power_on_idle.txt')
    time.sleep(5)
    power_monitor.terminate()
    power_on_idle = read_power_stats('power_on_idle.txt')
    return power_on_idle


def start_cpu_monitor(cpu_file):
    monitor = subprocess.Popen(
        ['python3', 'cpu_usage.py', '--cpu_file', cpu_file])
    return monitor


def read_cpu_stats(cpu_file):
    time.sleep(1)  # make sure the monitor file is ready
    value = 0
    with open(cpu_file, 'r') as f:
        value = float(f.read().strip())
    return value


def read_cpu_on_idle():
    monitor = start_cpu_monitor('cpu_on_idle.txt')
    time.sleep(5)
    monitor.terminate()
    value = read_cpu_stats('cpu_on_idle.txt')
    return value


def get_image_paths(image):
    print(f"Reading images from: {image}")
    paths = []
    if os.path.isdir(image):
        exts = {'*.jpg', '*.jpeg', '*.png'}
        for ext in exts:
            paths.extend(glob.glob(os.path.join(image, ext)))
    else:
        paths = [image]
    print(f"Found {len(paths)} images.")
    return paths


def preprocess(image_path, ):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error Image: {image_path}")
        return None

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0
    if img.ndim == 3:
        img = np.expand_dims(img, 0)
    return img


if __name__ == '__main__':
    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str,
                        required=True, help='Engine Path')
    parser.add_argument(
        '--image', type=str, default='test_image.jpg', help='Image file or directory')
    parser.add_argument('--runs', type=int, default=1,
                        help='How many times to run per image')
    opt = parser.parse_args()
    print("Arguments parsed.")

    process = psutil.Process(os.getpid())

    # power idle
    power_on_idle = read_power_on_idle()

    # cpu idle
    cpu_on_idle = read_cpu_on_idle()

    # 2. Load the model
    batch_size = 1
    device = select_device("cpu", batch_size=batch_size)

    # memory usage
    mem_before_load = process.memory_info().rss / (1024 * 1024)

    # TRT Model
    model_trt = TRTModelWrapper(opt.engine, device)

    mem_after_load = process.memory_info().rss / (1024 * 1024)

    engine_size = os.path.getsize(opt.engine) / (1024 * 1024)

    print("Model loaded.")

    # 3. Run inference and measure performance

    # read all image paths in the folder
    image_paths = get_image_paths(opt.image)

    print("Warming up the model...")
    img = preprocess(image_paths[0])
    out = model_trt(img)

    print("Running...")

    # start power monitor
    power_monitor = start_power_monitor('power_on_process.txt')
    power_start_time = time.perf_counter()

    # start cpu monitor
    cpu_monitor = start_cpu_monitor('cpu_on_process.txt')

    times = []
    for image_path in image_paths:
        img = preprocess(image_path)
        if img is not None:
            print(f"Test: {image_path} (running {opt.runs} times)...")
            for _ in range(opt.runs):
                start_time = time.perf_counter()

                model_trt(img)

                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)

    power_monitor.terminate()
    cpu_monitor.terminate()

    avg_inference_time = np.mean(times)

    power_end_time = time.perf_counter()
    power_usage_time = (power_end_time - power_start_time) / \
        3600  # from s to h
    power_on_process = read_power_stats('power_on_process.txt')
    power_for_process = (power_on_process - power_on_idle)
    power_consumption = power_for_process * \
        power_usage_time  # power consumption (mWh) = mW * h

    cpu_on_process = read_cpu_stats('cpu_on_process.txt')

    # 4. Print results
    print(f"\n--- {opt.model} Experiment Results ---")
    print(f"- Engine Size: {engine_size:.4f} MB")
    print(f"- RAM Footprint: {(mem_after_load - mem_before_load):.4f} MB")
    print(f"- Inference Time: {avg_inference_time:.4f} ms")
    print(
        f"- Power Consumption: {power_consumption:.4f} mWh [Power for Inference: {power_for_process:.4f} mW, Time for Inference: {power_usage_time:.4f} h]")
    print(f"- CPU Usage: {(cpu_on_process - cpu_on_idle):.4f} %")
    print("--------------------------------------------------------------------")
