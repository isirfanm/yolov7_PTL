import cv2
import numpy as np
import psutil
import os
import time
import subprocess
import glob
import argparse
import torch
import torch.nn as nn
from models.experimental import attempt_load
from utils.datasets import create_dataloader, letterbox
from utils.general import check_img_size, colorstr
from utils.torch_utils import select_device
import xir
import vart


def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert root_subgraph is not None, "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [cs for cs in child_subgraphs if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"]


def runDPU(dpu, img):
    """get tensor"""
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim_0 = tuple(outputTensors[0].dims)
    output_ndim_1 = tuple(outputTensors[1].dims)
    output_ndim_2 = tuple(outputTensors[2].dims)

    batchSize = img.shape[0]
    count = 0

    outputData = [
        np.empty(output_ndim_0, dtype=np.int8, order="C"),
        np.empty(output_ndim_1, dtype=np.int8, order="C"),
        np.empty(output_ndim_2, dtype=np.int8, order="C"),
    ]

    """prepare batch input/output """
    inputData = []
    inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]

    """init input image to input buffer """
    # ? imageRun defined but never used?
    # for i in range(runSize):
    for i in range(batchSize):
        imageRun = inputData[0]
        count = count+1
        try:
            imageRun[i, ...] = img[i].reshape(input_ndim[1:])
        except:
            print(colorstr("red", "ERROR: Shape mismatch"),
                  colorstr("red", count), img.shape)
    """run """
    # job_id = dpu.execute_async(inputData, outputData)
    job_id = dpu.execute_async(imageRun, outputData)

    dpu.wait(job_id)

    # output scaling
    outputData[0] = torch.from_numpy(outputData[0].astype(
        np.float32) / (2 ** outputTensors[0].get_attr("fix_point"))).permute(0, 3, 1, 2)
    outputData[1] = torch.from_numpy(outputData[1].astype(
        np.float32) / (2 ** outputTensors[1].get_attr("fix_point"))).permute(0, 3, 1, 2)
    outputData[2] = torch.from_numpy(outputData[2].astype(
        np.float32) / (2 ** outputTensors[2].get_attr("fix_point"))).permute(0, 3, 1, 2)

    return outputData


def forward_detect(model_detect, x):
    m = model_detect.model[0]
    x = m(x)  # run
    return x


def start_power_monitor(power_file):
    power_monitor = subprocess.Popen(
        ['python3', 'power_via.py', '--power_file', power_file])
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


def preprocess(image_path, input_scale):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error Image: {image_path}")
        return None
    img = np.expand_dims(img, 0)

    # Input scaling
    # img_DPU.shape = batch size, height, width, channels
    img = img.permute(0, 2, 3, 1).float().numpy() / 255 * input_scale
    img = img.astype(np.int8)
    img = torch.from_numpy(img)

    return img


if __name__ == '__main__':
    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model Path')
    parser.add_argument('--xmodel', type=str,
                        required=True, help='XModel Path')
    parser.add_argument(
        '--image', type=str, default='test_image.jpg', help='Image file or directory')
    parser.add_argument('--runs', type=int, default=1,
                        help='How many times to run per image')
    opt = parser.parse_args()

    # 2. Load the model
    batch_size = 1
    device = select_device("cpu", batch_size=batch_size)

    power_on_idle = read_power_on_idle()

    process = psutil.Process(os.getpid())
    mem_before_load = process.memory_info().rss / (1024 * 1024)

    # Detect model on cpu
    model = attempt_load(opt.model, map_location=device)  # load FP32 model
    model.model = nn.Sequential(model.model[-1])  # Last layer

    # Create DPU runner
    g = xir.Graph.deserialize(opt.xmodel)
    subgraphs = get_child_subgraph_dpu(g)
    dpu_runner = vart.Runner.create_runner(subgraphs[0], "run")

    mem_after_load = process.memory_info().rss / (1024 * 1024)

    model_size = os.path.getsize(opt.model) / (1024 * 1024)
    xmodel_size = os.path.getsize(opt.xmodel) / (1024 * 1024)

    # 3. Run inference and measure performance

    # detect model cpu
    model.eval()

    # input scaling
    input_fixpos = dpu_runner.get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2**input_fixpos

    # read all image paths in the folder
    image_paths = get_image_paths(opt.image)

    print("Warming up the model...")
    img = preprocess(image_paths[0], input_scale)
    out_DPU = runDPU(dpu_runner, img)
    out, train_out = forward_detect(model, out_DPU)

    # start power monitor
    power_monitor = start_power_monitor('power_on_process.txt')
    power_start_time = time.perf_counter()

    times = []
    for image_path in image_paths:
        img = preprocess(image_path, input_scale)
        if img is not None:
            print(f"Test: {image_path} (running {opt.runs} times)...")
            for _ in range(opt.runs):
                start_time = time.perf_counter()

                out_DPU = runDPU(dpu_runner, img)
                forward_detect(model, out_DPU)

                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)

    power_monitor.terminate()
    avg_inference_time = np.mean(times)

    power_end_time = time.perf_counter()
    power_usage_time = (power_end_time - power_start_time) / \
        3600  # from s to h
    power_on_process = read_power_stats('power_on_process.txt')
    power_for_process = (power_on_process - power_on_idle)
    power_consumption = power_for_process * \
        power_usage_time  # power consumption (mWh) = mW * h

    # 4. Print results
    print(f"\n--- {opt.model} Experiment Results ---")
    print(f"- Model Size: {model_size:.4f} MB")
    print(f"- XModel Size: {xmodel_size:.4f} MB")
    print(f"- RAM Footprint: {(mem_after_load - mem_before_load):.4f} MB")
    print(f"- Inference Time: {avg_inference_time:.4f} ms")
    print(
        f"- Power Consumption: {power_consumption:.4f} mWh [Power for Inference: {power_for_process:.4f} mW, Time for Inference: {power_usage_time:.4f} h]")
    print("--------------------------------------------------------------------")
