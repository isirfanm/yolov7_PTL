from data_loader_trt import load_data
import tensorrt as trt
import numpy as np
import ctypes


class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_generator, cache_file):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.data_gen = data_generator
        self.device_input = None

        # Pre-allocate a buffer on GPU using TRT's context if needed
        # or handle it dynamically in get_batch

    def get_batch(self, names):
        try:
            data = next(self.data_gen)
            # Allocate on device if not done yet
            if self.device_input is None:
                self.device_input = trt.Runtime(
                    trt.Logger()).allocate_memory(data.nbytes)

            # Copy from CPU to GPU
            # Note: For simplicity in standalone scripts, we use a simple copy
            # In production, you'd use a dedicated CUDA wrapper, but TRT can handle this:
            self.memcpy_host_to_device(self.device_input, data)
            return [int(self.device_input)]
        except StopIteration:
            return None

    def memcpy_host_to_device(self, device_ptr, host_data):
        # Fallback to a simple ctypes copy if no cuda wrapper is available
        ctypes.memmove(device_ptr, host_data.ctypes.data, host_data.nbytes)

    def read_calibration_cache(self):
        return None  # Force re-calibration to generate fresh cache

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            print(f"\n[SUCCESS] Calibration cache saved to: {self.cache_file}")


def generate_cache(onnx_path, cache_path):
    logger = trt.Logger(trt.Logger.VERBOSE)  # Use VERBOSE log
    builder = trt.Builder(logger)
    config = builder.create_builder_config()

    data = load_data()

    # Enable INT8
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = MyCalibrator(data, cache_path)
    config.max_workspace_size = 1 << 30  # 1GB

    # Parse Network
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    print("Building engine to trigger calibration...")
    # build_serialized_network runs the calibration phase
    _ = builder.build_serialized_network(network, config)


if __name__ == "__main__":
    generate_cache("quantization/model/yolov7-tiny-base.onnx",
                   "../calib_data.cache")
