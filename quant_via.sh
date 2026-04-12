# Base
# Inspect
python test_via_quant.py --data "data/furnas-val.yaml" --img-size 640 --batch-size 16 --conf-thres 0.001 --iou 0.65 --device cpu --weights "quantization/model/yolov7-tiny-base.pt" --name yolov7-tiny-via-inspect --inspect --quant_mode float --output_dir "quantization/vitis_ai/yolov7-tiny"

# Calib
python test_via_quant.py --data data/furnas-val.yaml --img-size 640 --batch-size 16 --conf-thres 0.001 --iou 0.65 --device cpu --weights "quantization/model/yolov7-tiny-base.pt" --name yolov7-tiny-via-calib --quant_mode calib  --output_dir "quantization/vitis_ai/yolov7-tiny"

# Test deploy
python test_via_quant.py --data data/furnas-val.yaml --img-size 640 --batch-size 1 --conf-thres 0.001 --iou 0.65 --device cpu --weights "quantization/model/yolov7-tiny-base.pt" --name yolov7-tiny-via-test --quant_mode test --deploy --output_dir "quantization/vitis_ai/yolov7-tiny"

# Compilation
vai_c_xir -x "quantization/vitis_ai/yolov7-tiny/Model_int.xmodel" -a "/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json" -o "quantization/vitis_ai/yolov7-tiny/vai_compile_out" -n yolov7-tiny-via-compile

# Pruned
# Inspect
python test_via_quant.py --data "data/furnas-val.yaml" --img 640 --batch-size 16 --conf-thres 0.001 --iou 0.65 --device cpu --weights "quantization/model/yolov7-tiny-pruned.pt" --name yolov7-tiny-pruned-via-inspect --inspect --quant_mode float --output_dir "quantization/vitis_ai/yolov7-tiny-pruned"

# Calib
python test_via_quant.py --data data/furnas-val.yaml --img 640 --batch-size 16 --conf-thres 0.001 --iou 0.65 --device cpu --weights "quantization/model/yolov7-tiny-pruned.pt" --name yolov7-tiny-pruned-via-calib --quant_mode calib  --output_dir "quantization/vitis_ai/yolov7-tiny-pruned"

# Test deploy
python test_via_quant.py --data data/furnas-val.yaml --img 640 --batch-size 1 --conf-thres 0.001 --iou 0.65 --device cpu --weights "quantization/model/yolov7-tiny-pruned.pt" --name yolov7-tiny-pruned-via-test --quant_mode test --deploy --output_dir "quantization/vitis_ai/yolov7-tiny-pruned"

# Compilation
vai_c_xir -x "quantization/vitis_ai/yolov7-tiny-pruned/Model_int.xmodel" -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json -o "quantization/vitis_ai/yolov7-tiny-pruned/vai_compile_out" -n yolov7-tiny-pruned-via-compile
