import torch


def quantize_onnx_model(onnx_model_path, quantized_model_path):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnx
    onnx_opt_model = onnx.load(onnx_model_path)
    quantize_dynamic(
        onnx_model_path,
        quantized_model_path,
        #  {torch.nn.Linear},
        op_types_to_quantize=['Linear', 'MatMul'],
        weight_type=QuantType.QInt8)

    # logger.info(f"quantized model saved to:{quantized_model_path}")


quantize_onnx_model('model.onnx', 'model_m.quant.onnx')
