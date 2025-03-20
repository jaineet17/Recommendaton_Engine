import os
import logging
import torch
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import time
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)

def convert_pytorch_to_onnx(model, model_name, dummy_input, output_dir="models/onnx", opset_version=12):
    """
    Convert PyTorch model to ONNX format for faster inference
    
    Args:
        model: PyTorch model
        model_name: Name for the ONNX model file
        dummy_input: Example input tensor(s) for tracing
        output_dir: Directory to save the ONNX model
        opset_version: ONNX opset version to use
        
    Returns:
        Path to the saved ONNX model
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Set model to evaluation mode
        model.eval()
        
        # Prepare output path
        onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
        
        # Export the model to ONNX
        with torch.no_grad():
            torch.onnx.export(
                model,                     # PyTorch model
                dummy_input,               # Example input
                onnx_path,                 # Output path
                export_params=True,        # Export model parameters
                opset_version=opset_version,  # ONNX opset version
                do_constant_folding=True,  # Fold constants for optimization
                input_names=['input'],     # Model input names
                output_names=['output'],   # Model output names
                dynamic_axes={
                    'input': {0: 'batch_size'},  # Dynamic batch size
                    'output': {0: 'batch_size'}
                }
            )
        
        # Verify the ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Log success
        logger.info(f"Successfully converted {model_name} to ONNX format at {onnx_path}")
        
        # Save model metadata
        metadata = {
            "name": model_name,
            "conversion_time": datetime.now().isoformat(),
            "pytorch_version": torch.__version__,
            "onnx_version": onnx.__version__,
            "opset_version": opset_version,
            "input_shape": [list(x.shape) if isinstance(x, torch.Tensor) else None for x in dummy_input] 
                           if isinstance(dummy_input, tuple) else list(dummy_input.shape)
        }
        
        metadata_path = os.path.join(output_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return onnx_path
    
    except Exception as e:
        logger.error(f"Error converting model to ONNX: {e}")
        return None

def quantize_onnx_model(onnx_model_path, quantized_model_path=None, quantization_type=QuantType.QInt8):
    """
    Quantize ONNX model to reduce size and improve inference speed
    
    Args:
        onnx_model_path: Path to the ONNX model
        quantized_model_path: Path to save the quantized model (if None, adds '_quantized' suffix)
        quantization_type: Type of quantization (QInt8, QUInt8, etc.)
        
    Returns:
        Path to the quantized model and stats dictionary
    """
    try:
        if not os.path.exists(onnx_model_path):
            logger.error(f"ONNX model not found at {onnx_model_path}")
            return None, {}
        
        # Determine output path if not provided
        if quantized_model_path is None:
            path = Path(onnx_model_path)
            quantized_model_path = str(path.parent / f"{path.stem}_quantized{path.suffix}")
        
        # Get original model size
        original_size = os.path.getsize(onnx_model_path)
        
        # Perform quantization
        start_time = time.time()
        quantize_dynamic(
            onnx_model_path,
            quantized_model_path,
            weight_type=quantization_type
        )
        quantization_time = time.time() - start_time
        
        # Get quantized model size
        if os.path.exists(quantized_model_path):
            quantized_size = os.path.getsize(quantized_model_path)
            size_reduction = (original_size - quantized_size) / original_size * 100
            
            # Collect stats
            stats = {
                "original_size_bytes": original_size,
                "quantized_size_bytes": quantized_size,
                "size_reduction_percentage": size_reduction,
                "quantization_time_seconds": quantization_time,
                "quantization_type": str(quantization_type)
            }
            
            logger.info(f"Quantized model saved at {quantized_model_path} "
                      f"(size reduced by {size_reduction:.2f}%)")
            
            return quantized_model_path, stats
        else:
            logger.error(f"Failed to create quantized model at {quantized_model_path}")
            return None, {}
    
    except Exception as e:
        logger.error(f"Error quantizing ONNX model: {e}")
        return None, {}

def optimize_onnx_model(onnx_model_path, optimized_model_path=None):
    """
    Optimize ONNX model using ONNX Runtime
    
    Args:
        onnx_model_path: Path to the ONNX model
        optimized_model_path: Path to save the optimized model (if None, adds '_optimized' suffix)
        
    Returns:
        Path to the optimized model
    """
    try:
        if not os.path.exists(onnx_model_path):
            logger.error(f"ONNX model not found at {onnx_model_path}")
            return None
        
        # Determine output path if not provided
        if optimized_model_path is None:
            path = Path(onnx_model_path)
            optimized_model_path = str(path.parent / f"{path.stem}_optimized{path.suffix}")
        
        # Load the model
        model = onnx.load(onnx_model_path)
        
        # Optimize with ONNX Runtime
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.optimized_model_filepath = optimized_model_path
        
        # Create session to optimize and save model
        _ = ort.InferenceSession(onnx_model_path, sess_options)
        
        if os.path.exists(optimized_model_path):
            logger.info(f"Optimized model saved at {optimized_model_path}")
            return optimized_model_path
        else:
            logger.error(f"Failed to create optimized model at {optimized_model_path}")
            return None
    
    except Exception as e:
        logger.error(f"Error optimizing ONNX model: {e}")
        return None

def load_onnx_model(model_path, providers=None):
    """
    Load an ONNX model for inference
    
    Args:
        model_path: Path to the ONNX model
        providers: List of execution providers (if None, uses CUDA if available, else CPU)
        
    Returns:
        ONNX Runtime InferenceSession
    """
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}")
            return None
        
        # Set default providers if not specified
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        
        # Configure session options
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Create and return session
        session = ort.InferenceSession(model_path, session_options, providers=providers)
        
        # Log model metadata
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_name = session.get_outputs()[0].name
        output_shape = session.get_outputs()[0].shape
        
        logger.info(f"Loaded ONNX model from {model_path}")
        logger.info(f"Input: {input_name}, shape: {input_shape}")
        logger.info(f"Output: {output_name}, shape: {output_shape}")
        logger.info(f"Using providers: {session.get_providers()}")
        
        return session
    
    except Exception as e:
        logger.error(f"Error loading ONNX model: {e}")
        return None

def benchmark_onnx_model(session, example_inputs, num_runs=100):
    """
    Benchmark ONNX model inference performance
    
    Args:
        session: ONNX Runtime InferenceSession
        example_inputs: Dictionary of example inputs for the model
        num_runs: Number of inference runs for benchmarking
        
    Returns:
        Dictionary of benchmark results
    """
    try:
        # Warm-up runs
        for _ in range(10):
            _ = session.run(None, example_inputs)
        
        # Benchmark runs
        start_time = time.time()
        for _ in range(num_runs):
            _ = session.run(None, example_inputs)
        total_time = time.time() - start_time
        
        # Calculate statistics
        avg_time = total_time / num_runs
        throughput = num_runs / total_time
        
        results = {
            "num_runs": num_runs,
            "total_time_seconds": total_time,
            "average_time_seconds": avg_time,
            "throughput_inferences_per_second": throughput,
            "providers_used": session.get_providers()
        }
        
        logger.info(f"ONNX model benchmark: {throughput:.2f} inferences/sec, {avg_time*1000:.2f} ms/inference")
        
        return results
    
    except Exception as e:
        logger.error(f"Error benchmarking ONNX model: {e}")
        return {} 