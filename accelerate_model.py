import os
import logging
from typing import Tuple, Dict, Optional
import torch
import torch.onnx
import tensorrt as trt
from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from tqdm import tqdm
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

def adjust_image_size(image_size: int) -> int:
    """
    Adjust image size to be compatible with model's patch size.

    Args:
        image_size (int): Original image size.

    Returns:
        int: Adjusted image size.
    """
    patch_size = 14
    adjusted_size = (image_size // patch_size) * patch_size
    if image_size % patch_size != 0:
        adjusted_size += patch_size
    return adjusted_size

def get_user_input() -> Tuple[int, str, int, int]:
    """
    Get user input for model version, size, width, and height.

    Returns:
        Tuple[int, str, int, int]: Model version, model size, width, and height.
    """
    while True:
        try:
            model_version = int(input("Enter 1 for DepthAnything v1 or 2 for DepthAnything v2: "))
            if model_version not in [1, 2]:
                raise ValueError("Invalid model version")
            
            valid_sizes = ['s', 'b', 'l'] if model_version == 1 else ['s', 'b', 'l', 'g']
            model_size = input(f"Enter model size ({'/'.join(valid_sizes)}): ").lower()
            if model_size not in valid_sizes:
                raise ValueError("Invalid model size")
            
            width = int(input("Enter the width of the input: "))
            height = int(input("Enter the height of the input: "))
            
            return model_version, model_size, width, height
        except ValueError as e:
            logger.error(f"Invalid input: {e}")

def initialize_model(model_version: int, encoder: str) -> torch.nn.Module:
    """
    Initialize the appropriate model based on user choices.

    Args:
        model_version (int): Version of DepthAnything (1 or 2).
        encoder (str): Encoder type (e.g., 'vits', 'vitb', 'vitl', 'vitg').

    Returns:
        torch.nn.Module: Initialized model.
    """
    if model_version == 1:
        from depth_anything.dpt import DPT_DINOv2
        return DPT_DINOv2(**MODEL_CONFIGS[encoder], localhub=False)
    else:
        from depth_anything_v2.dpt import DepthAnythingV2
        return DepthAnythingV2(**MODEL_CONFIGS[encoder])

def export_to_onnx(model: torch.nn.Module, dummy_input: torch.Tensor, onnx_path: str):
    """
    Export the PyTorch model to ONNX format.

    Args:
        model (torch.nn.Module): PyTorch model to export.
        dummy_input (torch.Tensor): Dummy input tensor for the model.
        onnx_path (str): Path to save the ONNX model.
    """
    try:
        with tqdm(total=100, desc="Exporting to ONNX") as pbar:
            def update_progress(current, total):
                pbar.update(int(100 * current / total) - pbar.n)

            # Export to ONNX opset 17 or later
            torch.onnx.export(model, dummy_input, onnx_path, opset_version=17, 
                              input_names=["input"], output_names=["output"], verbose=True)
        logger.info(f"Model exported to {onnx_path}")
    except Exception as e:
        logger.error(f"Failed to export model to ONNX: {e}")
        raise

def build_tensorrt_engine(onnx_path: str, engine_path: str):
    try:
        p = Profile()
        with tqdm(total=100, desc="Building TensorRT engine") as pbar:
            engine = engine_from_network(
                network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]),
                config=CreateConfig(fp16=True, refittable=False, profiles=[p], load_timing_cache=None),
                save_timing_cache=None,
            )
            for layer in engine:
                if isinstance(layer, trt.ILayer):  # Check if layer is a TensorRT ILayer object
                    if layer.type == trt.LayerType.LAYERNORM:
                        layer.precision = trt.float32
            pbar.update(90)  # Assume engine creation is 90% of the process
            save_engine(engine, path=engine_path)
            pbar.update(10)  # Remaining 10% for saving the engine
        logger.info(f"Finished building TensorRT engine: {engine_path}")
    except Exception as e:
        logger.error(f"Failed to build TensorRT engine: {e}")
        raise

def cleanup_intermediate_files(onnx_path: str):
    """
    Clean up intermediate files (ONNX model).

    Args:
        onnx_path (str): Path to the ONNX model to be removed.
    """
    try:
        os.remove(onnx_path)
        logger.info(f"Removed intermediate ONNX file: {onnx_path}")
    except Exception as e:
        logger.warning(f"Failed to remove ONNX file {onnx_path}: {e}")

def main(cleanup: bool = False):
    """
    Main function to run the model acceleration process.

    Args:
        cleanup (bool, optional): Whether to clean up intermediate files. Defaults to False.
    """
    torch.hub.set_dir('torchhub')
    os.makedirs("onnx_models", exist_ok=True)
    os.makedirs("engines", exist_ok=True)

    model_version, model_size, width, height = get_user_input()
    encoder = f'vit{model_size}'
    
    width = adjust_image_size(width)
    height = adjust_image_size(height)
    image_shape = (3, height, width)
    logger.info(f'Image shape is {width}x{height}')

    load_from = f'./checkpoints/depth_anything{"_v2" if model_version == 2 else ""}_vit{model_size}{"14" if model_version == 1 else ""}.pth'
    outputs = f"{os.path.splitext(os.path.basename(load_from))[0]}"
    onnx_path = f"onnx_models/{outputs}_{width}x{height}.onnx"
    engine_path = f"engines/{outputs}_{width}x{height}.engine"

    model = initialize_model(model_version, encoder)
    total_params = sum(param.numel() for param in model.parameters())
    logger.info(f'Total parameters: {total_params / 1e6:.2f}M')

    try:
        model.load_state_dict(torch.load(load_from, map_location='cpu'), strict=True)
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        return

    model.eval()
    dummy_input = torch.ones(image_shape).unsqueeze(0)

    export_to_onnx(model, dummy_input, onnx_path)
    build_tensorrt_engine(onnx_path, engine_path)

    if cleanup:
        cleanup_intermediate_files(onnx_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Acceleration Script")
    parser.add_argument('--cleanup', action='store_true', help='Clean up intermediate ONNX files after processing')
    args = parser.parse_args()

    main(cleanup=args.cleanup)