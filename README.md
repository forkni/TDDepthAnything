# Depth-Anything TensorRT in TouchDesigner

TouchDesigner implementation for Depth Anything and Depth Anything v2 with TensorRT monocular depth estimation. 

![Screenshot_68](https://github.com/olegchomp/TDDepthAnything/assets/11017531/fa457aa2-d10a-4f54-a93a-27d672501f16)

## Features
* One click install and run script
* In-TouchDesigner inference
  
## Usage
Tested with TouchDesigner 2023.11880 & Python 3.11

#### Installation process:
1. Install [Python 3.11](https://www.python.org/downloads/release/python-3118/);
2. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-11-8-0-download-archive) 11.8;
3. Install [GIT](https://git-scm.com/downloads);
4. Clone [TDDepthAnything](https://github.com/forkni/TDDepthAnything.git) repository
5. Download [Depth-Anything model](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints) or [Depth-Anything v2 model](https://github.com/DepthAnything/Depth-Anything-V2?tab=readme-ov-file#pre-trained-models)
6. Run ```install.bat```. When prompted to select version of Python, type in "3.11" and hit "Enter";
7. When installation will be finished, copy model to ```checkpoints``` folder

#### Acceleration process:
1. Run ```accelerate.bat```.
2. Select model version (1 - Depth-Anything, 2 - Depth-Anything v2)
3. Select model size (s - small, b - base, l - large, g - giant)
4. Select width & height (default is 518x518)
5. Wait for acceleration to finish

#### TouchDesigner inference:
1. Add TDDepthAnything.tox to project
2. On ```Settings``` page change path to ```TDDepthAnything``` folder and click Re-init
3. On ```Depth Anything``` page select path to engine file (for ex. ```engines/depth_anything_vits14.engine```) and click Load Engine

## Acknowledgement
Based on the following projects:
* [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) - Unleashing the Power of Large-Scale Unlabeled Data.
* [Depth-Anything TensorRT C++](https://github.com/spacewalk01/depth-anything-tensorrt) - Leveraging the TensorRT API for efficient real-time inference.
* [TopArray](https://github.com/IntentDev/TopArray) - Interaction between Python/PyTorch tensor operations and TouchDesigner TOPs.

