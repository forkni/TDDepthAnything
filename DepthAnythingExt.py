'''
License for https://github.com/IntentDev/TopArray

MIT License

Copyright (c) 2024 Keith Lostracco

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import tensorrt as trt
import torch
import numpy as np
import torchvision.transforms as transforms
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DepthAnythingExt:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self.ownerComp.par.Dimensions = ''
        self.trt_path = self.ownerComp.par.Enginefile.val
        
        if torch.cuda.is_available():
            cuda_device_count = torch.cuda.device_count()
            if cuda_device_count > 0:
                self.device = torch.device(f"cuda:0")  # Use the first available CUDA device
                torch.cuda.set_device(0)  # Set to the first CUDA device
                # logger.info(f"Using CUDA device 0 out of {cuda_device_count} available devices.")
            else:
                logger.warning("CUDA is available but no CUDA devices found. Falling back to CPU.")
                self.device = torch.device("cpu")
        else:
            logger.info("CUDA is not available. Running on CPU.")
            self.device = torch.device("cpu")
        
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        
        try:
            self.engine = self._load_engine()
            self.get_dimensions(self.engine)
            self.context = self.engine.create_execution_context()
            if self.device.type == 'cuda':
                self.stream = torch.cuda.Stream(device=self.device)
            else:
                self.stream = None
        except Exception as e:
            logger.error(f"Error initializing DepthAnythingExt: {e}")
            raise

        self.source = op('prepared_texture')
        
        # Preallocate tensors
        self.trt_input = torch.zeros((1, 1, self.source.height, self.source.width), dtype=torch.float32, device=self.device)
        self.trt_output = torch.zeros((1, 1, self.source.height, self.source.width), dtype=torch.float32, device=self.device)
        
        # Use pinned memory for faster transfers if CUDA is available
        self.to_tensor = TopArrayInterface(self.source, stream=self.stream, pinned=self.device.type == 'cuda')
        
        # Normalize as a part of the transform
        self.transform = transforms.Compose([transforms.Normalize(mean=[0.5] * 4, std=[0.5] * 4)])

    def get_dimensions(self, engine):
        shapes = [engine.get_binding_shape(binding) for binding in range(engine.num_bindings)]
        dimensions = shapes[0][2:]
        self.ownerComp.par.Dimensions = f"{dimensions[1]}x{dimensions[0]}"
        op('fit2').par.resolutionw = dimensions[1]
        op('fit2').par.resolutionh = dimensions[0]

    def _load_engine(self):
        with open(self.trt_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    @torch.no_grad()
    def infer(self, img, output):
        self.bindings = [img.data_ptr(), output.data_ptr()]
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.cuda_stream)

    @torch.no_grad()
    def run(self, scriptOp):
        if not self.ownerComp.par.Enginefile.val or not self.ownerComp.par.Venvpath.val:
            logger.info("Engine file or Venv path not set. Skipping inference.")
            return

        if self.device.type == 'cuda':
            with torch.cuda.stream(self.stream):
                self._prepare_input()
                self.infer(self.trt_input, self.trt_output)
                self._process_output()
                self._copy_output_to_scriptOp(scriptOp)
            self.stream.synchronize()
        else:
            self._prepare_input_cpu()
            self._infer_cpu()
            self._process_output()
            self._copy_output_to_scriptOp_cpu(scriptOp)

    def _prepare_input(self):
        self.to_tensor.update()
        input_tensor = self.to_tensor.get_tensor()
        
        # Check if we need to adjust the number of channels
        if input_tensor.shape[0] != self.trt_input.shape[1]:
            # Adjust the trt_input tensor to match the input shape
            self.trt_input = torch.zeros((1, input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2]), 
                                        dtype=torch.float32, device=self.device)
        
        if self.device.type == 'cuda':
            with torch.cuda.stream(self.stream):
                self.trt_input.copy_(input_tensor.unsqueeze(0), non_blocking=True)
                self.trt_input = self.transform(self.trt_input)
        else:
            self.trt_input.copy_(input_tensor.unsqueeze(0))
            self.trt_input = self.transform(self.trt_input)
        
        # Log the shapes for debugging
        # logger.info(f"Input tensor shape: {input_tensor.shape}")
        # logger.info(f"TRT input shape after preparation: {self.trt_input.shape}")

    def _process_output(self):
        if self.ownerComp.par.Normalize == 'normal':
            if self.device.type == 'cuda':
                with torch.cuda.stream(self.stream):
                    tensor_min = self.trt_output.min()
                    tensor_max = self.trt_output.max()
                    self.trt_output.sub_(tensor_min).div_(tensor_max - tensor_min)
            else:
                tensor_min = self.trt_output.min()
                tensor_max = self.trt_output.max()
                self.trt_output.sub_(tensor_min).div_(tensor_max - tensor_min)

    def _copy_output_to_scriptOp(self, scriptOp):
        output = TopCUDAInterface(self.source.width, self.source.height, 1, np.float32)
        scriptOp.copyCUDAMemory(self.trt_output.squeeze().data_ptr(), output.size, output.mem_shape)

    def _prepare_input_cpu(self):
        # Implement CPU version of input preparation
        pass

    def _infer_cpu(self):
        # Implement CPU version of inference
        pass

    def _copy_output_to_scriptOp_cpu(self, scriptOp):
        # Implement CPU version of output copying
        pass


class TopCUDAInterface:
    def __init__(self, width, height, num_comps, dtype):
        self.mem_shape = CUDAMemoryShape()
        self.mem_shape.width = width
        self.mem_shape.height = height
        self.mem_shape.numComps = num_comps
        self.mem_shape.dataType = dtype
        self.bytes_per_comp = np.dtype(dtype).itemsize
        self.size = width * height * num_comps * self.bytes_per_comp

class TopArrayInterface:
    def __init__(self, top, stream=None, pinned=False):
        self.top = top
        self.pinned = pinned
        self.stream = stream
        mem = top.cudaMemory(stream=self.stream.cuda_stream if self.stream else 0)
        self.w, self.h = mem.shape.width, mem.shape.height
        self.num_comps = mem.shape.numComps
        self.dtype = mem.shape.dataType
        self.shape = (mem.shape.numComps, self.h, self.w)
        dtype_info = {'descr': [('', '<f4')], 'num_bytes': 4}
        dtype_descr = dtype_info['descr']
        num_bytes = dtype_info['num_bytes']
        num_bytes_px = num_bytes * mem.shape.numComps
        
        self.__cuda_array_interface__ = {
            "version": 3,
            "shape": self.shape,
            "typestr": dtype_descr[0][1],
            "descr": dtype_descr,
            "stream": self.stream.cuda_stream if self.stream else 0,
            "strides": (num_bytes, num_bytes_px * self.w, num_bytes_px),
            "data": (mem.ptr, False),
        }

        if pinned and torch.cuda.is_available():
            self.pinned_memory = torch.empty(self.shape, dtype=torch.float32, pin_memory=True)
        else:
            self.pinned_memory = None
        
        self.cuda_tensor = torch.empty(self.shape, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

    def update(self):
        mem = self.top.cudaMemory(stream=self.stream.cuda_stream if self.stream else 0)
        self.__cuda_array_interface__['stream'] = self.stream.cuda_stream if self.stream else 0
        self.__cuda_array_interface__['data'] = (mem.ptr, False)
        
        if torch.cuda.is_available():
            with torch.cuda.stream(self.stream):
                if self.pinned_memory is not None:
                    # Copy from device to pinned memory
                    self.pinned_memory.copy_(torch.as_tensor(self, device='cuda'), non_blocking=True)
                    # Copy from pinned memory to device
                    self.cuda_tensor.copy_(self.pinned_memory, non_blocking=True)
                else:
                    # Direct copy to device memory
                    self.cuda_tensor.copy_(torch.as_tensor(self, device='cuda'), non_blocking=True)
        else:
            # CPU fallback
            self.cuda_tensor.copy_(torch.as_tensor(self, device='cpu'))

    def get_tensor(self):
        return self.cuda_tensor