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


class DepthAnythingExt:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self.ownerComp.par.Dimensions = ''
        self.trt_path = self.ownerComp.par.Enginefile.val
        self.device = "cuda"
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        try:
            self.engine = self._load_engine()
            self.get_dimensions(self.engine)
            self.context = self.engine.create_execution_context()
            self.stream = torch.cuda.current_stream(device=self.device)
        except Exception as e:
            debug(e)

        self.source = op('prepared_texture')
        self.trt_input = torch.zeros((self.source.height, self.source.width), device=self.device)
        self.trt_output = torch.zeros((self.source.height, self.source.width), device=self.device)
        self.to_tensor = TopArrayInterface(self.source)
        self.normalize = transforms.Normalize((0.5,), (0.5,))  # Updated normalization for depth data

    def get_dimensions(self, engine):
        shapes = [engine.get_binding_shape(binding) for binding in range(engine.num_bindings)]
        dimensions = shapes[0][2:]
        self.ownerComp.par.Dimensions = f"{dimensions[1]}x{dimensions[0]}"
        op('fit2').par.resolutionw = dimensions[1]
        op('fit2').par.resolutionh = dimensions[0]

    def _load_engine(self):
        with open(self.trt_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def infer(self, img, output):
        self.bindings = [img.data_ptr(), output.data_ptr()]
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()

    def run(self, scriptOp):
        if self.ownerComp.par.Enginefile.val and self.ownerComp.par.Venvpath.val:
            self._prepare_input()
            self.infer(self.trt_input, self.trt_output)
            self._process_output()
            self._copy_output_to_scriptOp(scriptOp)

    def _prepare_input(self):
        self.to_tensor.update(self.stream.cuda_stream)
        self.trt_input = torch.as_tensor(self.to_tensor, device=self.device)
        self.trt_input = self.normalize(self.trt_input).ravel()  # Updated normalization for depth data

    def _process_output(self):
        if self.ownerComp.par.Normalize == 'normal':
            tensor_min = self.trt_output.min()
            tensor_max = self.trt_output.max()
            self.trt_output = (self.trt_output - tensor_min) / (tensor_max - tensor_min)

    def _copy_output_to_scriptOp(self, scriptOp):
        output = TopCUDAInterface(self.source.width, self.source.height, 1, np.float32)
        scriptOp.copyCUDAMemory(self.trt_output.data_ptr(), output.size, output.mem_shape)


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
    def __init__(self, top, stream=0):
        self.top = top
        mem = top.cudaMemory(stream=stream)
        self.w, self.h = mem.shape.width, mem.shape.height
        self.num_comps = mem.shape.numComps
        self.dtype = mem.shape.dataType
        shape = (mem.shape.numComps, self.h, self.w)
        dtype_info = {'descr': [('', '<f4')], 'num_bytes': 4}
        dtype_descr = dtype_info['descr']
        num_bytes = dtype_info['num_bytes']
        num_bytes_px = num_bytes * mem.shape.numComps
        
        self.__cuda_array_interface__ = {
            "version": 3,
            "shape": shape,
            "typestr": dtype_descr[0][1],
            "descr": dtype_descr,
            "stream": stream,
            "strides": (num_bytes, num_bytes_px * self.w, num_bytes_px),
            "data": (mem.ptr, False),
        }

    def update(self, stream=0):
        mem = self.top.cudaMemory(stream=stream)
        self.__cuda_array_interface__['stream'] = stream
        self.__cuda_array_interface__['data'] = (mem.ptr, False)