'''
Modified from https://raw.githubusercontent.com/pytorch/vision/v0.9.1/torchvision/models/resnet.py

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import sys
import torch.nn as nn
from typing import Union, Tuple
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

class QuantizedTensor:
    def __init__(self, tensor_data: torch.Tensor, scale: float, zero_point: int):
        if tensor_data.dtype != torch.uint8:
            raise TypeError("Quantized activation must be torch.uint8")        
        self.tensor = tensor_data
        self.scale = scale
        self.zero_point = zero_point

    def dequantize(self) -> torch.Tensor:
        return (self.tensor.to(torch.float32) - self.zero_point) * self.scale
    
    def __repr__(self):
        return (f"QuantizedTensor(shape={self.tensor.shape}, "
                f"scale={self.scale:.4f}, zp={self.zero_point})")

class QuantizeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('output_scale', torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer('output_zero_point', torch.tensor(0, dtype=torch.int32))

    def set_output_quant_params(self, output_scale: float, output_zero_point: int):
        self.output_scale.data = torch.tensor(output_scale, dtype=torch.float32).to(self.output_scale.device)
        self.output_zero_point.data = torch.tensor(output_zero_point, dtype=torch.int32).to(self.output_zero_point.device)

    def forward(self, x: torch.Tensor) -> QuantizedTensor:
        if x.dtype != torch.float32:
            raise TypeError(f"Input tensor must be float32, but got {x.dtype}")    
        x_quant = torch.round(x / self.output_scale) + self.output_zero_point  
        output_quant_tensor = torch.clamp(x_quant, 0, 255).to(torch.uint8)        
        return QuantizedTensor(
            tensor_data=output_quant_tensor,
            scale=self.output_scale.item(),
            zero_point=self.output_zero_point.item()
        )

    def extra_repr(self) -> str:
        return f'output_scale={self.output_scale.item():.6f}, output_zero_point={self.output_zero_point.item()}'

class QuantizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        weight_shape = (out_channels, in_channels // groups, *self.kernel_size)
        self.register_buffer('weight_int8', torch.zeros(weight_shape, dtype=torch.int8))
        quant_param_shape = (out_channels, 1, 1, 1)
        self.register_buffer('weight_scale', torch.ones(quant_param_shape, dtype=torch.float32))
        self.register_buffer('weight_zero_point', torch.zeros(quant_param_shape, dtype=torch.int32))
        if bias:
            self.bias_fp32 = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32))
        else:
            self.register_parameter('bias_fp32', None)
        self.register_buffer('output_scale', torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer('output_zero_point', torch.tensor(0, dtype=torch.int32))

    def set_weight_quant_params(self, weight_scale, weight_zero_point):
        if weight_scale.dtype != torch.float32:
            raise TypeError(f"Invalid weight_scale dtype! Require torch.float32, but get {weight_scale.dtype}")
        if weight_zero_point.dtype != torch.int32:
            raise TypeError(f"Invalid weight_zero_point dtype! Require torch.int32, but get {weight_zero_point.dtype}")
        self.weight_scale.data = weight_scale.reshape(self.out_channels, 1, 1, 1).to(self.weight_scale.device)
        self.weight_zero_point.data = weight_zero_point.reshape(self.out_channels, 1, 1, 1).to(self.weight_zero_point.device)

    def set_output_quant_params(self, output_scale: float, output_zero_point: int):
        self.output_scale.data = torch.tensor(output_scale, dtype=torch.float32).to(self.output_scale.device)
        self.output_zero_point.data = torch.tensor(output_zero_point, dtype=torch.int32).to(self.output_zero_point.device)

    def set_int8_weight(self, weight_int8: torch.Tensor):
        if weight_int8.shape != self.weight_int8.shape:
            raise ValueError(f"Invalid weight shape! Require {self.weight_int8.shape}, but get {weight_int8.shape}")
        if weight_int8.dtype != torch.int8:
            raise TypeError(f"Invalid weight dtype! Require torch.int8, but get {weight_int8.dtype}")
        self.weight_int8.data.copy_(weight_int8)
        self.weight_int8.data = self.weight_int8.data.to(self.weight_scale.device)

    def set_fp32_bias(self, bias_fp32: torch.Tensor):
        if self.bias_fp32 is not None:
            if bias_fp32 is None:
                raise ValueError("bias=True but bias_fp32 is None during setup.")
            if bias_fp32.shape != self.bias_fp32.shape:
                raise ValueError(f"Invalid bias shape! Require {self.bias_fp32.shape}, but get {bias_fp32.shape}")
            if bias_fp32.dtype != torch.float32:
                 raise TypeError(f"Invalid bias dtype! Require torch.float32, but get {bias_fp32.dtype}")
            self.bias_fp32.data.copy_(bias_fp32)
        elif bias_fp32 is not None:
             print("Warning: bias=False but bias_fp32 was provided. Bias will be ignored.")

    def forward(self, x: QuantizedTensor) -> QuantizedTensor:
        x_quant = x.tensor
        input_scale = x.scale
        input_zero_point = x.zero_point
        x_sub_zp_float = x_quant.to(torch.float32) - input_zero_point
        w_sub_zp_float = self.weight_int8.to(torch.float32) - self.weight_zero_point.to(torch.float32)
        conv_out_float = F.conv2d(x_sub_zp_float, w_sub_zp_float, bias=None, stride=self.stride,
                                  padding=self.padding, dilation=self.dilation,
                                  groups=self.groups)
        requant_scale = (input_scale * self.weight_scale) / self.output_scale
        requant_scale_reshaped = requant_scale.view(1, self.out_channels, 1, 1)
        requant_conv_part = conv_out_float * requant_scale_reshaped
        if self.bias_fp32 is not None:
            bias_term = self.bias_fp32 / self.output_scale
            bias_term_reshaped = bias_term.view(1, -1, 1, 1)
            final_float_before_round = requant_conv_part + bias_term_reshaped
        else:
            final_float_before_round = requant_conv_part
        requant_out = torch.round(final_float_before_round) + self.output_zero_point
        output_quant = torch.clamp(requant_out, 0, 255).to(torch.uint8)
        return QuantizedTensor(
            tensor_data=output_quant,
            scale=self.output_scale.item(),
            zero_point=self.output_zero_point.item()
        )

    def extra_repr(self) -> str:
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}, '
             'stride={stride}, padding={padding}, groups={groups}, bias={bias_fp32 is not None}')
        bias_status = self.bias_fp32 is not None
        format_dict = {
            **self.__dict__,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'groups': self.groups,
            'bias_fp32 is not None': bias_status
        }
        return s.format(**format_dict)
    
class QuantizedConvReLU2d(QuantizedConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups)
        self.output_zero_point.data = torch.tensor(0, dtype=torch.int32)

    def set_output_quant_params(self, output_scale: float):
        self.output_scale.data = torch.tensor(output_scale, dtype=torch.float32).to(self.output_scale.device)
        if self.output_zero_point.item() != 0:
             self.output_zero_point.data = torch.tensor(0, dtype=torch.int32).to(self.output_zero_point.device)

    def forward(self, x: QuantizedTensor) -> QuantizedTensor:
        x_quant = x.tensor
        input_scale = x.scale
        input_zero_point = x.zero_point
        x_sub_zp_float = x_quant.to(torch.float32) - input_zero_point
        w_sub_zp_float = self.weight_int8.to(torch.float32) - self.weight_zero_point.to(torch.float32)
        conv_out_float = F.conv2d(x_sub_zp_float, w_sub_zp_float, bias=None, stride=self.stride,
                                  padding=self.padding, dilation=self.dilation,
                                  groups=self.groups)
        requant_scale = (input_scale * self.weight_scale) / self.output_scale
        requant_scale_reshaped = requant_scale.view(1, self.out_channels, 1, 1)
        requant_conv_part = conv_out_float * requant_scale_reshaped
        if self.bias_fp32 is not None:
            bias_term = self.bias_fp32 / self.output_scale
            bias_term_reshaped = bias_term.view(1, -1, 1, 1)
            final_float_before_round = requant_conv_part + bias_term_reshaped
        else:
            final_float_before_round = requant_conv_part
        final_float_before_round = F.relu(final_float_before_round)
        requant_out = torch.round(final_float_before_round) + self.output_zero_point
        output_quant = torch.clamp(requant_out, 0, 255).to(torch.uint8)
        return QuantizedTensor(
            tensor_data=output_quant,
            scale=self.output_scale.item(),
            zero_point=self.output_zero_point.item()
        )

class QuantizedReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: QuantizedTensor) -> QuantizedTensor:
        if not isinstance(x, QuantizedTensor):
            raise TypeError(f"Input must be a QuantizedTensor, but got {type(x)}")
        output_tensor = torch.clamp(x.tensor, min=x.zero_point)
        return QuantizedTensor(
            tensor_data=output_tensor,
            scale=x.scale,
            zero_point=x.zero_point
        )

    def extra_repr(self) -> str:
        return "Quantized ReLU (uint8 clamp at zero_point)"

class QuantizedAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('output_scale', torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer('output_zero_point', torch.tensor(0, dtype=torch.int32))

    def set_output_quant_params(self, output_scale: float, output_zero_point: int):
        self.output_scale.data = torch.tensor(output_scale, dtype=torch.float32).to(self.output_scale.device)
        self.output_zero_point.data = torch.tensor(output_zero_point, dtype=torch.int32).to(self.output_zero_point.device)

    def forward(self, x: QuantizedTensor, y: QuantizedTensor) -> QuantizedTensor:
        x_float = x.dequantize()
        y_float = y.dequantize()
        result_float = x_float + y_float
        scale = self.output_scale
        zero_point = self.output_zero_point        
        result_quant = torch.round(result_float / scale) + zero_point
        result_quant_tensor = torch.clamp(result_quant, 0, 255).to(torch.uint8)
        return QuantizedTensor(
            tensor_data=result_quant_tensor,
            scale=scale.item(),
            zero_point=zero_point.item()
        )

class QuantizedAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size: Union[int, Tuple[int, int]]):
        super().__init__()
        self.output_size = output_size

    def forward(self, x: QuantizedTensor) -> QuantizedTensor:
        if not isinstance(x, QuantizedTensor):
            raise TypeError(f"Input must be a QuantizedTensor, but got {type(x)}")
        x_quant = x.tensor
        x_float_for_pool = x_quant.float()
        pooled_float = F.adaptive_avg_pool2d(x_float_for_pool, self.output_size)
        pooled_rounded = torch.round(pooled_float)
        output_quant = torch.clamp(pooled_rounded, 0, 255).to(torch.uint8)
        return QuantizedTensor(
            tensor_data=output_quant,
            scale=x.scale,
            zero_point=x.zero_point
        )

    def extra_repr(self) -> str:
        return f'output_size={self.output_size}'

class QuantizedFlatten(nn.Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: QuantizedTensor) -> QuantizedTensor:
        output_tensor = torch.flatten(x.tensor, self.start_dim, self.end_dim)
        return QuantizedTensor(
            tensor_data=output_tensor,
            scale=x.scale,
            zero_point=x.zero_point
        )

class QuantizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        weight_shape = (out_features, in_features)
        self.register_buffer('weight_int8', torch.zeros(weight_shape, dtype=torch.int8))
        quant_param_shape = (out_features, 1) 
        self.register_buffer('weight_scale', torch.ones(quant_param_shape, dtype=torch.float32))
        self.register_buffer('weight_zero_point', torch.zeros(quant_param_shape, dtype=torch.int32))
        self.bias_fp32 = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    def set_weight_quant_params(self, weight_scale, weight_zero_point):
        if weight_scale.dtype != torch.float32:
            raise TypeError(f"Invalid weight_scale dtype! Require torch.float32, but get {weight_scale.dtype}")
        if weight_zero_point.dtype != torch.int32:
            raise TypeError(f"Invalid weight_zero_point dtype! Require torch.int32, but get {weight_zero_point.dtype}")
        self.weight_scale.data = weight_scale.reshape(self.out_features, 1).to(self.weight_scale.device)
        self.weight_zero_point.data = weight_zero_point.reshape(self.out_features, 1).to(self.weight_zero_point.device)

    def set_int8_weight(self, weight_int8: torch.Tensor):
        if weight_int8.shape != self.weight_int8.shape:
            raise ValueError(f"Invalid weight shape! Require {self.weight_int8.shape}, but get {weight_int8.shape}")
        if weight_int8.dtype != torch.int8:
            raise TypeError(f"Invalid weight dtype! Require torch.int8, but get {weight_int8.dtype}")
        self.weight_int8.data.copy_(weight_int8)
        
    def set_fp32_bias(self, bias_fp32: torch.Tensor):
        if bias_fp32.shape != self.bias_fp32.shape:
            raise ValueError(f"Invalid bias shape! Require {self.bias_fp32.shape}, but get {bias_fp32.shape}")
        if bias_fp32.dtype != torch.float32:
            raise TypeError(f"Invalid bias dtype! Require torch.float32, but get {bias_fp32.dtype}")
        self.bias_fp32.data.copy_(bias_fp32)

    def forward(self, x: QuantizedTensor) -> torch.Tensor:
        x_quant = x.tensor
        input_scale = x.scale
        input_zero_point = x.zero_point
        x_sub_zp_float = x_quant.to(torch.float32) - input_zero_point
        w_sub_zp_float = self.weight_int8.to(torch.float32) - self.weight_zero_point.to(torch.float32)
        linear_out_float_simulating_int32 = F.linear(x_sub_zp_float, w_sub_zp_float, bias=None)
        dequant_scale = input_scale * self.weight_scale.T
        output_fp32_no_bias = linear_out_float_simulating_int32 * dequant_scale
        output_fp32 = output_fp32_no_bias + self.bias_fp32
        return output_fp32

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias=True'

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return QuantizedConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return QuantizedConv2d(in_planes, out_planes, kernel_size=1, stride=stride)

def conv3x3relu(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return QuantizedConvReLU2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3relu(inplanes, planes, stride)
        #self.bn1 = nn.BatchNorm2d(planes)
        #self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.relu = QuantizedReLU()
        self.downsample = downsample
        self.add = QuantizedAdd()
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        #out = self.bn1(out)
        #out = self.relu(out)
        out = self.conv2(out)
        #out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.add(out, identity)
        out = self.relu(out)
        return out

class QuantizedCifarResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(QuantizedCifarResNet, self).__init__()
        block = BasicBlock
        layers = [3]*3
        self.inplanes = 16
        self.quant = QuantizeLayer()
        self.conv1 = conv3x3relu(3, 16)
        #self.bn1 = nn.BatchNorm2d(16)
        #self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = QuantizedAdaptiveAvgPool2d((1, 1))
        self.flat = QuantizedFlatten()
        self.fc = QuantizedLinear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                #nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.flat(x)
        x = self.fc(x)
        return x