import torch
import model_architectures as M

# 0. Init test Tensor X
X = torch.ones([64, 8, 224, 224])

# 1. BN Process Block
model = M.ConvolutionalProcessingBatchNormBlock(
    input_shape=X.shape, num_filters=16, kernel_size=3, 
    padding=1, bias=True, dilation=1
    )

try:
    Output = model.forward(X)
except Exception as e:
    print(e)

# 2. BN Reduction Block
model = M.ConvolutionalDimensionalityReductionBatchNormBlock(
    input_shape=X.shape, num_filters=16, kernel_size=3, 
    padding=1, bias=True, dilation=1, reduction_factor=2
    )

try:
    Output = model.forward(X)
except Exception as e:
    print(e)

# 3. BN Residual Process Block
model = M.ConvolutionalProcessingBatchNormResidualBlock(
    input_shape=X.shape, num_filters=16, kernel_size=3, 
    padding=1, bias=True, dilation=1
    )

try:
    Output = model.forward(X)
except Exception as e:
    print(e)
