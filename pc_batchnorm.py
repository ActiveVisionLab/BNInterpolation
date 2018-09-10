"""
Forked from PyTorch 0.3.0's torch.nn.modules.batchnorm implementation.
Last updated to be compatible with PyTorch 0.4.0.
"""

import torch
from torch.nn import BatchNorm2d, Module, Parameter, functional


class _PCBatchNorm(Module):
    """
    batch_norms must be an iterable containing batch norm layers with the same
    number of parameters

    If init='manual', self.weight and self.bias parameters are initialized as
    original_params['weight'] and original_params['bias'] transformed to PC
    space.
    """

    def __init__(self, batch_norms, eps=1e-5, num_pc=0, momentum=0.1,
                 init='equal', original_params=None):
        super(_PCBatchNorm, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.init = init
        self.original_params = original_params
        self.num_composition = len(batch_norms)
        self.num_features = next(iter(batch_norms)).num_features
        self.affine = next(iter(batch_norms)).affine
        if num_pc == 0:
            self.num_pc = min(self.num_composition, self.num_features)
        else:
            self.num_pc = min(num_pc, self.num_composition, self.num_features)

        self.register_buffer('running_mean', torch.zeros(self.num_features))
        self.register_buffer('running_var', torch.ones(self.num_features))
        if self.affine:
            self.weight = Parameter(torch.empty(self.num_pc))
            self.bias = Parameter(torch.empty(self.num_pc))
        else:
            self.weight = None
            self.bias = None

        # Check that batchnorm objects in batch_norms are all identical
        # (e.g. all has affine layers and same number of features)
        # TODO: implement and expose 1D and 3D API
        # (also see forward())
        for batch_norm in batch_norms:
            assert self.num_features == batch_norm.num_features
            assert self.affine == batch_norm.affine
            assert isinstance(batch_norm, BatchNorm2d)

        # Extract constituent batch normalization layer parameters/buffers to
        # this module
        if self.affine:
            composite_weight = torch.zeros(self.num_composition,
                                           self.num_features)
            composite_bias = torch.zeros(self.num_composition,
                                         self.num_features)
            for i, batch_norm in enumerate(batch_norms):
                composite_weight[i, :] = batch_norm.weight.data
                composite_bias[i, :] = batch_norm.bias.data

        # Register parameter means and eigenvector matrices as buffers
        if self.affine:
            mu_weight = torch.mean(composite_weight, 0)
            mu_bias = torch.mean(composite_bias, 0)
            _, _, V_weight = torch.svd(composite_weight - mu_weight)
            _, _, V_bias = torch.svd(composite_bias - mu_bias)
            V_weight = V_weight[:, 0:self.num_pc]
            V_bias = V_bias[:, 0:self.num_pc]
        else:
            mu_weight = None
            mu_bias = None
            V_weight = None
            V_bias = None

        self.register_buffer('mu_weight', mu_weight)
        self.register_buffer('mu_bias', mu_bias)
        self.register_buffer('V_weight', V_weight)
        self.register_buffer('V_bias', V_bias)

        self.reset_parameters()

    def _check_input_dim(self, x):
        return NotImplemented

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            if self.init == 'equal':
                self.weight.data.fill_(1 / self.num_pc)
                self.bias.data.fill_(1 / self.num_pc)
            elif self.init == 'standard':
                self.weight.data = torch.matmul(
                    torch.empty(self.num_features).uniform_()
                    - self.mu_weight,
                    self.V_weight)
                self.bias.data = torch.matmul(
                    torch.empty(self.num_features).zero_()
                    - self.mu_bias,
                    self.V_bias)
            elif self.init == 'manual':
                self.weight.data = torch.matmul(
                    torch.tensor(self.original_params['weight']).to(torch.device('cpu'))
                    - self.mu_weight,
                    self.V_weight)
                self.bias.data = torch.matmul(
                    torch.tensor(self.original_params['bias']).to(torch.device('cpu'))
                    - self.mu_bias,
                    self.V_bias)
            else:
                NameError('init {} not supported.'.format(self.init))

    def forward(self, x):
        self._check_input_dim(x)

        # This forward operation only works for 2D PC batchnorm
        # TODO: implement and expose 1D and 3D PC batchnorm API
        # (also see __init__())
        weight = torch.matmul(self.weight, self.V_weight.t()) + self.mu_weight
        bias = torch.matmul(self.bias, self.V_bias.t()) + self.mu_bias

        return functional.batch_norm(
            x, self.running_mean, self.running_var, weight, bias,
            self.training, self.momentum, self.eps)

    def __repr__(self):
        return ('{name}({num_composition}, num_features={num_features},'
                ' num_pc={num_pc}, eps={eps}, affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))


class PCBatchNorm2d(_PCBatchNorm):
    """Applies PC Batch Normalization over a 4d input that is seen as a
    mini-batch of 3d inputs
    """

    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(x.dim()))
