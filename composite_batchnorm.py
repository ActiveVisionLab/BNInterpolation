"""
Forked from PyTorch 0.3.0's torch.nn.modules.batchnorm implementation.
Last updated to be compatible with PyTorch 0.4.1.
"""

import torch
from torch.nn import BatchNorm2d, Module, Parameter


class _CompositeBatchNorm(Module):
    """
    batch_norms must be an iterable containing batch norm layers with the same
    number of parameters.

    If init='manual', weights are initialized to manual_params, which must be
    a list-like object containing weights.
    """

    def __init__(self, batch_norms, eps=1e-5, mode='naive', init='auto', manual_params=None):
        super(_CompositeBatchNorm, self).__init__()
        self.eps = eps
        self.mode = mode
        self.init = init
        self.manual_params = manual_params
        self.num_composition = len(batch_norms)
        self.num_features = next(iter(batch_norms)).num_features
        self.affine = next(iter(batch_norms)).affine
        if mode == 'naive':
            self.weight = Parameter(torch.empty(self.num_composition))
        elif mode == 'channel':
            self.weight = Parameter(torch.empty(self.num_composition,
                                                self.num_features))
        else:
            NameError('mode {} not supported.'.format(mode))
        if init == 'auto':
            pass
        elif init == 'manual':
            assert manual_params is not None
            assert self.num_composition == len(manual_params)
        else:
            NameError('init {} not supported.'.format(init))

        # Check that batchnorm objects in batch_norms are all identical
        # (e.g. all has affine layers and same number of features)
        # TODO: implement and expose 1D and 3D composite batchnorm API
        # (also see forward())
        for batch_norm in batch_norms:
            assert self.num_features == batch_norm.num_features
            assert self.affine == batch_norm.affine
            assert isinstance(batch_norm, BatchNorm2d)

        # Register constituent batch normalization layer parameters/buffers to
        # this module
        composite_mean = torch.zeros(self.num_composition, self.num_features)
        composite_var = torch.zeros(self.num_composition, self.num_features)
        if self.affine:
            composite_weight = torch.zeros(self.num_composition, self.num_features)
            composite_bias = torch.zeros(self.num_composition, self.num_features)
        else:
            composite_weight = None
            composite_bias = None

        for i, batch_norm in enumerate(batch_norms):
            composite_mean[i, :] = batch_norm.running_mean
            composite_var[i, :] = batch_norm.running_var
            if self.affine:
                composite_weight[i, :] = batch_norm.weight.data
                composite_bias[i, :] = batch_norm.bias.data

        self.register_buffer('composite_mean', composite_mean)
        self.register_buffer('composite_var', composite_var)
        self.register_buffer('composite_weight', composite_weight)
        self.register_buffer('composite_bias', composite_bias)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init == 'auto':
            if self.mode in {'naive', 'channel'}:
                self.weight.data.fill_(1 / self.num_composition)
        elif self.init == 'manual':
            if self.mode == 'naive':
                self.weight.data = torch.tensor(self.manual_params)

    def _check_input_dim(self, x):
        return NotImplemented

    def forward(self, x):
        self._check_input_dim(x)

        # This forward operation only works for 2D composite batchnorm
        # Logic also applies to both 'naive' and 'channel' modes
        # TODO: implement and expose 1D and 3D composite batchnorm API
        # (also see __init__())
        sz = x.size()
        flat_x = x.transpose(0, 1).contiguous().view(sz[1], -1)
        flat_y = torch.zeros_like(flat_x)

        for i in range(self.num_composition):
            if self.mode == 'naive':
                flat_y += self.weight[i] * (
                    (flat_x - self.composite_mean[i, :][:, None])
                    * torch.rsqrt(self.composite_var[i, :][:, None] + self.eps)
                    * self.composite_weight[i, :][:, None]
                    + self.composite_bias[i, :][:, None]
                    )
            elif self.mode == 'channel':
                flat_y += self.weight[i][:, None] * (
                    (flat_x - self.composite_mean[i, :][:, None])
                    * torch.rsqrt(self.composite_var[i, :][:, None] + self.eps)
                    * self.composite_weight[i, :][:, None]
                    + self.composite_bias[i, :][:, None]
                    )
        y = flat_y.view(sz[1], sz[0], sz[2], sz[3]).transpose(0, 1).contiguous()
        return y

    def __repr__(self):
        return ('{name}({num_composition}, num_features={num_features},'
                ' mode={mode}, eps={eps}, affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))


class CompositeBatchNorm2d(_CompositeBatchNorm):
    """Applies Composite Batch Normalization over a 4d input that is seen as a
    mini-batch of 3d inputs
    """

    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(x.dim()))
