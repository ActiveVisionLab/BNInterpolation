# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree (or rather,
# https://raw.githubusercontent.com/facebookresearch/low-shot-shrink-hallucinate/master/LICENSE).
#
#
# Modified by gwpd for compatibility with PyTorch 0.4.1 and own scripts:
#   Added support for torch.device.
#   Added dim=1 to nn.Softmax argument.
#   Removed torch.autograd.Variable wrappers.
#   Removed get_classifier_weight() from loss function parameters.
#   Made GenericLoss output model scores.
#
# Originally downloaded from
# https://raw.githubusercontent.com/facebookresearch/low-shot-shrink-hallucinate/master/losses.py

import torch
import torch.nn as nn


def l2_loss(feat):
    return feat.pow(2).sum() / (2.0 * feat.size(0))


def get_one_hot(labels, num_classes):
    one_hot = torch.range(0, num_classes - 1).unsqueeze(0).expand(labels.size(0), num_classes)
    one_hot = one_hot.to(labels.device)
    one_hot = one_hot.eq(labels.unsqueeze(1).expand_as(one_hot).float()).float()
    return one_hot


class BatchSGMLoss(nn.Module):
    def __init__(self, num_classes, device='cpu'):
        super(BatchSGMLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1).to(torch.device(device))
        self.num_classes = num_classes

    def forward(self, feats, scores, labels):
        one_hot = get_one_hot(labels, self.num_classes)
        p = self.softmax(scores)
        p = p.to(scores.device)

        G = (one_hot - p).transpose(0, 1).mm(feats)
        G = G.div(feats.size(0))
        return G.pow(2).sum()


class SGMLoss(nn.Module):
    def __init__(self, num_classes, device='cpu'):
        super(SGMLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1).to(torch.device(device))
        self.num_classes = num_classes

    def forward(self, feats, scores, labels):
        one_hot = get_one_hot(labels, self.num_classes)
        p = self.softmax(scores)
        p = p.to(scores.device)

        pereg_wt = (one_hot - p).pow(2).sum(1)
        sqrXnorm = feats.pow(2).sum(1)
        loss = pereg_wt.mul(sqrXnorm).mean()
        return loss


class GenericLoss:
    def __init__(self, aux_loss_type, aux_loss_wt, num_classes, device='cpu'):
        device = torch.device(device)
        aux_loss_fns = dict(l2=l2_loss,
                            sgm=SGMLoss(num_classes, device),
                            batchsgm=BatchSGMLoss(num_classes, device))
        self.aux_loss_fn = aux_loss_fns[aux_loss_type]
        self.aux_loss_type = aux_loss_type
        self.cross_entropy_loss = nn.CrossEntropyLoss().to(device)
        self.aux_loss_wt = aux_loss_wt

    def __call__(self, model, x_var, y_var):
        scores, feats = model(x_var)
        if self.aux_loss_type in ['l2']:
            aux_loss = self.aux_loss_fn(feats)
        else:
            aux_loss = self.aux_loss_fn(feats, scores, y_var)
        orig_loss = self.cross_entropy_loss(scores, y_var)
        return orig_loss + self.aux_loss_wt * aux_loss, scores
