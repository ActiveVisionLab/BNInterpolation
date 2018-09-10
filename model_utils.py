import copy
import re
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

import resnet_cifar
from composite_batchnorm import CompositeBatchNorm2d
from losses import GenericLoss
from pc_batchnorm import PCBatchNorm2d
from torchvision import models


def get_model(model_string, **kwargs):
    """Retrieves model from library.
    """

    if model_string == 'resnet18':
        return models.resnet18(**kwargs)
    elif model_string == 'resnet34':
        return models.resnet34(**kwargs)
    elif model_string == 'resnet50':
        return models.resnet50(**kwargs)
    elif model_string == 'resnet101':
        return models.resnet101(**kwargs)
    elif model_string == 'resnet152':
        return models.resnet152(**kwargs)
    elif model_string == 'resnet20':
        return resnet_cifar.resnet20(**kwargs)
    elif model_string == 'resnet32':
        return resnet_cifar.resnet32(**kwargs)
    elif model_string == 'resnet44':
        return resnet_cifar.resnet44(**kwargs)
    elif model_string == 'resnet56':
        return resnet_cifar.resnet56(**kwargs)
    elif model_string == 'resnet110':
        return resnet_cifar.resnet110(**kwargs)
    elif model_string == 'resnet1202':
        return resnet_cifar.resnet1202(**kwargs)
    else:
        raise NameError('{} is not recognized.'.format(model_string))


def get_model_trainable_parameters(model):
    """Retrieves all trainable parameters of a model to pass to an optimizer.
    """

    param_list = []

    for param in model.parameters():
        if param.requires_grad:
            param_list.append(param)
    return param_list


def freeze_model_parameters_(model):
    """Sets requires_grad attribute in all model parameters to False.
    """

    for param in model.parameters():
        param.requires_grad = False


def set_module_trainable_(model, target_module):
    """Sets all specified modules of model to trainable.
    """

    for module in model.modules():
        if isinstance(module, target_module):
            for param in module.parameters():
                param.requires_grad = True


def part_load_state_dict_(model, state_dict, prototype_module):
    """Loads only parameters of prototype module to model.
    """

    valid_names = []
    for name, module in model.named_modules():
        if isinstance(module, prototype_module):
            valid_names.append(name)

    valid_keys = []
    for key in state_dict:
        if any([name in key for name in valid_names]):
            valid_keys.append(key)

    new_state_dict = OrderedDict((key, state_dict[key]) for key in valid_keys)
    model.load_state_dict(new_state_dict, strict=False)


def replace_bn_with_combn_(model, state_dict_paths, mode='naive', init='auto',
                           manual_params=None):
    """
    Replace all BatchNorm2d modules in model with CompositeBatchNorm2d,
    initialised using a combination of BN layer weights retrieved from
    state_dict_paths.

    state_dict_paths is a collection of paths to model weights.
    """

    num_composition = len(state_dict_paths)
    batch_norms = [{} for i in range(num_composition)]

    # Collect BatchNorm2d modules from all state_dict(s)
    original_weights = model.state_dict()
    for i, path in enumerate(state_dict_paths):
        model.load_state_dict(torch.load(path))

        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                batch_norms[i][name] = copy.deepcopy(module)
    model.load_state_dict(original_weights)

    # Replace model's BatchNorm2d modules with CompositeBatchNorm2d
    regex = re.compile(r'\.(\d+)')
    for name, _ in model.named_modules():
        # Retrieve module dynamically instead of relying on generator
        # This is done to prevent undefined behaviour with iterating over a
        # mutating iterable
        stat = regex.sub(r'[\1]', name)
        transfer_dict = {'model': model}
        if stat != '':
            exec('module = model.' + stat, transfer_dict)
        else:
            exec('module = model', transfer_dict)
        module = transfer_dict['module']

        # Copy and replace Sequential module if the Sequential module contains
        # BatchNorm2d
        if isinstance(module, nn.Sequential):
            # Check that there is BatchNorm2d in the Sequential module
            has_bn = False
            for submodule in module.children():
                if isinstance(submodule, nn.BatchNorm2d):
                    has_bn = True
                    break

            # Replace module if check passes
            if has_bn:
                seq_copy = OrderedDict(module.named_children())
                for subname, submodule in seq_copy.items():
                    if isinstance(submodule, nn.BatchNorm2d):
                        bn_list = ([batch_norms[i][name + '.' + subname]
                                    for i in range(num_composition)])
                        seq_copy[subname] = CompositeBatchNorm2d(bn_list,
                                                                 mode=mode,
                                                                 init=init,
                                                                 manual_params=manual_params)
                stat = regex.sub(r'[\1]', name)
                exec('model.' + stat
                     + ' = nn.Sequential(seq_copy)')

        # Otherwise replace BatchNorm2d directly
        elif isinstance(module, nn.BatchNorm2d):
            bn_list = [batch_norms[i][name] for i in range(num_composition)]
            stat = regex.sub(r'[\1]', name)
            exec('model.' + stat
                 + ' = CompositeBatchNorm2d(bn_list, mode=mode, init=init, manual_params=manual_params)')


def replace_bn_with_pcbn_(model, state_dict_paths, num_pc=0, init='equal'):
    """
    Replace all BatchNorm2d modules in model with PCBatchNorm2d,
    initialised using a combination of BN layer weights retrieved from
    state_dict_paths.

    state_dict_paths is a collection of paths to model weights.

    If init='manual', PCBatchNorm2d parameters will be initialized to
    BatchNorm2d parameters found in model, transformed to PC space.
    """

    num_composition = len(state_dict_paths)
    batch_norms = [{} for i in range(num_composition)]
    original_params = None

    # Collect BatchNorm2d modules from all state_dict(s)
    original_weights = model.state_dict()
    for i, path in enumerate(state_dict_paths):
        model.load_state_dict(torch.load(path))

        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                batch_norms[i][name] = copy.deepcopy(module)
    model.load_state_dict(original_weights)

    # Replace model's BatchNorm2d modules with CompositeBatchNorm2d
    regex = re.compile(r'\.(\d+)')
    for name, _ in model.named_modules():
        # Retrieve module dynamically instead of relying on generator
        # This is done to prevent undefined behaviour with iterating over a
        # mutating iterable
        stat = regex.sub(r'[\1]', name)
        transfer_dict = {'model': model}
        if stat != '':
            exec('module = model.' + stat, transfer_dict)
        else:
            exec('module = model', transfer_dict)
        module = transfer_dict['module']

        # Copy and replace Sequential module if the Sequential module contains
        # BatchNorm2d
        if isinstance(module, nn.Sequential):
            # Check that there is BatchNorm2d in the Sequential module
            has_bn = False
            for submodule in module.children():
                if isinstance(submodule, nn.BatchNorm2d):
                    has_bn = True
                    break

            # Replace module if check passes
            if has_bn:
                seq_copy = OrderedDict(module.named_children())
                for subname, submodule in seq_copy.items():
                    if isinstance(submodule, nn.BatchNorm2d):
                        bn_list = ([batch_norms[i][name + '.' + subname]
                                    for i in range(num_composition)])
                        if init == 'manual':
                            original_params = {'weight': submodule.weight.data,
                                               'bias': submodule.bias.data}
                        seq_copy[subname] = PCBatchNorm2d(
                            bn_list,
                            num_pc=num_pc,
                            init=init,
                            original_params=original_params)
                stat = regex.sub(r'[\1]', name)
                exec('model.' + stat
                     + ' = nn.Sequential(seq_copy)')

        # Otherwise replace BatchNorm2d directly
        elif isinstance(module, nn.BatchNorm2d):
            bn_list = [batch_norms[i][name] for i in range(num_composition)]
            if init == 'manual':
                original_params = {'weight': module.weight.data,
                                   'bias': module.bias.data}
            stat = regex.sub(r'[\1]', name)
            exec('model.' + stat
                 + ' = PCBatchNorm2d(bn_list, num_pc=num_pc, init=init,'
                 + 'original_params=original_params)')


def train_model(model, criterion, optimizer, dataloader, dataset_sizes,
                scheduler=None, num_epochs=30, device='cuda', verbose=True,
                log_dir=None):
    """Trainer function.
    """

    device = torch.device(device)
    use_tensorboardx = log_dir is not None
    summary = {x: [] for x in ('train_loss',
                               'train_acc',
                               'val_loss',
                               'val_acc',
                               'wall_time')}
    if 'lr' in optimizer.param_groups[0]:
        summary['lr'] = []

    if use_tensorboardx:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir=log_dir)

    best_model_wts = model.state_dict()
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        for phase in ('train', 'val'):
            if phase == 'train':
                model.train()
                if scheduler is not None:
                    scheduler.step()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                if isinstance(criterion, GenericLoss):
                    loss, outputs = criterion(model, inputs, labels)
                else:
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                outputs_d, labels_d = outputs.detach(), labels.detach()
                if isinstance(criterion, (nn.CrossEntropyLoss, GenericLoss)):
                    _, preds = torch.max(outputs_d, 1)
                    running_corrects += torch.sum(preds == labels_d).item()
                elif isinstance(criterion, nn.MultiLabelSoftMarginLoss):
                    preds = (outputs_d > 0.5).to(torch.long)
                    running_corrects += torch.sum(
                        torch.sum(preds == labels_d.to(torch.long), 1) == labels.size(1)).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            # Update training summaries
            summary['{}_loss'.format(phase)].append(epoch_loss)
            summary['{}_acc'.format(phase)].append(epoch_acc)

            # Deep copy best performing model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # Update end-of-epoch summaries
        summary['wall_time'].append(time.time() - start_time)
        if 'lr' in summary:
            summary['lr'].append(optimizer.param_groups[0]['lr'])

        if use_tensorboardx:
            writer.add_scalars('summary/loss',
                               {'train': summary['train_loss'][-1],
                                'val': summary['val_loss'][-1]},
                               epoch)
            writer.add_scalars('summary/acc',
                               {'train': summary['train_acc'][-1],
                                'val': summary['val_acc'][-1]},
                               epoch)

        if verbose:
            print(('Epoch {}/{} [{:.0f}m {:.0f}s] - train_loss: {:.4f} train_acc: {:.4f} '
                   'val_loss: {:.4f} val_acc: {:.4f}').format(
                       epoch + 1, num_epochs,
                       summary['wall_time'][-1] // 60, summary['wall_time'][-1] % 60,
                       summary['train_loss'][-1], summary['train_acc'][-1],
                       summary['val_loss'][-1], summary['val_acc'][-1]))

    # Clean up
    if use_tensorboardx:
        writer.close()

    print('Best validation accuracy: {:.4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model, summary


def eval_model(model, dataloader, dataset_size, criterion=nn.CrossEntropyLoss(),
               device='cuda', verbose=True, return_preds=False, return_loss=False):
    """Evaluator function.
    """

    device = torch.device(device)
    criterion = criterion.to(device)
    model.eval()

    start_time = time.time()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            if isinstance(criterion, GenericLoss):
                this_loss = criterion(model, inputs, labels)
            else:
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                this_loss = criterion(outputs, labels)

            running_loss += this_loss.item() * inputs.size(0)

            if isinstance(criterion, (nn.CrossEntropyLoss, GenericLoss)):
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels).item()
            elif isinstance(criterion, nn.MultiLabelSoftMarginLoss):
                preds = (outputs > 0.5).to(torch.long)
                running_corrects += torch.sum(
                    torch.sum(preds == labels.to(torch.long), 1) == labels.size(1)).item()
            all_preds += [preds.cpu().numpy()]

    all_preds = np.concatenate(all_preds, axis=0)
    loss = running_loss / dataset_size
    accuracy = running_corrects / dataset_size
    time_elapsed = time.time() - start_time

    if verbose:
        print('Evaluation complete in {:.0f}m {:.3f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Accuracy: {:.4f}'.format(accuracy))
        if return_loss:
            print('Loss: {:.4f}'.format(loss))

    if return_preds:
        return accuracy, all_preds
    elif return_loss:
        return accuracy, loss
    return accuracy
