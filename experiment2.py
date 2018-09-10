import argparse
import json
import os
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader, sampler

import model_utils as mutil
from imagenet32_dataset import BinaryImageNet32, get_imagenet32_labels
from losses import GenericLoss
from torchvision import datasets, transforms

COMPONENT_DIRNAME = 'component'

args = None
device = None


def parse_arguments():
    valid_models = ('resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110')
    experiments = ('last', 'full', 'bn', 'sgm', 'l2',
                   'combn_loss_3', 'combn_loss_5', 'combn_loss_10',
                   'combn_accuracy_3', 'combn_accuracy_5', 'combn_accuracy_10',
                   'combn_threshold_0.75',
                   'pcbn_loss_3', 'pcbn_loss_5', 'pcbn_loss_10',
                   'pcbn_accuracy_3', 'pcbn_accuracy_5', 'pcbn_accuracy_10',
                   'pcbn_threshold_0.75')

    parser = argparse.ArgumentParser(description='Learning ImageNet32 from CIFAR10 Template')
    parser.add_argument('--arch', default='resnet32', choices=valid_models,
                        help='model architecture to use (default: resnet32)')
    parser.add_argument('--shot', default=1, type=int, metavar='N',
                        help='execute for N-shot classification (default: 1)')
    parser.add_argument('--experiments', default=experiments, nargs='*', type=str, metavar='S',
                        help='which experiments to run (default: run all experiments)')
    parser.add_argument('--cifar10-dir', default='./datasets', type=str,
                        help='directory where cifar-10-batches-py exists (default: ./datasets)')
    parser.add_argument('--imagenet32-dir', default='./datasets', type=str,
                        help='directory where imagenet-32-batches-py exists (default: ./datasets)')
    parser.add_argument('--indices-dir', default='./indices/experiment2', type=str,
                        help='directory to look for index files (default: ./indices/experiment2)')
    parser.add_argument('--model-dir', default='./models/experiment2', type=str,
                        help='directory to save models (default: ./models/experiment2)')
    parser.add_argument('--device', default='cuda', type=str,
                        help='device to run models on (default: cuda)')
    parser.add_argument('--evaluate', dest='train', action='store_false',
                        help='evaluate models (after training)')
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite existing model weights')
    parser.add_argument('--num-workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=1e-2, type=float,
                        metavar='F', help='initial learning rate (default: 1e-2)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='F',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='F',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--epoch', default=60, type=int, metavar='N',
                        help='number of epochs (default: 60)')
    parser.add_argument('--decay-milestones', default=[30, 45], nargs='*', type=int, metavar='N',
                        help='epochs to apply learning rate decay at (default: 30 45)')
    parser.add_argument('--decay-factor', default=0.1, type=float, metavar='F',
                        help='multiplicative factor of learning rate decay (default: 0.1)')
    return parser.parse_args()


def train_template_network(loss='default'):
    """Obtain CIFAR10-trained template network.

    Training parameters follow original ResNet paper.

    Args:
        loss: Choose from 'default'/'sgm'/'l2'
    """

    # Use training parameters of original ResNet paper
    split_index = 45000
    batch_size = 128
    lr = 1e-1
    momentum = 0.9
    weight_decay = 1e-4
    epoch = 180
    decay_milestones = [90, 120]
    decay_factor = 0.1

    # SGM/L2 specific parameters
    aux_loss_wt = 0.02

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    image_datasets = {x: datasets.CIFAR10(root=args.cifar10_dir, train=y, download=True,
                                          transform=z)
                      for x, y, z in zip([0, 1], [True, False], [train_transform, test_transform])}
    dataloaders = {x: DataLoader(image_datasets[y], batch_size=batch_size, sampler=z,
                                 num_workers=args.num_workers, pin_memory=('cpu' not in args.device))
                   for x, y, z in zip(['train', 'val', 'test'],
                                      [0, 0, 1],
                                      [sampler.SubsetRandomSampler(range(split_index)),
                                       sampler.SubsetRandomSampler(range(split_index,
                                                                         len(image_datasets[0]))),
                                       sampler.SequentialSampler(image_datasets[1])])}
    dataset_sizes = {'train': split_index,
                     'val'  : len(image_datasets[0]) - split_index,
                     'test' : len(image_datasets[1])}

    model = mutil.get_model(args.arch).to(device)
    if loss == 'default':
        criterion = torch.nn.CrossEntropyLoss().to(device)
    elif loss in ('sgm', 'l2'):
        criterion = GenericLoss(loss, aux_loss_wt, model.linear.out_features)
    else:
        raise NameError('{} is not recognized.'.format(loss))

    optimizer = torch.optim.SGD(mutil.get_model_trainable_parameters(model),
                                lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_milestones,
                                                     gamma=decay_factor)
    model, _ = mutil.train_model(model,
                                 criterion,
                                 optimizer,
                                 dataloaders,
                                 dataset_sizes,
                                 scheduler=scheduler,
                                 num_epochs=epoch,
                                 device=device)
    mutil.eval_model(model, dataloaders['test'], dataset_sizes['test'], device=device)

    return model


def get_binary_imagenet32(pos_class, pos_size=0, train=True, transform=None):
    """Load binary ImageNet32 dataset.
    """

    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    dataset = BinaryImageNet32(args.imagenet32_dir, pos_class, pos_size=pos_size,
                               train=train, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=train,
                            num_workers=args.num_workers, pin_memory=('cpu' not in args.device))
    dataset_size = len(dataset)
    return dataset, dataloader, dataset_size


def get_component_network(template_file=None, binary_linear_file=None, freeze_kernels=False):
    model = mutil.get_model(args.arch)
    if template_file is not None:
        model.load_state_dict(torch.load(template_file, map_location='cpu'))
    if freeze_kernels:
        mutil.freeze_model_parameters_(model)
        mutil.set_module_trainable_(model, torch.nn.BatchNorm2d)
    model.linear = torch.nn.Linear(model.linear.in_features, 2)
    if binary_linear_file is not None:
        model.linear.load_state_dict(torch.load(binary_linear_file, map_location='cpu'))
    return model


def train_component_network(pos_class, template_file=None, binary_linear_file=None):
    dataloaders = {}
    dataset_sizes = {}
    _, dataloaders['train'], dataset_sizes['train'] = get_binary_imagenet32(pos_class, train=True)
    _, dataloaders['val'], dataset_sizes['val'] = get_binary_imagenet32(pos_class, train=False)

    model = get_component_network(template_file, binary_linear_file, freeze_kernels=True).to(device)

    optimizer = torch.optim.SGD(mutil.get_model_trainable_parameters(model), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.decay_milestones,
                                                     gamma=args.decay_factor)
    model, _ = mutil.train_model(model,
                                 torch.nn.CrossEntropyLoss().to(device),
                                 optimizer,
                                 dataloaders,
                                 dataset_sizes,
                                 scheduler=scheduler,
                                 num_epochs=args.epoch,
                                 device=device,
                                 verbose=False)

    return model


def rank_component_networks(component_indices, target_indices, pos_size=0, method='accuracy'):
    shot_type = 'Max-shot' if pos_size == 0 else '{}-shot'.format(pos_size)
    model = get_component_network().to(device)
    all_metrics = []
    for pos_class in target_indices:
        _, dataloader, dataset_size = get_binary_imagenet32(pos_class, pos_size=pos_size,
                                                            train=True)
        pos_class_metrics = []
        for comp in component_indices:
            component_file = os.path.join(args.model_dir, COMPONENT_DIRNAME,
                                          '{}_{}.pth'.format(args.arch, comp))
            model.load_state_dict(torch.load(component_file, map_location='cpu'))

            if method == 'loss':
                _, metric = mutil.eval_model(model, dataloader, dataset_size,
                                             device=device, return_loss=True,
                                             verbose=False)
                print('{} {} of network {} on class {}: {:.4f}'.format(
                    shot_type, method, comp, pos_class, metric))
            elif method == 'accuracy':
                metric = mutil.eval_model(model, dataloader, dataset_size,
                                          device=device, verbose=False)
                print('{} {} of network {} on class {}: {:.1f}%'.format(
                    shot_type, method, comp, pos_class, metric * 100))
            else:
                raise NameError('{} is not recognized.'.format(method))
            pos_class_metrics.append(metric)
        all_metrics.append(pos_class_metrics)
    return all_metrics


def select_components(pos_class, metrics, target_classes, component_classes,
                      method='accuracy', num_components=3, threshold=0.75):
    """Return paths to selected component networks.

    if method='threshold', num_components specify the minimum number of
    components above the accuracy threshold for the component selection to be
    valid.
    """

    pos_class_metrics = metrics[target_classes.index(pos_class)]
    if method == 'loss':
        comp_indices = np.argsort(pos_class_metrics)[:num_components]
    elif method == 'accuracy':
        comp_indices = np.argsort(pos_class_metrics)[::-1][:num_components]
    elif method == 'threshold':
        th_indices = np.flatnonzero(np.array(pos_class_metrics) > threshold)
        if th_indices.size >= num_components:
            comp_indices = np.argsort(pos_class_metrics)[::-1][:min(th_indices.size, 10)]
        else:
            return None
    else:
        raise NameError('{} is not recognized.'.format(method))
    comps = np.array(component_classes)[comp_indices]
    comp_paths = [os.path.join(
        args.model_dir, COMPONENT_DIRNAME, '{}_{}.pth'.format(args.arch, x)) for x in comps]
    return comp_paths


def get_bn_combination_network(component_paths, method='combn', template_file=None,
                               binary_linear_file=None):
    model = get_component_network(template_file=template_file,
                                  binary_linear_file=binary_linear_file,
                                  freeze_kernels=True)
    if method == 'combn':
        mutil.replace_bn_with_combn_(model, component_paths)
    elif method == 'pcbn':
        mutil.replace_bn_with_pcbn_(model, component_paths)
    else:
        raise NameError('{} is not recognized.'.format(method))
    return model


def main():
    global args, device
    args = parse_arguments()
    device = torch.device(args.device)

    imagenet32_labels = get_imagenet32_labels(args.imagenet32_dir)

    # Generate template network weights and last layer initialization weights
    template_file = os.path.join(args.model_dir, '{}.pth'.format(args.arch))
    if args.overwrite or not os.path.isfile(template_file):
        print('Preparing {} template weights...'.format(args.arch))
        model = train_template_network()
        pathlib.Path(os.path.dirname(template_file)).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), template_file)
    else:
        print('{} template weights exist.'.format(args.arch))

    if 'sgm' in args.experiments:
        sgm_file = os.path.join(args.model_dir, '{}_sgm.pth'.format(args.arch))
        if args.overwrite or not os.path.isfile(sgm_file):
            print('Preparing {} SGM template weights...'.format(args.arch))
            model = train_template_network(loss='sgm')
            pathlib.Path(os.path.dirname(sgm_file)).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), sgm_file)
        else:
            print('{} SGM template weights exist.'.format(args.arch))

    if 'l2' in args.experiments:
        l2_file = os.path.join(args.model_dir, '{}_l2.pth'.format(args.arch))
        if args.overwrite or not os.path.isfile(l2_file):
            print('Preparing {} L2 template weights...'.format(args.arch))
            model = train_template_network(loss='l2')
            pathlib.Path(os.path.dirname(l2_file)).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), l2_file)
        else:
            print('{} L2 template weights exist.'.format(args.arch))

    binary_linear_file = os.path.join(args.model_dir, '{}_binary_linear.pth'.format(args.arch))
    if args.overwrite or not os.path.isfile(binary_linear_file):
        print('Preparing binary {} fully-connected weights...'.format(args.arch))
        model = mutil.get_model(args.arch)
        linear = torch.nn.Linear(model.linear.in_features, 2)
        pathlib.Path(os.path.dirname(binary_linear_file)).mkdir(parents=True, exist_ok=True)
        torch.save(linear.state_dict(), binary_linear_file)
    else:
        print('Binary {} fully-connected weights exist.'.format(args.arch))

    # Load component network class indices and experiment target indices.
    with open(os.path.join(args.indices_dir, 'imagenet_component_classes.json'), 'r') as f:
        component_classes = json.load(f)
        print('Index of {} component classes loaded.'.format(len(component_classes)))
    with open(os.path.join(args.indices_dir, 'imagenet_target_classes.json'), 'r') as f:
        target_classes = json.load(f)
        print('Index of {} target classes loaded.'.format(len(target_classes)))

    if any('combn' in x for x in args.experiments) or any('pcbn' in x for x in args.experiments):
        # Generate component networks (if haven't)
        for pos_class in component_classes:
            component_file = os.path.join(args.model_dir, COMPONENT_DIRNAME,
                                          '{}_{}.pth'.format(args.arch, pos_class))
            if args.overwrite or not os.path.isfile(component_file):
                print('Training component network ({} {})...'.format(
                    pos_class, imagenet32_labels[pos_class]))
                model = train_component_network(pos_class, template_file=template_file,
                                                binary_linear_file=binary_linear_file)
                pathlib.Path(os.path.dirname(component_file)).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), component_file)

        # Evaluate component networks to rank them for selection, or load the
        # evaluations if they already exist
        if (any('accuracy' in x for x in args.experiments)
                or any('threshold' in x for x in args.experiments)):
            max_shot_eval_file = os.path.join(args.indices_dir, 'max_shot_accuracies.json')
            if args.overwrite or not os.path.isfile(max_shot_eval_file):
                print('Generating max-shot component accuracies...')
                all_accuracies = rank_component_networks(component_classes, target_classes,
                                                         pos_size=0, method='accuracy')
                with open(max_shot_eval_file, 'w') as f:
                    json.dump(all_accuracies, f)
            else:
                with open(max_shot_eval_file, 'r') as f:
                    all_accuracies = json.load(f)
                    print('Max-shot accuracy component evaluations loaded.')
        if any('loss' in x for x in args.experiments):
            few_shot_eval_file = os.path.join(args.indices_dir,
                                              '{}-shot_losses.json'.format(args.shot))
            if args.overwrite or not os.path.isfile(few_shot_eval_file):
                print('Generating {}-shot component losses...'.format(args.shot))
                all_losses = rank_component_networks(component_classes, target_classes,
                                                     pos_size=args.shot, method='loss')
                with open(few_shot_eval_file, 'w') as f:
                    json.dump(all_losses, f)
            else:
                with open(few_shot_eval_file, 'r') as f:
                    all_losses = json.load(f)
                    print('{}-shot loss component evaluations loaded.'.format(args.shot))

    # Main experiment loop
    for experiment in args.experiments:
        shot_dir = 'max-shot' if args.shot == 0 else '{}-shot'.format(args.shot)

        if not args.train:
            # Perform evaluation by reading off the training summaries
            accuracies = []
            for pos_class in target_classes:
                summary_file = os.path.join(args.model_dir, shot_dir, experiment,
                                            '{}_{}.summary'.format(args.arch, pos_class))
                if os.path.isfile(summary_file):
                    entry = torch.load(summary_file)
                    accuracies.append(np.amax(entry['val_acc']))
            if accuracies:
                print('Mean validation accuracy of {} ({} classes): {:.1f}%'.format(
                    experiment, len(accuracies), np.mean(accuracies) * 100))
            else:
                print('Mean validation accuracy of {} ({} classes): N/A'.format(
                    experiment, len(accuracies)))
        else:
            for pos_class in target_classes:
                weights_file = os.path.join(args.model_dir, shot_dir, experiment,
                                            '{}_{}.pth'.format(args.arch, pos_class))
                summary_file = os.path.join(args.model_dir, shot_dir, experiment,
                                            '{}_{}.summary'.format(args.arch, pos_class))
                if not args.overwrite and os.path.isfile(weights_file):
                    print('Weights found for {} ({} {}). Skipping...'.format(
                        experiment, pos_class, imagenet32_labels[pos_class]))
                    continue

                print('Preparing {} ({} {})...'.format(
                    experiment, pos_class, imagenet32_labels[pos_class]))

                # Define model for this experiment
                if any(x in experiment for x in ('combn', 'pcbn')):
                    # Parse experiment text to set up the proper BN combination
                    # configuration
                    exp_params = experiment.split('_')
                    comb_method = exp_params[0]
                    selection_params = {'method': exp_params[1]}

                    if selection_params['method'] == 'loss':
                        metrics = all_losses
                        selection_params['num_components'] = int(exp_params[2])
                    elif selection_params['method'] == 'accuracy':
                        metrics = all_accuracies
                        selection_params['num_components'] = int(exp_params[2])
                    elif selection_params['method'] == 'threshold':
                        metrics = all_accuracies
                        selection_params['threshold'] = float(exp_params[2])

                    print('Selecting components...')
                    comp_paths = select_components(pos_class, metrics, target_classes,
                                                   component_classes, **selection_params)
                    if comp_paths is None:
                        print('No valid components. Skipping...')
                        continue

                    model = get_bn_combination_network(comp_paths, method=comb_method,
                                                       template_file=template_file,
                                                       binary_linear_file=binary_linear_file)
                elif experiment == 'last':
                    model = mutil.get_model(args.arch)
                    model.load_state_dict(torch.load(template_file, map_location='cpu'))
                    mutil.freeze_model_parameters_(model)
                    model.linear = torch.nn.Linear(model.linear.in_features, 2)
                    model.linear.load_state_dict(torch.load(binary_linear_file, map_location='cpu'))
                elif experiment == 'full':
                    model = mutil.get_model(args.arch)
                    model.load_state_dict(torch.load(template_file, map_location='cpu'))
                    model.linear = torch.nn.Linear(model.linear.in_features, 2)
                    model.linear.load_state_dict(torch.load(binary_linear_file, map_location='cpu'))
                elif experiment == 'bn':
                    model = mutil.get_model(args.arch)
                    model.load_state_dict(torch.load(template_file, map_location='cpu'))
                    mutil.freeze_model_parameters_(model)
                    mutil.set_module_trainable_(model, torch.nn.BatchNorm2d)
                    model.linear = torch.nn.Linear(model.linear.in_features, 2)
                    model.linear.load_state_dict(torch.load(binary_linear_file, map_location='cpu'))
                elif experiment == 'sgm':
                    model = mutil.get_model(args.arch)
                    model.load_state_dict(torch.load(sgm_file, map_location='cpu'))
                    mutil.freeze_model_parameters_(model)
                    model.linear = torch.nn.Linear(model.linear.in_features, 2)
                    model.linear.load_state_dict(torch.load(binary_linear_file, map_location='cpu'))
                elif experiment == 'l2':
                    model = mutil.get_model(args.arch)
                    model.load_state_dict(torch.load(l2_file, map_location='cpu'))
                    mutil.freeze_model_parameters_(model)
                    model.linear = torch.nn.Linear(model.linear.in_features, 2)
                    model.linear.load_state_dict(torch.load(binary_linear_file, map_location='cpu'))
                else:
                    raise NameError('{} is not recognized.'.format(experiment))
                model.to(device)

                # Prepare dataset
                dataloaders = {}
                dataset_sizes = {}
                _, dataloaders['train'], dataset_sizes['train'] = get_binary_imagenet32(
                    pos_class, pos_size=args.shot, train=True)
                _, dataloaders['val'], dataset_sizes['val'] = get_binary_imagenet32(
                    pos_class, pos_size=0, train=False)

                # Train model and save weights
                optimizer = torch.optim.SGD(mutil.get_model_trainable_parameters(model), lr=args.lr,
                                            momentum=args.momentum, weight_decay=args.weight_decay)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                 milestones=args.decay_milestones,
                                                                 gamma=args.decay_factor)
                print('Training...')
                model, summary = mutil.train_model(model,
                                                   torch.nn.CrossEntropyLoss().to(device),
                                                   optimizer,
                                                   dataloaders,
                                                   dataset_sizes,
                                                   scheduler=scheduler,
                                                   num_epochs=args.epoch,
                                                   device=device,
                                                   verbose=False)
                pathlib.Path(os.path.dirname(weights_file)).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), weights_file)
                torch.save(summary, summary_file)
    print('Script complete.')


if __name__ == '__main__':
    main()
