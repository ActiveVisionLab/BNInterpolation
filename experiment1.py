import argparse
import os
import pathlib

import torch
from torch.utils.data import DataLoader

import model_utils as mutil
from data_utils import BinaryCIFAR10Subset
from torchvision import datasets, transforms


def parse_arguments():
    valid_models = ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
    experiments = ('last', 'full', 'bn', 'combn', 'pcbn',
                   'bn_random', 'combn_random', 'pcbn_random')
    # experiments = ('last', 'full', 'bn', 'combn', 'pcbn',
    #                'bn_random', 'combn_random', 'pcbn_random',
    #                'combn_semi_random', 'pcbn_semi_random',
    #                'combn_full_random', 'pcbn_full_random',
    #                'bn_imagenet_random')

    parser = argparse.ArgumentParser(description='Learning CIFAR10 from ImageNet Template')
    parser.add_argument('--arch', default='resnet34', choices=valid_models,
                        help='model architecture to use (default: resnet34)')
    parser.add_argument('--experiments', default=experiments, nargs='*', type=str, metavar='S',
                        help='which experiments to run (default: run all experiments)')
    parser.add_argument('--cifar10-dir', default='./datasets', type=str,
                        help='directory where cifar-10-batches-py exists (default: ./datasets)')
    parser.add_argument('--model-dir', default='./models/experiment1', type=str,
                        help='directory to load/save models (default: ./models/experiment1)')
    parser.add_argument('--device', default='cuda', type=str,
                        help='device to run models on (default: cuda)')
    parser.add_argument('--evaluate', dest='train', action='store_false',
                        help='evaluate models (after training)')
    parser.add_argument('--overwrite', action='store_true',
                        help='when training, overwrite existing model weights')
    parser.add_argument('--num-workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--batch-size', default=8, type=int, metavar='N',
                        help='mini-batch size (default: 8)')
    parser.add_argument('--lr', default=1e-3, type=float,
                        metavar='F', help='initial learning rate (default: 1e-3)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='F',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', default=0.0, type=float, metavar='F',
                        help='weight decay (default: 0.0)')
    parser.add_argument('--epoch', default=30, type=int, metavar='N',
                        help='number of epochs (default: 30)')
    parser.add_argument('--decay-step', default=20, type=int, metavar='N',
                        help='learning rate decay epoch period (default: 20)')
    parser.add_argument('--decay-factor', default=0.1, type=float, metavar='F',
                        help='multiplicative factor of learning rate decay (default: 0.1)')
    return parser.parse_args()


def main():
    args = parse_arguments()
    device = torch.device(args.device)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    image_datasets = {x: datasets.CIFAR10(root=args.cifar10_dir, train=y,
                                          download=True, transform=transform)
                      for x, y in zip([0, 1], [True, False])}
    class_names = ('airplane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

    # Prepare initial template weights
    random_file = os.path.join(args.model_dir, '{}_random.pth'.format(args.arch))
    if args.overwrite or not os.path.isfile(random_file):
        print('Preparing {} random weights...'.format(args.arch))
        model = mutil.get_model(args.arch, pretrained=False)
        pathlib.Path(os.path.dirname(random_file)).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), random_file)
    binary_fc_file = os.path.join(args.model_dir, '{}_binary_fc.pth'.format(args.arch))
    if args.overwrite or not os.path.isfile(binary_fc_file):
        print('Preparing binary {} fully-connected weights...'.format(args.arch))
        model = mutil.get_model(args.arch, pretrained=False)
        fc = torch.nn.Linear(model.fc.in_features, 2)
        pathlib.Path(os.path.dirname(binary_fc_file)).mkdir(parents=True, exist_ok=True)
        torch.save(fc.state_dict(), binary_fc_file)

    # Run experiments
    for experiment in args.experiments:
        save_dir = os.path.join(args.model_dir, experiment)
        for pos_class in range(10):
            weights_file = os.path.join(save_dir, '{}_{}.pth'.format(args.arch, pos_class))
            if args.train and not args.overwrite and os.path.isfile(weights_file):
                print('Weights found for {} ({} {}). Skipping...'.format(
                    experiment, pos_class, class_names[pos_class]))
                continue
            if not args.train and not os.path.isfile(weights_file):
                print('Accuracy of {} ({} {}): N/A'.format(
                    experiment, pos_class, class_names[pos_class]))
                continue

            # Setup binary dataset
            binary_datasets = {a: BinaryCIFAR10Subset(image_datasets[b], pos_class,
                                                      start_index=c, end_index=d,
                                                      sample_size=e, balanced=True, random=False)
                               for a, b, c, d, e in zip(['train', 'val', 'test'],
                                                        [0, 0, 1],
                                                        [0, 40000, 0],
                                                        [40000, None, None],
                                                        [1000, 0, 0])}
            dataloaders = {x: DataLoader(binary_datasets[x], batch_size=args.batch_size,
                                         shuffle=True, num_workers=args.num_workers,
                                         pin_memory=('cpu' not in args.device))
                           for x in ['train', 'val', 'test']}
            dataset_sizes = {x: len(binary_datasets[x]) for x in ['train', 'val', 'test']}

            # Setup model
            if experiment == 'last':
                model = mutil.get_model(args.arch, pretrained=True)
                mutil.freeze_model_parameters_(model)
                model.fc = torch.nn.Linear(model.fc.in_features, 2)
                model.fc.load_state_dict(torch.load(binary_fc_file, map_location='cpu'))
            elif experiment == 'full':
                model = mutil.get_model(args.arch, pretrained=True)
                model.fc = torch.nn.Linear(model.fc.in_features, 2)
                model.fc.load_state_dict(torch.load(binary_fc_file, map_location='cpu'))
            elif experiment == 'bn':
                model = mutil.get_model(args.arch, pretrained=True)
                mutil.freeze_model_parameters_(model)
                model.fc = torch.nn.Linear(model.fc.in_features, 2)
                model.fc.load_state_dict(torch.load(binary_fc_file, map_location='cpu'))
                mutil.set_module_trainable_(model, torch.nn.BatchNorm2d)
            elif experiment == 'combn':
                model = mutil.get_model(args.arch, pretrained=True)
                mutil.freeze_model_parameters_(model)
                model.fc = torch.nn.Linear(model.fc.in_features, 2)
                model.fc.load_state_dict(torch.load(binary_fc_file, map_location='cpu'))
                component_classes = [x for x in range(10) if x != pos_class]
                mutil.replace_bn_with_combn_(
                    model, [os.path.join(args.model_dir, 'bn', '{}_{}.pth'.format(args.arch, x))
                            for x in component_classes])
            elif experiment == 'pcbn':
                model = mutil.get_model(args.arch, pretrained=True)
                mutil.freeze_model_parameters_(model)
                model.fc = torch.nn.Linear(model.fc.in_features, 2)
                model.fc.load_state_dict(torch.load(binary_fc_file, map_location='cpu'))
                component_classes = [x for x in range(10) if x != pos_class]
                mutil.replace_bn_with_pcbn_(
                    model, [os.path.join(args.model_dir, 'bn', '{}_{}.pth'.format(args.arch, x))
                            for x in component_classes])
            elif experiment == 'bn_random':
                model = mutil.get_model(args.arch, pretrained=False)
                model.load_state_dict(torch.load(random_file, map_location='cpu'))
                mutil.freeze_model_parameters_(model)
                model.fc = torch.nn.Linear(model.fc.in_features, 2)
                model.fc.load_state_dict(torch.load(binary_fc_file, map_location='cpu'))
                mutil.set_module_trainable_(model, torch.nn.BatchNorm2d)
            elif experiment == 'combn_random':
                model = mutil.get_model(args.arch, pretrained=False)
                model.load_state_dict(torch.load(random_file, map_location='cpu'))
                mutil.freeze_model_parameters_(model)
                model.fc = torch.nn.Linear(model.fc.in_features, 2)
                model.fc.load_state_dict(torch.load(binary_fc_file, map_location='cpu'))
                component_classes = [x for x in range(10) if x != pos_class]
                mutil.replace_bn_with_combn_(
                    model, [os.path.join(args.model_dir, 'bn_random',
                                         '{}_{}.pth'.format(args.arch, x))
                            for x in component_classes])
            elif experiment == 'pcbn_random':
                model = mutil.get_model(args.arch, pretrained=False)
                model.load_state_dict(torch.load(random_file, map_location='cpu'))
                mutil.freeze_model_parameters_(model)
                model.fc = torch.nn.Linear(model.fc.in_features, 2)
                component_classes = [x for x in range(10) if x != pos_class]
                mutil.replace_bn_with_pcbn_(
                    model, [os.path.join(args.model_dir, 'bn_random',
                                         '{}_{}.pth'.format(args.arch, x))
                            for x in component_classes])
            # elif experiment == 'combn_semi_random':
            #     model = mutil.get_model(args.arch, pretrained=False)
            #     model.load_state_dict(torch.load(random_file, map_location='cpu'))
            #     mutil.freeze_model_parameters_(model)
            #     model.fc = torch.nn.Linear(model.fc.in_features, 2)
            #     model.fc.load_state_dict(torch.load(binary_fc_file, map_location='cpu'))
            #     component_classes = [x for x in range(10) if x != pos_class]
            #     mutil.replace_bn_with_combn_(
            #         model, [os.path.join(args.model_dir, 'bn', '{}_{}.pth'.format(args.arch, x))
            #                 for x in component_classes])
            # elif experiment == 'pcbn_semi_random':
            #     model = mutil.get_model(args.arch, pretrained=False)
            #     model.load_state_dict(torch.load(random_file, map_location='cpu'))
            #     mutil.freeze_model_parameters_(model)
            #     model.fc = torch.nn.Linear(model.fc.in_features, 2)
            #     model.fc.load_state_dict(torch.load(binary_fc_file, map_location='cpu'))
            #     component_classes = [x for x in range(10) if x != pos_class]
            #     mutil.replace_bn_with_pcbn_(
            #         model, [os.path.join(args.model_dir, 'bn', '{}_{}.pth'.format(args.arch, x))
            #                 for x in component_classes])
            # elif experiment == 'combn_full_random':
            #     model = mutil.get_model(args.arch, pretrained=False)
            #     model.load_state_dict(torch.load(random_file, map_location='cpu'))
            #     mutil.freeze_model_parameters_(model)
            #     model.fc = torch.nn.Linear(model.fc.in_features, 2)
            #     model.fc.load_state_dict(torch.load(binary_fc_file, map_location='cpu'))
            #     component_classes = [x for x in range(10) if x != pos_class]
            #     mutil.replace_bn_with_combn_(
            #         model, [os.path.join(args.model_dir, 'full', '{}_{}.pth'.format(args.arch, x))
            #                 for x in component_classes])
            # elif experiment == 'pcbn_full_random':
            #     model = mutil.get_model(args.arch, pretrained=False)
            #     model.load_state_dict(torch.load(random_file, map_location='cpu'))
            #     mutil.freeze_model_parameters_(model)
            #     model.fc = torch.nn.Linear(model.fc.in_features, 2)
            #     model.fc.load_state_dict(torch.load(binary_fc_file, map_location='cpu'))
            #     component_classes = [x for x in range(10) if x != pos_class]
            #     mutil.replace_bn_with_pcbn_(
            #         model, [os.path.join(args.model_dir, 'full', '{}_{}.pth'.format(args.arch, x))
            #                 for x in component_classes])
            # elif experiment == 'bn_imagenet_random':
            #     model = mutil.get_model(args.arch, pretrained=False)
            #     model.load_state_dict(torch.load(random_file, map_location='cpu'))
            #     mutil.freeze_model_parameters_(model)
            #     mutil.set_module_trainable_(model, torch.nn.BatchNorm2d)
            #     mutil.part_load_state_dict_(
            #         model,
            #         mutil.get_model(args.arch, pretrained=True).state_dict(),
            #         torch.nn.BatchNorm2d)
            #     model.fc = torch.nn.Linear(model.fc.in_features, 2)
            #     model.fc.load_state_dict(torch.load(binary_fc_file, map_location='cpu'))
            else:
                raise NameError('{} is not recognized.'.format(experiment))
            model.to(device)

            # Train and save model
            if args.train:
                print('Training {} ({} {})...'.format(
                    experiment, pos_class, class_names[pos_class]))
                optimizer = torch.optim.SGD(mutil.get_model_trainable_parameters(model),
                                            lr=args.lr, momentum=args.momentum,
                                            weight_decay=args.weight_decay)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step,
                                                            gamma=args.decay_factor)
                model, _ = mutil.train_model(model,
                                             torch.nn.CrossEntropyLoss().to(device),
                                             optimizer,
                                             dataloaders,
                                             dataset_sizes,
                                             scheduler=scheduler,
                                             num_epochs=args.epoch,
                                             device=device)
                mutil.eval_model(model, dataloaders['test'], dataset_sizes['test'],
                                 device=device)
                pathlib.Path(os.path.dirname(weights_file)).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), weights_file)
            # Evaluate model
            else:
                model.load_state_dict(torch.load(weights_file, map_location='cpu'))
                accuracy = mutil.eval_model(model, dataloaders['test'], dataset_sizes['test'],
                                            device=device, verbose=False)
                print('Accuracy of {} ({} {}): {:.1f}%'.format(
                    experiment, pos_class, class_names[pos_class], accuracy * 100))


if __name__ == '__main__':
    main()
