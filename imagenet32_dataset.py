import os
import pickle

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

_base_folder = 'imagenet-32-batches-py'
_train_list = ['train_data_batch_1',
               'train_data_batch_2',
               'train_data_batch_3',
               'train_data_batch_4',
               'train_data_batch_5',
               'train_data_batch_6',
               'train_data_batch_7',
               'train_data_batch_8',
               'train_data_batch_9',
               'train_data_batch_10']
_val_list = ['val_data']
_label_file = 'map_clsloc.txt'


class ImageNet32(Dataset):
    """`ImageNet32 <https://patrykchrabaszcz.github.io/Imagenet32/>`_ dataset.

    Warning: this will load the whole dataset into memory! Please ensure that
    4 GB of memory is available before loading.

    Refer to ``map_clsloc.txt`` for label information.

    The integer labels in this dataset are offset by -1 from ``map_clsloc.txt``
    to make indexing start from 0.

    Args:
        root (string): Root directory of dataset where directory
            ``imagenet-32-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from validation set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        exclude (list, optional): List of class indices to omit from dataset.
        remap_labels (bool, optional): If True and exclude is not None, remaps
            remaining class labels so it is contiguous.

    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, exclude=None, remap_labels=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # Training set or validation set

        # Now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for f in _train_list:
                file = os.path.join(self.root, _base_folder, f)
                with open(file, 'rb') as fo:
                    entry = pickle.load(fo, encoding='latin1')
                    self.train_data.append(entry['data'])
                    self.train_labels += entry['labels']
            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((-1, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # Convert to HWC
            self.train_labels = np.array(self.train_labels) - 1
        else:
            f = _val_list[0]
            file = os.path.join(self.root, _base_folder, f)
            with open(file, 'rb') as fo:
                entry = pickle.load(fo, encoding='latin1')
                self.val_data = entry['data']
                self.val_labels = entry['labels']
            self.val_data = self.val_data.reshape((-1, 3, 32, 32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))  # Convert to HWC
            self.val_labels = np.array(self.val_labels) - 1

        if exclude is not None:
            if self.train:
                include_idx = np.isin(self.train_labels, exclude, invert=True)
                self.train_data = self.train_data[include_idx]
                self.train_labels = self.train_labels[include_idx]

                if remap_labels:
                    mapping = {y: x for x, y in enumerate(np.unique(self.train_labels))}
                    self.train_labels = remap(self.train_labels, mapping)

            else:
                include_idx = np.isin(self.val_labels, exclude, invert=True)
                self.val_data = self.val_data[include_idx]
                self.val_labels = self.val_labels[include_idx]

                if remap_labels:
                    mapping = {y: x for x, y in enumerate(np.unique(self.val_labels))}
                    self.val_labels = remap(self.val_labels, mapping)

        if self.train:
            self.train_labels = self.train_labels.tolist()
        else:
            self.val_labels = self.val_labels.tolist()


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.val_data[index], self.val_labels[index]

        # Doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        if self.train:
            return len(self.train_data)
        return len(self.val_data)


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'val'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class BinaryImageNet32(ImageNet32):
    """Binarized ImageNet32 dataset.

    Refer to ``map_clsloc.txt`` for label information.

    The integer labels in this dataset are offset by -1 from ``map_clsloc.txt``
    to make indexing start from 0.

    Args:
        root (string): Root directory of dataset where directory
            ``imagenet-32-batches-py`` exists.
        pos_label (int): Label (0-999) of positive class.
        pos_size (int, optional): Number of positive examples. If pos_size=0, all valid
            examples are used.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from validation set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    """
    def __init__(self, root, pos_label, pos_size=0, train=True,
                 transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        # Count the number of positive entries and validate
        all_labels = []
        for f in _train_list if self.train else _val_list:
            file = os.path.join(self.root, _base_folder, f)
            with open(file, 'rb') as fo:
                entry = pickle.load(fo, encoding='latin1')
                all_labels += entry['labels']
        all_labels = np.array(all_labels)
        max_pos_size = np.sum(all_labels == (pos_label + 1))
        assert pos_size <= max_pos_size
        self.pos_size = max_pos_size if pos_size == 0 else pos_size
        self.neg_size = self.pos_size

        # Now load the picked numpy arrays
        pos_count = 0
        neg_count = 0
        data = []
        labels = []
        for f in _train_list if self.train else _val_list:
            file = os.path.join(self.root, _base_folder, f)
            with open(file, 'rb') as fo:
                entry = pickle.load(fo, encoding='latin1')

            for i, label in enumerate(entry['labels']):
                if label == (pos_label + 1) and pos_count < self.pos_size:
                    pos_count += 1
                    data.append(entry['data'][i])
                    labels += [1]
                elif label != (pos_label + 1) and neg_count < self.neg_size:
                    neg_count += 1
                    data.append(entry['data'][i])
                    labels += [0]
                elif pos_count >= self.pos_size and neg_count >= self.neg_size:
                    break
        data = np.concatenate(data)
        data = data.reshape((-1, 3, 32, 32))
        data = data.transpose((0, 2, 3, 1))  # convert to HWC

        # Assign loaded data into the correct attributes
        if self.train:
            self.train_data = data
            self.train_labels = labels
        else:
            self.val_data = data
            self.val_labels = labels


def get_imagenet32_labels(root):
    file = os.path.join(root, _base_folder, _label_file)
    return np.loadtxt(file, dtype=str)[:, 2].tolist()


def remap(old_array, mapping):
    new_array = np.copy(old_array)
    for k, v in mapping.items():
        new_array[old_array == k] = v
    return new_array
