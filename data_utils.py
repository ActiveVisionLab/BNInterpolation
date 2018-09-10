import numpy as np
import torch
from PIL import Image


class ChunkSampler(torch.utils.data.sampler.Sampler):
    """Samples elements sequentially from some offset.

    Args:
        num_samples: # of desired datapoints.
        start: offset where we should start selecting from.
    """

    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


class SubsetSequentialSampler(torch.utils.data.sampler.Sampler):
    """Samples elements sequentially from a given list of indices, without
    replacement.

    Args:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class BinaryCIFAR10Subset(torch.utils.data.Dataset):
    """Samples elements from CIFAR10 and returns a binarised Dataset.

    Args:
        CIFAR10_dataset:
            torchvision.datasets.CIFAR10 object to subset.
        target_label:
            target index that becomes a positive example.
        start_index:
            index of dataset from which subset begins.
        end_index:
            index of dataset to end subset.
        sample_size:
            number of positive examples to draw from.
        negative_size:
            number of negaive examples to draw from.
        balanced:
            whether to create a balanced dataset.
        random:
            whether to sample randomly (without replacement) or sequentially
            from the original dataset.
        exclude:
            list of class indices to exclude from dataset
            before binarization.

    If end_index=None, the subset is created until the end of the dataset.

    If sample_size=0, then all available positive examples within start and end
    indices will be sampled.

    If negative_size=0, then all available negative examples within start and
    end indices will be sampled.

    If balanced=True, negative_size=sample_size is enforced regardless of
    initial value of negative_size.
    """

    def __init__(self, CIFAR10_dataset, target_label, start_index=0, end_index=None,
                 sample_size=0, negative_size=0, balanced=False, random=True,
                 exclude=None):
        # Ensure that most inputs are valid
        if end_index is not None:
            assert start_index < end_index
        assert isinstance(balanced, bool)
        assert isinstance(random, bool)
        target_label = int(target_label)
        sample_size = int(sample_size)

        self.transform = CIFAR10_dataset.transform
        self.target_transform = CIFAR10_dataset.target_transform

        # Reference the correct CIFAR10 data attributes
        if CIFAR10_dataset.train:
            CIFAR10_labels = np.array(
                CIFAR10_dataset.train_labels[start_index:end_index])
            CIFAR10_data = np.array(
                CIFAR10_dataset.train_data[start_index:end_index])
        else:
            CIFAR10_labels = np.array(
                CIFAR10_dataset.test_labels[start_index:end_index])
            CIFAR10_data = np.array(
                CIFAR10_dataset.test_data[start_index:end_index])

        # Exclude classes if specified.
        if exclude is not None:
            include_idx = np.isin(CIFAR10_labels, exclude, invert=True)
            CIFAR10_labels = CIFAR10_labels[include_idx]
            CIFAR10_data = CIFAR10_data[include_idx]

        indices = CIFAR10_labels == target_label

        # Ensure that indices exist
        assert np.sum(indices) != 0

        not_indices = np.flatnonzero(np.logical_not(indices))
        indices = np.flatnonzero(indices)

        # Ensure that specified sample sizes are not larger than available
        assert sample_size <= indices.size
        if not balanced:
            assert negative_size <= not_indices.size

        # Sample CIFAR10 into indices; if statement ordering matters here
        if sample_size == 0:
            pos_sample_indices = indices
        else:
            if random:
                pos_sample_indices = np.random.choice(
                    indices, sample_size, replace=False)
            else:
                pos_sample_indices = indices[:sample_size]

        if balanced:
            if random:
                neg_sample_indices = np.random.choice(
                    not_indices, pos_sample_indices.size, replace=False)
            else:
                neg_sample_indices = not_indices[:pos_sample_indices.size]
        elif negative_size == 0:
            neg_sample_indices = not_indices
        else:
            if random:
                neg_sample_indices = np.random.choice(
                    not_indices, negative_size, replace=False)
            else:
                neg_sample_indices = not_indices[:negative_size]

        # Use sampled indices to build a subset of the Dataset
        combined_indices = np.sort(np.concatenate(
            (pos_sample_indices, neg_sample_indices)))
        self.target_tensor = CIFAR10_labels[combined_indices]
        self.target_tensor = np.array(
            [1 if x == target_label else 0 for x in self.target_tensor])
        self.data_tensor = CIFAR10_data[combined_indices]


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data_tensor[index], self.target_tensor[index]

        # Doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return self.target_tensor.size
