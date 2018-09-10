# Interpolating Convolutional Neural Networks using Batch Normalization

This repository provides training and utility scripts to reproduce results reported in [1]. For brevity, we will only mention the most important argument options in this readme file. Users wishing to use other options or adapt the techniques implemented here for their own work can reuse our modules and refer to the main scripts (`experiment1.py` and `experiment2.py`).

## Prerequisites

- Python 3.6
- [PyTorch 0.4.1](https://pytorch.org) (preferably with GPU support)
- [ImageNet32](https://patrykchrabaszcz.github.io/Imagenet32/) (see below)

## ImageNet32 Setup

- Download [ImageNet32](http://image-net.org/download-images).
- Extract both `imagenet32_train.zip` and `imagenet32_val.zip` to `./datasets/imagenet-32-batches-py/`.
- Download `map_clsloc.txt` from [here](https://raw.githubusercontent.com/PatrykChrabaszcz/Imagenet32_Scripts/master/map_clsloc.txt) and save to `./datasets/imagenet-32-batches-py/`.

## Experiment 1 (Learning CIFAR10 from ImageNet Template)

### Training

Simply run the following script to begin training models that were reported in Table 1 of [1].

```bash
python experiment1.py
```

This will run all experiments and will take some time to complete. By default, the script supports resuming by not overwriting models that have been trained (checkpointing is not supported yet).

If you wish to select only a few experiments to run, pass `--experiments` as an argument, e.g.

```bash
python experiment1.py --experiments last full bn
```

This will only train models for "Last", "Full", and "BN" in Table 1.

Here are all the keywords for `experiment1.py`:

```bash
last full bn combn pcbn bn_random combn_random pcbn_random
```

However, note that `combn` and `pcbn` requires models for `bn` to have finished training, and similarly for `combn_random` and `pcbn_random` requiring `bn_random`.

### Evaluation

Running the following script will print out the test accuracies of all models that have finished training.

```bash
python experiment1.py --evaluate
```

As before, one can select which results to print:

```bash
python experiment1.py --evaluate --experiments last full bn
```

This will only print results for "Last", "Full", and "BN".

## Experiment 2 (Few-shot Learning ImageNet32 from CIFAR10 Template)

### Training

The following script executes the necessary steps to reproduce results in Table 2 of [1].

```bash
python experiment2.py
```

By default, the script performs 1-shot experiments. To change this, use the `--shot` argument, e.g.

```bash
python experiment2.py --shot 5
```

This causes the script to perform 5-shot experiments.

As before, specifying `--experiments` allow choice in which experiments which will be run. The full list includes:

```bash
last
full
bn
combn_loss_3
combn_loss_5
combn_loss_10
combn_accuracy_3
combn_accuracy_5
combn_accuracy_10
combn_threshold_0.75
pcbn_loss_3
pcbn_loss_5
pcbn_loss_10
pcbn_accuracy_3
pcbn_accuracy_5
pcbn_accuracy_10
pcbn_threshold_0.75
sgm
l2
```

Note that the script attempts to parse experiments that contain `combn`/`pcbn` as a substring. The experiment string should follow the format `[module]_[component_selection]_[num_components]`. For example, `combn_loss_3` means that 3 BN components will be selected by few-shot loss and combined using PCBN. This allows quick testing of different configurations without modifying the source code.

Note that the only valid entries for `[module]` are `combn`/`pcbn`. For `[component_selection]`, the valid values are `loss` (few-shot loss), and `accuracy`/`threshold` (max-shot accuracy). If `threshold` is specified, `[num_components]` instead changes the accuracy percentage threshold for component selection.

### Evaluation

Running the following script will print out mean validation accuracy of all models per experiment that has finished training.

```bash
python experiment2.py --evaluate
```

## References

[1] Data, G. W. P., Ngu, K., Murray, D. W., Prisacariu, V. A.: Interpolating Convolutional Neural Networks Using Batch Normalization. ECCV 2018.