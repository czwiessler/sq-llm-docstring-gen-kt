import cProfile
import sys
import os
import ast
import time
import yaml
import numpy as np
import configargparse

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from pysemseg import datasets
from pysemseg.metrics import SegmentationMetrics, TorchSegmentationMetrics
from pysemseg.loggers import TensorboardLogger, VisdomLogger, ConsoleLogger
from pysemseg.evaluate import evaluate
from pysemseg.utils import (
    prompt_delete_dir, restore, tensor_to_numpy, import_type,
    flatten_dict, get_latest_checkpoint, save
)


def define_args():
    parser = configargparse.ArgParser(description='PyTorch Segmentation Framework',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add_argument(
        '--config', is_config_file=True, required=False, help='Config file')
    parser.add_argument('--model', type=str, required=True,
                        help=('A path to the model including the module. '
                              'Should be resolvable'))
    parser.add_argument('--model-args', type=ast.literal_eval, required=False, default={},
                        help=('Args passed to the model constructor'))
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to the dataset root dir.')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Path to store output data.')
    parser.add_argument('--dataset', type=str, required=True,
                        help=('Path to the dataset class including the module'))
    parser.add_argument('--dataset-args', type=ast.literal_eval, default={},
                        required=False,
                        help='Dataset args.')
    parser.add_argument('--criterion', type=str, required=False,
                        default='torch.nn.CrossEntropyLoss',
                        help=('Path to the loss class including the module'))
    parser.add_argument('--criterion-args', type=ast.literal_eval,
                        default={'reduction': 'sum'}, required=False,
                        help='Criterion args.')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--max-gpu-batch-size', type=int, default=None,
                        help='Effective GPU batch size. Gradients will be'
                             'accumulated to the batch-size before update.')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--optimizer', type=str, default='RMSprop',
                        required=False,
                        help='Optimizer type.')
    parser.add_argument('--optimizer-args', type=ast.literal_eval, default={},
                        required=False,
                        help='Optimizer args.')
    parser.add_argument('--lr-scheduler', type=str, required=False,
                        default='lr_schedulers.ConstantLR',
                        help='Learning rate scheduler type.')
    parser.add_argument('--lr-scheduler-args', type=ast.literal_eval, default={},
                        required=False,
                        help='Learning rate scheduler args.')
    parser.add_argument('--transformer', type=str, required=False,
                        help='Transformer type')
    parser.add_argument('--transformer-args', type=ast.literal_eval, default={},
                        required=False,
                        help='Transformer args.')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=8861, metavar='S',
                        help='random seed (default: 8861)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='logging training status frequency')
    parser.add_argument('--log-images-interval', type=int, default=200, metavar='N',
                        help='Frequency of logging images and larger plots')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of CPU data workers')
    parser.add_argument('--checkpoint', type=str,
                        required=False,
                        help='Load model on checkpoint.')
    parser.add_argument('--save-model-frequency', type=int,
                        required=False, default=5,
                        help='Save model checkpoint every nth epoch.')
    parser.add_argument('--profile', type=str, required=False,
                        help='Runs a profiler and saves the results in a file.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--allow-missing-keys', action='store_true', default=False,
                        help='Whether to allow module keys to differ from checkpoint keys'
                             ' when loading a checkpoint')
    group.add_argument('--continue-training', action='store_true', default=False,
                       help='Continue experiment from the last checkpoint in the model dir')
    return parser


def train_epoch(
        model, loader, criterion, optimizer, lr_scheduler,
        epoch, console_logger, visual_logger, device, log_interval,
        log_images_interval, accumulate_steps):
    model.train()

    ignore_index = loader.dataset.ignore_index

    metrics = TorchSegmentationMetrics(
        loader.dataset.number_of_classes,
        loader.dataset.labels,
        ignore_index=ignore_index,
        device=device
    )
    epoch_metrics = TorchSegmentationMetrics(
        loader.dataset.number_of_classes,
        loader.dataset.labels,
        ignore_index=ignore_index,
        device=device
    )

    start_time = time.time()
    for step, (ids, data, target) in enumerate(loader):
        data , target = Variable(data.to(device)), Variable(target.to(device, non_blocking=True))
        output = model(data)
        loss = criterion(output, target)
        num_targets = torch.sum(target != ignore_index).float()
        loss = loss / (num_targets * accumulate_steps)
        loss.backward()

        if step % accumulate_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        output = F.softmax(output.detach(), dim=1)
        predictions = torch.argmax(output, dim=1)

        predictions, target, loss = (
            predictions.detach(), target.detach(),
            (loss * accumulate_steps).detach()
        )

        metrics.add(predictions, target, loss)
        epoch_metrics.add(predictions, target, loss)

        if step % (accumulate_steps * log_interval) == 0 and step > 0:
            avg_time = (
                time.time() - start_time) / log_interval
            start_time = time.time()
            metrics_dict = metrics.metrics()
            metrics_dict['time'] = avg_time
            metrics_dict.pop('class')
            console_logger.log(step, epoch, loader, tensor_to_numpy(data), metrics_dict)

            metrics = TorchSegmentationMetrics(
                loader.dataset.number_of_classes,
                loader.dataset.labels,
                ignore_index=ignore_index,
                device=device
            )

        if step % (accumulate_steps * log_images_interval) == 0 and step > 0:
            visual_logger.log_prediction_images(
                step,
                tensor_to_numpy(data.data),
                tensor_to_numpy(target.data),
                tensor_to_numpy(predictions.data),
                name='images',
                prefix='Train'
            )

    visual_logger.log_metrics(epoch, epoch_metrics.metrics(), 'Train')
    visual_logger.log_learning_rate(epoch, optimizer.param_groups[0]['lr'])


def _create_data_loaders(
        data_dir, dataset_cls, dataset_args, transformer_cls, transformer_args,
        train_batch_size, val_batch_size, num_workers, pin_memory=False):
    train_dataset = datasets.create_dataset(
        data_dir, dataset_cls, dataset_args,
        transformer_cls, transformer_args, mode='train')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    validate_dataset = datasets.create_dataset(
        data_dir, dataset_cls, dataset_args,
        transformer_cls, transformer_args, mode='val')

    validate_loader = torch.utils.data.DataLoader(
        validate_dataset, batch_size=val_batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, validate_loader


def _store_args(args, model_dir):
    with open(os.path.join(model_dir, 'args.yaml'), 'w') as args_file:
        yaml.dump(
            {**args.__dict__, 'command': " ".join(sys.argv)},
            args_file
        )


def _set_seed(seed, cuda):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def train(args):
    if not args.continue_training:
        prompt_delete_dir(args.model_dir)
        os.makedirs(args.model_dir)

    _store_args(args, args.model_dir)

    # seed torch and cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    _set_seed(args.seed, args.cuda)

    device = torch.device('cuda:0' if args.cuda else 'cpu:0')

    dataset_cls = import_type(args.dataset, ['pysemseg.datasets'])
    transformer_cls = import_type(args.transformer, ['pysemseg.datasets'])

    args.max_gpu_batch_size = args.max_gpu_batch_size or args.batch_size

    train_loader, validate_loader = _create_data_loaders(
        args.data_dir, dataset_cls, args.dataset_args, transformer_cls,
        args.transformer_args, args.max_gpu_batch_size,
        args.test_batch_size, args.num_workers, pin_memory=args.cuda
    )

    visual_logger = VisdomLogger(
        log_directory=args.model_dir,
        color_palette=train_loader.dataset.color_palette,
        continue_logging=args.continue_training
    )

    visual_logger.log_args(args.__dict__)

    model_class = import_type(args.model, ['pysemseg.models'])
    model = model_class(
        in_channels=train_loader.dataset.in_channels,
        n_classes=train_loader.dataset.number_of_classes,
        **args.model_args
    )

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    criterion_cls = import_type(args.criterion, ['pysemseg.losses'])
    criterion = criterion_cls(
        ignore_index=train_loader.dataset.ignore_index,
        **args.criterion_args
    )

    criterion = criterion.to(device)

    optimizer_class = import_type(args.optimizer, ['torch.optim'])
    optimizer = optimizer_class(
        model.parameters(), lr=args.lr, **args.optimizer_args
    )

    start_epoch = 0

    if args.continue_training:
        args.checkpoint = get_latest_checkpoint(args.model_dir)
        assert args.checkpoint is not None

    lr_scheduler_cls = import_type(
        args.lr_scheduler, ['pysemseg.lr_schedulers', 'torch.optim.lr_scheduler']
    )
    lr_scheduler = lr_scheduler_cls(optimizer, **args.lr_scheduler_args)

    if args.checkpoint:
        start_epoch = restore(
            args.checkpoint, model, optimizer, lr_scheduler,
            strict=not args.allow_missing_keys) + 1

    log_filepath = os.path.join(args.model_dir, 'train.log')

    with ConsoleLogger(filename=log_filepath) as logger:
        for epoch in range(start_epoch, start_epoch + args.epochs):
            train_epoch(
                model, train_loader, criterion, optimizer, lr_scheduler,
                epoch, logger, visual_logger, device, args.log_interval,
                args.log_images_interval,
                args.batch_size // args.max_gpu_batch_size)
            evaluate(
                model, validate_loader, criterion, logger, epoch,
                visual_logger, device, args.log_images_interval)
            if epoch % args.save_model_frequency == 0:
                save(model, optimizer, lr_scheduler, args.model_dir,
                     train_loader.dataset.in_channels,
                     train_loader.dataset.number_of_classes, epoch, args)
            lr_scheduler.step()


def main():
    sys.path.append('./')
    parser = define_args()
    args = parser.parse_args()
    if args.profile is not None:
        cProfile.runctx('train(args)', globals(), locals(), args.profile)
    else:
        train(args)


if __name__ == '__main__':
    main()
