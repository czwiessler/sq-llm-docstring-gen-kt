import datetime
import logging
import os
import pprint
import time

import torch
import torch.distributed as dist
import torch.utils.data
from configargparse import ArgumentParser
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageNet, CIFAR10
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

import benchmarks.utils as utils
from benchmarks.models.resnets import oct_resnet50, oct_resnet101, oct_resnet152
from benchmarks.models.resnets_small import resnet20_small, resnet44_small, resnet56_small

models = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnet20_small': resnet20_small,
    'resnet44_small': resnet44_small,
    'resnet56_small': resnet56_small,
    'oct_resnet50': oct_resnet50,
    'oct_resnet101': oct_resnet101,
    'oct_resnet152': oct_resnet152
}


def get_model(arch, **kwargs):
    return models[arch](**kwargs)


def make_data_loader(root,
                     batch_size,
                     dataset='imagenet',
                     workers=4,
                     is_train=True,
                     download=False,
                     distributed=False):
    if dataset == 'imagenet':
        loader = _make_data_loader_imagenet(root=root, batch_size=batch_size, workers=workers,
                                            is_train=is_train, download=download, distributed=distributed)
    elif dataset == 'cifar10':
        loader = _make_data_loader_cifar10(root=root, batch_size=batch_size, workers=workers,
                                           is_train=is_train, download=download, distributed=distributed)
    else:
        raise ValueError('Invalid dataset name')

    return loader


def _make_data_loader_imagenet(root,
                               batch_size,
                               workers=4,
                               is_train=True,
                               download=False,
                               distributed=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    logger = logging.getLogger('octconv')

    if is_train:
        logger.info("Loading ImageNet training data")

        st = time.time()
        scale = (0.08, 1.0)

        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=scale),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        dataset = ImageNet(root=root, split='train', download=download, transform=transform)

        logger.info("Took: {}".format(time.time() - st))

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = torch.utils.data.RandomSampler(dataset)

        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=workers,
                            sampler=sampler,
                            pin_memory=True)
    else:
        logger.info("Loading ImageNet validation data")

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        dataset = ImageNet(root=root, split='val', download=download, transform=transform)

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=workers,
                            sampler=sampler,
                            pin_memory=True)

    return loader


def _make_data_loader_cifar10(root,
                              batch_size,
                              workers=4,
                              is_train=True,
                              download=False,
                              distributed=False):
    logger = logging.getLogger('octconv')

    if is_train:
        logger.info('Loading CIFAR10 Training')

        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        dataset = CIFAR10(root=root, train=True,
                          download=download, transform=transform)

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = torch.utils.data.RandomSampler(dataset)

        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             sampler=sampler,
                                             num_workers=workers)
    else:
        logger.info('Loading CIFAR10 Test')

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        dataset = CIFAR10(root=root, train=False,
                          download=download, transform=transform)

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             sampler=sampler,
                                             num_workers=workers)

    return loader


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = utils.get_world_size()

    if world_size < 2:
        return loss_dict

    with torch.no_grad():
        loss_names = []
        all_losses = []

        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])

        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)

        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size

        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}

    return reduced_losses


def train_one_epoch(model, criterion, optimizer, data_loader,
                    device, epoch, print_freq, writer=None,
                    iter_start=0, mixed_precision=False):
    model.train()
    meters = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    logger = logging.getLogger("octconv")

    iter_nr = iter_start
    for image, target in meters.log_every(data_loader, logger, print_freq, header):

        image, target = image.to(device), target.to(device)
        output = model(image)

        loss = criterion(output, target)

        loss_dict = {'loss': loss}
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced)

        optimizer.zero_grad()

        if mixed_precision:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]

        meters.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        meters.meters['acc1'].update(acc1.item(), n=batch_size)
        meters.meters['acc5'].update(acc5.item(), n=batch_size)

        if writer is not None:
            writer.add_scalar('loss', losses_reduced, global_step=iter_nr)
            writer.add_scalar('train_top1_accuracy', meters.acc1.global_avg, global_step=iter_nr)
            writer.add_scalar('train_top5_accuracy', meters.acc5.global_avg, global_step=iter_nr)

        iter_nr += 1

    return iter_nr


def evaluate(model, criterion, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    logger = logging.getLogger('octconv')
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, logger, 100, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    logger.info(' * Acc@1 {top1.global_avg:.3f} Acc@5 '
                '{top5.global_avg:.3f}'.format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    return metric_logger.acc1.global_avg, metric_logger.acc5.global_avg


def main(args):
    is_distributed = int(os.environ["WORLD_SIZE"]) > 1 if 'WORLD_SIZE' in os.environ else False
    args.distributed = is_distributed

    num_gpus = int(os.environ["WORLD_SIZE"]) if is_distributed else len(args.devices)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        utils.synchronize()

    torch.backends.cudnn.benchmark = True

    if args.mixed_precision:
        try:
            import apex
            from apex import amp
        except ImportError:
            raise ImportError('Mixed precision training requires APEX to be installed')

        assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    logger = utils.setup_logger("octconv", args.output_dir, utils.get_rank())

    logger.info("Using {} GPUs".format(num_gpus))
    logger.info("Arguments: {}".format(pprint.pformat(args.__dict__)))

    # Data loading code
    logger.info("Loading data")

    data_loader = make_data_loader(root=args.root,
                                   batch_size=args.batch_size,
                                   dataset=args.dataset,
                                   workers=args.workers,
                                   is_train=True,
                                   download=args.download,
                                   distributed=args.distributed)

    data_loader_test = make_data_loader(root=args.root,
                                        batch_size=args.batch_size,
                                        dataset=args.dataset,
                                        workers=args.workers,
                                        is_train=False,
                                        download=args.download,
                                        distributed=args.distributed)

    if args.dataset == 'imagenet':
        num_classes = 1000
    elif args.dataset == 'cifar10':
        num_classes = 10
    else:
        raise ValueError('Invalid dataset')

    logger.info("Creating model")

    kwargs = {'num_classes': num_classes}
    if args.arch.startswith('oct'):
        kwargs['alpha'] = args.alpha

    model = get_model(args.arch, **kwargs)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = utils.WarmupMultiStepLR(optimizer,
                                           milestones=args.lr_steps,
                                           gamma=args.lr_gamma,
                                           warmup_iters=args.lr_warmup_epochs)

    if args.devices is not None:
        if isinstance(args.devices, list):
            device = torch.device('cuda')
            if not args.distributed:
                model = nn.DataParallel(model, args.devices)
        else:
            if args.distributed:
                device = torch.device('cuda')
            else:
                device = torch.device('cuda:{}'.format(args.devices))
    else:
        device = 'cpu'

    model = model.to(device)

    if args.mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    logger.info(model)

    model_without_ddp = model
    if args.distributed:
        if args.mixed_precision:
            model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank], output_device=args.local_rank,
                broadcast_buffers=False
            )
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    if args.test_only:
        evaluate(model, criterion, data_loader_test, device=device)
        return

    writer = None
    if args.tensorboard and utils.is_main_process():
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
        writer = SummaryWriter(log_dir='{}/runs/{}_{}'.format(args.output_dir, args.arch, timestamp))

    logger.info("Start training")
    best_acc1 = 0
    best_acc5 = 0
    best_epoch = 0
    iter_nr = 0
    start_time = time.time()
    for epoch in range(args.epochs):

        if args.distributed:
            data_loader.sampler.set_epoch(epoch)

        iter_nr = train_one_epoch(model=model, criterion=criterion,
                                  optimizer=optimizer, data_loader=data_loader, device=device,
                                  epoch=epoch, print_freq=args.print_freq, writer=writer,
                                  iter_start=iter_nr, mixed_precision=args.mixed_precision)

        acc1, acc5 = evaluate(model, criterion, data_loader_test, device=device)

        if writer is not None:
            writer.add_scalar('test_top1_accuracy', acc1, global_step=iter_nr)
            writer.add_scalar('test_top5_accuracy', acc5, global_step=iter_nr)
            writer.add_scalar('lr', optimizer.param_groups[0]["lr"], global_step=iter_nr)

        lr_scheduler.step()
        utils.synchronize()

        if acc1 > best_acc1:
            best_epoch = epoch
            best_acc1 = acc1
            best_acc5 = acc5

        if args.output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args},
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    logger.info('Best Top-1 Accuracy: {} [Top-5 Accuracy: {}] (Epoch {})'.format(best_acc1, best_acc5, best_epoch))


if __name__ == "__main__":
    parser = ArgumentParser(description='ImageNet training')

    parser.add_argument('-c', '--config', is_config_file=True, help='config file')

    # Data
    parser.add_argument('--root', required=True, help='dataset')
    parser.add_argument('--dataset', default='imagenet', help='dataset name')
    parser.add_argument('--download', action='store_true', default=False, help='download ImageNet')
    parser.add_argument('--workers', default=16, type=int, help='number of data loading workers (default: 16)')

    # Model
    parser.add_argument('--arch', default='resnet18', help='model')
    parser.add_argument('--alpha', default=0.125, type=float, help='OctConv alpha parameter')
    parser.add_argument('--devices', default=0, type=int, help='GPU devices', nargs='*')

    # Batch Size & Epochs
    parser.add_argument('--batch-size', default=32, type=int, help='total batch size')
    parser.add_argument('--epochs', default=90, type=int, help='number of total epochs to run')

    # Learning Rate
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='learning rate gamma scaling factor')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-steps', default=None, type=int, action='append', help='decrease lr every step epochs')
    parser.add_argument('--lr-warmup-epochs', default=2, type=int, help='number of warmup epochs')

    # Others
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./output', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument("--test-only", dest="test_only", action="store_true", default=False, help="Only test the model")
    parser.add_argument('--tensorboard', action='store_true', help='log to tensorboard')
    parser.add_argument('--mixed-precision', action='store_true', help='use APEX automatic mixed precision')

    # distributed training parameters
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    if not args.lr_steps:
        args.lr_steps = [40, 80]

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)
