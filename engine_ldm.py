# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

#Adapted from https://github.com/1zb/3DShape2VecSet

import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F

import utils_vecset.misc as misc
import utils_vecset.lr_sched as lr_sched


def train_one_epoch_skel(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, cur_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        points, labels, categories, skeletons = cur_data
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        assert skeletons is not None
        skeletons = skeletons.to(device, non_blocking=True)
        categories = categories.to(device, non_blocking=True)

        #print(skeletons.max(), skeletons.min())
        loss = criterion(model, skeletons, skeletons=None, class_labels=categories)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_skel(data_loader, model, criterion, device, args=None):


    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for cur_data in metric_logger.log_every(data_loader, 50, header):

        points, labels, categories, skeletons = cur_data
        #points = points.to(device, non_blocking=True)
        #labels = labels.to(device, non_blocking=True).long()
        assert skeletons is not None
        skeletons = skeletons.to(device, non_blocking=True)
        categories = categories.to(device, non_blocking=True)

        loss = criterion(model, skeletons, skeletons=None, class_labels=categories)
            
        batch_size = skeletons.shape[0]
        
        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_latent(model: torch.nn.Module, ae: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, cur_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if len(cur_data) == 5:
            points, labels, surface, categories, skeletons = cur_data
        else:
            points, labels, surface, categories = cur_data
            skeletons = None
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # points = points.to(device, non_blocking=True)
        # labels = labels.to(device, non_blocking=True).long()
        surface = surface.to(device, non_blocking=True)
        if skeletons is not None:
            skeletons = skeletons.to(device, non_blocking=True)
        categories = categories.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                # print(skeletons.shape, surface.shape)
                # _, x = ae.encode(skeletons, surface)
                x, centers, _ = ae.encoder.forward(surface, skeletons)
                if args.distributed:
                    _, x_encoded = model.module.latent_encoder.encode(x)
                else:
                    _, x_encoded = model.latent_encoder.encode(x)
            if skeletons is not None:
                loss = criterion(model, x_encoded, skeletons=skeletons, class_labels=categories)
            else:
                loss = criterion(model, x, categories)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_latent(data_loader, model, ae, criterion, device, args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for cur_data in metric_logger.log_every(data_loader, 50, header):
        if len(cur_data) == 4:
            points, labels, surface, categories = cur_data
            skeletons = None
        else:
            points, labels, surface, categories, skeletons = cur_data
        # points = points.to(device, non_blocking=True)
        # labels = labels.to(device, non_blocking=True).long()
        surface = surface.to(device, non_blocking=True)
        skeletons = skeletons.to(device, non_blocking=True)
        categories = categories.to(device, non_blocking=True)
        # compute output

        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():

                x, centers, _ = ae.encoder.forward(surface, skeletons)
                if args.distributed:
                    _, x_encoded = model.module.latent_encoder.encode(x)
                else:
                    _, x_encoded = model.latent_encoder.encode(x)
                # print(x.shape, x_encoded.shape)
                # _, x = ae.encode(surface)

            if skeletons is not None:
                # print(skeletons.shape, labels.shape)
                # print(labels)
                loss = criterion(model, x_encoded, skeletons=skeletons, class_labels=categories)
            else:
                loss = criterion(model, x, categories)

        batch_size = surface.shape[0]

        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}