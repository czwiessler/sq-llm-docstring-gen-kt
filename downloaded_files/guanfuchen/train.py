# -*- coding: utf-8 -*-_resnet18_32s
import datetime

import torch
import os
import argparse

import cv2
import time
import numpy as np
import visdom
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR

from semseg.dataloader.camvid_loader import camvidLoader
from semseg.dataloader.cityscapes_loader import cityscapesLoader
from semseg.dataloader.freespace_loader import freespaceLoader
from semseg.dataloader.movingmnist_loader import movingmnistLoader
from semseg.dataloader.segmpred_loader import segmpredLoader
from semseg.loss import cross_entropy2d
from semseg.metrics import scores
from semseg.modelloader.EDANet import EDANet
from semseg.modelloader.bisenet import BiSeNet
from semseg.modelloader.deconvnet import DeConvResNet50, DeConvResNet18
from semseg.modelloader.deeplabv3 import Res_Deeplab_101, Res_Deeplab_50
from semseg.modelloader.drn import drn_d_22, DRNSeg, drn_a_asymmetric_18, drn_a_asymmetric_ibn_a_18, drnseg_a_50, drnseg_a_18, drnseg_a_34, drnseg_e_22, drnseg_a_asymmetric_18, drnseg_a_asymmetric_ibn_a_18, drnseg_d_22, drnseg_d_38
from semseg.modelloader.drn_a_irb import drnsegirb_a_18
from semseg.modelloader.drn_a_refine import drnsegrefine_a_18
from semseg.modelloader.duc_hdc import ResNetDUC, ResNetDUCHDC
from semseg.modelloader.enet import ENet
from semseg.modelloader.enetv2 import ENetV2
from semseg.modelloader.erfnet import erfnet
from semseg.modelloader.fc_densenet import fcdensenet103, fcdensenet56, fcdensenet_tiny
from semseg.modelloader.fcn import fcn, fcn_32s, fcn_16s, fcn_8s
from semseg.modelloader.fcn_mobilenet import fcn_MobileNet, fcn_MobileNet_32s, fcn_MobileNet_16s, fcn_MobileNet_8s
from semseg.modelloader.fcn_resnet import fcn_resnet18, fcn_resnet34, fcn_resnet18_32s, fcn_resnet18_16s, \
    fcn_resnet18_8s, fcn_resnet34_32s, fcn_resnet34_16s, fcn_resnet34_8s, fcn_resnet50_32s, fcn_resnet50_16s, fcn_resnet50_8s
from semseg.modelloader.fcn_shufflenet import fcn_shufflenet_32s, fcn_shufflenet_16s, fcn_shufflenet_8s
from semseg.modelloader.gcn import gcn_resnet18, gcn_resnet34, gcn_resnet50, gcn_resnet101
from semseg.modelloader.lrn import lrn_vgg16
from semseg.modelloader.segnet import segnet, segnet_squeeze, segnet_alignres, segnet_vgg19
from semseg.modelloader.segnet_unet import segnet_unet
from semseg.modelloader.sqnet import sqnet
from semseg.schedulers import ConstantLR, PolynomialLR
from semseg.utils.get_class_weights import median_frequency_balancing, ENet_weighing


def train(args):
    now = datetime.datetime.now()
    now_str = '{}-{}-{} {}:{}:{}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    # print('now:', now)
    # print('now_str:', now_str)
    if args.vis:
        # start visdom and close all window
        vis = visdom.Visdom(env=now_str)
        vis.close()

    class_weight = None
    local_path = os.path.expanduser(args.dataset_path)
    train_dst = None
    val_dst = None
    if args.dataset == 'CamVid':
        train_dst = camvidLoader(local_path, is_transform=True, is_augment=args.data_augment, split='train')
        val_dst = camvidLoader(local_path, is_transform=True, is_augment=False, split='val')

        trainannot_image_dir = os.path.expanduser(os.path.join(local_path, "trainannot"))
        trainannot_image_files = [os.path.join(trainannot_image_dir, file) for file in os.listdir(trainannot_image_dir) if file.endswith('.png')]
        if args.class_weighting=='MFB':
            class_weight = median_frequency_balancing(trainannot_image_files, num_classes=12)
            class_weight = torch.tensor(class_weight)
        elif args.class_weighting=='ENET':
            class_weight = ENet_weighing(trainannot_image_files, num_classes=12)
            class_weight = torch.tensor(class_weight)
    elif args.dataset == 'CityScapes':
        train_dst = cityscapesLoader(local_path, is_transform=True, split='train')
        val_dst = cityscapesLoader(local_path, is_transform=True, split='val')
    elif args.dataset == 'SegmPred':
        train_dst = segmpredLoader(local_path, is_transform=True, split='train')
        val_dst = segmpredLoader(local_path, is_transform=True, split='train')
    elif args.dataset == 'MovingMNIST':
        # class_weight = [0.1, 0.5]
        # class_weight = torch.tensor(class_weight)
        train_dst = movingmnistLoader(local_path, is_transform=True, split='train')
        val_dst = movingmnistLoader(local_path, is_transform=True, split='val')
    elif args.dataset == 'FreeSpace':
        train_dst = freespaceLoader(local_path, is_transform=True, split='train')
        val_dst = freespaceLoader(local_path, is_transform=True, split='val')
    else:
        print('{} dataset does not implement'.format(args.dataset))
        exit(0)

    if args.cuda:
        if class_weight is not None:
            class_weight = class_weight.cuda()
    print('class_weight:', class_weight)

    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dst, batch_size=1, shuffle=True)

    start_epoch = 0
    best_mIoU = 0
    if args.resume_model != '':
        model = torch.load(args.resume_model)
        start_epoch_id1 = args.resume_model.rfind('_')
        start_epoch_id2 = args.resume_model.rfind('.')
        start_epoch = int(args.resume_model[start_epoch_id1+1:start_epoch_id2])
    else:
        # model = eval(args.structure)(n_classes=args.n_classes, pretrained=args.init_vgg16)
        try:
            model = eval(args.structure)(n_classes=args.n_classes, pretrained=args.init_vgg16)
        except:
            print('missing structure or not support')
            exit(0)

        # ---------------for testing SegmPred---------------
        if args.dataset == 'MovingMNIST':
            input_channel = 1*9
        elif args.dataset == 'SegmPred':
            input_channel = 19*4
        if args.structure == 'drnseg_a_18':
            model = drnseg_a_18(n_classes=args.n_classes, pretrained=args.init_vgg16, input_channel=input_channel)
        # ---------------for testing SegmPred---------------

        if args.resume_model_state_dict != '':
            try:
                # from model save format get useful information, such as miou, epoch
                miou_model_name_str = '_miou_'
                class_model_name_str = '_class_'
                miou_id1 = args.resume_model_state_dict.find(miou_model_name_str)+len(miou_model_name_str)
                miou_id2 = args.resume_model_state_dict.find(class_model_name_str)
                best_mIoU = float(args.resume_model_state_dict[miou_id1:miou_id2])

                start_epoch_id1 = args.resume_model_state_dict.rfind('_')
                start_epoch_id2 = args.resume_model_state_dict.rfind('.')
                start_epoch = int(args.resume_model_state_dict[start_epoch_id1 + 1:start_epoch_id2])
                pretrained_dict = torch.load(args.resume_model_state_dict, map_location='cpu')
                model.load_state_dict(pretrained_dict)
            except KeyError:
                print('missing resume_model_state_dict or wrong type')



    if args.cuda:
        model.cuda()
    print('start_epoch:', start_epoch)
    print('best_mIoU:', best_mIoU)

    if args.solver == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.99, weight_decay=5e-4)
    elif args.solver == 'RMSprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.99, weight_decay=5e-4)
    elif args.solver == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-4)
    else:
        print('missing solver or not support')
        exit(0)
    # when observerd object dose not decrease scheduler will let the optimizer learing rate decrease
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=100, min_lr=1e-10, verbose=True)
    if args.lr_policy == 'Constant':
        scheduler = ConstantLR(optimizer)
    elif args.lr_policy == 'Polynomial':
        scheduler = PolynomialLR(optimizer, max_iter=args.training_epoch, power=0.9) # base lr=0.01 power=0.9 like PSPNet
    elif args.lr_policy == 'MultiStep':
        scheduler = MultiStepLR(optimizer, milestones=[10, 50, 90], gamma=0.1) # base lr=0.01 power=0.9 like PSPNet

    # scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    data_count = int(train_dst.__len__() * 1.0 / args.batch_size)
    print('data_count:', data_count)
    # iteration_step = 0
    train_gts, train_preds = [], []
    for epoch in range(start_epoch+1, args.training_epoch, 1):
        loss_epoch = 0
        scheduler.step()

        optimizer.zero_grad() # when train next time zero all grad, just acc the grad when the epoch training
        for i, (imgs, labels) in enumerate(train_loader):
            # if i==1:
            #     break
            model.train()

            # 最后的几张图片可能不到batch_size的数量，比如batch_size=4，可能只剩3张
            imgs_batch = imgs.shape[0]
            if imgs_batch != args.batch_size:
                break
            # iteration_step += 1

            imgs = Variable(imgs)
            labels = Variable(labels)

            if args.cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()
            outputs = model(imgs)
            # print('imgs.size:', imgs.size())
            # print('labels.size:', labels.size())
            # print('outputs.size:', outputs.size())

            loss = cross_entropy2d(outputs, labels, weight=class_weight)

            # add grad backward the avg loss
            loss_grad_acc_avg = loss*1.0/args.grad_acc_steps
            loss_grad_acc_avg.backward()

            loss_np = loss.cpu().data.numpy()
            loss_epoch += loss_np


            if (i+1)%args.grad_acc_steps == 0:
                optimizer.step()
                # 一次backward后如果不清零，梯度是累加的
                optimizer.zero_grad()

            # ------------------train metris-------------------------------
            train_pred = outputs.cpu().data.max(1)[1].numpy()
            train_gt = labels.cpu().data.numpy()

            for train_gt_, train_pred_ in zip(train_gt, train_pred):
                train_gts.append(train_gt_)
                train_preds.append(train_pred_)
            # ------------------train metris-------------------------------

            if args.vis and i%50==0:
                pred_labels = outputs.cpu().data.max(1)[1].numpy()
                label_color = train_dst.decode_segmap(labels.cpu().data.numpy()[0]).transpose(2, 0, 1)
                pred_label_color = train_dst.decode_segmap(pred_labels[0]).transpose(2, 0, 1)
                win = 'label_color'
                vis.image(label_color, win=win, opts=dict(title='Gt', caption='Ground Truth'))
                win = 'pred_label_color'
                vis.image(pred_label_color, win=win, opts=dict(title='Pred', caption='Prediction'))

            # 显示一个周期的loss曲线
            if args.vis:
                win = 'loss_iteration'
                loss_np_expand = np.expand_dims(loss_np, axis=0)
                win_res = vis.line(X=np.ones(1)*(i+data_count*(epoch-1)+1), Y=loss_np_expand, win=win, update='append')
                if win_res != win:
                    vis.line(X=np.ones(1)*(i+data_count*(epoch-1)+1), Y=loss_np_expand, win=win, opts=dict(title=win, xlabel='iteration', ylabel='loss'))

        # val result on val dataset and pick best to save
        if args.val_interval > 0  and epoch % args.val_interval == 0:
            print('----starting val----')
            model.eval()

            val_gts, val_preds = [], []
            for val_i, (val_imgs, val_labels) in enumerate(val_loader):
                # print(val_i)
                val_imgs = Variable(val_imgs, volatile=True)
                val_labels = Variable(val_labels, volatile=True)

                if args.cuda:
                    val_imgs = val_imgs.cuda()
                    val_labels = val_labels.cuda()

                val_outputs = model(val_imgs)
                val_pred = val_outputs.cpu().data.max(1)[1].numpy()
                val_gt = val_labels.cpu().data.numpy()
                for val_gt_, val_pred_ in zip(val_gt, val_pred):
                    val_gts.append(val_gt_)
                    val_preds.append(val_pred_)

            score, class_iou = scores(val_gts, val_preds, n_class=args.n_classes)
            for k, v in score.items():
                print(k, v)
                if k == 'Mean IoU : \t':
                    v_iou = v
                    if v > best_mIoU:
                        best_mIoU = v_iou
                        torch.save(model.state_dict(), '{}_{}_miou_{}_class_{}_{}.pt'.format(args.structure, args.dataset, best_mIoU, args.n_classes, epoch))
                    # 显示校准周期的mIoU
                    if args.vis:
                        win = 'mIoU_epoch'
                        v_iou_expand = np.expand_dims(v_iou, axis=0)
                        win_res = vis.line(X=np.ones(1)*epoch*args.val_interval, Y=v_iou_expand, win=win, update='append')
                        if win_res != win:
                            vis.line(X=np.ones(1)*epoch*args.val_interval, Y=v_iou_expand, win=win, opts=dict(title=win, xlabel='epoch', ylabel='mIoU'))

            for class_i in range(args.n_classes):
                print(class_i, class_iou[class_i])
            print('----ending   val----')

        # 显示多个周期的loss曲线
        loss_avg_epoch = loss_epoch / (data_count * 1.0)
        # print(loss_avg_epoch)
        if args.vis:
            win = 'loss_epoch'
            loss_avg_epoch_expand = np.expand_dims(loss_avg_epoch, axis=0)
            win_res = vis.line(X=np.ones(1)*epoch, Y=loss_avg_epoch_expand, win=win, update='append')
            if win_res != win:
                vis.line(X=np.ones(1)*epoch, Y=loss_avg_epoch_expand, win=win, opts=dict(title=win, xlabel='epoch', ylabel='loss'))

        if args.vis:
            win = 'lr_epoch'
            lr_epoch = np.array(scheduler.get_lr())
            lr_epoch_expand = np.expand_dims(lr_epoch, axis=0)
            win_res = vis.line(X=np.ones(1)*epoch, Y=lr_epoch_expand, win=win, update='append')
            if win_res != win:
                vis.line(X=np.ones(1)*epoch, Y=lr_epoch_expand, win=win, opts=dict(title=win, xlabel='epoch', ylabel='lr'))

        # ------------------train metris-------------------------------
        if args.vis:
            score, class_iou = scores(train_gts, train_preds, n_class=args.n_classes)
            for k, v in score.items():
                print(k, v)
                if k == 'Overall Acc : \t':
                    # 显示校准周期的mIoU
                    overall_acc = v
                    if args.vis:
                        win = 'acc_epoch'
                        overall_acc_expand = np.expand_dims(overall_acc, axis=0)
                        win_res = vis.line(X=np.ones(1) * epoch, Y=overall_acc_expand, win=win,
                                           update='append')
                        if win_res != win:
                            vis.line(X=np.ones(1) * epoch, Y=overall_acc_expand, win=win,
                                     opts=dict(title=win, xlabel='epoch', ylabel='accuracy'))
            # clear for new training metrics
            train_gts, train_preds = [], []
        # ------------------train metris-------------------------------

        if args.save_model and epoch%args.save_epoch==0:
            torch.save(model.state_dict(), '{}_{}_class_{}_{}.pt'.format(args.structure, args.dataset, args.n_classes, epoch))


# best training: python train.py --resume_model fcn32s_camvid_9.pkl --save_model True
# --init_vgg16 True --dataset_path /home/cgf/Data/CamVid --batch_size 1 --vis True
if __name__=='__main__':
    # print('train----in----')
    parser = argparse.ArgumentParser(description='training parameter setting')
    parser.add_argument('--structure', type=str, default='ENetV2', help='use the net structure to segment [ fcn_32s ResNetDUC segnet ENet drn_d_22 ]')
    parser.add_argument('--solver', type=str, default='Adam', help='use the solver to optimizer net [ SGD Adam RMSprop ]')
    parser.add_argument('--grad_acc_steps', type=int, default=1, help='gpu memory not enough use grad accumulation act like large batch [ 1 ]')
    parser.add_argument('--resume_model', type=str, default='', help='resume model path [ fcn32s_camvid_9.pkl ]')
    parser.add_argument('--resume_model_state_dict', type=str, default='', help='resume model state dict path [ fcn32s_camvid_9.pt ]')
    parser.add_argument('--save_model', type=bool, default=False, help='save model [ False ]')
    parser.add_argument('--save_epoch', type=int, default=1, help='save model after epoch [ 1 ]')
    parser.add_argument('--training_epoch', type=int, default=500, help='training epoch end training model [ 30000 ]')
    parser.add_argument('--init_vgg16', type=bool, default=False, help='init model using vgg16 weights [ False ]')
    parser.add_argument('--dataset', type=str, default='CamVid', help='train dataset [ CamVid CityScapes FreeSpace SegmPred MovingMNIST ]')
    parser.add_argument('--dataset_path', type=str, default='~/Data/CamVid', help='train dataset path [ ~/Data/CamVid ~/Data/cityscapes ~/Data/FreeSpaceDataset ~/Data/SegmPred ~/Data/mnist_test_seq.npy]')
    parser.add_argument('--data_augment', type=bool, default=True, help='enlarge the training data [ True False ]')
    parser.add_argument('--class_weighting', type=str, default='MFB', help='weighting class [ MFB ENET ]')
    parser.add_argument('--batch_size', type=int, default=1, help='train dataset batch size [ 1 ]')
    parser.add_argument('--val_interval', type=int, default=-1, help='val dataset interval unit epoch [ 3 ]')
    parser.add_argument('--n_classes', type=int, default=12, help='train class num [ 12 ]')
    parser.add_argument('--lr', type=float, default=1e-4, help='train learning rate [ 0.00001 ]')
    parser.add_argument('--lr_policy', type=str, default='Polynomial', help='train learning policy [ Constant Polynomial MultiStep ]')
    parser.add_argument('--vis', type=bool, default=True, help='visualize the training results [ False ]')
    parser.add_argument('--cuda', type=bool, default=False, help='use cuda [ False ]')
    args = parser.parse_args()
    print(args)
    train(args)
    # print('train----out----')
