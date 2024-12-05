# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import time
import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
# import sklearn.metrics as sm
# import pandas as pd
# import sklearn.metrics as sm
import random
import numpy as np

# from wideresnet import WideResNet, VNet
import sys
import logging
from copy import deepcopy
from encoder_resnet import *
from wideresnet import WideResNet
from load_corrupted_data import CIFAR10, CIFAR100
from dataloader import *
from torch.utils.data import random_split
from sam.example.animal10.dataset import Animal10
from vgg import vgg19_bn
from cutout import Cutout

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='animal', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--corruption_prob', type=float, default=0.0,
                    help='label noise')
parser.add_argument('--corruption_type', '-ctype', type=str, default='flip2',
                    help='Type of corruption ("unif" or "flip" or "flip2").')
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--epochs', default=120, type=int,
                    help='number of total epochs to run')
parser.add_argument('--iters', default=60000, type=int,
                    help='number of total iters to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', '--batch-size', default=100, type=int,
                    help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='./log/result-vgg-animal-0.0-20220926-114340', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='WideResNet-28-10', type=str,
                    help='name of experiment')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')

parser.add_argument('--save', type=str, default='./log', help='save path')
parser.add_argument('--note', type=str, default='vgg-animal', help='experiment name')
parser.add_argument('--gpu', type=str, default='1', help='gpu device id')
parser.set_defaults(augment=True)

args = parser.parse_args()
use_cuda = True
torch.manual_seed(args.seed)
# device = torch.device("cuda" if use_cuda else "cpu")
torch.cuda.set_device(int(args.gpu))
args.save = '{}/result-{}-{}-{}'.format(args.save, args.note, args.corruption_prob, time.strftime("%Y%m%d-%H%M%S"))
print()


def save_checkpoint(state, is_best, checkpoint=args.save, filename='best_checkpoint.pth.tar'):
    if is_best:
        filepath = os.path.join(checkpoint, filename)
        torch.save(state, filepath)


def _compute_x(encoder, input_train, input_valid):
    train_embed = encoder(input_train)
    val_embed = encoder(input_valid)
    x = torch.matmul(train_embed, torch.transpose(val_embed, 0, 1))
    m = torch.nn.Softmax(dim=1)
    # return torch.nn.softmax()
    x = m(x)
    return x  # x_out
    # return size (ntr * nval)


def _compute_z(target_train, target_valid):
    # print('target_train shape ', target_train.shape, target_valid.shape)
    z = (target_train.view(-1, 1) == target_valid).type(torch.float)
    return z
    ## return z (n_tr * n_val dim tensor)


def _compute_a(model, encoder, r_vec, input_var, target_var, input_validation_var, target_validation_var):
    weight_x = _compute_x(encoder, input_var, input_validation_var)
    weight_z = _compute_z(target_var, target_validation_var)
    y = model(input_validation_var)
    weight_u = F.cross_entropy(y, target_validation_var.long(), reduction='none')
    weight_a = torch.sigmoid(r_vec(weight_x * weight_z * weight_u))

    return weight_a


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


def _hessian_vector_product1(model, vector, input, target, i, r=1e-2):
    R = r / _concat(vector).norm()
    if _concat(vector).norm() == 0:
        print('norm 0 : hessian1')
        return [torch.zeros_like(x) for x in model.b_parameters()]

    for p, v in zip(model.parameters(), vector):
        p.data.add_(v, alpha=R)
    y_f = model(input)
    cost_w = F.cross_entropy(y_f, target.long(), reduction='none')
    # cost_v = torch.reshape(cost_w, (len(cost_w), 1))
    loss = (cost_w * model.weight_b[:, i]).mean()
    grads_p = torch.autograd.grad(loss, model.b_parameters(), create_graph=True)

    for p, v in zip(model.parameters(), vector):
        p.data.sub_(v, alpha=2 * R)
    y_f = model(input)
    cost_w = F.cross_entropy(y_f, target.long(), reduction='none')
    # cost_v = torch.reshape(cost_w, (len(cost_w), 1))
    loss = (cost_w * model.weight_b[:, i]).mean()
    grads_n = torch.autograd.grad(loss, model.b_parameters(), allow_unused=True)

    for p, v in zip(model.parameters(), vector):
        p.data.add_(v, alpha=R)

    return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]


def build_dataset():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.augment:
        train_transform = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            # transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.ToTensor(),
        normalize,
    ])

    if args.dataset == 'cifar10':
        train_data_meta = CIFAR10(
            root='../data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True)
        train_data = CIFAR10(
            root='../data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)
        test_data = CIFAR10(root='../data', train=False, transform=test_transform, download=True)

    elif args.dataset == 'cifar100':
        train_data_meta = CIFAR100(
            root='../data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True)
        train_data = CIFAR100(
            root='../data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)
        test_data = CIFAR100(root='../data', train=False, transform=test_transform, download=True)

    elif args.dataset == 'SVHN':
        train_x, train_y, test_x, test_y = load_svhn_data("../data/SVHN", train_transform, test_transform)
        train_data_all = torch.utils.data.TensorDataset(train_x, train_y)
        test_data = torch.utils.data.TensorDataset(test_x, test_y)
        valid_num = 10000
        train_num = len(train_data_all) - valid_num
        train_data, train_data_meta = torch.utils.data.random_split(
            dataset = train_data_all,
            lengths = [train_num, valid_num],
            generator=torch.Generator().manual_seed(42),
        )
    
    elif args.dataset == 'animal':
        ##### Animal-10 dataset
        transform_train = transforms.Compose([
            # transforms.RandomCrop(size=(32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            Cutout(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_data = Animal10(split='train', data_path='../data/animal10', transform=transform_train)
        
        valid_num = 1000
        train_num = len(train_data) - valid_num
        train_data, train_data_meta = torch.utils.data.random_split(
            dataset = train_data,
            lengths = [train_num, valid_num],
            generator=torch.Generator().manual_seed(42),
        )
        
        test_data = Animal10(split='test', data_path='../data/animal10', transform=transform_test)


    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True, drop_last=True)
    train_meta_loader = torch.utils.data.DataLoader(
        train_data_meta, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.prefetch, pin_memory=True, drop_last=True)

    return train_loader, train_meta_loader, test_loader


def build_model(len_train):
    # model = WideResNet(args.layers, 10, len_train, args.batch_size, args.widen_factor, dropRate=args.droprate)

    model = vgg19_bn(num_classes=10, pretrained=False, len_train=len_train, batch_size=args.batch_size)
    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer, epochs):
    # lr = args.lr * ((0.1 ** int(epochs >= 80)) * (0.1 ** int(epochs >= 100)))
    lr = args.lr * ((0.2 ** int(epochs >= 40)) * (0.2 ** int(epochs >= 80)) * (0.2 ** int(epochs >= 100)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def test(model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.float().cuda(), targets.float().cuda()
            outputs = model(inputs)
            test_loss += F.cross_entropy(outputs, targets.long()).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
        correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    return accuracy

criterion = torch.nn.CrossEntropyLoss()

def train(train_loader,
          train_meta_loader,
          model,
          meta_model,
          encoder,
          r_vec,
          optimizer_model,
          optimizer_meta,
          optimizer_v,
          optimizer_r,
          optimizer_b,
          epoch):
    logging.info('\nEpoch: %d' % epoch)

    train_loss = 0
    meta_loss = 0

    train_meta_loader_iter = iter(train_meta_loader)
    model_old = build_model(len_train)
    meta_model_old = build_model(len_train)
    model_cal = build_model(len_train)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        model.train()
        # lr = optimizer_model.param_groups[0]['lr']
        # lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 90)))
        lr = args.lr * ((0.1 ** int(epoch >= 40)) * (0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 100)))
        inputs, targets = inputs.float().cuda(), targets.float().cuda()
        
        try:
            inputs_val, targets_val = next(train_meta_loader_iter)
        except StopIteration:
            train_meta_loader_iter = iter(train_meta_loader)
            inputs_val, targets_val = next(train_meta_loader_iter)
        inputs_val, targets_val = inputs_val.float().cuda(), targets_val.float().cuda()
        
        # step 1
        model_old.load_state_dict(deepcopy(model.state_dict()))
        y_f = model(inputs)
        cost_w = F.cross_entropy(y_f, targets.long(), reduction='none')
        prec_train = accuracy(y_f.data, targets.data, topk=(1,))[0]
        l_f = (cost_w * model.weight_b[:, batch_idx]).mean()
        optimizer_model.zero_grad()
        l_f.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer_model.step()
        model_cal.load_state_dict(deepcopy(model.state_dict()))

        # step 2
        meta_model.load_state_dict(deepcopy(model.state_dict()))
        meta_model_old.load_state_dict(deepcopy(meta_model.state_dict()))
        y_f_hat = meta_model(inputs)
        cost = F.cross_entropy(y_f_hat, targets.long(), reduction='none')
        cost_v = torch.reshape(cost, (len(cost), 1))
        v_lambda = _compute_a(model_cal, encoder, r_vec, inputs, targets, inputs_val, targets_val)
        l_f_meta = (cost_v * v_lambda).mean()
        optimizer_meta.zero_grad()
        l_f_meta.backward()
        nn.utils.clip_grad_norm_(meta_model.parameters(), 5)
        optimizer_meta.step()

        # step 3
        optimizer_v.zero_grad()
        optimizer_r.zero_grad()
        optimizer_b.zero_grad()

        y_f_hat = meta_model_old(inputs)
        cost = F.cross_entropy(y_f_hat, targets.long(), reduction='none')
        cost_v = torch.reshape(cost, (len(cost), 1))
        v_lambda = _compute_a(model_cal, encoder, r_vec, inputs, targets, inputs_val, targets_val)
        l_f_meta = torch.sum(cost_v * v_lambda).mean()
        grads = torch.autograd.grad(l_f_meta, (meta_model_old.parameters()), create_graph=True)
        y_g_hat = meta_model(inputs_val)
        l_g_meta = F.cross_entropy(y_g_hat, targets_val.long())
        prec_meta = accuracy(y_g_hat.data, targets_val.data, topk=(1,))[0]

        grad_l_meta = torch.autograd.grad(l_g_meta, (meta_model.parameters()))

        assert _concat(grad_l_meta).shape == _concat(grads).shape

        l_v_r_final = torch.dot(_concat(grad_l_meta), _concat(grads))
        l_v_r_final.backward()
        # Encoder grads populated
        for g in encoder.parameters():
            g.grad.mul_(-lr)
        # r grad populated
        for g in r_vec.parameters():
            g.grad.mul_(-lr)

        d_lvalw2p_alpha = [v.grad for v in model.b_parameters()]

        d_ailtr_w2_dot_d_lval_w2p__w1p = [v.grad.data for v in model.parameters()]
        finite_diff_1 = _hessian_vector_product1(model_old, d_ailtr_w2_dot_d_lval_w2p__w1p, inputs, targets, batch_idx)
        # caluculate implicit grads1 and 2 and populate gradients
        for g, g_fd1 in zip(d_lvalw2p_alpha, finite_diff_1):
            g.data.sub_(g_fd1.data.mul(-1), alpha=1)

        for v, g in zip(model.b_parameters(), d_lvalw2p_alpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)
        # print(vnet.linear1.weight.grad)
        nn.utils.clip_grad_norm_(model.b_parameters(), 5)
        nn.utils.clip_grad_norm_(encoder.parameters(), 5)
        nn.utils.clip_grad_norm_(r_vec.parameters(), 5)

        optimizer_v.step()
        optimizer_r.step()
        optimizer_b.step()

        train_loss += l_f.item()
        meta_loss += l_g_meta.item()

        if (batch_idx + 1) % 50 == 0:
            logging.info('Epoch: [%d/%d]\t'
                         'Iters: [%d/%d]\t'
                         'Loss: %.4f\t'
                         'MetaLoss:%.4f\t'
                         'Prec@1 %.2f\t'
                         'Prec_meta@1 %.2f' % (
                            (epoch + 1), args.epochs, batch_idx + 1, len(train_loader.dataset) / args.batch_size,
                            (train_loss / (batch_idx + 1)),
                            (meta_loss / (batch_idx + 1)), prec_train, prec_meta))
        

train_loader, train_meta_loader, test_loader = build_dataset()
len_train = len(train_loader)
# create model
model = build_model(len_train)
meta_model = build_model(len_train)

if args.dataset == 'cifar10' or args.dataset == 'SVHN':
    num_classes = 10
if args.dataset == 'cifar100':
    num_classes = 100

optimizer_model = torch.optim.SGD(model.parameters(), args.lr,
                                  momentum=args.momentum, weight_decay=args.weight_decay)
optimizer_meta = torch.optim.SGD(meta_model.parameters(), args.lr,
                                  momentum=args.momentum, weight_decay=args.weight_decay)
# create LFM related model
encoder = resnet18(pretrained=True).cuda()
r_vec = nn.Sequential(nn.Linear(args.batch_size, 1, bias=False)).cuda()
r_vec[0].weight = nn.Parameter(torch.ones_like(r_vec[0].weight, requires_grad=True))

optimizer_v = torch.optim.SGD(encoder.parameters(), 1e-5,
                              momentum=args.momentum, nesterov=args.nesterov,
                              weight_decay=args.weight_decay)
optimizer_r = torch.optim.SGD(r_vec.parameters(), 1e-5,
                              momentum=args.momentum, nesterov=args.nesterov,
                              weight_decay=args.weight_decay)
optimizer_b = torch.optim.SGD(model.b_parameters(), 1e-2,
                              momentum=args.momentum, nesterov=args.nesterov,
                              weight_decay=args.weight_decay)

scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_b, float(args.epochs), eta_min=1e-5)
scheduler_model = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_model, float(args.epochs), eta_min=1e-5)
scheduler_meta = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_meta, float(args.epochs), eta_min=1e-5)

def main():
    logging.info(args)
    best_acc = 0.0
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(os.path.join(args.resume, 'best_checkpoint.pth.tar'))
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best-acc']

        model.load_state_dict(checkpoint['model'])
        model.weight_b.data = checkpoint['weight-b']
        meta_model.load_state_dict(checkpoint['meta-model'])
        encoder.load_state_dict(checkpoint['encoder'])
        r_vec.load_state_dict(checkpoint['r-vec'])
        scheduler_b.load_state_dict(checkpoint['scheduler-b'])
        meta_model.load_state_dict(checkpoint['meta-model'])

        optimizer_model.load_state_dict(checkpoint['optimizer-a'])
        optimizer_meta.load_state_dict(checkpoint['optimizer-meta'])
        optimizer_b.load_state_dict(checkpoint['optimizer-b'])
        optimizer_v.load_state_dict(checkpoint['optimizer-V'])
        optimizer_r.load_state_dict(checkpoint['optimizer-r'])
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer_model, epoch + 1)
        adjust_learning_rate(optimizer_meta, epoch + 1)
        train(train_loader, train_meta_loader,
              model, meta_model, encoder, r_vec,
              optimizer_model, optimizer_meta, optimizer_v, optimizer_r, optimizer_b, epoch)
        scheduler_b.step()
        # scheduler_model.step()
        # scheduler_meta.step()
        test_acc = test(model=model, test_loader=test_loader)
        if test_acc >= best_acc:
            best_acc = test_acc
            save_checkpoint({
                'epoch': epoch + 1,
                'best-acc': best_acc,

                'model': model.state_dict(),
                'meta-model': meta_model.state_dict(),
                'encoder': encoder.state_dict(),
                'r-vec': r_vec.state_dict(),
                'weight-b': model.weight_b,
                'scheduler-b': scheduler_b.state_dict(),

                'optimizer-a': optimizer_model.state_dict(),
                'optimizer-meta': optimizer_meta.state_dict(),
                'optimizer-b': optimizer_b.state_dict(),
                'optimizer-V': optimizer_v.state_dict(),
                'optimizer-r': optimizer_r.state_dict(),
            }, is_best=True)

    logging.info('best accuracy: %.2f' % best_acc)


if __name__ == '__main__':
    os.mkdir(args.save)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    main()
