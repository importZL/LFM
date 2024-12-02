import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import copy
from model_search_lfm import Network
from genotypes import PRIMITIVES
from genotypes import Genotype
from encoder_resnet import *
from types import SimpleNamespace
from architect_lfm import *
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--workers', type=int, default=2, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=192, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=25, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='./', help='experiment path')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--tmp_data_dir', type=str, default='../data', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--dropout_rate', action='append', default=[0.1, 0.4, 0.7], help='dropout rate of skip connect')
parser.add_argument('--add_width', action='append', default=['0'], help='add channels')
parser.add_argument('--add_layers', action='append', default=['0', '6', '12'], help='add layers')
parser.add_argument('--gpu', type=str, default='1', help='gpu device id')


# new hyperparams.
parser.add_argument('--learning_rate_w1', type=float, default=0.25)
parser.add_argument('--learning_rate_w2', type=float, default=0.1)
parser.add_argument('--learning_rate_A', type=float, default=6e-4)
parser.add_argument('--learning_rate_V', type=float, default=1e-2)
parser.add_argument('--learning_rate_r', type=float, default=1e-2)
parser.add_argument('--momentum_w1', type=float, default=0.9, help='momentum')
parser.add_argument('--momentum_w2', type=float, default=0.9, help='momentum')
parser.add_argument('--momentum_A', type=float, default=0.9, help='momentum')
parser.add_argument('--momentum_V', type=float, default=0.9, help='momentum')
parser.add_argument('--momentum_r', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay_w1', type=float, default=1e-4)
parser.add_argument('--weight_decay_w2', type=float, default=1e-4)
parser.add_argument('--weight_decay_A', type=float, default=1e-5)
parser.add_argument('--weight_decay_V', type=float, default=1e-4)
parser.add_argument('--weight_decay_r', type=float, default=1e-4)
parser.add_argument('--grad_clip_w1', type=float, default=5)
parser.add_argument('--grad_clip_w2', type=float, default=5)
parser.add_argument('--grad_clip_A', type=float, default=5)
parser.add_argument('--grad_clip_V', type=float, default=5)
parser.add_argument('--grad_clip_r', type=float, default=5)
parser.add_argument('--is_parallel', type=int, default=0)
parser.add_argument('--encoder_size', type=str, default='18')
parser.add_argument('--is_cifar100', type=int, default=1)
parser.add_argument('--resume', type=str, default='')

args = parser.parse_args()

args.save = '{}search-{}-{}-cifar100'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(filename_suffix=time.strftime("%Y%m%d-%H%M%S"))

if args.is_cifar100:
    CIFAR_CLASSES = 100
    data_folder = 'cifar-100-python'
else:
    CIFAR_CLASSES = 10
    data_folder = 'cifar-10-batches-py'


def save_checkpoint(state, checkpoint=args.save, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


def main():

    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
        
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(int(args.gpu))
    logging.info("args = %s", args)

    #  prepare dataset
    if args.is_cifar100:
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
    else:
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
    if args.is_cifar100:
        train_data = dset.CIFAR100(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
    else:
        train_data = dset.CIFAR10(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=args.workers, drop_last=True)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=args.workers, drop_last=True)

    # encoder contains V
    if args.encoder_size == '18':
        encoder = resnet18(pretrained=True).cuda()
    elif args.encoder_size == '34':
        encoder = resnet34(pretrained=True).cuda()
    elif args.encoder_size == '50':
        encoder = resnet50(pretrained=True).cuda()
    elif args.encoder_size == '101':
        encoder = resnet101(pretrained=True).cuda()

    # contains r
    # TODO: check input size
    r_vec = nn.Sequential(nn.Linear(64, 1, bias=False)).cuda()
    r_vec[0].weight = nn.Parameter(torch.ones_like(r_vec[0].weight, requires_grad=True))

    # build Network
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    switches = []
    for i in range(14):
        switches.append([True for j in range(len(PRIMITIVES))])
    switches_normal = copy.deepcopy(switches)
    switches_reduce = copy.deepcopy(switches)
    # To be moved to args
    num_to_keep = [5, 3, 1]
    num_to_drop = [3, 2, 2]
    if len(args.add_width) == 3:
        add_width = args.add_width
    else:
        add_width = [0, 0, 0]
    if len(args.add_layers) == 3:
        add_layers = args.add_layers
    else:
        add_layers = [0, 6, 12]
    if len(args.dropout_rate) == 3:
        drop_rate = args.dropout_rate
    else:
        drop_rate = [0.0, 0.0, 0.0]
    eps_no_archs = [10, 10, 10]  # epochs train with fixed architecture
    sp = 0
    while sp < len(num_to_keep):
        
        if args.resume:
            checkpoint = torch.load(os.path.join(args.resume, 'checkpoint.pth.tar'))
            sp = checkpoint['stage'] - 1
            switches_normal = checkpoint['switches_normal']
            switches_reduce = checkpoint['switches_reduce']

        model = Network(args.init_channels + int(add_width[sp]), CIFAR_CLASSES, args.layers + int(add_layers[sp]),
                        criterion, switches_normal=switches_normal, switches_reduce=switches_reduce,
                        p=float(drop_rate[sp]))

        if args.resume:
            model = torch.load(os.path.join(args.resume, 'weights_model.pt')).cuda()
            encoder = torch.load(os.path.join(args.resume, 'weights_encoder.pt')).cuda()
            r_vec = torch.load(os.path.join(args.resume, 'weights_r.pt')).cuda()

        # model = nn.DataParallel(model)
        model = model.cuda()
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
        
        sm_dim = -1
        epochs = args.epochs
        eps_no_arch = eps_no_archs[sp]
        scale_factor = 0.2

        optimizers = SimpleNamespace(
            w1=torch.optim.SGD(
                model.w1.parameters(),
                args.learning_rate_w1,
                momentum=args.momentum_w1,
                weight_decay=args.weight_decay_w1),
            w2=torch.optim.SGD(
                model.w2.parameters(),
                args.learning_rate_w2,
                momentum=args.momentum_w2,
                weight_decay=args.weight_decay_w2),
            A=torch.optim.Adam(
                model.arch_parameters(),
                lr=args.learning_rate_A, betas=(0.5, 0.999),
                weight_decay=args.weight_decay_A),
            V=torch.optim.Adam(
                encoder.parameters(),
                lr=args.learning_rate_V, betas=(0.5, 0.999),
                weight_decay=args.weight_decay_V),
            r=torch.optim.Adam(
                r_vec.parameters(),
                lr=args.learning_rate_r, betas=(0.5, 0.999),
                weight_decay=args.weight_decay_r)
        )

        lr = SimpleNamespace(
            w1=args.learning_rate_w1,
            w2=args.learning_rate_w2,
            A=args.learning_rate_A,
            V=args.learning_rate_V,
            r=args.learning_rate_r
        )

        schedulers = SimpleNamespace(
            w1=torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizers.w1, float(args.epochs), eta_min=args.learning_rate_min),
            w2=torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizers.w2, float(args.epochs), eta_min=args.learning_rate_min),
            A=torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizers.A, float(args.epochs), eta_min=args.learning_rate_min),
            V=torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizers.V, float(args.epochs), eta_min=args.learning_rate_min),
            r=torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizers.r, float(args.epochs), eta_min=args.learning_rate_min)
        )

        architect = Architect(model, encoder, r_vec, args, optimizers, lr)

        start_epoch = 0
        if args.resume:
            checkpoint = torch.load(os.path.join(args.resume, 'checkpoint.pth.tar'))
            start_epoch = checkpoint['epoch']

            optimizers.w1.load_state_dict(checkpoint['optimizer-w1'])
            optimizers.w2.load_state_dict(checkpoint['optimizer-w2'])
            optimizers.A.load_state_dict(checkpoint['optimizer-A'])
            optimizers.V.load_state_dict(checkpoint['optimizer-V'])
            optimizers.r.load_state_dict(checkpoint['optimizer-r'])

            schedulers.w1.load_state_dict(checkpoint['scheduler-w1'])
            schedulers.w2.load_state_dict(checkpoint['scheduler-w2'])
            schedulers.A.load_state_dict(checkpoint['scheduler-A'])
            schedulers.V.load_state_dict(checkpoint['scheduler-V'])
            schedulers.r.load_state_dict(checkpoint['scheduler-r'])

            # model = torch.load(os.path.join(args.resume, 'weights_model.pt')).cuda()
            # encoder = torch.load(os.path.join(args.resume, 'weights_encoder.pt')).cuda()
            # r_vec = torch.load(os.path.join(args.resume, 'weights_r.pt')).cuda()
            args.resume = None

        for epoch in range(start_epoch, epochs):
            for i in schedulers.__dict__:
                lr.__dict__[i] = schedulers.__dict__[i].get_last_lr()[0]
            logging.info('Stage %d Epoch %d lr_w1 %f lr_w2 %f lr_A %f lr_V %f lr_r %f',
                         sp, epoch, lr.w1, lr.w2, lr.A, lr.V, lr.r)

            epoch_start = time.time()

            # change batch_size in architecture update
            if epoch == eps_no_archs[sp]:
                train_queue = torch.utils.data.DataLoader(
                    train_data, batch_size=64,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                    pin_memory=True, num_workers=args.workers, drop_last=True)

                valid_queue = torch.utils.data.DataLoader(
                    train_data, batch_size=64,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
                    pin_memory=True, num_workers=args.workers, drop_last=True)
            # training
            if epoch < eps_no_arch:
                model.w1.p = float(drop_rate[sp]) * (epochs - epoch - 1) / epochs
                model.update_p('w1')
                # model.w2.p = float(drop_rate[sp]) * (epochs - epoch - 1) / epochs
                # model.update_p('w2')
                train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizers, lr, epoch, train_arch=False)
            else:
                model.w1.p = float(drop_rate[sp]) * np.exp(-(epoch - eps_no_arch) * scale_factor)
                model.update_p('w1')
                model.w2.p = float(drop_rate[sp]) * np.exp(-(epoch - eps_no_arch) * scale_factor)
                model.update_p('w2')
                train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizers, lr, epoch, train_arch=True)
            logging.info('Train_acc %f', train_acc)
            epoch_duration = time.time() - epoch_start
            logging.info('Epoch time: %ds', epoch_duration)
            # validation
            if epochs - epoch < 5:
                valid_acc, valid_obj = infer(valid_queue, model, criterion)
                logging.info('Valid_acc %f', valid_acc)
            # save for the re-training
            torch.save(model, os.path.join(args.save, 'weights_model.pt'))
            torch.save(encoder, os.path.join(args.save, 'weights_encoder.pt'))
            torch.save(r_vec, os.path.join(args.save, 'weights_r.pt'))
            save_checkpoint({
                'stage': sp + 1,
                'epoch': epoch + 1,

                'scheduler-w1': schedulers.w1.state_dict(),
                'scheduler-w2': schedulers.w2.state_dict(),
                'scheduler-A': schedulers.A.state_dict(),
                'scheduler-V': schedulers.V.state_dict(),
                'scheduler-r': schedulers.r.state_dict(),

                'optimizer-w1': optimizers.w1.state_dict(),
                'optimizer-w2': optimizers.w2.state_dict(),
                'optimizer-A': optimizers.A.state_dict(),
                'optimizer-V': optimizers.V.state_dict(),
                'optimizer-r': optimizers.r.state_dict(),

                'switches_normal': switches_normal,
                'switches_reduce': switches_reduce,
            })
            writer.close()
        # utils.save(model, os.path.join(args.save, 'weights.pt'))
        print('------Dropping %d paths------' % num_to_drop[sp])
        # Save switches info for s-c refinement. 
        if sp == len(num_to_keep) - 1:
            switches_normal_2 = copy.deepcopy(switches_normal)
            switches_reduce_2 = copy.deepcopy(switches_reduce)
        # drop operations with low architecture weights
        arch_param = model.arch_parameters()
        normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()
        for i in range(14):
            idxs = []
            for j in range(len(PRIMITIVES)):
                if switches_normal[i][j]:
                    idxs.append(j)
            if sp == len(num_to_keep) - 1:
                # for the last stage, drop all Zero operations
                drop = get_min_k_no_zero(normal_prob[i, :], idxs, num_to_drop[sp])
            else:
                drop = get_min_k(normal_prob[i, :], num_to_drop[sp])
            for idx in drop:
                switches_normal[i][idxs[idx]] = False
        reduce_prob = F.softmax(arch_param[1], dim=-1).data.cpu().numpy()
        for i in range(14):
            idxs = []
            for j in range(len(PRIMITIVES)):
                if switches_reduce[i][j]:
                    idxs.append(j)
            if sp == len(num_to_keep) - 1:
                drop = get_min_k_no_zero(reduce_prob[i, :], idxs, num_to_drop[sp])
            else:
                drop = get_min_k(reduce_prob[i, :], num_to_drop[sp])
            for idx in drop:
                switches_reduce[i][idxs[idx]] = False
        logging.info('switches_normal = %s', switches_normal)
        logging_switches(switches_normal)
        logging.info('switches_reduce = %s', switches_reduce)
        logging_switches(switches_reduce)

        if sp == len(num_to_keep) - 1:
            arch_param = model.arch_parameters()
            normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()
            reduce_prob = F.softmax(arch_param[1], dim=sm_dim).data.cpu().numpy()
            normal_final = [0 for idx in range(14)]
            reduce_final = [0 for idx in range(14)]
            # remove all Zero operations
            for i in range(14):
                if switches_normal_2[i][0] == True:
                    normal_prob[i][0] = 0
                normal_final[i] = max(normal_prob[i])
                if switches_reduce_2[i][0] == True:
                    reduce_prob[i][0] = 0
                reduce_final[i] = max(reduce_prob[i])
                # Generate Architecture, similar to DARTS
            keep_normal = [0, 1]
            keep_reduce = [0, 1]
            n = 3
            start = 2
            for i in range(3):  # choice the two input nodes of every node
                end = start + n
                tbsn = normal_final[start:end]
                tbsr = reduce_final[start:end]
                edge_n = sorted(range(n), key=lambda x: tbsn[x])
                keep_normal.append(edge_n[-1] + start)
                keep_normal.append(edge_n[-2] + start)
                edge_r = sorted(range(n), key=lambda x: tbsr[x])
                keep_reduce.append(edge_r[-1] + start)
                keep_reduce.append(edge_r[-2] + start)
                start = end
                n = n + 1
            # set switches according the ranking of arch parameters
            for i in range(14):
                if not i in keep_normal:
                    for j in range(len(PRIMITIVES)):
                        switches_normal[i][j] = False
                if not i in keep_reduce:
                    for j in range(len(PRIMITIVES)):
                        switches_reduce[i][j] = False
            # translate switches into genotype
            genotype = parse_network(switches_normal, switches_reduce)
            logging.info(genotype)
            ## restrict skipconnect (normal cell only)
            logging.info('Restricting skipconnect...')
            # generating genotypes with different numbers of skip-connect operations
            for sks in range(0, 9):
                max_sk = 8 - sks
                num_sk = check_sk_number(switches_normal)
                if not num_sk > max_sk:
                    continue
                while num_sk > max_sk:
                    normal_prob = delete_min_sk_prob(switches_normal, switches_normal_2, normal_prob)
                    switches_normal = keep_1_on(switches_normal_2, normal_prob)
                    switches_normal = keep_2_branches(switches_normal, normal_prob)
                    num_sk = check_sk_number(switches_normal)
                logging.info('Number of skip-connect: %d', max_sk)
                genotype = parse_network(switches_normal, switches_reduce)
                logging.info(genotype)
        sp = sp + 1


def train(train_queue, valid_queue, model, architect, criterion, optimizers, lr, epoch, train_arch=True):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, ((input, target), (input_search, target_search)) in enumerate(zip(train_queue, valid_queue)):
        model.train()
        n = input.size(0)
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        input_search = input_search.cuda()
        target_search = target_search.cuda(non_blocking=True)

        architect.step(input, target, input_search, target_search, train_arch, epoch)

        logits = model.forward(input, 'w1')
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        writer.add_scalar("train_loss", objs.avg, step)
        writer.add_scalar("train_top1", top1.avg, step)
        writer.add_scalar("train_top5", top5.avg, step)

        if step % args.report_freq == 0:
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits = model(input, 'w1')
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        writer.add_scalar("val_top5", top5.avg, step)
        writer.add_scalar("val_loss", objs.avg, step)
        writer.add_scalar("val_top1", top1.avg, step)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def parse_network(switches_normal, switches_reduce):
    def _parse_switches(switches):
        n = 2
        start = 0
        gene = []
        step = 4
        for i in range(step):
            end = start + n
            for j in range(start, end):
                for k in range(len(switches[j])):
                    if switches[j][k]:
                        gene.append((PRIMITIVES[k], j - start))
            start = end
            n = n + 1
        return gene

    gene_normal = _parse_switches(switches_normal)
    gene_reduce = _parse_switches(switches_reduce)

    concat = range(2, 6)

    genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
    )

    return genotype


def get_min_k(input_in, k):
    input = copy.deepcopy(input_in)
    index = []
    for i in range(k):
        idx = np.argmin(input)
        index.append(idx)
        input[idx] = 1

    return index


def get_min_k_no_zero(w_in, idxs, k):
    w = copy.deepcopy(w_in)
    index = []
    if 0 in idxs:
        zf = True
    else:
        zf = False
    if zf:
        w = w[1:]
        index.append(0)
        k = k - 1
    for i in range(k):
        idx = np.argmin(w)
        w[idx] = 1
        if zf:
            idx = idx + 1
        index.append(idx)
    return index


def logging_switches(switches):
    for i in range(len(switches)):
        ops = []
        for j in range(len(switches[i])):
            if switches[i][j]:
                ops.append(PRIMITIVES[j])
        logging.info(ops)


def check_sk_number(switches):
    count = 0
    for i in range(len(switches)):
        if switches[i][3]:
            count = count + 1

    return count


def delete_min_sk_prob(switches_in, switches_bk, probs_in):
    def _get_sk_idx(switches_in, switches_bk, k):
        if not switches_in[k][3]:
            idx = -1
        else:
            idx = 0
            for i in range(3):
                if switches_bk[k][i]:
                    idx = idx + 1
        return idx

    probs_out = copy.deepcopy(probs_in)
    sk_prob = [1.0 for i in range(len(switches_bk))]
    for i in range(len(switches_in)):
        idx = _get_sk_idx(switches_in, switches_bk, i)
        if not idx == -1:
            sk_prob[i] = probs_out[i][idx]
    d_idx = np.argmin(sk_prob)
    idx = _get_sk_idx(switches_in, switches_bk, d_idx)
    probs_out[d_idx][idx] = 0.0

    return probs_out


def keep_1_on(switches_in, probs):
    switches = copy.deepcopy(switches_in)
    for i in range(len(switches)):
        idxs = []
        for j in range(len(PRIMITIVES)):
            if switches[i][j]:
                idxs.append(j)
        drop = get_min_k_no_zero(probs[i, :], idxs, 2)
        for idx in drop:
            switches[i][idxs[idx]] = False
    return switches


def keep_2_branches(switches_in, probs):
    switches = copy.deepcopy(switches_in)
    final_prob = [0.0 for i in range(len(switches))]
    for i in range(len(switches)):
        final_prob[i] = max(probs[i])
    keep = [0, 1]
    n = 3
    start = 2
    for i in range(3):
        end = start + n
        tb = final_prob[start:end]
        edge = sorted(range(n), key=lambda x: tb[x])
        keep.append(edge[-1] + start)
        keep.append(edge[-2] + start)
        start = end
        n = n + 1
    for i in range(len(switches)):
        if not i in keep:
            for j in range(len(PRIMITIVES)):
                switches[i][j] = False
    return switches


if __name__ == '__main__':
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Total searching time: %ds', duration)
