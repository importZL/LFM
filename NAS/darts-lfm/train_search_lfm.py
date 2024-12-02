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

from torch.autograd import Variable
from model_search_lfm import Network, Network_w
from architect_lfm import Architect
from encoder_resnet import *
from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate_min', type=float,
                    default=0.00025, help='minimum learning rate')
parser.add_argument('--report_freq', type=float,
                    default=1, help='report frequency')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--epochs', type=int, default=50,
                    help='num of training epochs')
parser.add_argument('--init_channels', type=int,
                    default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8,
                    help='total number of layers')
parser.add_argument('--model_path', type=str,
                    default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true',
                    default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int,
                    default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float,
                    default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--train_portion', type=float,
                    default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true',
                    default=False, help='use one-step unrolled validation loss')

# new hyperparams.
parser.add_argument('--learning_rate_w1', type=float, default=1e-2)
parser.add_argument('--learning_rate_w2', type=float, default=1e-2)
parser.add_argument('--learning_rate_A', type=float, default=1e-3)
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
parser.add_argument('--is_cifar100', type=int, default=0)
parser.add_argument('--resume', type=str, default='')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(filename_suffix=time.strftime("%Y%m%d-%H%M%S"))

CIFAR_CLASSES = 10
CIFAR100_CLASSES = 100


def save_checkpoint(state, checkpoint=args.save, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    if not args.is_parallel:
        torch.cuda.set_device(int(args.gpu))
        logging.info('gpu device = %d' % int(args.gpu))
    else:
        logging.info('gpu device = %s' % args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    # model contains w1, w2 and A
    if args.is_cifar100:
        model = Network(args.init_channels, CIFAR100_CLASSES, args.layers, criterion, args.is_parallel, args.gpu)
    else:
        model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, args.is_parallel, args.gpu)
    torch.save(model.w_temp, os.path.join(args.save, 'w_temp.pt'))
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
    r_vec = nn.Sequential(nn.Linear(args.batch_size, 1, bias=False)).cuda()
    r_vec[0].weight = nn.Parameter(torch.ones_like(r_vec[0].weight) + 1e-3*torch.randn_like(r_vec[0].weight))

    if args.is_parallel:
        args.gpu = '0,1'
        gpus = [int(i) for i in args.gpu.split(',')]
        encoder = nn.parallel.DataParallel(
            encoder, device_ids=gpus, output_device=gpus[1])
        model.w1 = nn.parallel.DataParallel(
            model.w1, device_ids=gpus, output_device=gpus[1])
        model.w2 = nn.parallel.DataParallel(
            model.w2, device_ids=gpus, output_device=gpus[1])
        encoder = encoder.module
        model.w1 = model.w1.module
        model.w2 = model.w2.module

    # logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

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

    if args.is_cifar100:
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
    else:
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
    if args.is_cifar100:
        train_data = dset.CIFAR100(root=args.data, train=True,
                                   download=True, transform=train_transform)
    else:
        train_data = dset.CIFAR10(root=args.data, train=True,
                                  download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=False, num_workers=4, drop_last=True)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=False, num_workers=4, drop_last=True)

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

        model = torch.load(os.path.join(args.resume, 'weights_model.pt')).cuda()
        encoder = torch.load(os.path.join(args.resume, 'weights_encoder.pt')).cuda()
        r_vec = torch.load(os.path.join(args.resume, 'weights_r.pt')).cuda()

    for epoch in range(start_epoch, args.epochs):
        for i in schedulers.__dict__:
            lr.__dict__[i] = schedulers.__dict__[i].get_last_lr()[0]
        # TODO: verify the loop above and then delete below
        ####lr.w1 = schedulers.w1.get_lr()[0]
        ####lr.w2 = schedulers.w2.get_lr()[0]
        ####lr.A = schedulers.A.get_lr()[0]
        ####lr.V = schedulers.V.get_lr()[0]
        ####lr.r = schedulers.r.get_lr()[0]
        logging.info('epoch %d lr_w1 %f lr_w2 %f lr_A %f lr_V %f lr_r %f', epoch, lr.w1, lr.w2, lr.A, lr.V, lr.r)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)
        # TODO: log genotypes to a folder and use some good file format -> make it usable with visualize

        # print(F.softmax(model.alphas_normal, dim=-1))
        # print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        train_acc, train_obj = train(
            train_queue, valid_queue, model,
            architect, criterion, optimizers, lr)

        logging.info('train_acc %f', train_acc)
        logging.info('train_loss %f', train_obj)

        for i in schedulers.__dict__:
            schedulers.__dict__[i].step()

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, architect, criterion)
        logging.info('valid_acc %f', valid_acc)
        logging.info('valid_loss %f', valid_obj)

        # save for the re-training
        torch.save(model, os.path.join(args.save, 'weights_model.pt'))
        torch.save(encoder, os.path.join(args.save, 'weights_encoder.pt'))
        torch.save(r_vec, os.path.join(args.save, 'weights_r.pt'))
        save_checkpoint({
            'epoch': epoch + 1,

            'scheduler_w1': schedulers.w1.state_dict(),
            'scheduler-w2': schedulers.w2.state_dict(),
            'scheduler-A': schedulers.A.state_dict(),
            'scheduler-V': schedulers.V.state_dict(),
            'scheduler-r': schedulers.r.state_dict(),

            'optimizer-w1': optimizers.w1.state_dict(),
            'optimizer-w2': optimizers.w2.state_dict(),
            'optimizer-A': optimizers.A.state_dict(),
            'optimizer-V': optimizers.V.state_dict(),
            'optimizer-r': optimizers.r.state_dict(),
        })

    writer.close()


def train(train_queue, valid_queue,
          model, architect, criterion, optimizers, lr):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    g_step = 0

    # for step, ((input, target), (input_val, target_val)) in enumerate(zip(train_queue, valid_queue)):
    for step, (input, target) in enumerate(train_queue):
        model.train()
        architect.encoder.train()

        n = input.size(0)

        input = input.cuda()
        target = target.cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        input_val, target_val = next(iter(valid_queue))
        input_val = input_val.cuda()
        target_val = target_val.cuda(non_blocking=True)

        ###Architect.step will perform W1, W2, V, r, and A updates.
        ###because equations are all linked, its better to have their updates in a single place
        ### be careful of leaking gradients!!
        architect.step(input, target, input_val, target_val, unrolled=args.unrolled, save_dir=args.save)

        # TODO: think on using w1, w2, or average results
        logits = model.forward(input, 'w2')
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        writer.add_scalar("train_loss", objs.avg, g_step)
        writer.add_scalar("train_top1", top1.avg, g_step)
        writer.add_scalar("train_top5", top5.avg, g_step)

        if step % args.report_freq == 0:
            logging.info('train (on w2) %03d %e %f %f', g_step, objs.avg, top1.avg, top5.avg)

        g_step += 1

    return top1.avg, objs.avg


def infer(valid_queue, model, architect, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    architect.encoder.eval()
    g_step = 0
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            # TODO: w1 or w2 or average the two
            logits = model.forward(input, 'w2')
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            writer.add_scalar("val_top5", top5.avg, g_step)
            writer.add_scalar("val_loss", objs.avg, g_step)
            writer.add_scalar("val_top1", top1.avg, g_step)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', g_step, objs.avg, top1.avg, top5.avg)

            g_step += 1

    return top1.avg, objs.avg


if __name__ == '__main__':
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    main()
