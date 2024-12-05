import os
import sys
import logging
import time
import argparse
import random
import copy
import torch
import torchvision
import numpy as np
import pandas as pd
import sklearn.metrics as sm
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from data_utils import *
from resnet import *
from encoder_resnet import *
from copy import deepcopy
from sklearn.metrics import confusion_matrix

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# parse arguments
parser = argparse.ArgumentParser(description='Imbalanced Example')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--num_meta', type=int, default=10,
                    help='The number of meta data for each class.')
parser.add_argument('--imb_factor', type=float, default=0.005)
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=120, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disabl    es CUDA training')
parser.add_argument('--split', type=int, default=1000)
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')

parser.add_argument('--save', type=str, default='./log-base', help='save path')
parser.add_argument('--note', type=str, default='cifar10', help='experiment name')

parser.add_argument('--encoder-size', type=int, default=18, help='.')
parser.add_argument('--resume', type=str, default='./log/result-cifar10-LFM-200.0-20210825-112745')
args = parser.parse_args()

args.save = '{}/result-{}-{}-{}'.format(args.save, args.note, 1 / args.imb_factor, time.strftime("%Y%m%d-%H%M%S"))
kwargs = {'num_workers': 0, 'pin_memory': False}
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

train_data_meta, train_data, test_dataset = build_dataset(args.dataset, args.num_meta)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, **kwargs)

# make imbalanced data
torch.manual_seed(args.seed)
classe_labels = range(args.num_classes)

data_list = {}

for j in range(args.num_classes):
    data_list[j] = [i for i, label in enumerate(train_loader.dataset.targets) if label == j]

img_num_list = get_img_num_per_cls(args.dataset, args.imb_factor, args.num_meta * args.num_classes)
im_data = {}
idx_to_del = []
for cls_idx, img_id_list in data_list.items():
    random.shuffle(img_id_list)
    img_num = img_num_list[int(cls_idx)]
    im_data[cls_idx] = img_id_list[img_num:]
    idx_to_del.extend(img_id_list[img_num:])

imbalanced_train_dataset = copy.deepcopy(train_data)
imbalanced_train_dataset.targets = np.delete(train_loader.dataset.targets, idx_to_del, axis=0)
imbalanced_train_dataset.data = np.delete(train_loader.dataset.data, idx_to_del, axis=0)
imbalanced_train_loader = torch.utils.data.DataLoader(
    imbalanced_train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs)

validation_loader = torch.utils.data.DataLoader(
    train_data_meta, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

best_prec1 = 0
len_train = len(imbalanced_train_loader)

train_targets = imbalanced_train_dataset.targets[0:len_train*args.batch_size]
num_target = []
index_target = []
for i in range(args.num_classes):
    num_target.append(np.sum(train_targets == i))
    index_target.append((np.where(train_targets == i)))


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


def _compute_a(model, encoder, r_vec, input_var, target_var, input_validation_var, target_validation_var):
    weight_x = _compute_x(encoder, input_var, input_validation_var)
    weight_z = _compute_z(target_var, target_validation_var)
    y = model(input_validation_var)
    weight_u = F.cross_entropy(y, target_validation_var.long(), reduction='none')
    weight_a = torch.sigmoid(r_vec(weight_x * weight_z * weight_u))

    return weight_a


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


def main():
    global args, best_prec1
    args = parser.parse_args()

    logging.info("args = %s", args)
    logging.info("image number of per classes: %s", img_num_list)
    logging.info("total image number of train: %s", sum(img_num_list))
    logging.info("total image number to delete: %s", len(idx_to_del))
    # create model
    model = build_model(len_train)
    meta_model = build_model(len_train)
    optimizer_a = torch.optim.SGD(model.params(), args.lr,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)
    optimizer_meta = torch.optim.SGD(meta_model.params(), args.lr,
                                     momentum=args.momentum, nesterov=args.nesterov,
                                     weight_decay=args.weight_decay)

    # encoder contains V
    encoder = resnet18(pretrained=True).cuda()

    # contains r
    # TODO: check input size
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

    scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_a, float(args.epochs), eta_min=1e-4)
    scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_b, float(args.epochs), eta_min=1e-5)
    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(os.path.join(args.resume, 'best_checkpoint.pth.tar'))
        start_epoch = checkpoint['epoch']

        model.load_state_dict(checkpoint['model'])
        model.weight_b.data = checkpoint['weight-b']
        meta_model.load_state_dict(checkpoint['meta-model'])
        encoder.load_state_dict(checkpoint['encoder'])
        r_vec.load_state_dict(checkpoint['r-vec'])
        scheduler_b.load_state_dict(checkpoint['scheduler-b'])
        meta_model.load_state_dict(checkpoint['meta-model'])

        optimizer_a.load_state_dict(checkpoint['optimizer-a'])
        optimizer_meta.load_state_dict(checkpoint['optimizer-meta'])
        optimizer_b.load_state_dict(checkpoint['optimizer-b'])
        optimizer_v.load_state_dict(checkpoint['optimizer-V'])
        optimizer_r.load_state_dict(checkpoint['optimizer-r'])

    draw_cm(test_loader, model)
    sys.exit()

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer_a, epoch + 1)
        adjust_learning_rate(optimizer_meta, epoch + 1)
        train(imbalanced_train_loader,
              validation_loader,
              model,
              meta_model,
              encoder,
              r_vec,
              optimizer_a,
              optimizer_meta,
              optimizer_v,
              optimizer_r,
              optimizer_b,
              epoch)
        # scheduler_a.step()
        # scheduler_b.step()
        # evaluate on validation set
        prec1 = validate(test_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if best_prec1 >= 60:
            sys.exit()
        save_checkpoint({
            'epoch': epoch + 1,

            'model': model.state_dict(),
            'meta-model': meta_model.state_dict(),
            'encoder': encoder.state_dict(),
            'r-vec': r_vec.state_dict(),
            'weight-b': model.weight_b,
            'scheduler-b': scheduler_b.state_dict(),

            'optimizer-a': optimizer_a.state_dict(),
            'optimizer-meta': optimizer_meta.state_dict(),
            'optimizer-b': optimizer_b.state_dict(),
            'optimizer-V': optimizer_v.state_dict(),
            'optimizer-r': optimizer_r.state_dict(),
        }, is_best)
        ave_weights = []
        sample_weights = model.weight_b.cpu().detach().numpy().flatten()
        for i in range(args.num_classes):
            ave_weights.append(np.sum(sample_weights[index_target[i]]) / num_target[i])
        logging.info(ave_weights)
    logging.info(model.weight_b)
    logging.info('Best accuracy: %s', best_prec1)


def train(train_loader,
          validation_loader,
          model,
          meta_model,
          encoder,
          r_vec,
          optimizer_a,
          optimizer_meta,
          optimizer_v,
          optimizer_r,
          optimizer_b,
          epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    meta_losses = AverageMeter()
    top1 = AverageMeter()
    meta_top1 = AverageMeter()
    model.train()
    lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 90)))

    input_validation, target_validation = next(iter(validation_loader))

    for i, (input, target) in enumerate(train_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)
        '''
        try:
            input_validation, target_validation = next(iter(validation_loader))
        except StopIteration:
            sample_valid_iter = iter(validation_loader)
            input_validation, target_validation = next(iter(validation_loader))
        '''
        input_validation_var = to_var(input_validation, requires_grad=False)
        target_validation_var = to_var(target_validation, requires_grad=False)

        # step 1
        y_f = model(input_var)
        l_f = F.cross_entropy(y_f, target_var.long(), reduction='mean')
        prec_train = accuracy(y_f.data, target_var.data, topk=(1,))[0]
        optimizer_a.zero_grad()
        l_f.backward()
        nn.utils.clip_grad_norm_(model.params(), 5)
        optimizer_a.step()

        losses.update(l_f.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))

        if i % args.print_freq == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader),
                loss=losses, top1=top1))


def _hessian_vector_product1(model, vector, input, target, i, r=1e-2):
    R = r / _concat(vector).norm()
    if _concat(vector).norm() == 0:
        print('norm 0 : hessian1')
        return [torch.zeros_like(x) for x in model.b_parameters()]

    for p, v in zip(model.params(), vector):
        p.data.add_(v, alpha=R)
    y_f = model(input)
    cost_w = F.cross_entropy(y_f, target.long(), reduction='none')
    # cost_v = torch.reshape(cost_w, (len(cost_w), 1))
    loss = (cost_w * model.weight_b[:, i]).mean()
    grads_p = torch.autograd.grad(loss, model.b_parameters(), create_graph=True)

    for p, v in zip(model.params(), vector):
        p.data.sub_(v, alpha=2 * R)
    y_f = model(input)
    cost_w = F.cross_entropy(y_f, target.long(), reduction='none')
    # cost_v = torch.reshape(cost_w, (len(cost_w), 1))
    loss = (cost_w * model.weight_b[:, i]).mean()
    grads_n = torch.autograd.grad(loss, model.b_parameters(), allow_unused=True)

    for p, v in zip(model.params(), vector):
        p.data.add_(v, alpha=R)

    return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('Test: [{0}/{1}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    logging.info(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log-ResNet32 to TensorBoard

    return top1.avg


def build_model(len_train):
    model = ResNet32(args.dataset == 'cifar10' and 10 or 100, len_train)
    # print('Number of model parameters: {}'.format(
    #     sum([p.data.nelement() for p in model.parameters()])))

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    # lr = args.lr * ((0.2 ** int(epoch >= 60)) * (0.2 ** int(epoch >= 120))* (0.2 ** int(epoch >= 160)))
    lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 90)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


def draw_cm(val_loader, model):
    """Perform validation on the validation set"""
    # switch to evaluate mode
    model.eval()
    maxk = max((1,))
    pred_label = []
    true_label = []

    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        true_label.extend(target_var.cpu().numpy())
        # compute output
        with torch.no_grad():
            output = model(input_var)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()[0].cpu().numpy()
            pred_label.extend(pred)

    classes = range(10)
    my_cm = confusion_matrix(true_label, pred_label)
    plt.imshow(my_cm, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
    plt.title('Ours')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)  # 倾斜
    plt.yticks(tick_marks, classes)

    thresh = my_cm.max() / 2.
    # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
    # ij配对，遍历矩阵迭代器
    iters = np.reshape([[[i, j] for j in range(10)] for i in range(10)], (my_cm.size, 2))
    for i, j in iters:
        my_text = ("%.1f%%" % (my_cm[i, j]/10))
        plt.text(j, i, my_text, va='center', ha='center', fontsize=8)  # 显示对应的数字

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    # plt.show()
    plt.savefig('confusion_matrix.png', format='png')



if __name__ == '__main__':
    os.mkdir(args.save)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log-ResNet32.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Total searching time: %ds', duration)
