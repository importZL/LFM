import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as dset


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_data(args):

    if args.is_cifar100:
        train_transform, valid_transform = _data_transforms_cifar100(args)
    else:
        train_transform, valid_transform = _data_transforms_cifar10(args)
    if args.is_cifar100:
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    else:
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

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
    return train_queue, valid_queue


def cal_similarity(
        data_encoder,
        input_train, input_valid,
        target_train, target_valid):
    """
    Calculates the visual and label similarity
        between training dataset and validation dataset.
    :param data_encoder: Date encoder used to extract the features of input
    :param input_train: Training dataset
    :param input_valid: Validation dataset
    :param target_train: Target of training dataset
    :param target_valid: Target of validation dataset
    :return: Visual and label similarity
    """
    simi_visual = []
    simi_label = []
    input_train = input_train.cuda()
    input_valid = input_valid.cuda()
    feature_train = torch.squeeze(data_encoder(input_train))
    feature_valid = torch.squeeze(data_encoder(input_valid))
    for feature_train_single in feature_train:
        simi_every_train = feature_train_single * feature_valid
        simi_every_train = F.softmax(simi_every_train, dim=0)
        simi_visual.append(np.squeeze(simi_every_train.tolist()))
    for target_train_single in target_train:
        simi_every_train = []
        for target_valid_single in target_valid:
            if torch.equal(target_train_single, target_valid_single):
                simi_every_train.append(1)
            else:
                simi_every_train.append(0)
        simi_label.append(simi_every_train)
    simi_label = torch.FloatTensor(simi_label).cuda()
    simi_visual = torch.FloatTensor(simi_visual).cuda()
    return simi_visual, simi_label


def normalize_list(data):
    max_list = np.amax(data)
    min_list = np.amin(data)
    return (data - min_list) / (max_list - min_list)

def cal_weight_a(model, simi_visual, simi_label, pred_performance):
    """
    Calculates the relation weight between
        training dataset and validation dataset
    :param model: Network
    :param simi_visual: Visual similarity
    :param simi_label:  Label similarity
    :param pred_performance:  Predict performance
    :return: relation weight a
    """
    weight_a = torch.tensor([], requires_grad=True).cuda()
    for visual_single, label_single in zip(simi_visual, simi_label):
        a = visual_single * label_single * pred_performance * -1
        # a = F.softmax(visual_single+label_single+pred_performance, dim=0)
        # a = torch.FloatTensor(2 * (normalize_list((visual_single+F.softmax(label_single)+pred_performance).tolist()))).cuda()
        a = torch.mm(a.view(1, -1), model.weight_r)
        # a = torch.tensor([1.0, ], requires_grad=True).cuda()  # fix the weight a to 1
        weight_a = torch.cat((weight_a, a), 0)
    # return weight_a.cuda()
    return torch.sigmoid(weight_a)


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def _data_transforms_cifar100(args):
    CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
    CIFAR_STD = [0.2675, 0.2565, 0.2761]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    # if scripts_to_save is not None:
    #   os.mkdir(os.path.join(path, 'scripts'))
    #   for script in scripts_to_save:
    #     dst_file = os.path.join(path, 'scripts', os.path.basename(script))
    #     shutil.copyfile(script, dst_file)

