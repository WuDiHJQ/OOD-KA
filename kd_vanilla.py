from torchvision import datasets, transforms as T
from engine.utils import DataIter, get_logger, flatten_dict, move_to_device, set_mode
from engine.criterions import kldiv

import torch, time, os
import registry
import torch.nn as nn

import argparse

parser = argparse.ArgumentParser()
# model & dataset
parser.add_argument('--data_root', default='data')
parser.add_argument('--model', default='wrn16_2')
parser.add_argument('--dataset', default='cifar100')
parser.add_argument('--unlabeled', default='cifar10')
# train detail
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256, type=int)
parser.add_argument('--lr', default=1e-3, type=float,
                    help='initial learning rate')
# test only
parser.add_argument('--test_only', action='store_true', default=False)
parser.add_argument('--ckpt', type=str, default=None)
args = parser.parse_args()


def main():
    # ==================================================
    # ==================== dataset =====================
    # ==================================================
    part0_num_classes, part0_train, part0_val = registry.get_dataset(name='%s_part0' % (args.dataset),
                                                                     data_root=args.data_root)
    part1_num_classes, part1_train, part1_val = registry.get_dataset(name='%s_part1' % (args.dataset),
                                                                     data_root=args.data_root)
    _, ood_with_aug, _ = registry.get_dataset(name=args.unlabeled, data_root=args.data_root)

    # ==================================================
    # ===================== model ======================
    # ==================================================
    part0_teacher = registry.get_model(args.model, num_classes=part0_num_classes, pretrained=False)
    part1_teacher = registry.get_model(args.model, num_classes=part1_num_classes, pretrained=False)
    student = registry.get_model(args.model, num_classes=part0_num_classes + part1_num_classes, pretrained=False)

    part0_teacher.load_state_dict(
        torch.load('checkpoints/pretrained/%s_part0_%s.pth' % (args.dataset, args.model))['state_dict'])
    part1_teacher.load_state_dict(
        torch.load('checkpoints/pretrained/%s_part1_%s.pth' % (args.dataset, args.model))['state_dict'])

    # ==================================================
    # =================== DataLoader ===================
    # ==================================================
    ood_with_aug.transforms = ood_with_aug.transform = part0_train.transform  # with aug
    ood_with_aug_loader = torch.utils.data.DataLoader(ood_with_aug, batch_size=args.batch_size, shuffle=True,
                                                      num_workers=4)
    ood_with_aug_iter = DataIter(ood_with_aug_loader)
    epoch_length = len(ood_with_aug_loader)

    part0_val_loader = torch.utils.data.DataLoader(part0_val, batch_size=args.batch_size, shuffle=False, num_workers=4)
    part1_val_loader = torch.utils.data.DataLoader(part1_val, batch_size=args.batch_size, shuffle=False, num_workers=4)

    val_loaders = [part0_val_loader, part1_val_loader]
    val_num_classes = [part0_num_classes, part1_num_classes]

    TOTAL_ITERS = len(ood_with_aug_loader) * args.epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optim_s = torch.optim.Adam(student.parameters(), lr=args.lr, weight_decay=1e-4)
    sched_s = torch.optim.lr_scheduler.CosineAnnealingLR(optim_s, T_max=TOTAL_ITERS)

    teachers = nn.ModuleList([part0_teacher, part1_teacher]).to(device)
    student = student.to(device)

    # ==================================================
    # ==================== Logger ======================
    # ==================================================
    output_dir = 'run/vanilla_kd-%s' % (time.asctime().replace(' ', '_'))
    logger = get_logger(name='vanilla-kd', output=os.path.join(output_dir, 'log.txt'))

    for k, v in flatten_dict(vars(args)).items():  # print args
        logger.info("%s: %s" % (k, v))

    if args.ckpt is not None:
        student.load_state_dict(torch.load(args.ckpt)['state_dict'])
        print("Load student model from %s" % args.ckpt)
    if args.test_only:
        validate(0, student, val_loaders, val_num_classes, logger, device)
        return

    best_acc1 = 0
    with set_mode(student, training=True), \
            set_mode(teachers, training=False):
        for iter in range(0, TOTAL_ITERS):
            data = ood_with_aug_iter.next()[0].to(device)
            s_out = student(data)
            with torch.no_grad():
                t_out = [teacher(data) for teacher in teachers]

            loss_kd = kldiv(s_out, torch.cat(t_out, dim=1))

            optim_s.zero_grad()
            loss_kd.backward()
            optim_s.step()
            sched_s.step()

            # STEP END
            if iter % 100 == 0:
                logger.info('loss_kd: %.4f, optim_s_lr: %.6f' % (loss_kd, optim_s.param_groups[0]['lr']))

            # EPOCH END
            if (iter + 1) % epoch_length == 0:
                acc1 = validate(iter // epoch_length, student, val_loaders, val_num_classes, logger, device)
                is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)
                save_checkpoint({
                    'epoch': iter // epoch_length,
                    's_state_dict': student.state_dict(),
                    'best_acc1': float(best_acc1),
                    'optim_s': optim_s.state_dict(),
                    'sched_s': sched_s.state_dict(),
                }, is_best, os.path.join(output_dir, 'best.pth'))


def validate(epoch, student, val_loaders, val_num_classes, logger, device):
    losses = AverageMeter('Loss', ':.4e')
    part_top1 = [AverageMeter('Part0_Acc@1', ':6.2f'), AverageMeter('Part1_Acc@1', ':6.2f')]
    part_top5 = [AverageMeter('Part0_Acc@5', ':6.2f'), AverageMeter('Part1_Acc@5', ':6.2f')]
    total_top1 = AverageMeter('Total_Acc@1', ':6.2f')
    total_top5 = AverageMeter('Total_Acc@5', ':6.2f')

    with set_mode(student, training=False):
        with torch.no_grad():
            for i, val_loader in enumerate(val_loaders):

                for batch in val_loader:
                    batch = move_to_device(batch, device)
                    data, target = batch

                    output = student(data)[:, sum(val_num_classes[:i]):sum(val_num_classes[:i + 1])]

                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    part_top1[i].update(acc1[0], data.size(0))
                    part_top5[i].update(acc5[0], data.size(0))
                    total_top1.update(acc1[0], data.size(0))
                    total_top5.update(acc5[0], data.size(0))

            logger.info(' [Eval] Epoch={}'.format(epoch))
            logger.info(' [Eval] Part0 Acc@1={:.4f} Acc@5={:.4f}'
                        .format(part_top1[0].avg, part_top5[0].avg))
            logger.info(' [Eval] Part1 Acc@1={:.4f} Acc@5={:.4f}'
                        .format(part_top1[1].avg, part_top5[1].avg))
            logger.info(' [Eval] Total Acc@1={:.4f} Acc@5={:.4f}'
                        .format(total_top1.avg, total_top5.avg))
    return total_top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()