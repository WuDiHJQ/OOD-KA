import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import engine
import pickle
from engine.utils import get_logger, flatten_dict, set_bn_momentum, PolyLR
from PFA_amal_deeplab import PFA_Amalgamator_DeepLab
from engine.models import deeplab
import torch, time, os
import torch.nn as nn
import registry
from torch.utils.tensorboard import SummaryWriter
from engine.datasets import OOD_Segment

import argparse

parser = argparse.ArgumentParser()
# model & dataset
parser.add_argument('--data_root', default='data')
parser.add_argument('--model', default='deeplabv3plus_mobilenet')
parser.add_argument('--dataset0', default='voc')
parser.add_argument('--dataset1', default='nyu')
parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
# train detail
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=16, type=int)
parser.add_argument('--lr', default=1e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_g', default=1e-3, type=float)
parser.add_argument('--z_dim', default=256, type=int)
parser.add_argument('-k', '--k_step', default=5, type=int)
# loss weight
parser.add_argument('--oh', default=1.0, type=float)
parser.add_argument('--bn', default=1.0, type=float)
parser.add_argument('--local', default=1.0, type=float)
parser.add_argument('--adv', default=1.0, type=float)
parser.add_argument('--sim', default=1.0, type=float)
parser.add_argument('--balance', default=1.0, type=float)
parser.add_argument('--kd', default=1.0, type=float)
parser.add_argument('--amal', default=1.0, type=float)
parser.add_argument('--recons', default=1.0, type=float)

parser.add_argument('--test_only', action='store_true', default=False)
parser.add_argument('--ckpt', type=str, default=None)
args = parser.parse_args()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ==================================================
    # ==================== Dataset =====================
    # ==================================================
    part0_num_classes, part0_train, part0_val = registry.get_dataset(name=args.dataset0, data_root=args.data_root)
    part1_num_classes, part1_train, part1_val = registry.get_dataset(name=args.dataset1, data_root=args.data_root)

    # ==================================================
    # ===================== Model ======================
    # ==================================================
    part0_teacher = deeplab.modeling.__dict__[args.model](num_classes=part0_num_classes,
                                                          output_stride=args.output_stride)
    part1_teacher = deeplab.modeling.__dict__[args.model](num_classes=part1_num_classes,
                                                          output_stride=args.output_stride)
    student = deeplab.modeling.__dict__[args.model](num_classes=part0_num_classes + part1_num_classes,
                                                    output_stride=args.output_stride)
    set_bn_momentum(part0_teacher.backbone, momentum=0.01)
    set_bn_momentum(part1_teacher.backbone, momentum=0.01)
    set_bn_momentum(student.backbone, momentum=0.01)

    netG = engine.models.generator.DcGanGenerator(nz=args.z_dim, nc=3)
    netD = engine.models.generator.DeeperPatchDiscriminator(nc=3, ndf=64)

    part0_teacher.load_state_dict(
        torch.load('checkpoints/pretrained/%s_%s.pth' % (args.dataset0, args.model))['model_state'])
    part1_teacher.load_state_dict(
        torch.load('checkpoints/pretrained/%s_%s.pth' % (args.dataset1, args.model))['model_state'])

    # ==================================================
    # ================== OOD Dataset ===================
    # ==================================================
    normalizer = engine.utils.Normalizer(**registry.NORMALIZE_DICT[args.dataset0])
    args.normalizer = normalizer

    ood_with_aug = OOD_Segment("data/ImageNet_subset")
    ood_without_aug = OOD_Segment("data/ImageNet_subset")

    ood_with_aug.transforms = ood_with_aug.transform = part0_train.transform  # with aug
    ood_without_aug.transforms = ood_without_aug.transform = part0_val.transform  # without aug

    # ==================================================
    # =================== DataLoader ===================
    # ==================================================
    ood_with_aug_loader = torch.utils.data.DataLoader(ood_with_aug, batch_size=args.batch_size, shuffle=True,
                                                      num_workers=4)
    ood_without_aug_loader = torch.utils.data.DataLoader(ood_without_aug, batch_size=args.batch_size, shuffle=True,
                                                         num_workers=4)
    part0_val_loader = torch.utils.data.DataLoader(part0_val, batch_size=args.batch_size, shuffle=False, num_workers=4)
    part1_val_loader = torch.utils.data.DataLoader(part1_val, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ==================================================
    # =================== Optimizer ====================
    # ==================================================
    TOTAL_ITERS = len(ood_with_aug_loader) * args.epochs
    optim_s = torch.optim.SGD(params=[
        {'params': student.backbone.parameters(), 'lr': 0.1 * args.lr},
        {'params': student.classifier.parameters(), 'lr': args.lr},
    ], lr=args.lr, momentum=0.9, weight_decay=1e-4)
    sched_s = PolyLR(optim_s, TOTAL_ITERS, power=0.9)
    optim_g = torch.optim.Adam(netG.parameters(), lr=args.lr_g, betas=[0.5, 0.999])
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(optim_g, T_max=TOTAL_ITERS)
    optim_d = torch.optim.Adam(netD.parameters(), lr=args.lr_g, betas=[0.5, 0.999])
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(optim_d, T_max=TOTAL_ITERS)

    # ==================================================
    # ==================== Trainer =====================
    # ==================================================
    output_dir = 'run/PFA_deeplab_%s' % (time.asctime().replace(' ', '_'))
    trainer = PFA_Amalgamator_DeepLab(
        logger=get_logger(name='PFA-deeplab', output=os.path.join(output_dir, 'log.txt')),
        tb_writer=SummaryWriter(log_dir=output_dir),
        output_dir=output_dir
    )
    for k, v in flatten_dict(vars(args)).items():  # print args
        trainer.logger.info("%s: %s" % (k, v))

    trainer.setup(args=args,
                  student=student,
                  teachers=[part0_teacher, part1_teacher],
                  netG=netG,
                  netD=netD,
                  train_loader=[ood_with_aug_loader, ood_without_aug_loader],
                  val_loaders=[part0_val_loader, part1_val_loader],
                  val_num_classes=[part0_num_classes, part1_num_classes],
                  optimizers=[optim_s, optim_g, optim_d],
                  schedulers=[sched_s, sched_g, sched_d],
                  device=device)

    if args.ckpt is not None:
        trainer.student.load_state_dict(torch.load(args.ckpt)['state_dict'])
        print("Load student model from %s" % args.ckpt)
    if args.test_only:
        trainer.validate()
        return

    trainer.train(start_iter=0, max_iter=TOTAL_ITERS)


if __name__ == '__main__':
    main()