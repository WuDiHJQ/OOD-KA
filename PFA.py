import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import engine
import pickle
from engine.utils import get_logger, flatten_dict, prepare_ood_subset
from PFA_amal import PFA_Amalgamator

import torch, time, os
import torch.nn as nn
import registry
from torch.utils.tensorboard import SummaryWriter

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
parser.add_argument('--lr_g', default=1e-3, type=float)
parser.add_argument('--z_dim', default=100, type=int)
parser.add_argument('-k', '--k_step', default=5, type=int)
# loss weight
parser.add_argument('--oh', default=1.0, type=float)
parser.add_argument('--bn', default=1.0, type=float)
parser.add_argument('--local', default=1.0, type=float)
parser.add_argument('--adv', default=1.0, type=float)
parser.add_argument('--sim', default=1.0, type=float)
parser.add_argument('--balance', default=10.0, type=float)
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
    part0_num_classes, part0_train, part0_val = registry.get_dataset(name='%s_part0' % (args.dataset),
                                                                     data_root=args.data_root)
    part1_num_classes, part1_train, part1_val = registry.get_dataset(name='%s_part1' % (args.dataset),
                                                                     data_root=args.data_root)

    # ==================================================
    # ===================== Model ======================
    # ==================================================
    part0_teacher = registry.get_model(args.model, num_classes=part0_num_classes, pretrained=False)
    part1_teacher = registry.get_model(args.model, num_classes=part1_num_classes, pretrained=False)
    student = registry.get_model(args.model, num_classes=part0_num_classes + part1_num_classes, pretrained=False)

    netG = engine.models.generator.Generator(nz=args.z_dim, nc=3, img_size=32)
    netD = engine.models.generator.PatchDiscriminator(nc=3, ndf=128)

    part0_teacher.load_state_dict(
        torch.load('checkpoints/pretrained/%s_part0_%s.pth' % (args.dataset, args.model))['state_dict'])
    part1_teacher.load_state_dict(
        torch.load('checkpoints/pretrained/%s_part1_%s.pth' % (args.dataset, args.model))['state_dict'])

    # ==================================================
    # ================== OOD Dataset ===================
    # ==================================================
    normalizer = engine.utils.Normalizer(**registry.NORMALIZE_DICT[args.dataset])
    args.normalizer = normalizer

    _, ood_with_aug, _ = registry.get_dataset(name=args.unlabeled, data_root=args.data_root)
    _, ood_without_aug, _ = registry.get_dataset(name=args.unlabeled, data_root=args.data_root)

    ood_with_aug.transforms = ood_with_aug.transform = part0_train.transform  # with aug
    ood_without_aug.transforms = ood_without_aug.transform = part0_val.transform  # without aug

    if args.unlabeled in ['imagenet_32x32', 'places365_32x32']:
        ood_index_root = os.path.join(args.data_root, 'ood_index_%s_%s.pkl' % (args.unlabeled, args.model))

        if not os.path.exists(ood_index_root):
            ood_index = prepare_ood_subset(ood_without_aug, 50000,
                                           nn.ModuleList([part0_teacher, part1_teacher]).to(device))

            with open(ood_index_root, 'wb') as f:
                pickle.dump(ood_index, f)

        with open(ood_index_root, 'rb') as f:
            ood_index = pickle.load(f)
        ood_with_aug.samples = [ood_with_aug.samples[i] for i in ood_index]
        ood_without_aug.samples = [ood_without_aug.samples[i] for i in ood_index]

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
    optim_s = torch.optim.Adam(student.parameters(), lr=args.lr, weight_decay=1e-4)
    sched_s = torch.optim.lr_scheduler.CosineAnnealingLR(optim_s, T_max=TOTAL_ITERS)
    optim_g = torch.optim.Adam(netG.parameters(), lr=args.lr_g, betas=[0.5, 0.999])
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(optim_g, T_max=TOTAL_ITERS)
    optim_d = torch.optim.Adam(netD.parameters(), lr=args.lr_g, betas=[0.5, 0.999])
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(optim_d, T_max=TOTAL_ITERS)

    # ==================================================
    # ==================== Trainer =====================
    # ==================================================
    output_dir = 'run/PFA-%s' % (time.asctime().replace(' ', '_'))
    trainer = PFA_Amalgamator(
        logger=get_logger(name='PFA', output=os.path.join(output_dir, 'log.txt')),
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