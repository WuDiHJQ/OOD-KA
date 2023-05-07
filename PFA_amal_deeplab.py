import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.utils import DataIter, save_image_batch, move_to_device, set_mode
from engine.hooks import FeatureHook, DeepInversionHook
from engine.criterions import kldiv
from engine.metrics import StreamSegMetrics

import typing
import time
import os


class AmalBlock(nn.Module):
    # amalgamation block
    # cs channel student
    # cts channel teachers
    # fs feature student
    # fts feature teachers
    # enc encoder
    # fam feature adapt module
    # dec decoder
    # rep representation
    def __init__(self, cs, cts):
        super(AmalBlock, self).__init__()
        self.cs, self.cts = cs, cts
        self.enc = nn.Conv2d(in_channels=sum(self.cts), out_channels=self.cs, kernel_size=1, stride=1, padding=0,
                             bias=True)
        self.fam = nn.Conv2d(in_channels=self.cs, out_channels=self.cs, kernel_size=1, stride=1, padding=0, bias=True)
        self.dec = nn.Conv2d(in_channels=self.cs, out_channels=sum(self.cts), kernel_size=1, stride=1, padding=0,
                             bias=True)

    def forward(self, fs, fts):
        rep = self.enc(torch.cat(fts, dim=1))
        _fts = self.dec(rep)
        _fts = torch.split(_fts, self.cts, dim=1)
        _fs = self.fam(fs)
        return rep, _fs, _fts


class PFA_Amalgamator_DeepLab():
    def __init__(self, logger=None, tb_writer=None, output_dir=None):
        self.logger = logger if logger else get_logger(name='mosaic_amal', color=True)
        self.tb_writer = tb_writer
        self.output_dir = output_dir

    def setup(
            self,
            args,
            student,
            teachers: [],
            netG,
            netD,
            train_loader: [],
            val_loaders: [],
            val_num_classes: [],
            optimizers: [],
            schedulers: [],
            device=None,
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.ood_with_aug_loader, self.ood_without_aug_loader = train_loader
        self.ood_with_aug_iter = DataIter(self.ood_with_aug_loader)
        self.ood_without_aug_iter = DataIter(self.ood_without_aug_loader)
        self.val_loaders = val_loaders
        self.val_num_classes = val_num_classes
        self.student = student.to(self.device)
        self.teachers = nn.ModuleList(teachers).to(self.device)
        self.netG = netG.to(self.device)
        self.netD = netD.to(self.device)
        self.optim_s, self.optim_g, self.optim_d = optimizers
        self.sched_s, self.sched_g, self.sched_d = schedulers
        self.args = args
        self.z_dim = args.z_dim
        self.normalizer = args.normalizer
        self.batch_size = args.batch_size
        self.bn_hooks = []
        self.metrics = [StreamSegMetrics(val_num_classes[0]),
                        StreamSegMetrics(val_num_classes[1])]
        amal_blocks = []

        # add hook to amalgamation features
        with set_mode(self.student, training=True), \
                set_mode(self.teachers, training=False):
            rand_in = torch.randn([16, 3, 256, 256]).cuda()
            _, s_feas = self.student(rand_in, return_features=True)
            _, t0_feas = self.teachers[0](rand_in, return_features=True)
            _, t1_feas = self.teachers[1](rand_in, return_features=True)

            for s_fea, t0_fea, t1_fea in zip(s_feas.values(), t0_feas.values(), t1_feas.values()):
                cs = s_fea.shape[1]
                cts = [t0_fea.shape[1], t1_fea.shape[1]]
                amal_block = AmalBlock(cs=cs, cts=cts).to(self.device).train()
                amal_blocks.append(amal_block)
        self._amal_blocks = amal_blocks

    def train(self, max_iter, start_iter=0, epoch_length=None):
        self.iter = start_iter
        self.max_iter = max_iter
        self.epoch_length = epoch_length if epoch_length else len(self.ood_with_aug_loader)
        best_miou = 0

        block_params = []
        for block in self._amal_blocks:
            block_params.extend(list(block.parameters()))
        if isinstance(self.optim_s, torch.optim.SGD):
            self.optim_amal = torch.optim.SGD(block_params, lr=self.optim_s.param_groups[0]['lr'], momentum=0.9,
                                              weight_decay=1e-4)
        else:
            self.optim_amal = torch.optim.Adam(block_params, lr=self.optim_s.param_groups[0]['lr'], weight_decay=1e-4)
        self.sched_amal = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_amal, T_max=max_iter)

        for m in self.teachers[0].modules():
            if isinstance(m, nn.BatchNorm2d):
                self.bn_hooks.append(DeepInversionHook(m))

        for m in self.teachers[1].modules():
            if isinstance(m, nn.BatchNorm2d):
                self.bn_hooks.append(DeepInversionHook(m))

        with set_mode(self.student, training=True), \
                set_mode(self.teachers, training=False), \
                set_mode(self.netG, training=True), \
                set_mode(self.netD, training=True):
            for self.iter in range(start_iter, max_iter):

                ###############################
                # Patch Discrimination
                ###############################
                real = self.ood_without_aug_iter.next()[0].to(self.device)
                z = torch.randn(size=(self.batch_size, self.z_dim, 1, 1), device=self.device)
                images = self.netG(z)
                images = self.normalizer(images)
                d_out_fake = self.netD(images.detach())
                d_out_real = self.netD(real.detach())

                # patch discrimination loss
                loss_d = (F.binary_cross_entropy_with_logits(d_out_fake, torch.zeros_like(d_out_fake),
                                                             reduction='sum') + \
                          F.binary_cross_entropy_with_logits(d_out_real, torch.ones_like(d_out_real),
                                                             reduction='sum')) / \
                         (2 * len(d_out_fake)) * self.args.local

                self.optim_d.zero_grad()
                loss_d.backward()
                self.optim_d.step()

                ###############################
                # Generation
                ###############################
                t0_out, t0_feas = self.teachers[0](images, return_features=True)
                t1_out, t1_feas = self.teachers[1](images, return_features=True)
                t_out = [t0_out, t1_out]
                s_out = self.student(images)

                pyx = [F.softmax(i, dim=1) for i in t_out]
                py = [i.mean([0, 2, 3]) for i in pyx]

                d_out_fake = self.netD(images)
                # patch discrimination loss
                loss_local = F.binary_cross_entropy_with_logits(d_out_fake, torch.ones_like(d_out_fake),
                                                                reduction='sum') / len(d_out_fake)
                # bn loss
                loss_bn = sum([h.r_feature for h in self.bn_hooks])
                # adv loss
                loss_adv = -kldiv(s_out, torch.cat(t_out, dim=1)) / (256 * 256)
                # oh loss
                loss_oh = sum([F.cross_entropy(i, i.max(1)[1]) for i in t_out])
                # balance loss
                loss_balance = sum([(i * torch.log2(i)).sum() for i in py])
                # feature similarity loss
                loss_sim = 0.0
                for (f0, f1) in zip(t0_feas.values(), t1_feas.values()):
                    N, C, H, W = f0.shape
                    f0 = f0.view(N, C, -1)
                    f1 = f1.view(N, C, -1)
                    f0 = F.normalize(f0, p=2, dim=2)
                    f1 = F.normalize(f1, p=2, dim=2)
                    sim_mat = torch.abs(torch.matmul(f0, f1.permute(0, 2, 1)))
                    loss_sim += (1 - sim_mat).mean()

                # Final loss
                loss_g = self.args.adv * loss_adv + \
                         self.args.local * loss_local + \
                         self.args.balance * loss_balance + \
                         self.args.bn * loss_bn + \
                         self.args.oh * loss_oh + \
                         self.args.sim * loss_sim

                self.optim_g.zero_grad()
                loss_g.backward()
                self.optim_g.step()

                ###############################
                # Knowledge Amalgamation
                ###############################
                for _ in range(self.args.k_step):
                    z = torch.randn(size=(self.batch_size, self.z_dim, 1, 1), device=self.device)
                    vis_images = images = self.netG(z)
                    images = self.normalizer(images)
                    ood_images = self.ood_with_aug_iter.next()[0].to(self.device)
                    data = torch.cat([images, ood_images])

                    ka_output = self.ka_step(data)

                self.sched_s.step()
                self.sched_g.step()
                self.sched_d.step()
                self.sched_amal.step()

                # STEP END
                if self.iter % 100 == 0:
                    self.logger.info('loss_d: %.4f' % loss_d)
                    self.logger.info('loss_g: %.4f' % loss_g)
                    self.logger.info(
                        'loss_adv: %.4f, loss_local: %.4f, loss_oh: %.4f, loss_balance: %.4f, loss_bn: %.4f, loss_sim: %.4f' %
                        (loss_adv, loss_local, loss_oh, loss_balance, loss_bn, loss_sim))
                    self.logger.info('loss_ka: %.4f' % ka_output['loss_ka'])
                    self.logger.info('loss_kd: %.4f, loss_amal: %.4f, loss_recons: %.4f' %
                                     (ka_output['loss_kd'], ka_output['loss_amal'], ka_output['loss_recons']))
                    self.logger.info('optim_s_lr: %.6f, optim_g_lr: %.6f, optim_d_lr: %.6f, optim_amal_lr: %.6f' %
                                     (self.optim_s.param_groups[0]['lr'], self.optim_g.param_groups[0]['lr'],
                                      self.optim_d.param_groups[0]['lr'], self.optim_amal.param_groups[0]['lr']))

                # EPOCH END
                if self.epoch_length != None and (self.iter + 1) % self.epoch_length == 0:
                    scores = self.validate()

                    # Total MIOU on Multi Val_loaders
                    total_miou = 0
                    for i in range(len(scores)):
                        total_miou += scores[i]['Mean IoU'] * self.val_num_classes[i]
                    total_miou /= sum(self.val_num_classes)

                    self.logger.info(' [Eval] Epoch={}'.format(self.iter // self.epoch_length))
                    self.logger.info(' [Eval] Part0 MIoU={:.4f} Part1 MIoU={:.4f} Total MIoU={:.4f}'
                                     .format(scores[0]['Mean IoU'], scores[1]['Mean IoU'], total_miou))

                    is_best = total_miou > best_miou
                    best_miou = max(total_miou, best_miou)

                    save_checkpoint({
                        'epoch': self.iter // self.epoch_length,
                        's_state_dict': self.student.state_dict(),
                        'g_state_dict': self.netG.state_dict(),
                        'd_state_dict': self.netD.state_dict(),
                        'best_miou': float(best_miou),
                        'optim_s': self.optim_s.state_dict(),
                        'optim_g': self.optim_g.state_dict(),
                        'optim_d': self.optim_d.state_dict(),
                        'optim_amal': self.optim_amal.state_dict(),
                        'sched_s': self.sched_s.state_dict(),
                        'sched_g': self.sched_g.state_dict(),
                        'sched_d': self.sched_d.state_dict(),
                        'sched_amal': self.sched_amal.state_dict(),
                    }, is_best, os.path.join(self.output_dir, 'best.pth'))
                    save_image_batch(self.normalizer(real, True), os.path.join(self.output_dir, 'ood_data.png'))
                    save_image_batch(vis_images, os.path.join(self.output_dir, 'synthetic_data.png'))

        self.logger.info("Best: %.4f" % best_miou)

    def ka_step(self, data):
        s_out, s_feas = self.student(data, return_features=True)
        with torch.no_grad():
            t0_out, t0_feas = self.teachers[0](data, return_features=True)
            t1_out, t1_feas = self.teachers[1](data, return_features=True)

        loss_amal = 0
        loss_recons = 0
        for amal_block, s_fea, t0_fea, t1_fea in zip(self._amal_blocks, s_feas.values(), t0_feas.values(),
                                                     t1_feas.values()):
            fs, fts = s_fea, [t0_fea, t1_fea]
            rep, _fs, _fts = amal_block(fs, fts)
            # encoder loss
            loss_amal += F.mse_loss(_fs, rep.detach())
            # decoder loss
            loss_recons += sum([F.mse_loss(_ft, ft) for (_ft, ft) in zip(_fts, fts)])

        # kd loss
        loss_kd = kldiv(s_out, torch.cat([t0_out, t1_out], dim=1)) / (256 * 256)
        loss_dict = {"loss_kd": self.args.kd * loss_kd,
                     "loss_amal": self.args.amal * loss_amal,
                     "loss_recons": self.args.recons * loss_recons}
        loss_ka = sum(loss_dict.values())

        self.optim_s.zero_grad()
        self.optim_amal.zero_grad()
        loss_ka.backward()
        self.optim_s.step()
        self.optim_amal.step()

        metrics = {loss_name: loss_value.item() for (loss_name, loss_value) in loss_dict.items()}
        metrics.update({'loss_ka': loss_ka.item()})
        return metrics

    def validate(self):
        scores = []
        with set_mode(self.student, training=False):
            with torch.no_grad():
                for i, val_loader in enumerate(self.val_loaders):
                    self.metrics[i].reset()
                    for image, label in val_loader:
                        image = image.to(self.device, dtype=torch.float32)
                        label = label.to(self.device, dtype=torch.long)

                        output = self.student(image)[:, sum(self.val_num_classes[:i]):sum(self.val_num_classes[:i + 1]),
                                 :, :]
                        pred = output.detach().max(dim=1)[1].cpu().numpy()
                        target = label.cpu().numpy()

                        self.metrics[i].update(target, pred)

                    scores.append(self.metrics[i].get_results())

        return scores


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)