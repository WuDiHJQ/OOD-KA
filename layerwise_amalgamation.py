import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.hooks import FeatureHook
from engine.criterions import kldiv

import typing
import time
import os
from engine.utils import move_to_device, set_mode

class AmalBlock(nn.Module):
    # 特征融合模块 实现从cts 到 cs的学习
    # cs channel student
    # cts channel teachers
    # fs feature student
    # fts feature teachers
    # enc encoder conv 教师cat2学生
    # fam 特征自适应模块
    # dec decoder conv 学生2教师cat
    def __init__(self, cs, cts):
        super( AmalBlock, self ).__init__()
        self.cs, self.cts = cs, cts
        self.enc = nn.Conv2d( in_channels=sum(self.cts), out_channels=self.cs, kernel_size=1, stride=1, padding=0, bias=True )
        self.fam = nn.Conv2d( in_channels=self.cs, out_channels=self.cs, kernel_size=1, stride=1, padding=0, bias=True )
        self.dec = nn.Conv2d( in_channels=self.cs, out_channels=sum(self.cts), kernel_size=1, stride=1, padding=0, bias=True )
    
    def forward(self, fs, fts):
        rep = self.enc( torch.cat( fts, dim=1 ) )
        _fts = self.dec( rep )
        _fts = torch.split( _fts, self.cts, dim=1 )
        _fs = self.fam( fs )
        return rep, _fs, _fts

class LayerWiseAmalgamator():
    def __init__(self, logger=None, tb_writer=None, output_dir=None):
        self.logger = logger if logger else get_logger(name='kamal', color=True)
        self.tb_writer = tb_writer
        self.output_dir = output_dir
    
    def setup(
        self, 
        student,
        teachers,
        layer_groups: typing.Sequence[typing.Sequence],
        layer_channels: typing.Sequence[typing.Sequence],
        train_loader:  torch.utils.data.DataLoader, 
        val_loaders: [],
        val_num_classes: [],
        optimizer: torch.optim.Optimizer, 
        scheduler: torch.optim.lr_scheduler,
        weights = [1., 1., 1.],
        device=None,
    ):
        if device is None:
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.device = device
        self.train_loader = train_loader
        self.train_loader_iter = iter(train_loader)
        self.val_loaders = val_loaders
        self.val_num_classes = val_num_classes
        self.student = student.to(self.device)
        self.teachers = nn.ModuleList(teachers).to(self.device) 
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = dict()
        self.weights = weights
        amal_blocks = []
        
        # 为三个模型每个卷积层添加hook 模块间添加融合
        # hook获取中间特征
        # amal_blocks 包含融合模块hook和通道信息
        for group, C in zip(layer_groups, layer_channels):
            hooks = [ FeatureHook(layer) for layer in group ]
            amal_block = AmalBlock(cs=C[0], cts=C[1:]).to(self.device).train()
            amal_blocks.append( (amal_block, hooks, C)  )
        self._amal_blocks = amal_blocks
        
    def run(self, max_iter, start_iter=0, epoch_length=None ):
        self.iter = start_iter
        self.max_iter = max_iter
        self.epoch_length = epoch_length if epoch_length else len(self.train_loader)
        best_acc1 = 0
        
        # 获得所有优化参数
        block_params = []
        for block, _, _ in self._amal_blocks:
            block_params.extend( list(block.parameters()) )
        if isinstance( self.optimizer, torch.optim.SGD ):
            self._amal_optimimizer = torch.optim.SGD( block_params, lr=self.optimizer.param_groups[0]['lr'], momentum=0.9, weight_decay=1e-4 )
        else:
            self._amal_optimimizer = torch.optim.Adam( block_params, lr=self.optimizer.param_groups[0]['lr'], weight_decay=1e-4 )
        self._amal_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( self._amal_optimimizer, T_max=max_iter )

        with set_mode(self.student, training=True), \
             set_mode(self.teachers, training=False):
            for self.iter in range( start_iter, max_iter ):
                   
                self.batch = self.get_batch()
                step_output = self.train_step(self.batch)
                
                if isinstance(step_output, dict):
                    self.metrics.update(step_output)
                
                # STEP END
                if self.iter%100 == 0:
                    self.logger.info('loss_kd: %.4f, loss_amal: %.4f, loss_recons: %.4f, total_loss: %.4f, step_time: %.4f, lr: %.6f' %
                                      (step_output['loss_kd'], step_output['loss_amal'],
                                       step_output['loss_recons'], step_output['total_loss'],
                                       step_output['step_time'], step_output['lr']))
                    
                # EPOCH END
                if self.epoch_length!=None and (self.iter+1)%self.epoch_length==0:
                    acc1 = self.validate()
                    is_best = acc1 > best_acc1
                    best_acc1 = max(acc1, best_acc1)
                    save_checkpoint({
                        'epoch': self.iter//self.epoch_length,
                        'state_dict': self.student.state_dict(),
                        'best_acc1': float(best_acc1),
                        'optimizer' : self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict()
                    }, is_best, os.path.join(self.output_dir, 'best.pth'))
        
        self.logger.info("Best: %.4f"%best_acc1)
                    
    
    def get_batch(self):
        try:
            batch = next( self.train_loader_iter )
        except StopIteration:
            self.train_loader_iter = iter(self.train_loader) # reset iterator
            batch = next( self.train_loader_iter )
        if not isinstance(batch, (list, tuple)):
            batch = [ batch, ] # no targets
        return batch
    
    def train_step(self, batch):
        start_time = time.perf_counter()
        batch = move_to_device(batch, self.device)
        data = batch[0]
        # 获取ts 输出
        s_out = self.student( data )
        with torch.no_grad():
            t_out = [ teacher( data ) for teacher in self.teachers ]

        loss_amal = 0
        loss_recons = 0
        for amal_block, hooks, C in self._amal_blocks:
            features = [ h.output for h in hooks ]
            fs, fts = features[0], features[1:]
            rep, _fs, _fts = amal_block( fs, fts )
            # 自编码解码保持结构不变性 所以用融合后的紧实特征和目前特征计算
            loss_amal += F.mse_loss( _fs, rep.detach() )
            # 以及衡量教师特征的恢复能力
            loss_recons += sum( [ F.mse_loss( _ft, ft ) for (_ft, ft) in zip( _fts, fts ) ] )
        # 输出kd损失
        loss_kd = kldiv( s_out, torch.cat( t_out, dim=1 ) )
        #loss_kd = F.mse_loss( s_out, torch.cat( t_out, dim=1 ) )
        loss_dict = { "loss_kd":      self.weights[0] * loss_kd,
                      "loss_amal":    self.weights[1] * loss_amal,
                      "loss_recons":  self.weights[2] * loss_recons }
        loss = sum(loss_dict.values())
        self.optimizer.zero_grad()
        self._amal_optimimizer.zero_grad()
        loss.backward()
        # 先调整融合模块 再微调整个网络
        self.optimizer.step()
        self.scheduler.step()
        self._amal_optimimizer.step()
        self._amal_scheduler.step()
        step_time = time.perf_counter() - start_time
        metrics = { loss_name: loss_value.item() for (loss_name, loss_value) in loss_dict.items() }
        metrics.update({
            'total_loss': loss.item(),
            'step_time': step_time,
            'lr': float( self.optimizer.param_groups[0]['lr'] )
        })
        return metrics
    
    def validate(self):
        # batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        part_top1 = [AverageMeter('Part0_Acc@1', ':6.2f'), AverageMeter('Part1_Acc@1', ':6.2f')]
        part_top5 = [AverageMeter('Part0_Acc@5', ':6.2f'), AverageMeter('Part1_Acc@5', ':6.2f')]
        total_top1 =  AverageMeter('Total_Acc@1', ':6.2f')
        total_top5 =  AverageMeter('Total_Acc@5', ':6.2f')
        # switch to evaluate mode
        with set_mode(self.student, training=False), \
             set_mode(self.teachers, training=False):
            with torch.no_grad():
                for i, val_loader in enumerate(self.val_loaders):
                # end = time.time()
                    for batch in val_loader:
                        batch = move_to_device(batch, self.device)
                        data, target = batch
                        # compute & cut output
                        output = self.student(data)[:, sum(self.val_num_classes[:i]):sum(self.val_num_classes[:i+1])]
                        # measure accuracy and record loss
                        acc1, acc5 = accuracy(output, target, topk=(1, 5))
                        part_top1[i].update(acc1[0], data.size(0))
                        part_top5[i].update(acc5[0], data.size(0))          
                        total_top1.update(acc1[0], data.size(0))
                        total_top5.update(acc5[0], data.size(0))
                        # measure elapsed time
                        # batch_time.update(time.time() - end)
                        # end = time.time()
                self.logger.info(' [Eval] Epoch={}  Lr={:.6f}'
                        .format(self.iter//self.epoch_length, self.optimizer.param_groups[0]['lr']))
                self.logger.info(' [Eval] Part0 Acc@1={:.4f} Acc@5={:.4f}'
                        .format(part_top1[0].avg, part_top5[0].avg))
                self.logger.info(' [Eval] Part1 Acc@1={:.4f} Acc@5={:.4f}'
                        .format(part_top1[1].avg, part_top5[1].avg))
                self.logger.info(' [Eval] Total Acc@1={:.4f} Acc@5={:.4f}'
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

    
def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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

