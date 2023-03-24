from engine import metrics, evaluators
from engine.utils import get_logger, flatten_dict
from layerwise_amalgamation import LayerWiseAmalgamator
from torchvision import datasets, transforms as T

import torch, time, os
import registry
from torch.utils.tensorboard import SummaryWriter

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='data')
parser.add_argument('--model', default='resnet34')
parser.add_argument('--dataset', default='cifar100')
parser.add_argument('--unlabeled', default='cifar10')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument( '--lr', default=3e-4, type=float,
                    help='initial learning rate')
parser.add_argument( '--test_only', action='store_true', default=False)
parser.add_argument( '--ckpt', type=str, default=None)
args = parser.parse_args()

def main():
    # ==================================================
    # ==================== dataset =====================
    # ==================================================
    part0_num_classes, part0_train, part0_val = registry.get_dataset(name='%s_part0'%(args.dataset), data_root=args.data_root)
    part1_num_classes, part1_train, part1_val = registry.get_dataset(name='%s_part1'%(args.dataset), data_root=args.data_root)
    _, ood_train, _ = registry.get_dataset(name=args.unlabeled, data_root=args.data_root)
    # ==================================================
    # ===================== model ======================
    # ==================================================
    part0_teacher = registry.get_model(args.model, num_classes=part0_num_classes, pretrained=False)
    part1_teacher = registry.get_model(args.model, num_classes=part1_num_classes, pretrained=False)
    student = registry.get_model(args.model, num_classes=part0_num_classes+part1_num_classes, pretrained=False)
    
    part0_teacher.load_state_dict(torch.load('checkpoints/pretrained/%s_part0_%s.pth'%(args.dataset, args.model))['state_dict'])
    part1_teacher.load_state_dict(torch.load('checkpoints/pretrained/%s_part1_%s.pth'%(args.dataset, args.model))['state_dict'])
    
    # ==================================================
    # =================== transform ====================
    # ==================================================
    ood_train.transform = part0_train.transform

    # ==================================================
    # =================== DataLoader ===================
    # ==================================================
    # 训练集为二者合并 验证集分开 不需要label
    train_loader = torch.utils.data.DataLoader( ood_train, batch_size=256, shuffle=True, num_workers=4 )
    part0_val_loader = torch.utils.data.DataLoader( part0_val, batch_size=256, shuffle=False, num_workers=4 )
    part1_val_loader = torch.utils.data.DataLoader( part1_val, batch_size=256, shuffle=False, num_workers=4 )
    
    # 共计迭代200 epoch 使用合并数据集 保证预处理后输入一致即可
    TOTAL_ITERS=len(train_loader) * args.epochs
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    optim = torch.optim.Adam( student.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR( optim, T_max=TOTAL_ITERS )

    # ==================================================
    # ==================== Trainer =====================
    # ==================================================
    output_dir = 'run/ood_layerwise_ka-%s'%( time.asctime().replace( ' ', '_' ) )
    trainer = LayerWiseAmalgamator( 
        logger=get_logger(name='ood-layerwise-ka', output= os.path.join(output_dir, 'log.txt')), 
        tb_writer=SummaryWriter(log_dir=output_dir),
        output_dir = output_dir
    )
    for k, v in flatten_dict( vars(args) ).items(): # print args
        trainer.logger.info( "%s: %s"%(k,v) )
    
    # ==================================================
    # =================== 只动卷积层 ====================
    # ==================================================
    layer_groups = []
    layer_channels = []
    for stu_block, part0_block, part1_block in zip( student.modules(), part0_teacher.modules(), part1_teacher.modules() ):
        if isinstance( stu_block, torch.nn.Conv2d ):
            layer_groups.append( [ stu_block, part0_block, part1_block ] )
            layer_channels.append( [ stu_block.out_channels, part0_block.out_channels, part1_block.out_channels ] )
    
    trainer.setup( student=student, 
                   teachers=[part0_teacher, part1_teacher],
                   layer_groups=layer_groups,
                   layer_channels=layer_channels,
                   train_loader=train_loader,
                   val_loaders=[part0_val_loader, part1_val_loader],
                   val_num_classes=[part0_num_classes, part1_num_classes],
                   optimizer=optim,
                   scheduler=sched,
                   device=device,
                   weights=[1., 1., 1.] )
    
    if args.ckpt is not None:
        trainer.student.load_state_dict( torch.load(args.ckpt)['state_dict'] )
        print("Load student model from %s"%args.ckpt)
    if args.test_only:
        trainer.validate()
        return 
    
    trainer.run(start_iter=0, max_iter=TOTAL_ITERS)

if __name__=='__main__':
    main()