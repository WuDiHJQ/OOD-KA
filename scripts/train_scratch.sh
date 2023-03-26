# train scratch
python train_scratch.py --lr 0.1 --batch-size 128 --model resnet34 --dataset cifar100_part0 --gpu 0
python train_scratch.py --lr 0.1 --batch-size 128 --model resnet34 --dataset cifar100_part1 --gpu 0

python train_scratch.py --lr 0.1 --batch-size 128 --model wrn16_2 --dataset cifar100_part0 --gpu 0
python train_scratch.py --lr 0.1 --batch-size 128 --model wrn16_2 --dataset cifar100_part1 --gpu 0

