# PFA reproduce
python PFA.py --model vgg16 --unlabeled cifar10
python PFA.py --model vgg16 --unlabeled svhn
python PFA.py --model vgg16 --unlabeled imagenet_32x32
python PFA.py --model vgg16 --unlabeled places365_32x32


python PFA.py --model resnet18 --unlabeled cifar10
python PFA.py --model resnet18 --unlabeled svhn
python PFA.py --model resnet18 --unlabeled imagenet_32x32
python PFA.py --model resnet18 --unlabeled places365_32x32


python PFA.py --model wrn40_2 --unlabeled cifar10
python PFA.py --model wrn40_2 --unlabeled svhn
python PFA.py --model wrn40_2 --unlabeled imagenet_32x32
python PFA.py --model wrn40_2 --unlabeled places365_32x32


python PFA.py --model wrn16_2 --unlabeled cifar10
python PFA.py --model wrn16_2 --unlabeled svhn
python PFA.py --model wrn16_2 --unlabeled imagenet_32x32
python PFA.py --model wrn16_2 --unlabeled places365_32x32
