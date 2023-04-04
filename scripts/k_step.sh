# k step select
python PFA.py --model wrn16_2 --unlabeled svhn -k 1
python PFA.py --model wrn16_2 --unlabeled svhn -k 2
python PFA.py --model wrn16_2 --unlabeled svhn -k 3
python PFA.py --model wrn16_2 --unlabeled svhn -k 4
python PFA.py --model wrn16_2 --unlabeled svhn -k 5
python PFA.py --model wrn16_2 --unlabeled svhn -k 6


python PFA.py --model wrn16_2 --unlabeled cifar10 -k 1
python PFA.py --model wrn16_2 --unlabeled cifar10 -k 2
python PFA.py --model wrn16_2 --unlabeled cifar10 -k 3
python PFA.py --model wrn16_2 --unlabeled cifar10 -k 4
python PFA.py --model wrn16_2 --unlabeled cifar10 -k 5
python PFA.py --model wrn16_2 --unlabeled cifar10 -k 6
