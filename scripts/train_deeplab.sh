# train deeplab
python train_deeplab.py --model deeplabv3plus_mobilenet --dataset voc --output_stride 16 --lr 0.01 --batch_size 16
python train_deeplab.py --model deeplabv3plus_mobilenet --dataset nyu --output_stride 16 --lr 0.01 --batch_size 16