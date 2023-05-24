# OOD_KA deeplab
python PFA_deeplab.py \
--model deeplabv3plus_mobilenet \
--output_stride 16 \
--batch_size 16 \
--lr 1e-2 \
--lr_g 1e-3 \
--z_dim 256 \
--oh 1.0 \
--bn 0.5 \
--local 0.5 \
--adv 1.0 \
--sim 1.0 \
--balance 1.0 