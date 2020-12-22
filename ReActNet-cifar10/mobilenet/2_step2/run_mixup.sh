clear
mkdir models_cutmix
cp ../1_step1/models_cutmix/checkpoint.pth.tar ./models_cutmix/checkpoint_ba.pth.tar
mkdir log
python train.py --batch_size=128 --learning_rate=5e-3 --epochs=35 --weight_decay=0 --beta 1 --mixup_prob 1.0