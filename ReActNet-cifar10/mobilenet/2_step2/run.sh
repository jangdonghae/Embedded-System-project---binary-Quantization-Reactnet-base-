clear
mkdir models
cp ../1_step1/models/checkpoint.pth.tar ./models/checkpoint_ba.pth.tar
mkdir log
python train.py --batch_size=128 --learning_rate=5e-3 --epochs=35 --weight_decay=0
