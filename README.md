# Embedded-System-project---binary-Quantization-Reactnet-base-

To run Reactnet for cifar10
1. go to mobilenet/1_step1 directory
2. run run.sh, or execute python train.py --batch_size=128 --learning_rate=5e-3 --epochs=60 --weight_decay=1e-5
3. go to mobilenet/2_step2 directory
4. run run.sh, or execute python train.py --batch_size=128 --learning_rate=5e-3 --epochs=35 --weight_decay=0, after making /models directory and put the model trained on step1

To run Reactnet with cutmix(data augmentation technique)
1. go to mobilenet/1_step1 directory
2. run run_cutmix.sh, or execute python train_cutmix_binary.py --batch_size=128 --learning_rate=5e-3 --epochs=60 --weight_decay=1e-5 --beta 1 --cutmix_prob 1
3. go to mobilenet/2_step2 directory
4. run run_cutmix.sh, or execute python train_cutmix_binary.pyy --batch_size=128 --learning_rate=5e-3 --epochs=35 --weight_decay=0  --beta 1 --cutmix_prob 1, after making /models directory and put the model trained on step1

To run Reactnet with mixup(data augmentation technique)
1. go to mobilenet/1_step1 directory
2. run run_mixup.sh, or execute python train_cutmix_binary.py --batch_size=128 --learning_rate=5e-3 --epochs=60 --weight_decay=1e-5 --beta 1 --cutmix_prob 1
3. go to mobilenet/2_step2 directory
4. run run_mixup.sh, or execute python train_cutmix_binary.py --batch_size=128 --learning_rate=5e-3 --epochs=35 --weight_decay=0  --beta 1 --cutmix_prob 1, after making /models directory and put the model trained on step1
