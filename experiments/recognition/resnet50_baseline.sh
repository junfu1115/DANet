# baseline
python train_dist.py --dataset imagenet --model resnet50 --lr-scheduler cos --epochs 120 --checkname resnet50_check --lr 0.025 --batch-size 64

# rectify
python train_dist.py --dataset imagenet --model resnet50 --lr-scheduler cos --epochs 120 --checkname resnet50_rt --lr 0.1 --batch-size 256 --rectify

# warmup
python train_dist.py --dataset imagenet --model resnet50 --lr-scheduler cos --epochs 120 --checkname resnet50_rt_warm --lr 0.1 --batch-size 256 --warmup-epochs 5 --rectify 

# no-bn-wd
python train_dist.py --dataset imagenet --model resnet50 --lr-scheduler cos --epochs 120 --checkname resnet50_rt_nobnwd_warm --lr 0.1 --batch-size 256 --no-bn-wd --warmup-epochs 5 --rectify 

# LS
python train_dist.py --dataset imagenet --model resnet50 --lr-scheduler cos --epochs 120 --checkname resnet50_rt_ls --lr 0.1 --batch-size 256 --label-smoothing 0.1 --rectify

# Mixup + LS
python train_dist.py --dataset imagenet --model resnet50 --lr-scheduler cos --epochs 200 --checkname resnet50_rt_ls_mixup --lr 0.1 --batch-size 256 --label-smoothing 0.1 --mixup 0.2 --rectify

# last-gamma
python train_dist.py --dataset imagenet --model resnet50 --lr-scheduler cos --epochs 120 --checkname resnet50_rt_gamma --lr 0.1 --batch-size 256 --last-gamma  --rectify

# BoTs
python train_dist.py --dataset imagenet --model resnet50 --lr-scheduler cos --epochs 200 --checkname resnet50_rt_bots --lr 0.1 --batch-size 256 --label-smoothing 0.1 --mixup 0.2 --last-gamma --no-bn-wd --warmup-epochs 5 --rectify

# resnet50d
python train_dist.py --dataset imagenet --model resnet50d --lr-scheduler cos --epochs 200 --checkname resnet50d_rt_bots --lr 0.1 --batch-size 256 --label-smoothing 0.1 --mixup 0.2 --last-gamma --no-bn-wd --warmup-epochs 5 --rectify

# dropblock
python train_dist.py --dataset imagenet --model resnet50 --lr-scheduler cos --epochs 200 --checkname  --label-smoothing 0.1 --mixup 0.2 --lr 0.1 --batch-size 256 --label-smoothing 0.1 --mixup 0.2  --dropblock-prob 0.1 --rectify

# resnest50
python train_dist.py --dataset imagenet --model resnest50 --lr-scheduler cos --epochs 270 --checkname resnest50_rt_bots --lr 0.1 --batch-size 256 --label-smoothing 0.1 --mixup 0.2  --last-gamma --no-bn-wd --warmup-epochs 5 --dropblock-prob 0.1 --rectify
