python main.py --model naive-ensemble-sum-head --net resnet-cifar --dropout_rate 0.5 --dropout_iters 20 --dataset cifar10 --batch_size 1024 --loss cent --epochs 200 --num_workers 16
python main.py --model naive-ensemble-sum --net resnet-cifar --dropout_rate 0.5 --dropout_iters 20 --dataset cifar10 --batch_size 1024 --loss cent --epochs 200 --num_workers 16
python main.py --model naive-ensemble --net resnet-cifar --dropout_rate 0.5 --dropout_iters 20 --dataset cifar10 --batch_size 1024 --loss cent --epochs 200 --num_workers 16
