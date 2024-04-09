##train on densehaze##

python3 ./train/train_dehaze.py --arch AST_B --batch_size 4 --gpu '0' --train_ps 256 --train_dir ../dataset/Dense-Haze-v2/train_dense/ --val_ps 256 --val_dir ../dataset/Dense-Haze-v2/valid_dense/ --lr 0.0002 --env _1108_s1 --mode dehaze --nepoch 2000 --dataset DenseHaze --warmup --token_mlp frfn 

# python3 ./train/train_dehaze.py --arch AST_B --batch_size 4 --retrain --pretrain_weights ./logs/dehazing/DenseHaze/AST_B_1108_s1/models/model_best.pth --gpu '0' --train_ps 384 --train_dir ../dataset/Dense-Haze-v2/train_dense/ --val_ps 384 --val_dir ../dataset/Dense-Haze-v2/valid_dense/ --lr 0.00012 --env _1109_s2 --mode dehaze --nepoch 1200 --token_mlp frfn --dataset DenseHaze --warmup

# python3 ./train/train_dehaze.py --arch AST_B --batch_size 4 --retrain --pretrain_weights ./logs/dehazing/DenseHaze/AST_B_1109_s2/models/model_best.pth --gpu '0' --train_ps 512 --train_dir ../dataset/Dense-Haze-v2/train_dense/ --val_ps 512 --val_dir ../dataset/Dense-Haze-v2/valid_dense/ --lr 0.00008 --env _1110_s3 --mode dehaze --nepoch 800 --token_mlp frfn --dataset DenseHaze --warmup

# python3 ./train/train_dehaze.py --arch AST_B --retrain --pretrain_weights ./logs/dehazing/DenseHaze/AST_B_1110_s3/models/model_best.pth --batch_size 2 --gpu '0,1' --train_ps 768 --train_dir ../dataset/Dense-Haze-v2/train_dense/ --val_ps 768 --val_dir ../dataset/Dense-Haze-v2/valid_dense/ --lr 0.00003 --env _1105_s4 --mode dehaze --nepoch 300 --token_mlp frfn --dataset DenseHaze --warmup

# python3 ./train/train_dehaze.py --arch AST_B --retrain --pretrain_weights ./logs/dehazing/DenseHaze/AST_B_1105/models/model_best.pth --batch_size 2 --gpu '0,1' --train_ps 896 --train_dir ../dataset/Dense-Haze-v2/train_dense/ --val_ps 896 --val_dir ../dataset/Dense-Haze-v2/valid_dense/ --lr 0.00001 --env _1106_s5 --mode dehaze --nepoch 80 --token_mlp frfn --dataset DenseHaze --warmup
