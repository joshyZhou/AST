##train on SPAD##
python ./train/train_derain.py --arch AST_B --batch_size 32 --gpu 0,1 --train_ps 128 --train_dir ../derain_dataset/derain/ --env _1030_s1 --val_dir ../derain_dataset/derain/ --save_dir ./logs/ --dataset spad --warmup --token_mlp frfn --nepoch 20 --lr_initial 0.0002

# python ./train/train_derain.py --arch AST_B --retrain --pretrain_weights ./logs/derain/spad/AST_B_1030_s1/models/model_best.pth --batch_size 16 --gpu 0,1 --train_ps 256 --train_dir ../derain_dataset/derain/ --env _1030_s2 --val_ps 256 --val_dir ../derain_dataset/derain/ --save_dir ./logs/ --dataset spad --warmup --token_mlp frfn --nepoch 15 --lr_initial 0.0001
