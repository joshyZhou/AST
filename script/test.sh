##test on AGAN-Data##
python3 test/test_raindrop.py --arch AST_B --input_dir ../dataset/raindrop/test_a/ --result_dir ./results/rain_drop/AGAN-Data/ --weights ./logs/raindrop/AGAN-Data/AST_B/models/model_best.pth --token_mlp frfn

##test on densehaze##
python3 test/test_denseHaze.py --arch AST_B --input_dir ../dataset/Dense-Haze-v2/valid_dense --result_dir ./results/dehaze/DenseHaze/  --weights ./logs/dehazing/DenseHaze/AST_B/models/model_best.pth --token_mlp frfn

## test on SPAD##
python3 test/test_spad.py --arch AST_B --input_dir /PATH/TO/DATASET/ --result_dir ./results/deraining/SPAD/ --weights ./pretrained/rain/model_best.pth --token_mlp frfn



