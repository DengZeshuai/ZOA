# 8bit ViT
CUDA_VISIBLE_DEVICES=0 python3 main.py --output ./outputs --algorithm 'zoa_vit' --tag '_8bit' --lr 0.0005 --sc 0.02 --lambda_bp 30 --quant --bit 8 --domain_t 0.1
# 6bit ViT
CUDA_VISIBLE_DEVICES=0 python3 main.py --output ./outputs --algorithm 'zoa_vit' --tag '_6bit' --lr 0.0002 --sc 0.02 --lambda_bp 30 --quant --bit 6 --domain_t 0.1

# 8bit resnet50
CUDA_VISIBLE_DEVICES=0 python3 main.py --output ./outputs --algorithm 'zoa_resnet' --tag '_8bit' --arch resnet50 --lr 0.0001 --sc 0.01 --lambda_bp 1 --use_in1k_norm --use_in1k_norm_c --quant --bit 8 --domain_t 0.2
