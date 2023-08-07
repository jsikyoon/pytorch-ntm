CUDA_VISIBLE_DEVICES=6 python train.py --seed 1 --task copy --checkpoint-interval 500 --checkpoint-path ./notebooks/copy
CUDA_VISIBLE_DEVICES=7 python train.py --seed 10 --task copy --checkpoint-interval 500 --checkpoint-path ./notebooks/copy
CUDA_VISIBLE_DEVICES=6 python train.py --seed 100 --task copy --checkpoint-interval 500 --checkpoint-path ./notebooks/copy
CUDA_VISIBLE_DEVICES=7 python train.py --seed 1000 --task copy --checkpoint-interval 500 --checkpoint-path ./notebooks/copy
