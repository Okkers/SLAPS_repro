export CUDA_VISIBLE_DEVICES=1

python -u main.py \
    --is_training 1 \
    --train_epochs 2000 \
    --model_id OGBN-ARXIV_TEST \
    --dataset ogbn-arxiv \
    --model SLAPS_MLPD \
    --input_dim 128 \
    --hidden_dim 256 \
    --output_dim 40 \
    --r 1 \
    --eta 5 \
    --noise_type "zero" \
    --generator MLP-D \
    --gen_input_dim 128 \
    --gen_layers_size 2 \
    --gen_k 15 \
    --use_gpu True \
    --lr_c 0.01 \
    --lr_DAE 0.001 \
    --dropout_c 0.5 \
    --dropout_DAE 0.25 \
    --lambda_val 10 \
    --itr 10 \
    --patience 500
