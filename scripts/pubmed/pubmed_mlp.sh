export CUDA_VISIBLE_DEVICES=1

python -u main.py \
    --is_training 1 \
    --train_epochs 2000 \
    --model_id PUBMED_TEST \
    --dataset pubmed \
    --model SLAPS_MLP \
    --input_dim 500 \
    --hidden_dim 32 \
    --output_dim 3 \
    --r 10 \
    --eta 5 \
    --noise_type "zero" \
    --generator MLP \
    --gen_input_dim 500 \
    --gen_layers_size 2 \
    --gen_k 15 \
    --use_gpu True \
    --lr_c 0.01 \
    --lr_DAE 0.01 \
    --dropout_c 0.5 \
    --dropout_DAE 0.5 \
    --lambda_val 10 \
    --itr 10 \
    --patience 500
