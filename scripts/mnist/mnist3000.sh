export CUDA_VISIBLE_DEVICES=0

python -u main.py \
    --is_training 1 \
    --train_epochs 2000 \
    --model_id MNIST3000_TEST \
    --dataset mnist \
    --mnist_training_size 3000 \
    --model SLAPS_MLP \
    --input_dim 784 \
    --hidden_dim 32 \
    --output_dim 10 \
    --r 5 \
    --eta 5 \
    --noise_type "not_zero" \
    --generator MLP \
    --gen_input_dim 784 \
    --gen_layers_size 2 \
    --gen_k 15 \
    --use_gpu True \
    --lr_c 0.01 \
    --lr_DAE 0.01 \
    --dropout_c 0.5 \
    --dropout_DAE 0.5 \
    --lambda_val 10 \
    --itr 10 \
    --patience 15 \
    --weight_decay_c 0.003 \
    --hidden_dim_dae 512
