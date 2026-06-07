export CUDA_VISIBLE_DEVICES=0

python -u main.py \
    --is_training 1 \
    --train_epochs 2000 \
    --model_id MNIST2000_TEST \
    --dataset mnist \
    --mnist_training_size 2000 \
    --model SLAPS_MLPD \
    --input_dim 784 \
    --hidden_dim 32 \
    --output_dim 10 \
    --r 10 \
    --eta 5 \
    --noise_type "not_zero" \
    --generator MLP-D \
    --gen_input_dim 784 \
    --gen_layers_size 2 \
    --gen_k 15 \
    --use_gpu True \
    --lr_c 0.01 \
    --lr_DAE 0.001 \
    --dropout_c 0.5 \
    --dropout_DAE 0.5 \
    --lambda_val 100 \
    --itr 10 \
    --patience 15 \
    --weight_decay_c 0.05 \
    --hidden_dim_dae 1024

