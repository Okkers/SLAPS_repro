export CUDA_VISIBLE_DEVICES=1

python -u main.py \
    --is_training 1 \
    --train_epochs 2000 \
    --model_id CITESEER_TEST_1 \
    --dataset citeseer370 \
    --model SLAPS_FP \
    --input_dim 3703 \
    --hidden_dim 32 \
    --output_dim 6 \
    --r 10 \
    --eta 1 \
    --noise_type "not_zero" \
    --generator FP \
    --gen_input_dim 3703 \
    --gen_layers_size 2 \
    --gen_k 30 \
    --use_gpu True \
    --lr_c 0.01 \
    --lr_DAE 0.01 \
    --dropout_c 0.5 \
    --dropout_DAE 0.5 \
    --lambda_val 1 \
    --itr 10 \
    --patience 500 \
    --is_discrete \
    --hidden_dim_dae 1024 \
    --weight_decay_c 0.05

python -u main.py \
    --is_training 1 \
    --train_epochs 2000 \
    --model_id CITESEER_TEST_1 \
    --dataset citeseer370 \
    --model SLAPS_MLP \
    --input_dim 3703 \
    --hidden_dim 32 \
    --output_dim 6 \
    --r 10 \
    --eta 5 \
    --noise_type "not_zero" \
    --generator MLP \
    --gen_input_dim 3703 \
    --gen_layers_size 2 \
    --gen_k 30 \
    --use_gpu True \
    --lr_c 0.01 \
    --lr_DAE 0.001 \
    --dropout_c 0.25 \
    --dropout_DAE 0.5 \
    --lambda_val 10 \
    --itr 10 \
    --patience 200 \
    --is_discrete \
    --hidden_dim_dae 1024 \
    --weight_decay_c 0.0005

python -u main.py \
    --is_training 1 \
    --train_epochs 2000 \
    --model_id CORA_TEST_1 \
    --dataset cora \
    --model SLAPS_MLPD \
    --input_dim 1433 \
    --hidden_dim 32 \
    --output_dim 7 \
    --r 10 \
    --eta 5 \
    --noise_type "not_zero" \
    --generator MLP-D \
    --gen_input_dim 1433 \
    --gen_layers_size 2 \
    --gen_k 15 \
    --use_gpu True \
    --lr_c 0.01 \
    --lr_DAE 0.001 \
    --dropout_c 0.25 \
    --dropout_DAE 0.5 \
    --lambda_val 10 \
    --itr 10 \
    --patience 500 \
    --is_discrete \
    --weight_decay_c 0.05 \
    --hidden_dim_dae 512

python -u main.py \
    --is_training 1 \
    --train_epochs 2000 \
    --model_id CORA_TEST_1 \
    --dataset cora390 \
    --model SLAPS_MLPD \
    --input_dim 1433 \
    --hidden_dim 32 \
    --output_dim 7 \
    --r 10 \
    --eta 5 \
    --noise_type "not_zero" \
    --generator MLP-D \
    --gen_input_dim 1433 \
    --gen_layers_size 2 \
    --gen_k 20 \
    --use_gpu True \
    --lr_c 0.001 \
    --lr_DAE 0.001 \
    --dropout_c 0.25 \
    --dropout_DAE 0.5 \
    --lambda_val 10 \
    --itr 10 \
    --patience 500 \
    --is_discrete \
    --weight_decay_c 0.0005 \
    --hidden_dim_dae 512









