python pytorch_mlp_framework/train_evaluate_image_classification_system.py --batch_size 100 --seed 0 --num_filters 32 --num_stages 3 --num_blocks_per_stage 5 --experiment_name VGG_38_experiment --use_gpu True --num_classes 100 --block_type conv_block --continue_from_epoch -1

# --experiment_name new name
# --block_type conv_BN_block / conv_BN_Res_block
# add --lr 1e-3 / 1e-2

# VGG38 BN 1e-3
python pytorch_mlp_framework/train_evaluate_image_classification_system.py --batch_size 100 --seed 0 --num_filters 32 --num_stages 3 --num_blocks_per_stage 5 --experiment_name VGG_38_BN_1e-3_experiment --use_gpu True --num_classes 100 --block_type conv_BN_block --continue_from_epoch -1 --lr 0.001

# VGG38 BN RC 1e-2
python pytorch_mlp_framework/train_evaluate_image_classification_system.py --batch_size 100 --seed 0 --num_filters 32 --num_stages 3 --num_blocks_per_stage 5 --experiment_name VGG_38_BN_RC_1e-2_experiment --use_gpu True --num_classes 100 --block_type conv_BN_RC_block --continue_from_epoch -1 --lr 0.01