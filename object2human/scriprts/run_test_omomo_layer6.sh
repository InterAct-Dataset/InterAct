python trainer_marker_manip_diffusion.py \
--window=120 \
--batch_size=64 \
--project="./omomo_runs" \
--exp_name="omomo_layer4_kv256_dim512" \
--wandb_pj_name="omomo" \
--entity="your wandb account" \
--data_root_folder="../data" \
--use_object_split \
--for_quant_eval \
--test_sample_res \
--n_dec_layers 6 \
--d_k 256 \
--d_v 256 \
--d_model 512
