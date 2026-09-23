# for seed in 0 ; do
#     echo $seed
#     CUDA_VISIBLE_DEVICES="0" python inference_vbench.py \
#         --config_path configs/self_forcing_dmd_long.yaml \
#         --output_folder vbench_long/self_forcing_dmd/sf_cf_extended/ \
#         --checkpoint_path /mnt/ddn/zhenhao_team/models/Self-Forcing/checkpoints/self_forcing_dmd.pt \
#         --data_path /shared/john/VBench/custom_small_vbench_set/sf_cf_prompts_extended.txt \
#         --num_output_frames 123 \
#         --use_ema \
#         --seed $seed \
#         --save_prompt_name_x_prefix_idx 150
#     date
# done

seed=0
device="7"

CUDA_VISIBLE_DEVICES="${device}" python inference_vbench.py   \
    --config_path configs/self_forcing_dmd_long.yaml \
    --output_folder long_video/self_forcing_dmd/sink_0_window_21/random_sample_plus_patent_prompts/ \
    --checkpoint_path /mnt/ddn/zhenhao_team/models/Self-Forcing/checkpoints/self_forcing_dmd.pt \
    --data_path /shared/john/AR_dawood/AR/prompts/random_sample_plus_patent_prompts.txt \
    --num_output_frames 123 \
    --use_ema \
    --seed $seed \
    --save_prompt_name_x_prefix_idx 150
date

CUDA_VISIBLE_DEVICES="${device}" python inference_vbench.py   \
    --config_path configs/causal_forcing_dmd_chunkwise_long.yaml \
    --output_folder long_video/causal_forcing/sink_0_window_21/random_sample_plus_patent_prompts/ \
    --checkpoint_path /mnt/ddn/zhenhao_team/models/Causal-Forcing/chunkwise/causal_forcing.pt \
    --data_path /shared/john/AR_dawood/AR/prompts/random_sample_plus_patent_prompts.txt \
    --num_output_frames 123 \
    --seed $seed \
    --save_prompt_name_x_prefix_idx 150
date



# for seed in 0 1 ; do # 0 1 2 3 4 ; do
#     echo $seed
#     CUDA_VISIBLE_DEVICES="1" python inference_vbench.py \
#         --config_path configs/causal_forcing_dmd_chunkwise.yaml \
#         --output_folder vbench/causal_forcing_dmd/sf_cf_extended/ \
#         --checkpoint_path /mnt/ddn/zhenhao_team/models/Causal-Forcing/chunkwise/causal_forcing.pt \
#         --data_path /shared/john/VBench/custom_small_vbench_set/sf_cf_prompts_extended.txt \
#         --seed $seed \
#         --save_prompt_name_x_prefix_idx 150
#     date
# done

# CUDA_VISIBLE_DEVICES="5" python inference_vbench.py \
#     --config_path configs/causal_forcing_dmd_chunkwise.yaml \
#     --output_folder vbench/causal_forcing_dmd/sf_cf_extended/ \
#     --checkpoint_path /mnt/ddn/zhenhao_team/models/Causal-Forcing/chunkwise/causal_forcing.pt \
#     --data_path /shared/john/VBench/custom_small_vbench_set/sf_cf_prompts_extended.txt \
#     --use_ema \
#     --seed $seed \
#     --save_prompt_name_x_prefix_idx 150



# CUDA_VISIBLE_DEVICES="5" python inference_vbench.py \
#     --config_path configs/self_forcing_dmd.yaml \
#     --output_folder videos/vbench_self_forcing_dmd/ \
#     --checkpoint_path /mnt/ddn/zhenhao_team/models/Self-Forcing/checkpoints/self_forcing_dmd.pt \
#     --data_path prompts/vbench/all_dimension.txt \
#     --extended_prompt_path prompts/vbench/all_dimension_extended.txt \
#     --use_ema \
#     --seed 42

# CUDA_VISIBLE_DEVICES="5" python inference_vbench.py \
#     --config_path configs/causal_forcing_dmd_chunkwise.yaml \
#     --output_folder videos/vbench_causal_forcing_dmd/ \
#     --checkpoint_path /mnt/ddn/zhenhao_team/models/Causal-Forcing/chunkwise/causal_forcing.pt \
#     --data_path prompts/vbench/all_dimension.txt \
#     --extended_prompt_path prompts/vbench/all_dimension_extended.txt \
#     --seed 42
