# for seed in 0 1 2 3 4 ; do
#     echo $seed
    # CUDA_VISIBLE_DEVICES="5" python inference_vbench.py \
    #     --config_path configs/self_forcing_dmd.yaml \
    #     --output_folder vbench/self_forcing_dmd/sf_cf_extended/ \
    #     --checkpoint_path /mnt/ddn/zhenhao_team/models/Self-Forcing/checkpoints/self_forcing_dmd.pt \
    #     --data_path /shared/john/VBench/custom_small_vbench_set/sf_cf_prompts_extended.txt \
    #     --use_ema \
    #     --seed $seed \
    #     --save_prompt_name_x_prefix_idx 150
#     date
# done


for seed in 4 ; do # 0 1 2 3 4 ; do
    echo $seed
    CUDA_VISIBLE_DEVICES="1" python inference_vbench.py \
        --config_path configs/causal_forcing_dmd_chunkwise.yaml \
        --output_folder vbench/causal_forcing_dmd/sf_cf_extended/ \
        --checkpoint_path /mnt/ddn/zhenhao_team/models/Causal-Forcing/chunkwise/causal_forcing.pt \
        --data_path /shared/john/VBench/custom_small_vbench_set/sf_cf_prompts_extended.txt \
        --seed $seed \
        --save_prompt_name_x_prefix_idx 150
    date
done

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



CUDA_VISIBLE_DEVICES="6" python inference_vbench.py \
    --config_path configs/self_forcing_dmd.yaml \
    --output_folder /mnt/ddn/john/eval_training_infers/self_forcing/ \
    --checkpoint_path /mnt/ddn/zhenhao_team/models/Self-Forcing/checkpoints/self_forcing_dmd.pt \
    --data_path prompts/vidprom_filtered_extended.txt \
    --use_ema \
    --seed $seed \
    --save_prompt_name_x_prefix_idx 150


CUDA_VISIBLE_DEVICES="2" python inference_vbench.py \
    --config_path configs/causal_forcing_dmd_chunkwise.yaml \
    --output_folder videos/causal_forcing/patent_prompts \
    --checkpoint_path /mnt/ddn/zhenhao_team/models/Causal-Forcing/chunkwise/causal_forcing.pt \
    --data_path /mnt/ddn/john/AR_dawood/AR/prompts/patent_prompts.txt \
    --seed 0 \
    --save_prompt_name_x_prefix_idx 150
