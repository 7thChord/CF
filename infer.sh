CUDA_VISIBLE_DEVICES="1" python3 inference.py \
  --config_path configs/causal_forcing_dmd_chunkwise.yaml \
  --output_folder videos/causal_forcing \
  --checkpoint_path /mnt/ddn/zhenhao_team/models/Causal-Forcing/chunkwise/causal_forcing.pt \
  --data_path prompts/random_sample.txt

# note: SF needs use_ema to work, and the CF config is the same one used for SF for inference
CUDA_VISIBLE_DEVICES="1" python3 inference.py \
  --config_path configs/causal_forcing_dmd_chunkwise.yaml \
  --output_folder videos/self_forcing \
  --checkpoint_path /mnt/ddn/zhenhao_team/models/Self-Forcing/checkpoints/self_forcing_dmd.pt \
  --data_path prompts/random_sample.txt \
  --use_ema
