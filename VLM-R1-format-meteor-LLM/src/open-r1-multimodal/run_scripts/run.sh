conda activate r1-v


tmux new -s 3B_3k

bash src/open-r1-multimodal/run_scripts/run_grpo_nig_lora_3B_3k.sh > run_grpo_nig_lora_3B_3k.log 2>&1 &